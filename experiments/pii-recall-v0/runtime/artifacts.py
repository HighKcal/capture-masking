"""Serialize standard integration analyses and COORD02 mask records locally."""

import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import time
import uuid

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
EXPERIMENTS = EXPERIMENT.parent
sys.path.insert(0, str(EXPERIMENTS / 'ocr-integration-v0'))
import integration as api

sys.path.insert(0, str(EXPERIMENTS / 'ocr-coordinate-refinement-v1'))
import mask_method


MASK_METHOD = 'segment_hull_caps_v1'
RENDER_SCHEMA = 'ocr-coordinate-refinement-render-v1'
read, sha, digest = api.read, api.sha, api.digest


def output_path(path):
    path = Path(path).resolve()
    if path == EXPERIMENT or not path.is_relative_to(EXPERIMENT):
        raise ValueError('Output must be below the PII-RECALL01 experiment')
    return path


def write_json(path, payload):
    path = output_path(path)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    serialized.encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(serialized)


def failure(stage, error):
    return dict(status='failure', state='incomplete', stage=stage,
                type=type(error).__name__, message=str(error))


def runtime_versions(producer):
    return dict(dependencies=api.source_hashes(),
                integration_sources={p.name: sha(p) for p in api.HERE.glob('*.py')},
                producer=producer)


def execution_environment():
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages={name: metadata.version(name) for name in (
                    'rapidocr', 'onnxruntime', 'Pillow', 'numpy', 'opencv-python-headless')})


def save_analysis(directory, shared, detection, relations, arm, versions, environment):
    """Rebind the original OCR and final candidates before returning a usable analysis."""
    directory = output_path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    ocr, case, identity, detail, baseline, rgb = (
        shared[name] for name in ('ocr', 'case', 'identity', 'detail', 'baseline', 'rgb'))
    api.verify_detection(case, baseline)
    api.verify_detection(case, detection)
    if (relations['image_id'] != identity['image_id']
            or relations['ocr_run_id'] != identity['ocr_run_id']
            or relations['ocr_sha256'] != identity['ocr_sha256']):
        raise ValueError('Relations belong to another image or OCR result')
    final = detection['configs']['FULL']
    if any(not isinstance(p['id'], str) or not p['id'] for p in final):
        raise ValueError('Candidate IDs must be nonempty strings')
    refined = api.refine.refine_candidates(rgb, ocr, identity, final, detail)
    timing = dict(shared['timing'], refinement_seconds=time.perf_counter() - start)
    blobs = {name: shared[name] for name in (
        'ocr', 'baseline', 'detail', 'source', 'case', 'transforms')}
    blobs.update(detection=detection, relations=relations, refined=refined)
    files = {}
    for name, value in blobs.items():
        filename = name + '.json'
        write_json(directory / filename, value)
        files[filename] = sha(directory / filename)
    status = api.aggregate_status(ocr, baseline, detection, relations, detail, refined)
    analysis = dict(schema=api.ANALYSIS_SCHEMA, analysis_version=1,
                    analysis_id='analysis-' + uuid.uuid4().hex, identity=identity,
                    source=shared['source'], candidate_version=digest(final), statuses=status,
                    files=files, **versions, arm=arm,
                    execution=dict(**environment, **shared['execution'], timing=timing))
    analysis['analysis_digest'] = digest(analysis)
    write_json(directory / 'analysis.json', analysis)
    return api.load_analysis(directory)


def render_all(directory, analysis, blobs, rgb, fallback):
    """Select every current candidate and apply the same frozen method to each arm."""
    directory = output_path(directory)
    start = time.perf_counter()
    candidates = blobs['refined']['candidates']
    ids = [candidate['id'] for candidate in candidates]
    selection = api.make_selection(analysis, candidates, ids)
    api.validate_selection(analysis, candidates, selection)
    union = np.zeros(rgb.size[::-1], dtype=bool)
    statuses, geometry, substitutions = {}, {}, []
    for candidate in candidates:
        candidate_id = candidate['id']
        mask, applied = mask_method.candidate_mask(candidate, rgb.size, MASK_METHOD, fallback)
        if mask.dtype != bool or mask.shape != union.shape:
            raise ValueError('Candidate mask must match the original raster and be boolean')
        union |= mask
        statuses[candidate_id] = candidate['status']
        geometry[candidate_id] = mask_method.geometry(candidate, rgb.size, MASK_METHOD, fallback)
        for item in applied:
            segment = candidate['segment_results'][item['segment_index']]
            polygons = segment['coarse_polygons']
            substitutions.append(dict(item, candidate_id=candidate_id, segment=segment['segment'],
                                      polygons=polygons,
                                      region_pixels=int(api.support.raster(polygons, rgb.size).sum())))
    masked = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'selection.json', selection)
    verification = api.link.save_masked(rgb, union, directory)
    settings = dict(method=MASK_METHOD, algorithm=mask_method.settings(MASK_METHOD),
                    fallback=fallback, opaque_RGB=[0, 0, 0], comparison_only=True)
    sources = mask_method.method_source()
    sources.update({str(Path(module.__file__).resolve()): sha(module.__file__)
                    for module in (sys.modules[__name__], api, api.link)})
    render_id = digest(dict(schema=RENDER_SCHEMA, selection_id=selection['selection_id'],
                            settings=settings, render_sources=sources))
    selection_state = ('selected_candidates' if candidates else
                       'no_candidates' if analysis['statuses']['status'] == 'ok' else
                       'incomplete_no_candidates')
    record = dict(schema=RENDER_SCHEMA, render_version=1, render_id=render_id,
                  **{k: analysis[k] for k in ('analysis_id', 'analysis_digest',
                                              'candidate_version', 'identity')},
                  selection=selection, selected_ids=ids, method=MASK_METHOD, fallback=fallback,
                  settings=settings, render_sources=sources, analysis_statuses=analysis['statuses'],
                  selected_coordinate_statuses=statuses, selected_segment_geometry=geometry,
                  selection_state=selection_state, fallback_segments=substitutions,
                  unselected_mask_overlap=[], pixel_verification=verification,
                  mask_pixels=int(union.sum()), inference_performed=False,
                  timing=dict(mask_seconds=masked - start,
                              png_write_verify_seconds=time.perf_counter() - masked),
                  files={name: sha(directory / name) for name in ('mask.png', 'masked.png')})
    write_json(directory / 'render.json', record)
    return record
