"""Bind standard analyses and save verified segment-hull masks inside this runtime."""

import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
EXPERIMENTS = EXPERIMENT.parent
RECALL = EXPERIMENTS / 'pii-recall-v0'
sys.dont_write_bytecode = True
sys.path.insert(0, str(EXPERIMENTS / 'ocr-integration-v0'))
import integration as api
from dependencies import load

previous = load('account_previous_artifacts', RECALL / 'runtime/artifacts.py')
mask_method = previous.mask_method
MASK_METHOD = previous.MASK_METHOD
RENDER_SCHEMA = previous.RENDER_SCHEMA
digest, sha, read = api.digest, api.sha, api.read
failure = previous.failure
SHARED = ('case', 'ocr', 'transforms', 'baseline', 'detail', 'source')
FALLBACKS = {'no': False, 'yes': True}


def output_path(path):
    path = Path(path).resolve()
    if path == HERE or not path.is_relative_to(HERE):
        raise ValueError('Output must be below the ACCOUNT-01 runtime')
    return path


def write_json(path, payload):
    path = output_path(path)
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    data.encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(data)


def file_entry(root, path):
    return dict(path=path.relative_to(root).as_posix(), sha256=sha(path))


def save_shared(root, directory, item):
    index = dict(files={})
    for name in SHARED:
        if name not in item:
            continue
        path = directory / (name + '.json')
        write_json(path, item[name])
        index['files'][path.name] = file_entry(root, path)
    native = directory / 'native.json'
    if native.is_file():
        index['files'][native.name] = file_entry(root, native)
    for name in ('ocr', 'baseline', 'detail'):
        if name in item:
            index[name + '_sha256'] = digest(item[name])
    if 'case' in item:
        index['case_id'] = item['case']['id']
    return index


def verify_relation_binding(identity, relations):
    for key in ('image_id', 'ocr_run_id', 'ocr_sha256'):
        if relations[key] != identity[key]:
            raise ValueError('Relations belong to another image or OCR result')


def save_analysis(directory, item, detection, relations, arm, versions, environment):
    directory = output_path(directory)
    started = time.perf_counter()
    ocr, case, identity, detail, baseline, rgb = (
        item[name] for name in ('ocr', 'case', 'identity', 'detail', 'baseline', 'rgb'))
    api.verify_detection(case, detection)
    verify_relation_binding(identity, relations)
    final = detection['configs']['FULL']
    if any(not isinstance(candidate['id'], str) or not candidate['id'] for candidate in final):
        raise ValueError('Candidate IDs must be nonempty strings')
    refined = api.refine.refine_candidates(rgb, ocr, identity, final, detail)
    timing = dict(item['timing'], refinement_seconds=time.perf_counter() - started)
    blobs = {name: item[name] for name in SHARED}
    blobs.update(detection=detection, relations=relations, refined=refined)
    directory.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, payload in blobs.items():
        filename = name + '.json'
        write_json(directory / filename, payload)
        files[filename] = sha(directory / filename)
    statuses = api.aggregate_status(ocr, baseline, detection, relations, detail, refined)
    analysis = dict(schema=api.ANALYSIS_SCHEMA, analysis_version=1,
                    analysis_id='analysis-' + uuid.uuid4().hex, identity=identity,
                    source=item['source'], candidate_version=digest(final), statuses=statuses,
                    files=files, **versions, arm=arm,
                    execution=dict(**environment, **item['execution'], timing=timing))
    analysis['analysis_digest'] = digest(analysis)
    write_json(directory / 'analysis.json', analysis)
    return api.load_analysis(directory)


def load_own_analysis(directory):
    """Use the existing reader and additionally bind the new producer source hashes."""
    directory = output_path(directory)
    paths = tuple(directory / (name + '.json') for name in (*SHARED, 'analysis', 'detection', 'relations', 'refined'))
    if any(path.resolve() != path for path in paths):
        raise ValueError('Saved analysis artifact was redirected')
    analysis = read(directory / 'analysis.json')
    producer = analysis.get('producer', {})
    if (producer.get('name') != 'account-field-link-v0/runtime/run.py'
            or producer.get('mask_method') != MASK_METHOD or not producer.get('files')
            or any(sha(path) != expected for path, expected in producer['files'].items())):
        raise ValueError('Account analysis producer differs from saved sources')
    source = read(directory / 'source.json')
    confirmation = EXPERIMENT / 'confirmation'
    input_path = Path(source['input_path']).resolve()
    if input_path.is_relative_to(confirmation):
        allowed = confirmation / 'sealed/eval'
        if (input_path.parent != allowed / 'images' or source['input_manifest'] != str(allowed / 'manifest.jsonl')
                or not producer.get('method_freeze')
                or producer['method_freeze'] != load('account_method_freeze', EXPERIMENT / 'freeze_guard.py').verify_method()):
            raise ValueError('Confirmation analysis is missing the frozen prediction provenance')
    analysis, blobs, rgb = api.load_analysis(directory)
    verify_relation_binding(analysis['identity'], blobs['relations'])
    return analysis, blobs, rgb


def render(directory, analysis, blobs, rgb, selection, fallback):
    directory = output_path(directory)
    if type(fallback) is not bool:
        raise ValueError('Explicit fallback is required')
    candidates = blobs['refined']['candidates']
    api.validate_selection(analysis, candidates, selection)
    started = time.perf_counter()
    ids = selection['selected_ids']
    selected = set(ids)
    union = np.zeros(rgb.size[::-1], dtype=bool)
    statuses, geometry, substitutions, unselected = {}, {}, [], {}
    for candidate in candidates:
        candidate_id = candidate['id']
        mask, applied = mask_method.candidate_mask(candidate, rgb.size, MASK_METHOD, fallback)
        if mask.dtype != bool or mask.shape != union.shape:
            raise ValueError('Candidate mask must match the original raster and be boolean')
        if candidate_id not in selected:
            if selected:
                unselected[candidate_id] = mask
            continue
        union |= mask
        statuses[candidate_id] = candidate['status']
        geometry[candidate_id] = mask_method.geometry(candidate, rgb.size, MASK_METHOD, fallback)
        for item in applied:
            segment = candidate['segment_results'][item['segment_index']]
            polygons = segment['coarse_polygons']
            substitutions.append(dict(item, candidate_id=candidate_id, segment=segment['segment'],
                                      polygons=polygons,
                                      region_pixels=int(api.support.raster(polygons, rgb.size).sum())))
    overlap = []
    for candidate_id, mask in unselected.items():
        count = int((mask & union).sum())
        if count:
            overlap.append(dict(candidate_id=candidate_id, overlap_pixels=count,
                                meaning='overlap with unselected candidate mask; glyph coverage unproven'))
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
    selection_state = ('selected_candidates' if selected else 'user_selected_none' if candidates else
                       'no_candidates' if analysis['statuses']['status'] == 'ok' else
                       'incomplete_no_candidates')
    record = dict(schema=RENDER_SCHEMA, render_version=1, render_id=render_id,
                  **{key: analysis[key] for key in ('analysis_id', 'analysis_digest',
                                                    'candidate_version', 'identity')},
                  selection=selection, selected_ids=ids, method=MASK_METHOD, fallback=fallback,
                  settings=settings, render_sources=sources, analysis_statuses=analysis['statuses'],
                  selected_coordinate_statuses=statuses, selected_segment_geometry=geometry,
                  selection_state=selection_state, fallback_segments=substitutions,
                  unselected_mask_overlap=overlap, pixel_verification=verification,
                  mask_pixels=int(union.sum()), inference_performed=False,
                  timing=dict(mask_seconds=masked - started,
                              png_write_verify_seconds=time.perf_counter() - masked),
                  files={name: sha(directory / name) for name in ('mask.png', 'masked.png')})
    write_json(directory / 'render.json', record)
    return record
