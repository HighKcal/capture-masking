"""Write standard integration records and verified PNGs inside ACCOUNT-02."""

import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np


HERE = Path(__file__).resolve().parent
TASK = HERE.parent
EXPERIMENTS = TASK.parent
ACCOUNT01 = EXPERIMENTS / 'account-field-link-v0'
RECALL = EXPERIMENTS / 'pii-recall-v0'
CONFIRMATION = TASK / 'confirmation'
MANIFEST = CONFIRMATION / 'sealed/eval/manifest.jsonl'
PRODUCER = 'account-field-link-v1/runtime/run.py'
SCHEMA = 'account-field-run-v1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(EXPERIMENTS / 'ocr-integration-v0'))
from dependencies import load

old = load('artifacts', ACCOUNT01 / 'runtime/artifacts.py')
api, mm = old.api, old.mask_method
sha, digest, read, failure = old.sha, old.digest, old.read, old.failure
SHARED, FALLBACKS, METHOD, RENDER_SCHEMA = old.SHARED, old.FALLBACKS, old.MASK_METHOD, old.RENDER_SCHEMA
ANALYSIS_FILES = tuple(name + '.json' for name in (*SHARED, 'analysis', 'detection', 'relations', 'refined'))
PREVIOUS_SOURCES = (ACCOUNT01 / 'account_stage.py', ACCOUNT01 / 'runtime/artifacts.py',
                    ACCOUNT01 / 'runtime/inputs.py', RECALL / 'improvement.py', RECALL / 'runtime/artifacts.py',
                    Path(api.__file__).resolve(), api.HERE / 'dependencies.py', Path(mm.__file__).resolve(),
                    api.LINK / 'results/evaluation/ocr-run.json')
PRODUCER_SOURCES = (*PREVIOUS_SOURCES, HERE / 'run.py', HERE / 'account2_inputs.py', HERE / 'account2_output.py',
                    TASK / 'account_joint.py', TASK / 'joint_geometry.py', TASK / 'freeze_guard.py')


def confined(path):
    path = Path(path).resolve()
    if path == HERE or not path.is_relative_to(HERE):
        raise ValueError('Write output below the ACCOUNT-02 runtime')
    return path


def write_text(path, value):
    path = confined(path)
    value.encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(value)


def write(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def file_entry(root, path):
    return dict(path=path.relative_to(root).as_posix(), sha256=sha(path))


def save_shared(root, directory, item):
    files = {}
    for name in SHARED + ('native',):
        if name in item:
            path = directory / (name + '.json')
            write(path, item[name])
            files[path.name] = file_entry(root, path)
    fingerprints = {key + '_sha256': digest(item[key]) for key in ('ocr', 'baseline', 'detail') if key in item}
    if 'case' in item:
        fingerprints['case_id'] = item['case']['id']
    return dict(files=files, **fingerprints)


def save_analysis(directory, item, detection, relations, arm, versions, environment):
    directory = confined(directory)
    ocr, case, identity, detail, baseline, rgb = (
        item[name] for name in ('ocr', 'case', 'identity', 'detail', 'baseline', 'rgb'))
    api.verify_detection(case, detection)
    old.verify_relation_binding(identity, relations)
    final = detection['configs']['FULL']
    start = time.perf_counter()
    refined = api.refine.refine_candidates(rgb, ocr, identity, final, detail)
    timing = dict(item['timing'], refinement_seconds=time.perf_counter() - start)
    directory.mkdir(parents=True, exist_ok=False)
    blobs = {name: item[name] for name in SHARED}
    blobs.update(detection=detection, relations=relations, refined=refined)
    files = {}
    for name, payload in blobs.items():
        path = directory / (name + '.json')
        write(path, payload)
        files[path.name] = sha(path)
    statuses = api.aggregate_status(ocr, baseline, detection, relations, detail, refined)
    analysis = dict(schema=api.ANALYSIS_SCHEMA, analysis_version=1, arm=arm, files=files,
                    analysis_id='analysis-' + uuid.uuid4().hex, identity=identity,
                    source=item['source'], candidate_version=digest(final), statuses=statuses,
                    execution=dict(**environment, **item['execution'], timing=timing), **versions)
    analysis['analysis_digest'] = digest(analysis)
    write(directory / 'analysis.json', analysis)
    return analysis, blobs


def verify_producer(producer):
    files = producer.get('files', {})
    if (producer.get('name') != PRODUCER or producer.get('mask_method') != METHOD
            or set(files) != {str(path) for path in PRODUCER_SOURCES}):
        raise ValueError('ACCOUNT-02 producer differs from saved source versions')
    for path in PRODUCER_SOURCES:
        if path.resolve() != path or sha(path) != files[str(path)]:
            raise ValueError('ACCOUNT-02 producer differs from saved source versions')


def load_analysis(directory):
    directory = confined(directory)
    if any((directory / name).resolve() != directory / name for name in ANALYSIS_FILES):
        raise ValueError('Saved analysis artifact was redirected')
    manifest = read(directory / 'analysis.json')
    producer = manifest['producer']
    verify_producer(producer)
    source = read(directory / 'source.json')
    path = Path(source['input_path']).resolve()
    if path.is_relative_to(CONFIRMATION):
        guard = load('_account02_freeze', TASK / 'freeze_guard.py')
        if (path.parent != MANIFEST.parent / 'images' or source['input_manifest'] != str(MANIFEST)
                or not producer.get('method_freeze') or producer['method_freeze'] != guard.verify_method()):
            raise ValueError('Confirmation analysis is missing frozen prediction provenance')
    analysis, blobs, rgb = api.load_analysis(directory)
    old.verify_relation_binding(analysis['identity'], blobs['relations'])
    return analysis, blobs, rgb


def render(target, analysis, candidates, rgb, selection, fallback):
    target = confined(target)
    if type(fallback) is not bool:
        raise ValueError('Explicit fallback required')
    api.validate_selection(analysis, candidates, selection)
    start = time.perf_counter()
    ids = set(selection['selected_ids'])
    union = np.zeros(rgb.size[::-1], dtype=bool)
    statuses, geometry, substitutions, unselected = {}, {}, [], {}
    for candidate in candidates:
        cid = candidate['id']
        if not ids:
            break
        mask, applied = mm.candidate_mask(candidate, rgb.size, METHOD, fallback)
        if mask.dtype != bool or mask.shape != union.shape:
            raise ValueError('Mask must be a boolean source-size raster')
        if cid not in ids:
            unselected[cid] = mask
            continue
        union |= mask
        statuses[cid] = candidate['status']
        geometry[cid] = mm.geometry(candidate, rgb.size, METHOD, fallback)
        for entry in applied:
            segment = candidate['segment_results'][entry['segment_index']]
            polygons = segment['coarse_polygons']
            substitutions.append(dict(entry, candidate_id=cid, segment=segment['segment'], polygons=polygons,
                                      region_pixels=int(api.support.raster(polygons, rgb.size).sum())))
    overlap = [dict(candidate_id=cid, overlap_pixels=int((mask & union).sum()),
                    meaning='overlap with unselected candidate mask; glyph coverage unproven')
               for cid, mask in unselected.items() if (mask & union).any()]
    state = ('selected_candidates' if ids else 'user_selected_none' if candidates else
             'no_candidates' if analysis['statuses']['status'] == 'ok' else 'incomplete_no_candidates')
    masked = time.perf_counter()
    target.mkdir(parents=True, exist_ok=False)
    write(target / 'selection.json', selection)
    verification = api.link.save_masked(rgb, union, target)
    settings = dict(method=METHOD, algorithm=mm.settings(METHOD), fallback=fallback,
                    opaque_RGB=[0, 0, 0], comparison_only=True)
    sources = mm.method_source()
    sources.update({str(Path(path).resolve()): sha(path) for path in (__file__, api.__file__, api.link.__file__)})
    render_id = digest(dict(schema=RENDER_SCHEMA, selection_id=selection['selection_id'],
                            settings=settings, render_sources=sources))
    result = dict(schema=RENDER_SCHEMA, render_version=1, render_id=render_id,
                  **{key: analysis[key] for key in ('analysis_id', 'analysis_digest', 'candidate_version', 'identity')},
                  selection=selection, selected_ids=selection['selected_ids'], selection_state=state,
                  method=METHOD, fallback=fallback, settings=settings, render_sources=sources,
                  analysis_statuses=analysis['statuses'], selected_coordinate_statuses=statuses,
                  selected_segment_geometry=geometry, fallback_segments=substitutions, unselected_mask_overlap=overlap,
                  pixel_verification=verification, mask_pixels=int(union.sum()), inference_performed=False,
                  timing=dict(mask_seconds=masked - start, png_write_verify_seconds=time.perf_counter() - masked),
                  files={name: sha(target / name) for name in ('mask.png', 'masked.png')})
    write(target / 'render.json', result)
    return result
