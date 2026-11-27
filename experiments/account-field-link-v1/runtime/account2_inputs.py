"""Admit local images and bind previous evidence before using OCR/NER/detail."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

from account2_output import (ACCOUNT01, ANALYSIS_FILES, CONFIRMATION, MANIFEST, RECALL, SHARED, api,
                             confined, digest, failure, load, read, sha)


legacy = load('_account01_inputs', ACCOUNT01 / 'runtime/inputs.py')
OCR_CONFIGURATION = legacy.OCR_CONFIGURATION
PRIOR_METHOD = 'fe0bf123287d5ae33c419ef88d1a6647e9f34ff744cd8c32f3643077b7b9ad7e'
PRIOR_FINAL = '43ec790e45476af3890d5cbbccfd55d999a0e9c2c8e4d20fdc23e89e20bba3ae'
PRIOR_PREDICTIONS = '4d97721415f5f8bb67c705706e7f5484fd0e3f9567d26238946a7eef2955eb6d'
RECALL_FINAL = 'bae0933dc470e6bd6c7651b4c490933347e31c817f87bb3ca987d76cb7a39f2c'
RECALL_PREDICTIONS = 'e3a24dd7b90dad0927c3b98d94147271275bb3904f5303aa3a0dac2345d8f4f1'


def pinned(path, expected):
    if path.resolve() != path or sha(path) != expected:
        raise ValueError('Frozen file differs: ' + str(path))
    return path


def prior_method(used_sources):
    path = pinned(ACCOUNT01 / 'METHOD_FREEZE.json', PRIOR_METHOD)
    previous = read(path)['files']
    if any(previous.get(str(source)) != sha(source) for source in used_sources):
        raise ValueError('A reused ACCOUNT-01 source differs from its frozen method')
    return dict(path=str(path), sha256=PRIOR_METHOD)


def ordinary(path):
    lexical = Path(os.path.abspath(path))
    if lexical.is_relative_to(CONFIRMATION):
        raise ValueError('Confirmation input requires the frozen manifest route')
    resolved = legacy.local_path(path)
    if resolved.is_relative_to(CONFIRMATION):
        raise ValueError('Confirmation input requires the frozen manifest route')
    return resolved


def confirmation_inputs(path, verify_method):
    if Path(os.path.abspath(path)) != MANIFEST:
        raise ValueError('Use the exact ACCOUNT-02 confirmation manifest')
    proof = verify_method()
    freeze_path, verifier = CONFIRMATION / 'FREEZE.json', CONFIRMATION / 'verify.py'
    if any(legacy.local_path(file) != file for file in (freeze_path, verifier, MANIFEST)):
        raise ValueError('Confirmation file was redirected')
    subprocess.run([sys.executable, '-B', '-X', 'utf8', str(verifier)], cwd=CONFIRMATION,
                   check=True, capture_output=True, timeout=30)
    freeze = read(freeze_path)
    if freeze['dataset_version'] != 'account-confirmation-v2.0.0' or freeze['images'] != 12 or freeze['families'] != 12:
        raise ValueError('Unexpected ACCOUNT-02 confirmation dataset version or size')
    manifest_hash = freeze['input_manifest_sha256']
    payload = MANIFEST.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest_hash:
        raise ValueError('Confirmation manifest differs from the public commitment')
    rows = [json.loads(line) for line in payload.decode('utf-8').splitlines()]
    required = {'schema_version', 'image_id', 'image_path', 'width', 'height', 'split', 'family_id', 'sha256'}
    seen = set()
    for row in rows:
        if (set(row) != required or row['schema_version'] != 'ocr-input-v0' or row['split'] != 'eval'
                or not legacy.SAFE_ID.fullmatch(row['image_id']) or row['image_id'] in seen
                or not legacy.SAFE_ID.fullmatch(row['family_id']) or not legacy.SHA256.fullmatch(row['sha256'])
                or any(type(row[key]) is not int or row[key] <= 0 for key in ('width', 'height'))):
            raise ValueError('Invalid confirmation manifest row')
        relative = Path(row['image_path'])
        if (len(relative.parts) != 2 or relative.parts[0] != 'images' or relative.suffix != '.png'
                or not legacy.SAFE_ID.fullmatch(relative.stem)):
            raise ValueError('Confirmation image must be a PNG directly under images/')
        image = MANIFEST.parent / relative
        if legacy.local_path(image) != image:
            raise ValueError('Confirmation image was redirected')
        row.update(_path=image, _manifest=str(MANIFEST))
        seen.add(row['image_id'])
    if (len(rows) != freeze['images'] or len({row['sha256'] for row in rows}) != freeze['images']
            or len({row['family_id'] for row in rows}) != freeze['families']):
        raise ValueError('Confirmation count differs from the public commitment')
    return rows, dict(path=str(MANIFEST), sha256=manifest_hash, split='eval',
                      dataset_version=freeze['dataset_version'], freeze_path=str(freeze_path),
                      freeze_sha256=sha(freeze_path), public_verifier_sha256=sha(verifier)), proof


def reuse_source(directory):
    directory = ordinary(directory)
    for name in ANALYSIS_FILES:
        path = directory / name
        if ordinary(path) != path:
            raise ValueError('Saved analysis artifact was redirected')
    if not directory.is_relative_to(ACCOUNT01):
        if directory.is_relative_to(RECALL):
            pinned(RECALL / 'PREDICTIONS_FREEZE.json', RECALL_PREDICTIONS)
            pinned(RECALL / 'final-files.json', RECALL_FINAL)
        ordinary(read(directory / 'source.json')['input_path'])
        return legacy.load_reused(directory)

    seal_path = pinned(ACCOUNT01 / 'PREDICTIONS_FREEZE.json', PRIOR_PREDICTIONS)
    seal = read(seal_path)
    if directory.is_relative_to(Path(seal['run']).parent):
        files, key = seal['files'], str
    else:
        seal_path = pinned(ACCOUNT01 / 'final-files.json', PRIOR_FINAL)
        files = read(seal_path)['files']
        key = lambda path: path.relative_to(ACCOUNT01).as_posix()
    for name in ANALYSIS_FILES:
        path = directory / name
        pinned(path, files.get(key(path)))
    ordinary(read(directory / 'source.json')['input_path'])
    analysis, blobs, rgb = api.load_analysis(directory)
    if (analysis['arm'] != 'improved' or analysis['producer']['name'] != 'account-field-link-v0/runtime/run.py'
            or analysis['producer']['injected_test_stage'] or 'ACCOUNT' not in blobs['detection']['config_statuses']
            or 'account_stage' not in blobs['relations']):
        raise ValueError('Reuse the final ACCOUNT-01 improved analysis')
    detail_engine = blobs['detail'].get('settings', {}).get('baseline_model_settings_digest')
    if detail_engine is not None and detail_engine != digest(blobs['ocr']['engine']):
        raise ValueError('Saved detail and OCR settings differ')
    identity, ocr, source = analysis['identity'], blobs['ocr'], blobs['source']
    if not legacy.SAFE_ID.fullmatch(identity['image_id']):
        raise ValueError('Unsafe image ID')
    row = dict(ocr['input'], image_id=identity['image_id'], _path=Path(source['input_path']))
    reused = dict(directory=str(directory), analysis_id=analysis['analysis_id'],
                  analysis_digest=analysis['analysis_digest'], analysis_sha256=sha(directory / 'analysis.json'),
                  files={name + '.json': analysis['files'][name + '.json'] for name in SHARED},
                  prior_seal=dict(path=str(seal_path), sha256=sha(seal_path)))
    return dict({name: blobs[name] for name in SHARED}, metadata=legacy.metadata(row), identity=identity,
                rgb=rgb, timing={}, expected_account=(blobs['detection'], blobs['relations']),
                execution=dict(OCR='reused', NER='reused', detail='reused', reused_analysis=reused))


def prepare_reused(paths):
    seen = set()
    for index, path in enumerate(paths, 1):
        start = time.perf_counter()
        try:
            item = reuse_source(path)
            image_id = item['identity']['image_id']
            if image_id in seen:
                raise ValueError('Duplicate image identity in the reuse batch')
            seen.add(image_id)
        except Exception as error:
            item = legacy.unavailable_input(index, 'load reused analysis', error)
        item['timing']['reuse_load_seconds'] = time.perf_counter() - start
        yield item


@contextmanager
def local_caches(directory):
    directory = confined(directory)
    directory.mkdir(exist_ok=False)
    cache_names = ('HF_HOME', 'HF_HUB_CACHE', 'HF_ASSETS_CACHE', 'HF_XET_CACHE', 'HF_MODULES_CACHE',
                   'TRANSFORMERS_CACHE', 'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
                   'CUDA_CACHE_PATH', 'XDG_CACHE_HOME')
    env = {name: str(directory / name.lower()) for name in cache_names}
    env.update(TEMP=str(directory), TMP=str(directory), TMPDIR=str(directory), PYTHONDONTWRITEBYTECODE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    previous, temp = {name: os.environ.get(name) for name in env}, tempfile.tempdir
    os.environ.update(env)
    tempfile.tempdir = str(directory)
    try:
        yield
    finally:
        tempfile.tempdir = temp
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _decode(specs, from_manifest):
    seen = set()
    for index, spec in enumerate(specs, 1):
        start = time.perf_counter()
        try:
            if from_manifest:
                row = spec
                if legacy.local_path(row['_path']) != row['_path']:
                    raise ValueError('Confirmation source image was redirected')
                rgb, transforms = api.support.decode(row)
            else:
                row, rgb, transforms = api.decode_file(ordinary(spec))
            if row['image_id'] in seen:
                raise ValueError('Duplicate image identity in the fresh batch')
            seen.add(row['image_id'])
            item = dict(metadata=legacy.metadata(row), row=row, rgb=rgb, transforms=transforms, timing={})
        except Exception as error:
            item = legacy.unavailable_input(index, 'decode', error)
            if from_manifest:
                item['metadata'] = legacy.metadata(spec)
        item['timing']['decode_seconds'] = time.perf_counter() - start
        yield item


def _ocr(item, adapter, run_id):
    start = time.perf_counter()
    try:
        raw = adapter.infer(item['rgb'])
    except Exception as error:
        raw = dict(status='failure', units=[], errors=[failure('OCR', error)])
    item['timing']['OCR_inference_seconds'] = time.perf_counter() - start
    item['native'] = raw.get('native', {})
    item['transforms'].extend(raw.get('transforms', []))
    row = item['row']
    ocr = api.support.finalize(raw, row, run_id, adapter.metadata)
    identity, case = api.link.bind_ocr(ocr, row, item['transforms'], item['rgb'], run_id)
    item.update(ocr=ocr, identity=identity, case=case,
                source=dict(input_path=str(row['_path']), identity=identity, input_manifest=row['_manifest']))


def prepare_fresh(specs, target, from_manifest):
    items = list(_decode(specs, from_manifest))
    ready = [item for item in items if 'failure' not in item]
    run_id, timing = 'rapid-account02-' + uuid.uuid4().hex, {}
    if not ready:
        return items, run_id, timing
    stage, start = 'OCR_preparation', time.perf_counter()
    try:
        from adapters.rapid import Adapter
        adapter = Adapter(api.support.ASSETS)
        if adapter.metadata != read(OCR_CONFIGURATION)['metadata']:
            raise ValueError('OCR settings differ from the pinned configuration')
        timing[stage + '_seconds'] = time.perf_counter() - start
        for item in ready:
            try:
                _ocr(item, adapter, run_id)
            except Exception as error:
                item['failure'] = failure('OCR finalization', error)
        ready = [item for item in ready if 'failure' not in item]
        if not ready:
            return items, run_id, timing
        stage, start = 'NER_batch', time.perf_counter()
        worker = confined(target / 'ner-baseline')
        worker.mkdir(exist_ok=False)
        baseline = api.baseline_runner.ner([item['case'] for item in ready], worker)
        timing[stage + '_seconds'] = time.perf_counter() - start
    except Exception as error:
        timing[stage + '_seconds'] = time.perf_counter() - start
        for item in ready:
            item['failure'] = failure(stage, error)
        return items, run_id, timing

    start = time.perf_counter()
    extractor, preparation_error = None, None
    try:
        extractor = api.detail_adapter.DetailExtractor(api.support.ASSETS)
    except Exception as error:
        preparation_error = failure('detail preparation', error)
    timing['detail_preparation_seconds'] = time.perf_counter() - start
    for item in ready:
        start = time.perf_counter()
        identity, detail_error = item['identity'], preparation_error
        if extractor is not None:
            try:
                item['detail'] = extractor.extract(item['rgb'], item['ocr'], identity)
            except Exception as error:
                detail_error = failure('detail OCR', error)
        if detail_error is not None:
            item['detail'] = dict(schema_version='ocr-coordinate-details-v0', identity=identity, status='failure',
                                  units=[], reasons=[detail_error['type']], integration_error=detail_error)
        item['timing']['detail_extract_seconds'] = time.perf_counter() - start
        item['baseline'] = baseline[item['case']['id']]
        item['execution'] = dict(OCR='fresh', NER='fresh', detail='fresh', reused_analysis=None)
    return items, run_id, timing
