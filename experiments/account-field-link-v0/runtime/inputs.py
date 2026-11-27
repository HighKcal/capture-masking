"""Local image decoding, standard cache reuse, and shared fresh model stages."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid

from artifacts import EXPERIMENT, RECALL, SHARED, api, digest, failure, output_path, read, sha, write_json


CONFIRMATION = EXPERIMENT / 'confirmation'
CONFIRMATION_MANIFEST = CONFIRMATION / 'sealed/eval/manifest.jsonl'
OCR_CONFIGURATION = api.LINK / 'results/evaluation/ocr-run.json'
SAFE_ID = re.compile(r'[A-Za-z0-9_-]+')
SHA256 = re.compile(r'[0-9a-f]{64}')


def local_path(path):
    raw = Path(path)
    if raw.drive.startswith('\\\\'):
        raise ValueError('Local files are required')
    resolved = raw.resolve()
    if resolved.drive.startswith('\\\\'):
        raise ValueError('Local files are required')
    return resolved


def ordinary_input(path):
    if Path(path).absolute().is_relative_to(CONFIRMATION):
        raise ValueError('Confirmation files require the frozen manifest route')
    path = local_path(path)
    if path.is_relative_to(CONFIRMATION):
        raise ValueError('Confirmation files require the frozen manifest route')
    return path


def confirmation_inputs(path, verify_method):
    """Validate the method and public commitment before opening the inference manifest."""
    if Path(os.path.abspath(path)) != CONFIRMATION_MANIFEST:
        raise ValueError('Provide the exact ACCOUNT-01 confirmation manifest')
    proof = verify_method()
    verifier = CONFIRMATION / 'verify.py'
    freeze_path = CONFIRMATION / 'FREEZE.json'
    if any(local_path(file) != file for file in (verifier, freeze_path, CONFIRMATION_MANIFEST)):
        raise ValueError('Confirmation artifact was redirected')
    subprocess.run([sys.executable, '-B', '-X', 'utf8', str(verifier)], cwd=CONFIRMATION,
                   check=True, capture_output=True, timeout=30)
    freeze = read(freeze_path)
    payload = CONFIRMATION_MANIFEST.read_bytes()
    manifest_hash = hashlib.sha256(payload).hexdigest()
    if freeze['input_manifest_sha256'] != manifest_hash:
        raise ValueError('Confirmation manifest differs from public freeze')
    rows = [json.loads(line) for line in payload.decode('utf-8').splitlines()]
    required = {'schema_version', 'image_id', 'image_path', 'width', 'height', 'split', 'family_id', 'sha256'}
    seen = set()
    for row in rows:
        if (set(row) != required or row['schema_version'] != 'ocr-input-v0' or row['split'] != 'eval'
                or not SAFE_ID.fullmatch(row['image_id']) or row['image_id'] in seen
                or not SAFE_ID.fullmatch(row['family_id']) or not SHA256.fullmatch(row['sha256'])
                or any(type(row[key]) is not int or row[key] <= 0 for key in ('width', 'height'))):
            raise ValueError('Invalid confirmation manifest row')
        relative = Path(row['image_path'])
        if (len(relative.parts) != 2 or relative.parts[0] != 'images'
                or relative.suffix != '.png' or not SAFE_ID.fullmatch(relative.stem)):
            raise ValueError('Confirmation image must be a PNG directly below images/')
        image = CONFIRMATION_MANIFEST.parent / relative
        if local_path(image) != image:
            raise ValueError('Confirmation image was redirected')
        row.update(_path=image, _manifest=str(CONFIRMATION_MANIFEST))
        seen.add(row['image_id'])
    if len(rows) != freeze['images'] or len({row['family_id'] for row in rows}) != freeze['families']:
        raise ValueError('Confirmation count differs from public freeze')
    provenance = dict(path=str(CONFIRMATION_MANIFEST), sha256=manifest_hash, split='eval',
                      dataset_version=freeze['dataset_version'], freeze_path=str(freeze_path),
                      freeze_sha256=sha(freeze_path), public_verifier_sha256=sha(verifier))
    return rows, provenance, proof


@contextmanager
def local_temporary_files(directory):
    directory = output_path(directory)
    directory.mkdir(exist_ok=False)
    caches = dict(HF_HOME='hf', HF_HUB_CACHE='hf-hub', HF_ASSETS_CACHE='hf-assets',
                  HF_XET_CACHE='hf-xet', HF_MODULES_CACHE='hf-modules', TRANSFORMERS_CACHE='transformers',
                  TORCH_HOME='torch', TORCHINDUCTOR_CACHE_DIR='inductor', TRITON_CACHE_DIR='triton',
                  CUDA_CACHE_PATH='cuda', XDG_CACHE_HOME='xdg')
    changes = {key: str(directory / name) for key, name in caches.items()}
    changes.update(TMP=str(directory), TEMP=str(directory), TMPDIR=str(directory),
                   PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_HUB_DISABLE_TELEMETRY='1')
    previous = {name: os.environ.get(name) for name in changes}
    previous_tempdir = tempfile.tempdir
    os.environ.update(changes)
    tempfile.tempdir = str(directory)
    try:
        yield
    finally:
        tempfile.tempdir = previous_tempdir
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def metadata(row):
    return dict(image_id=row['image_id'], family_id=row['family_id'], split=row['split'],
                file_sha256=row['sha256'], image_path=str(row['_path']))


def verify_previous_files(directory, filenames):
    """Use existing external seals for published prior PII-RECALL inputs."""
    if not directory.is_relative_to(RECALL):
        return None
    seal_path = RECALL / 'PREDICTIONS_FREEZE.json'
    freeze = read(seal_path)
    if not directory.is_relative_to(Path(freeze['run']).parent):
        seal_path = RECALL / 'final-files.json'
        freeze = read(seal_path)
        key = lambda path: path.relative_to(RECALL).as_posix()
    else:
        key = str
    if any(freeze['files'].get(key(directory / name)) != sha(directory / name) for name in filenames):
        raise ValueError('Saved prior analysis differs from its published freeze')
    return dict(path=str(seal_path), sha256=sha(seal_path))


def load_reused(path):
    directory = ordinary_input(path)
    filenames = tuple(name + '.json' for name in (*SHARED, 'analysis', 'detection', 'relations', 'refined'))
    for name in filenames:
        if ordinary_input(directory / name) != directory / name:
            raise ValueError('Saved analysis record was redirected')
    seal = verify_previous_files(directory, filenames)
    ordinary_input(read(directory / 'source.json')['input_path'])
    analysis, blobs, rgb = api.load_analysis(directory)
    current = {p.name: sha(p) for p in api.HERE.glob('*.py')}
    saved = analysis['integration_sources']
    changes = {name: dict(saved=saved.get(name), current=current.get(name))
               for name in sorted(set(saved) | set(current)) if saved.get(name) != current.get(name)}
    settings = blobs['detail'].get('settings', {})
    if ('baseline_model_settings_digest' in settings
            and settings['baseline_model_settings_digest'] != digest(blobs['ocr']['engine'])):
        raise ValueError('Saved detail settings differ from OCR settings')
    identity, ocr, source = analysis['identity'], blobs['ocr'], blobs['source']
    if not SAFE_ID.fullmatch(identity['image_id']):
        raise ValueError('Saved image ID is unsafe as an output component')
    row = dict(ocr['input'], image_id=identity['image_id'], _path=Path(source['input_path']))
    reused = dict(directory=str(directory), analysis_id=analysis['analysis_id'],
                  analysis_digest=analysis['analysis_digest'], analysis_sha256=sha(directory / 'analysis.json'),
                  files={name + '.json': analysis['files'][name + '.json'] for name in SHARED},
                  integration_source_changes=changes, prior_seal=seal)
    shared = {name: blobs[name] for name in SHARED}
    shared.update(metadata=metadata(row), identity=identity, rgb=rgb, timing={},
                  execution=dict(OCR='reused', NER='reused', detail='reused', reused_analysis=reused))
    if analysis.get('producer', {}).get('name', '').startswith('pii-recall-v0/') and analysis.get('arm') == 'improved':
        shared['expected_recall'] = (blobs['detection'], blobs['relations'])
    return shared


def unavailable_input(index, stage, error):
    return dict(metadata=dict(image_id=f'failed-input-{index:03d}', family_id=None, split=None,
                              file_sha256=None, image_path=None),
                timing={}, failure=failure(stage, error))


def prepare_reused(directories):
    seen = set()
    for index, directory in enumerate(directories, 1):
        started = time.perf_counter()
        try:
            item = load_reused(directory)
            identity = item['identity']['image_id']
            if identity in seen:
                raise ValueError('Duplicate image identity in the reuse batch')
            seen.add(identity)
        except Exception as error:
            item = unavailable_input(index, 'load reused analysis', error)
        item['timing']['reuse_load_seconds'] = time.perf_counter() - started
        yield item


def prepare_fresh(specifications, target, from_manifest):
    """Run the existing batch NER worker and reuse OCR/detail setup across images."""
    prepared, timing, seen = [], {}, set()
    run_id = 'rapid-account-' + uuid.uuid4().hex
    for index, specification in enumerate(specifications, 1):
        started = time.perf_counter()
        try:
            if from_manifest:
                row = specification
                if local_path(row['_path']) != row['_path']:
                    raise ValueError('Frozen fixture image path was redirected')
                rgb, transforms = api.support.decode(row)
            else:
                row, rgb, transforms = api.decode_file(ordinary_input(specification))
            if row['image_id'] in seen:
                raise ValueError('Duplicate image identity in the fresh batch')
            seen.add(row['image_id'])
            item = dict(metadata=metadata(row), row=row, rgb=rgb, transforms=transforms, timing={})
        except Exception as error:
            item = unavailable_input(index, 'decode', error)
            if from_manifest:
                item['metadata'] = metadata(specification)
        item['timing']['decode_seconds'] = time.perf_counter() - started
        prepared.append(item)

    ready = [item for item in prepared if 'failure' not in item]
    if not ready:
        return prepared, run_id, timing

    started = time.perf_counter()
    try:
        from adapters.rapid import Adapter
        adapter = Adapter(api.support.ASSETS)
        if adapter.metadata != read(OCR_CONFIGURATION)['metadata']:
            raise ValueError('OCR settings differ from the pinned prior configuration')
    except Exception as error:
        for item in ready:
            item['failure'] = failure('OCR preparation', error)
        return prepared, run_id, timing
    finally:
        timing['OCR_preparation_seconds'] = time.perf_counter() - started

    for item in ready:
        fresh_ocr(item, adapter, run_id, target)
    ready = [item for item in ready if 'failure' not in item]
    if not ready:
        return prepared, run_id, timing

    started = time.perf_counter()
    try:
        ner_output = target / 'ner-baseline'
        ner_output.mkdir(exist_ok=False)
        baselines = api.baseline_runner.ner([item['case'] for item in ready], ner_output)
    except Exception as error:
        for item in ready:
            item['failure'] = failure('NER batch', error)
        return prepared, run_id, timing
    finally:
        timing['NER_batch_seconds'] = time.perf_counter() - started

    started = time.perf_counter()
    extractor, preparation_error = None, None
    try:
        extractor = api.detail_adapter.DetailExtractor(api.support.ASSETS)
    except Exception as error:
        preparation_error = failure('detail preparation', error)
    timing['detail_preparation_seconds'] = time.perf_counter() - started
    for item in ready:
        started = time.perf_counter()
        identity, detail_error = item['identity'], preparation_error
        if extractor is not None:
            try:
                item['detail'] = extractor.extract(item['rgb'], item['ocr'], identity)
            except Exception as error:
                detail_error = failure('detail OCR', error)
        if detail_error is not None:
            item['detail'] = dict(schema_version='ocr-coordinate-details-v0', identity=identity,
                                  status='failure', units=[], reasons=[detail_error['type']],
                                  integration_error=detail_error)
        item['timing']['detail_extract_seconds'] = time.perf_counter() - started
        item['baseline'] = baselines[item['case']['id']]
        item['execution'] = dict(OCR='fresh', NER='fresh', detail='fresh', reused_analysis=None)
    return prepared, run_id, timing


def fresh_ocr(item, adapter, run_id, target):
    started = time.perf_counter()
    row = item['row']
    try:
        raw = adapter.infer(item['rgb'])
    except Exception as error:
        raw = dict(status='failure', units=[], errors=[failure('OCR', error)])
    item['timing']['OCR_inference_seconds'] = time.perf_counter() - started
    try:
        write_json(target / 'cases' / row['image_id'] / 'shared/native.json', raw.get('native', {}))
        item['transforms'] += raw.get('transforms', [])
        ocr = api.support.finalize(raw, row, run_id, adapter.metadata)
        identity, case = api.link.bind_ocr(ocr, row, item['transforms'], item['rgb'], run_id)
        source = dict(input_path=str(row['_path']), identity=identity, input_manifest=row['_manifest'])
        item.update(ocr=ocr, identity=identity, case=case, source=source)
    except Exception as error:
        item['failure'] = failure('OCR finalization', error)
