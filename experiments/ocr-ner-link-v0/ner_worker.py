"""Runs in the existing, separate TXT-01 CUDA environment; prediction whitelist only."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import sys
import socket
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'text-detection-ab-v0'))
# This environment deliberately has no Pillow. Keep its bridge stdlib-only.
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

@contextmanager
def network_guard():
    # Same CPython socket guard as OCR run.py, without its image-library imports.
    old=(socket.socket.connect,socket.socket.connect_ex,socket.create_connection)
    def blocked(*args,**kw):
        raise RuntimeError('Network prohibited during local inference')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=blocked
    try:
        yield
    finally:
        socket.socket.connect,socket.socket.connect_ex,socket.create_connection=old

def validate_job(job):
    if set(job) != {'schema', 'cases'} or job['schema'] != 'ocr-link-ner-input-v1':
        raise ValueError('Unexpected job fields')
    ids = set()
    for case in job['cases']:
        if set(case) != {'id', 'blocks', 'links'} or case['links'] != [] or case['id'] in ids:
            raise ValueError('Only unique id, exact blocks, empty links are allowed')
        ids.add(case['id'])
        if any(set(b) != {'id', 'text'} for b in case['blocks']):
            raise ValueError('Gold/geometry/metadata must not enter detector')
    return job['cases']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cases = validate_job(read(args.input))
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    started = time.perf_counter()
    with network_guard():
        from model_adapter import NameModel
        from pipeline import single_path
        import_done = time.perf_counter()
        model = NameModel('A', device='cuda')
        if model.spec['revision'] != '95a3af685dccf4bb3f7fff6fc81146fed71500bb':
            raise ValueError('NER revision mismatch')
        meta = dict(input_sha256=sha(args.input), python=sys.version, executable=sys.executable,
            platform=platform.platform(), packages={p:md.version(p) for p in ('torch','transformers','tokenizers','safetensors')},
            cuda_device=model.torch.cuda.get_device_name(), metadata=model.metadata,
            library_import_seconds=import_done-started, constructor_seconds=time.perf_counter()-import_done,
            records=[])
        write(args.output / 'run.json', meta)
        for i, case in enumerate(cases):
            result = single_path(model, case)
            result['input_case_sha256'] = digest(case)
            record_started = time.perf_counter()
            path = args.output / f'{i:03d}.json'
            write(path, result)
            meta['records'].append(dict(id=case['id'], output=path.name, sha256=sha(path),
                status=result['status'], candidates=len(result['configs']['FULL']),
                detection_ms=result['timing']['single_path_ms'], record_seconds=time.perf_counter()-record_started))
            write(args.output / 'run.json', meta)
    print(json.dumps(dict(cases=len(cases), device=meta['cuda_device'], statuses=[r['status'] for r in meta['records']])))

if __name__ == '__main__':
    main()
