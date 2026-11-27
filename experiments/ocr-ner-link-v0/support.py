"""Local experiment paths and immutable source helpers; no dataset content reads on import."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OCR = EXP / 'ocr-adapters-v0'
TXT = EXP / 'text-detection-ab-v0'
DATA = EXP / 'ocr-fixtures-v0'
ASSETS = Path(os.environ['LOCALAPPDATA']) / 'capture-masking/ocr-adapters-v0'
RAPID_PY = ASSETS / 'env-rapid/Scripts/python.exe'
NER_PY = Path(os.environ['LOCALAPPDATA']) / 'capture-masking/TXT-01/venv/Scripts/python.exe'
SAVED = [OCR / 'results' / s / 'run.json' for s in
         ('rapid-fixture-dev', 'rapid-fixture-check', 'rapid-reference-final')]

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

sys.path.insert(0, str(OCR))
from common import decode, load_manifest, polygon_valid, validate_output, finalize, overlay
ocr_runner = module('existing_ocr_runner', OCR / 'run.py')
comparison = module('existing_comparison', EXP / 'ocr-comparison-v0/analyze.py')
raster = comparison.raster

def confined_output(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE) or path == HERE:
        raise ValueError('Outputs must be inside this experiment')
    return path

def verify_freeze():
    record = read(HERE / 'freeze.json')
    if record.get('status') != 'frozen_after_independent_review':
        raise ValueError('Independent pre-freeze review required')
    for path, expected in record['files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen file changed: ' + path)
    return record

def eval_rows(manifest):
    """Narrow new authorization; existing load_manifest/no_eval_path remain unchanged."""
    freeze = verify_freeze()
    manifest = Path(manifest).resolve()
    expected = (DATA / 'eval/manifest.jsonl').resolve()
    if manifest != expected or sha(manifest) != freeze['eval_manifest_sha256']:
        raise ValueError('Only the explicitly frozen evaluation manifest is authorized')
    rows = jsonl(manifest)
    if len(rows) != 12 or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Evaluation count/identity mismatch')
    for row in rows:
        required={'schema_version','image_id','image_path','width','height','split','family_id','sha256'}
        if not isinstance(row,dict) or not required <= row.keys():
            raise ValueError('Missing evaluation input fields')
        if row['schema_version'] != 'ocr-input-v0' or row['split'] != 'eval':
            raise ValueError('Evaluation schema/split mismatch')
        for key in ('image_id','image_path','family_id'):
            if not isinstance(row[key],str) or not row[key]:
                raise ValueError('Invalid input identity/path')
        iid=row['image_id']
        if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in iid):
            raise ValueError('Image ID must be a safe single output component')
        if not isinstance(row['sha256'],str) or len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256']):
            raise ValueError('Invalid input SHA-256')
        rel = Path(row['image_path'])
        if rel.anchor or '://' in str(rel) or '..' in rel.parts:
            raise ValueError('Evaluation path must be local and relative')
        path = (manifest.parent / rel).resolve()
        if path.drive.startswith('\\\\') or not path.is_relative_to(manifest.parent / 'images'):
            raise ValueError('Evaluation image outside frozen input folder')
        if any(type(row[k]) is not int or row[k] <= 0 for k in ('width', 'height')):
            raise ValueError('Invalid dimensions')
        if sha(path) != row['sha256'] or sha(path) != freeze['files'].get(str(path)):
            raise ValueError('Evaluation input hash mismatch')
        row.update(_path=path, _manifest=str(manifest))
    return rows
