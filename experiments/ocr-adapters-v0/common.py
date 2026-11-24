"""Experiment-only input, geometry, occurrence and persistence contracts."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
ASSET_ROOT = Path(os.environ['LOCALAPPDATA']) / 'capture-masking' / 'ocr-adapters-v0'
STATUSES = {'ok', 'partial_failure', 'failure', 'indeterminate'}
TYPES = {'line', 'word', 'character', 'fragment', 'unknown'}

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    payload=json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    payload.encode('utf-8')  # reject lone surrogates before opening/truncating a file
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(payload, encoding='utf-8')

def no_eval_path(path):
    raw=Path(path)
    if raw.drive.startswith('\\\\') or 'eval' in [s.lower() for s in raw.parts]:
        raise ValueError('eval/network share paths are prohibited')
    p = raw.resolve()
    if p.drive.startswith('\\\\'):
        raise ValueError('UNC/network share input is prohibited; use local files')
    if 'eval' in [s.lower() for s in p.parts]:
        raise ValueError('eval path is prohibited by this experiment')
    return p

def load_manifest(path):
    path = no_eval_path(path)
    rows, ids = [], set()
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        row = json.loads(line)
        required = {'schema_version','image_id','image_path','width','height','split','family_id','sha256'}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f'missing input fields at line {number}')
        if row['schema_version'] != 'ocr-input-v0' or row['split'] not in {'dev','check'}:
            raise ValueError(f'unsupported schema/split at line {number}; eval is refused')
        for key in ('image_id','family_id','image_path'):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f'invalid {key} at line {number}')
        if row['image_id'] in ids:
            raise ValueError('duplicate image_id')
        ids.add(row['image_id'])
        if Path(row['image_path']).anchor or '://' in row['image_path']:
            raise ValueError('image_path must be local and manifest-relative')
        if any(type(row[k]) is not int or row[k] <= 0 for k in ('width','height')):
            raise ValueError('invalid image dimensions')
        if not isinstance(row['sha256'], str) or len(row['sha256']) != 64 or any(c not in '0123456789abcdef' for c in row['sha256']):
            raise ValueError('invalid sha256')
        row = dict(row, _path=no_eval_path(path.parent / row['image_path']), _manifest=str(path))
        rows.append(row)
    if not rows:
        raise ValueError('manifest has no inputs')
    return rows

def decode(row):
    path = row['_path']
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != row['sha256']:
        raise ValueError('input SHA-256 mismatch')
    with Image.open(BytesIO(data)) as im:
        im.load()
        if im.size != (row['width'], row['height']):
            raise ValueError('manifest dimensions differ from decoded raster')
        if getattr(im, 'n_frames', 1) != 1:
            raise ValueError('multi-frame input is unsupported')
        if 'A' in im.getbands() and im.getchannel('A').getextrema() != (255,255):
            raise ValueError('non-opaque input requires an agreed compositing policy')
        if 'transparency' in im.info:
            raise ValueError('transparency metadata is unsupported')
        transforms = [{'stage':'common_decode','decoder':'Pillow','input_format':im.format,
            'source_mode':im.mode,'output_mode':'RGB','width':im.width,'height':im.height,
            'exif_orientation':im.getexif().get(274),
            'orientation_policy':'raw decoded raster; EXIF orientation not applied',
            'icc_policy':'no color-management transform',
            'rgb_sha256':hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()}]
        return im.convert('RGB').copy(), transforms

def polygon_valid(poly, width, height):
    if not isinstance(poly, list) or len(poly) != 4:
        return False
    if any(not isinstance(p, (list,tuple)) or len(p) != 2 or
           any(type(v) not in (int,float) or not math.isfinite(v) for v in p) for p in poly):
        return False
    if any(not (0 <= x <= width and 0 <= y <= height) for x,y in poly):
        return False
    cross = []
    for i in range(len(poly)):
        a,b,c = poly[i-2],poly[i-1],poly[i]
        cross.append((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0]))
    # Adapters return convex quads/rectangles; reject crossed/degenerate ordering.
    return (all(c > 0 for c in cross) or all(c < 0 for c in cross))

def error(stage, ex, **kw):
    return dict(stage=stage, type=type(ex).__name__, message=str(ex), **kw)

def finalize(result, row, run_id, metadata):
    if not isinstance(result, dict) or result.get('status') not in STATUSES:
        raise ValueError('malformed adapter result/status')
    result = copy.deepcopy(result)
    if not isinstance(result.get('units'), list) or not isinstance(result.get('errors'), list):
        raise ValueError('malformed units/errors')
    units, details, issues = result['units'], result.get('details',[]), []
    if not isinstance(result.get('metrics',{}),dict):
        raise ValueError('metrics must be an object')
    def check_unit(unit, uid):
        if not isinstance(unit, dict) or not isinstance(unit.get('text'), str) or unit.get('unit_type') not in TYPES:
            raise ValueError('malformed occurrence text/type')
        unit['id'] = uid
        if unit.get('location_status') not in {'valid','unavailable','uncertain'}:
            raise ValueError('invalid location_status')
        polygons = unit.get('polygons')
        if not isinstance(polygons, list):
            raise ValueError('polygons must be a list')
        if polygons and unit['location_status']!='valid':
            raise ValueError('unverified geometry must be retained in native output only')
        if any(not polygon_valid(p,row['width'],row['height']) for p in polygons) or (unit['location_status']=='valid' and not polygons):
            issues.append(dict(stage='geometry_validation',type='InvalidPolygon',message='invalid geometry retained only in native output',unit_id=uid))
            unit['polygons'] = []
            unit['location_status'] = 'unavailable'
        conf = unit.get('confidence')
        if conf is not None and (type(conf) not in (int,float) or not math.isfinite(conf)):
            raise ValueError('invalid native confidence')
        unit['confidence'] = conf
        unit.setdefault('geometry_source','unspecified')
    for i,u in enumerate(units):
        check_unit(u, f'{run_id}:{row["image_id"]}:u{i:04d}')
    if not isinstance(details, list):
        raise ValueError('details must be a list')
    for i,d in enumerate(details):
        check_unit(d, f'{run_id}:{row["image_id"]}:d{i:04d}')
        p = d.get('parent_index')
        if type(p) is not int or not 0 <= p < len(units):
            raise ValueError('invalid detail parent')
        d['parent_id'] = units[p]['id']
        span = d.get('span')
        if span is not None:
            if not isinstance(span,list) or len(span)!=2 or any(type(x) is not int for x in span) or not 0<=span[0]<span[1]<=len(units[p]['text']) or units[p]['text'][span[0]:span[1]] != d['text']:
                raise ValueError('child span is not an exact code-point slice')
    status = result['status']
    if issues and status == 'ok':
        status = 'partial_failure' if units else 'failure'
    return dict(schema_version='ocr-output-v0',image_id=row['image_id'],run_id=run_id,
        input={k:row[k] for k in ('image_path','width','height','sha256','split','family_id')},
        input_manifest=row['_manifest'],engine=metadata,status=status,
        errors=result['errors']+issues,units=units,details=details,
        artifacts={'native':'native.json','transforms':'transforms.json','overlay':'overlay.png'},
        metrics=result.get('metrics',{}))

def validate_output(obj):
    assert obj['schema_version']=='ocr-output-v0' and obj['status'] in STATUSES
    assert isinstance(obj['image_id'],str) and isinstance(obj['run_id'],str)
    w,h=obj['input']['width'],obj['input']['height']
    all_units = obj['units'] + obj['details']
    assert len({u['id'] for u in all_units}) == len(all_units)
    parents = {u['id']:u for u in obj['units']}
    for u in all_units:
        assert isinstance(u['text'],str) and u['unit_type'] in TYPES
        assert all(polygon_valid(p,w,h) for p in u['polygons'])
        assert u['location_status']!='valid' or u['polygons']
        if u.get('span') is not None:
            p=parents[u['parent_id']]['text']; a,b=u['span']
            assert 0<=a<b<=len(p) and p[a:b]==u['text']
    return True

def overlay(rgb, obj, path):
    im=rgb.copy(); draw=ImageDraw.Draw(im)
    for i,u in enumerate(obj['units']):
        for poly in u['polygons']:
            xy=[tuple(p) for p in poly]
            draw.line(xy+[xy[0]], fill=(255,50,70),width=2)
            draw.text(xy[0],str(i),fill=(255,50,70))
    for d in obj['details']:
        for poly in d['polygons']:
            xy=[tuple(p) for p in poly]
            draw.line(xy+[xy[0]],fill=(20,160,255),width=1)
    im.save(path)
