"""Prediction-only linkage. Never imports/opens gold, masks or field relationships."""
import copy
import hashlib
import itertools
import time
import numpy as np
from PIL import Image
from support import digest, raster, validate_output, polygon_valid

TYPES = {'NAME', 'PHONE', 'EMAIL', 'ACCOUNT_ID'}

def bind_ocr(ocr, row, transforms, rgb, expected_run_id):
    validate_output(ocr)
    if ocr['run_id'] != expected_run_id or ocr['image_id'] != row['image_id']:
        raise ValueError('Mixed image/run')
    for key in ('width', 'height', 'sha256', 'split', 'family_id'):
        if ocr['input'][key] != row[key]:
            raise ValueError('OCR input binding mismatch: ' + key)
    rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    decodes = [t for t in transforms if t.get('stage') == 'common_decode']
    if len(decodes) != 1 or decodes[0]['rgb_sha256'] != rgb_sha:
        raise ValueError('RGB provenance mismatch')
    if rgb.size != (row['width'], row['height']):
        raise ValueError('RGB size mismatch')
    for i, u in enumerate(ocr['units']):
        if u['id'] != f"{expected_run_id}:{row['image_id']}:u{i:04d}":
            raise ValueError('Mixed unit occurrence identity')
        if u['location_status'] != 'valid' and u['polygons']:
            raise ValueError('Uncertain geometry cannot be used')
    identity = dict(image_id=row['image_id'], ocr_run_id=expected_run_id,
                    file_sha256=row['sha256'], rgb_sha256=rgb_sha,
                    width=rgb.width, height=rgb.height, ocr_sha256=digest(ocr))
    case = dict(id=digest(identity), blocks=[dict(id=u['id'], text=u['text']) for u in ocr['units']], links=[])
    return identity, case

def map_candidates(ocr, identity, case, detection):
    started = time.perf_counter()
    if (identity['image_id'] != ocr['image_id'] or identity['ocr_run_id'] != ocr['run_id']
            or identity['ocr_sha256'] != digest(ocr) or case['id'] != digest(identity)
            or detection['id'] != case['id']):
        raise ValueError('Detection/image/run binding mismatch')
    expected_blocks = [dict(id=u['id'], text=u['text']) for u in ocr['units']]
    if case != dict(id=case['id'], blocks=expected_blocks, links=[]):
        raise ValueError('Blocks or links differ from exact OCR projection')
    units = {u['id']: u for u in ocr['units']}
    candidates, errors, rejected = [], [], []
    seen = set()
    for p in detection['configs']['FULL']:
        try:
            if p['id'] in seen or p['type'] not in TYPES or not p['segments']:
                raise ValueError('Invalid/duplicate candidate identity/type')
            seen.add(p['id'])
            regions, segment_regions = {}, []
            for s in p['segments']:
                u = units[s['block_id']]
                a, z = s['start'], s['end']
                if (type(a) is not int or type(z) is not int or not 0 <= a < z <= len(u['text'])
                        or u['text'][a:z] != s['text']):
                    raise ValueError('Candidate is not an exact code-point slice')
                good = u['location_status'] == 'valid' and bool(u['polygons'])
                if good and not all(polygon_valid(poly, identity['width'], identity['height']) for poly in u['polygons']):
                    raise ValueError('Invalid candidate geometry')
                segment_regions.append(dict(segment=copy.deepcopy(s), unit_id=u['id'],
                    location_status=u['location_status'], polygons=copy.deepcopy(u['polygons']) if good else []))
                if good:
                    regions[u['id']] = dict(unit_id=u['id'], polygons=copy.deepcopy(u['polygons']),
                        unit_text=u['text'], geometry_source=u['geometry_source'],
                        scope='whole OCR fragment; glyph completeness unproven',
                        wider_than_segment=(a != 0 or z != len(u['text'])))
            nvalid = sum(bool(s['polygons']) for s in segment_regions)
            status = 'valid' if nvalid == len(segment_regions) else ('partial' if nvalid else 'unavailable')
            candidates.append(dict(copy.deepcopy(p), id=f"{case['id']}:{p['id']}", detector_id=p['id'],
                                   regions=list(regions.values()), segment_regions=segment_regions, location_status=status))
            if status != 'valid':
                errors.append(dict(stage='location', candidate_id=p['id'], type='MissingLocation'))
        except (ValueError, KeyError, TypeError) as ex:
            rejected.append(copy.deepcopy(p))
            errors.append(dict(stage='link_validation', type=type(ex).__name__, message=str(ex)))
    size = (identity['width'], identity['height'])
    masks = {p['id']: raster([poly for r in p['regions'] for poly in r['polygons']], size) for p in candidates}
    shared = []
    for a, b in itertools.combinations(candidates, 2):
        overlap = int((masks[a['id']] & masks[b['id']]).sum())
        common = sorted({r['unit_id'] for r in a['regions']} & {r['unit_id'] for r in b['regions']})
        if common or overlap:
            shared.append(dict(a=a['id'], b=b['id'], types=[a['type'], b['type']], unit_ids=common,
                overlapping_pixels=overlap, identical_pixel_masks=bool(np.array_equal(masks[a['id']], masks[b['id']])),
                independent_masking=False))
    area = np.zeros((size[1], size[0]), bool)
    for mask in masks.values():
        area |= mask
    location_status = 'failure' if rejected and not candidates else ('partial_failure' if errors else 'ok')
    stages = dict(ocr=ocr['status'], detection=detection['status'],
                  names=detection['config_statuses']['N1'], contacts=detection['config_statuses']['C1'], location=location_status)
    if all(v == 'ok' for v in stages.values()):
        status = 'ok'
    elif candidates:
        status = 'partial_failure'
    elif ocr['status'] in ('failure', 'indeterminate'):
        status = ocr['status']
    else:
        status = 'partial_failure' if any(v == 'ok' for v in stages.values()) else 'failure'
    state = ('incomplete' if status != 'ok' else 'normal_no_text' if not ocr['units']
             else 'normal_zero_candidates' if not candidates else 'candidates')
    return dict(**identity, case_id=case['id'], status=status, state=state, stage_statuses=stages,
        upstream_errors=dict(ocr=ocr['errors'], detection=detection['errors']), errors=errors,
        candidates=candidates, rejected_candidates=rejected, shared_regions=shared,
        provenance_units=copy.deepcopy(ocr['units']), mask_pixels=int(area.sum()),
        link_mask_seconds=time.perf_counter()-started), area

def save_masked(rgb, mask, target):
    source = np.array(rgb)
    if mask.dtype != bool or mask.shape != source.shape[:2]:
        raise ValueError('Invalid mask shape/type')
    out = source.copy()
    out[mask] = 0
    Image.fromarray(out).save(target / 'masked.png')
    Image.fromarray(mask.astype(np.uint8) * 255).save(target / 'mask.png')
    with Image.open(target / 'masked.png') as im:
        saved = np.array(im)
        if im.mode != 'RGB' or im.size != rgb.size:
            raise ValueError('Saved masked raster mismatch')
    with Image.open(target / 'mask.png') as im:
        saved_mask = np.array(im)
        if im.mode != 'L' or not np.array_equal(saved_mask, mask.astype(np.uint8)*255):
            raise ValueError('Saved mask mismatch')
    if np.any(saved[mask] != 0) or not np.array_equal(saved[~mask], source[~mask]):
        raise ValueError('Actual pixels do not implement mask')
    return dict(inside_mask_black=True, outside_mask_identical=True, mask_round_trip=True,
                size=list(rgb.size), originally_black_mask_pixels=int(np.all(source == 0, axis=2)[mask].sum()))
