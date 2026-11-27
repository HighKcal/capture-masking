"""Offline OCR-COMP01 analysis. No OCR/torch/ONNX imports or inference entry points."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'ocr-fixtures-v0'
CONFIGS = ['rapid', 'easy', 'windows2x', 'vl_no2x', 'windows_old', 'vl_old']
LABELS = dict(zip(CONFIGS, ['RapidOCR — 한국어 PP-OCRv5', 'EasyOCR — CRAFT + korean_g2',
    'Windows OCR — 입력 2배', 'PaddleOCR-VL-1.6 — 확대 해제',
    'Windows OCR — 이전 원본 입력', 'PaddleOCR-VL-1.6 — 이전 조건부 2배']))
COLORS = dict(zip(CONFIGS, ['#009B83', '#DF8B00', '#3188FF', '#C15CD8', '#3188FF', '#C15CD8']))
TYPES = ['NAME', 'PHONE', 'EMAIL', 'ACCOUNT_ID']


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)


def raster(polygons, size):
    """Union of convex quads; pixel centers on edges included, original coordinates."""
    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    for poly in polygons:
        p = np.asarray(poly, dtype=float)
        if p.shape != (4, 2) or not np.isfinite(p).all():
            raise ValueError('invalid quad')
        if np.any(p < 0) or np.any(p[:, 0] > w) or np.any(p[:, 1] > h):
            raise ValueError('quad out of bounds')
        edges = np.roll(p, -1, axis=0) - p
        cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
        if not (np.all(cross > 0) or np.all(cross < 0)):
            raise ValueError('nonconvex or degenerate quad')
        x0, y0 = np.floor(p.min(axis=0)).astype(int)
        x1, y1 = np.ceil(p.max(axis=0)).astype(int)
        xx, yy = np.meshgrid(np.arange(x0, x1)+.5, np.arange(y0, y1)+.5)
        cs = np.array([(b[0]-a[0])*(yy-a[1])-(b[1]-a[1])*(xx-a[0])
                       for a, b in zip(p, np.roll(p, -1, axis=0))])
        mask[y0:y1, x0:x1] |= np.all(cs >= -1e-9, axis=0) | np.all(cs <= 1e-9, axis=0)
    return mask


def bbox(polys):
    p = np.array([xy for poly in polys for xy in poly])
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def boxpoly(b):
    l, t, r, bot = b
    return [[l, t], [r, t], [r, bot], [l, bot]]


def reading_order(items):
    """Geometry only: anchored rows, half-min-height center tolerance; stable ties."""
    pending = sorted(items, key=lambda v: (v['bbox'][1], v['bbox'][0], v['index']))
    rows = []
    for item in pending:
        l, t, r, b = item['bbox']; cy = (t+b)/2; height = b-t
        match = next((row for row in rows if abs(cy-row['cy']) <= .5*min(height, row['height'])), None)
        if match is None:
            rows.append(dict(cy=cy, height=height, items=[item]))
        else:
            match['items'].append(item)
    return [item for row in rows for item in sorted(row['items'], key=lambda v: (v['bbox'][0], v['index']))]


def edit_alignment(gold, pred):
    n, m = len(gold), len(pred)
    d = np.zeros((n+1, m+1), dtype=np.int32)
    d[:, 0] = np.arange(n+1); d[0, :] = np.arange(m+1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            d[i, j] = min(d[i-1, j-1]+(gold[i-1] != pred[j-1]), d[i-1, j]+1, d[i, j-1]+1)
    i, j = n, m; mapping = {}; ops = []
    while i or j:
        if i and j and d[i, j] == d[i-1, j-1]+(gold[i-1] != pred[j-1]):
            kind = 'equal' if gold[i-1] == pred[j-1] else 'substitution'
            mapping[i-1] = j-1; ops.append(dict(kind=kind, gold=i-1, pred=j-1)); i -= 1; j -= 1
        elif i and d[i, j] == d[i-1, j]+1:
            ops.append(dict(kind='deletion', gold=i-1, pred=None)); i -= 1
        else:
            ops.append(dict(kind='insertion', gold=None, pred=j-1)); j -= 1
    ops.reverse()
    counts = {k: sum(o['kind'] == k for o in ops) for k in ['substitution', 'deletion', 'insertion']}
    return dict(gold_codepoints=n, pred_codepoints=m, edits=int(d[n, m]), **counts), mapping, ops


def text_stream(items):
    raw = ''; refs = []; separators = []
    for i, item in enumerate(items):
        if i:
            separators.append(len(raw)); raw += '\n'; refs.append(None)
        raw += item['text']; refs.extend((item['id'], cp) for cp in range(len(item['text'])))
    keep = [i for i, c in enumerate(raw) if not c.isspace()]
    return dict(raw=raw, refs=refs, compact=''.join(raw[i] for i in keep),
                compact_refs=[refs[i] for i in keep], compact_raw_indices=keep,
                inserted_boundaries=separators,
                native_whitespace=sum(c.isspace() and refs[i] is not None for i, c in enumerate(raw)))


def associate(out, blocks):
    size = (out['input']['width'], out['input']['height'])
    bitems = [dict(b, index=i) for i, b in enumerate(blocks)]
    units = out['units']; umasks = {}
    bmask = {b['id']: raster([boxpoly(b['bbox'])], size) for b in blocks}
    edges = {}; unassigned = []
    for i, u in enumerate(units):
        if u['location_status'] == 'valid' and u['polygons']:
            umasks[u['id']] = raster(u['polygons'], size)
            hits = [b['id'] for b in blocks if np.any(umasks[u['id']] & bmask[b['id']])]
        else:
            hits = []
        if hits:
            edges[u['id']] = hits
        else:
            unassigned.append(dict(id=u['id'], text=u['text'], location_status=u['location_status'],
                                   reason='no_location' if not u['polygons'] else 'outside_blocks'))
    # Bipartite connected components; every native unit is counted at most once.
    groups = []; todo = {b['id'] for b in blocks}
    while todo:
        seed = next(b['id'] for b in blocks if b['id'] in todo)
        bids = {seed}; uids = set()
        while True:
            moreu = {uid for uid, bs in edges.items() if bids.intersection(bs)}
            moreb = {bid for uid in moreu for bid in edges[uid]}
            if moreu <= uids and moreb <= bids:
                break
            bids |= moreb; uids |= moreu
        todo -= bids
        gb = reading_order([b for b in bitems if b['id'] in bids])
        pu = reading_order([dict(u, index=i, bbox=bbox(u['polygons'])) for i, u in enumerate(units) if u['id'] in uids])
        gs, ps = text_stream(gb), text_stream(pu)
        raw, _, _ = edit_alignment(gs['raw'], ps['raw'])
        compact, mapping, ops = edit_alignment(gs['compact'], ps['compact'])
        groups.append(dict(id=f'c{len(groups):02d}', block_ids=[b['id'] for b in gb],
            unit_ids=[u['id'] for u in pu], gold_raw=gs['raw'], predicted_raw=ps['raw'],
            gold_compact=gs['compact'], predicted_compact=ps['compact'], raw_metrics=raw,
            compact_metrics=compact, gold_inserted_boundaries=gs['inserted_boundaries'],
            pred_inserted_boundaries=ps['inserted_boundaries'],
            gold_native_whitespace=gs['native_whitespace'], pred_native_whitespace=ps['native_whitespace'],
            units=[dict(id=u['id'], text=u['text']) for u in pu],
            _gold=gs, _pred=ps, _mapping=mapping, alignment=ops))
    return groups, unassigned


def segment_evidence(segment, groups):
    group = next(c for c in groups if segment['block_id'] in c['block_ids'])
    gs, ps, mapping = group['_gold'], group['_pred'], group['_mapping']
    expected = segment['text']; bid = segment['block_id']; start, end = segment['start'], segment['end']
    gi = [i for i, ref in enumerate(gs['compact_refs']) if ref[0] == bid and start <= ref[1] < end]
    positions = [mapping.get(i) for i in gi]
    selected_refs = []; raw_span = None; predicted = ''; compact_predicted = ''
    aligned_positions = [i for i in positions if i is not None]
    if aligned_positions:
        a, b = aligned_positions[0], aligned_positions[-1]+1
        compact_predicted = ps['compact'][a:b]
        raw_a, raw_b = ps['compact_raw_indices'][a], ps['compact_raw_indices'][b-1]+1
        # Ignore analysis-introduced separators; retain all native whitespace.
        predicted = ''.join(c for i, c in enumerate(ps['raw'][raw_a:raw_b], raw_a) if ps['refs'][i] is not None)
        selected_refs = ps['compact_refs'][a:b]
        raw_span = [raw_a, raw_b]
        exact = positions == list(range(a, b)) and compact_predicted == ''.join(c for c in expected if not c.isspace())
    else:
        exact = False
    return dict(segment=segment, component_id=group['id'], exact_no_whitespace=exact,
        exact_strict=exact and predicted == expected, predicted=predicted,
        predicted_compact=compact_predicted, raw_component_span=raw_span,
        source_codepoints=[list(r) for r in selected_refs],
        missing_alignment_codepoints=sum(i is None for i in positions),
        assembly_unit_ids=list(dict.fromkeys(r[0] for r in selected_refs)))


def atoms(out):
    if out['engine']['engine'] != 'Windows.Media.Ocr':
        return [u for u in out['units'] if u['location_status'] == 'valid' and u['polygons']]
    parents = {u['id']: u for u in out['units']}
    result = []
    for d in out['details']:
        if d['location_status'] != 'valid' or not d['polygons']:
            continue
        span = d.get('span'); parent = parents.get(d.get('parent_id'))
        if parent is None or span is None or parent['text'][span[0]:span[1]] != d['text']:
            raise ValueError('Windows word/parent CP linkage not verified')
        result.append(d)
    return result


def linked(evidence, chosen):
    if not all(e['exact_no_whitespace'] for e in evidence):
        return False
    covered = set()
    for a in chosen:
        if 'parent_id' in a:
            covered.update((a['parent_id'], i) for i in range(*a['span']))
        else:
            covered.update((a['id'], i) for i in range(len(a['text'])))
    return all(set(map(tuple, e['source_codepoints'])) <= covered for e in evidence)


def load_mask(rel, size):
    p = (DATA / rel).resolve()
    if not p.is_relative_to(DATA/'masks') or not p.parent.name.startswith('dev-') or 'foreground' not in p.name:
        raise ValueError('only explicitly referenced dev foreground masks allowed')
    with Image.open(p) as im:
        if im.size != size or im.mode != 'L':
            raise ValueError('unexpected foreground dimensions/mode')
        return np.array(im) > 0


def mask_stats(area, target, text):
    den = int(target.sum()); hit = int((area & target).sum()); pixels = int(area.sum())
    return dict(target_pixels=den, covered_target_pixels=hit, missed_target_pixels=den-hit,
                non_target_text_pixels=int((area & text & ~target).sum()),
                region_area_pixels=pixels, non_target_area_pixels=pixels-hit)


def clean_components(groups):
    return [{k: v for k, v in g.items() if not k.startswith('_')} for g in groups]


def score_dev(out, gold):
    size = (out['input']['width'], out['input']['height'])
    groups, unassigned = associate(out, gold['blocks'])
    block_map = {b['id']: b for b in gold['blocks']}
    text = load_mask(gold['masks']['text_foreground'], size)
    minimal = atoms(out); masks = {a['id']: raster(a['polygons'], size) for a in minimal}
    target_union = np.zeros(text.shape, bool); selection_union = np.zeros(text.shape, bool)
    targets = {}; areas = {}; choices = {}; occurrences = []
    for occ in gold['occurrences']:
        for s in occ['segments']:
            if block_map[s['block_id']]['text'][s['start']:s['end']] != s['text']:
                raise ValueError('gold CP span mismatch')
        target = load_mask(occ['foreground_mask'], size)
        assert int(target.sum()) == occ['foreground_pixels']
        evidence = [segment_evidence(s, groups) for s in occ['segments']]
        chosen = [a for a in minimal if np.any(masks[a['id']] & target)]
        area = raster([p for a in chosen for p in a['polygons']], size)
        exact = all(e['exact_no_whitespace'] for e in evidence)
        strict = all(e['exact_strict'] for e in evidence)
        link = linked(evidence, chosen); stats = mask_stats(area, target, text)
        occurrences.append(dict(id=occ['id'], type=occ['type'],
            gold_segments=[s['text'] for s in occ['segments']], segments=evidence,
            exact_no_whitespace=exact, exact_strict=strict, text_location_link=link,
            joint_exact_link_full=exact and link and stats['missed_target_pixels'] == 0,
            joint_strict_link_full=strict and link and stats['missed_target_pixels'] == 0,
            selected_unit_ids=[a['id'] for a in chosen],
            selected_text=[a['text'] for a in chosen], geometry=stats))
        targets[occ['id']] = target; areas[occ['id']] = area; choices[occ['id']] = {a['id'] for a in chosen}
        target_union |= target; selection_union |= area
    pairs = []; names = [o for o in gold['occurrences'] if o['type'] == 'NAME']
    for i, a in enumerate(names):
        for b in names[i+1:]:
            at = ''.join(s['text'] for s in a['segments']); bt = ''.join(s['text'] for s in b['segments'])
            if at != bt and not {s['block_id'] for s in a['segments']}.intersection(s['block_id'] for s in b['segments']):
                continue
            aid, bid = a['id'], b['id']; shared = choices[aid] & choices[bid]
            ab, ba = int((areas[aid] & targets[bid]).sum()), int((areas[bid] & targets[aid]).sum())
            separate = bool(choices[aid]) and bool(choices[bid]) and not shared and ab == ba == 0
            full = not np.any(targets[aid] & ~areas[aid]) and not np.any(targets[bid] & ~areas[bid])
            pairs.append(dict(a=aid, b=bid, shared_unit_ids=sorted(shared), a_covers_b=ab, b_covers_a=ba,
                              separate=separate, separate_and_full=separate and full))
    all_area = raster([p for a in minimal for p in a['polygons']], size)
    keys = ['gold_codepoints', 'pred_codepoints', 'edits', 'substitution', 'deletion', 'insertion']
    return dict(occurrences=occurrences, components=clean_components(groups), unassigned=unassigned,
        text_metrics={k: sum(g['compact_metrics'][k] for g in groups) for k in keys},
        raw_text_metrics={k: sum(g['raw_metrics'][k] for g in groups) for k in keys},
        native_whitespace=sum(g['pred_native_whitespace'] for g in groups),
        inserted_boundaries=sum(len(g['pred_inserted_boundaries']) for g in groups),
        unique_geometry=mask_stats(selection_union, target_union, text),
        occurrence_sum_geometry={k: sum(o['geometry'][k] for o in occurrences) for k in occurrences[0]['geometry']},
        all_text_geometry=mask_stats(all_area, text, text), name_pairs=pairs,
        blocks=len(gold['blocks']), multi_block_components=sum(len(g['block_ids']) > 1 for g in groups))


def smoke_score(out):
    texts = ['김민수  김민수', '연락처 010-1234-5678', 'sample+tag_2@example.com', '드문 음절: 똠 쀍']
    bands = [(0, 80), (80, 145), (145, 210), (210, 320)]
    blocks = [dict(id=f'row{i+1}', text=t, bbox=[0, a, 880, b]) for i, (t, (a, b)) in enumerate(zip(texts, bands))]
    groups, unassigned = associate(out, blocks)
    return dict(components=clean_components(groups), unassigned=unassigned)


def aggregate(results):
    summaries = []
    for cfg in CONFIGS:
        rr = [r for r in results if r['config'] == cfg]; dev = [r['dev'] for r in rr if 'dev' in r]
        occ = [o for d in dev for o in d['occurrences']]
        row = dict(config=cfg, label=LABELS[cfg], group='main' if cfg in CONFIGS[:4] else 'control',
            results=len(rr), states=dict(Counter(r['status'] for r in rr)), occurrences=len(occ),
            exact_no_whitespace=sum(o['exact_no_whitespace'] for o in occ), exact_strict=sum(o['exact_strict'] for o in occ),
            joint_exact_link_full=sum(o['joint_exact_link_full'] for o in occ),
            joint_strict_link_full=sum(o['joint_strict_link_full'] for o in occ),
            type_counts={t: dict(denominator=sum(o['type'] == t for o in occ),
                exact=sum(o['exact_no_whitespace'] for o in occ if o['type'] == t),
                strict=sum(o['exact_strict'] for o in occ if o['type'] == t)) for t in TYPES},
            text_metrics={k: sum(d['text_metrics'][k] for d in dev) for k in dev[0]['text_metrics']},
            raw_text_metrics={k: sum(d['raw_text_metrics'][k] for d in dev) for k in dev[0]['raw_text_metrics']},
            unique_geometry={k: sum(d['unique_geometry'][k] for d in dev) for k in dev[0]['unique_geometry']},
            occurrence_sum_geometry={k: sum(d['occurrence_sum_geometry'][k] for d in dev) for k in dev[0]['occurrence_sum_geometry']},
            unassigned_units=sum(len(d['unassigned']) for d in dev),
            unassigned_codepoints=sum(len(u['text']) for d in dev for u in d['unassigned']),
            unassigned_nonspace_codepoints=sum(sum(not c.isspace() for c in u['text']) for d in dev for u in d['unassigned']),
            name_pairs=sum(len(d['name_pairs']) for d in dev),
            separate_name_pairs=sum(p['separate'] for d in dev for p in d['name_pairs']),
            separate_full_name_pairs=sum(p['separate_and_full'] for d in dev for p in d['name_pairs']),
            location_unavailable_units=sum(r['unlocated_units'] for r in rr),
            location_unavailable_codepoints=sum(r['unlocated_codepoints'] for r in rr))
        summaries.append(row)
    return summaries


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', size)


def create_visuals(inv, outputs, golds):
    dest = HERE / 'visuals'; dest.mkdir(exist_ok=True)
    for image_id, inp in inv['inputs'].items():
        base = Image.open(inp['path']).convert('RGB')
        tiles = [('원본 — '+image_id, base)]
        for cfg in CONFIGS[:4]:
            out = outputs[cfg, image_id]
            im = base.copy(); d = ImageDraw.Draw(im)
            for i, atom in enumerate(atoms(out)):
                for p in atom['polygons']:
                    xy = [tuple(v) for v in p]; d.line(xy+[xy[0]], fill=COLORS[cfg], width=2)
                l, t, _, _ = bbox(atom['polygons'])
                d.text((l, max(0, t-13)), str(i), fill=COLORS[cfg], font=font(11), stroke_width=1, stroke_fill='#111111')
            im.save(dest / f'{image_id}--{cfg}--overlay.png')
            tiles.append((LABELS[cfg], im))
        tile_w = 450; tile_h = round(base.height*tile_w/base.width)
        sheet = Image.new('RGB', (tile_w*5, tile_h+92), '#101720'); draw = ImageDraw.Draw(sheet)
        for i, (label, im) in enumerate(tiles):
            draw.text((i*tile_w+8, 8), label, font=font(20), fill='white')
            sheet.paste(im.resize((tile_w, tile_h), Image.Resampling.LANCZOS), (i*tile_w, 44))
        draw.text((8, tile_h+52), '축소 개요 · 영역=엔진별 색 · Windows=단어, 나머지=조각 · 숫자는 해당 최소 단위의 순번 · 정밀 경계는 원본 크기 overlay 확인', font=font(17), fill='#e0e7ee')
        sheet.save(dest / f'{image_id}--comparison.png')
    # Pixel evidence at 5x nearest-neighbor. No masks for private photographs.
    for image_id, crop in [('dev-01', (188, 320, 530, 367)), ('dev-03', (73, 102, 132, 133)), ('dev-04', (56, 265, 233, 330))]:
        g = golds[image_id]; inp = inv['inputs'][image_id]; size = (inp['width'], inp['height'])
        target = load_mask(g['masks']['pii_foreground'], size)
        original = Image.open(inp['path']).convert('RGB'); panels = []
        for cfg in CONFIGS[:4]:
            out = outputs[cfg, image_id]; aa = atoms(out)
            chosen = [a for a in aa if np.any(raster(a['polygons'], size) & target)]
            area = raster([p for a in chosen for p in a['polygons']], size)
            arr = np.array(original); arr[target & area] = (0, 222, 146); arr[target & ~area] = (255, 40, 54)
            im = Image.fromarray(arr); dr = ImageDraw.Draw(im)
            for a in chosen:
                for p in a['polygons']:
                    xy = [tuple(v) for v in p]; dr.line(xy+[xy[0]], fill=COLORS[cfg], width=1)
            panels.append((LABELS[cfg], im.crop(crop).resize(((crop[2]-crop[0])*5, (crop[3]-crop[1])*5), Image.Resampling.NEAREST)))
        pw, ph = panels[0][1].size
        sheet = Image.new('RGB', (max(pw, 760), (ph+40)*4+40), '#101720'); dr = ImageDraw.Draw(sheet)
        for i, (label, im) in enumerate(panels):
            dr.text((5, i*(ph+40)+5), label, font=font(18), fill='white'); sheet.paste(im, (0, i*(ph+40)+35))
        dr.text((5, (ph+40)*4), '초록=포함한 대상 전경 / 빨강=영역 밖 대상 전경 · 5배 nearest · 정답 보조 선택', font=font(15), fill='white')
        sheet.save(dest/f'{image_id}--foreground-detail.png')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--no-visuals', action='store_true'); args = ap.parse_args()
    from inventory import load_inventory
    inv = load_inventory(write=False)
    if read(HERE/'inventory.json') != inv:
        raise ValueError('Stored provenance differs from current sources: investigate and explicitly regenerate inventory before analysis')
    golds = {g['image_id']: g for g in (json.loads(l) for l in (DATA/'ground_truth/dev.jsonl').read_text(encoding='utf-8').splitlines())}
    assert set(golds) == {'dev-01', 'dev-02', 'dev-03', 'dev-04'}
    assert all(g['dataset_version'] == 'ocr-fixtures-v0.1.0' and g['split'] == 'dev' for g in golds.values())
    outputs = {}; results = []; occurrence_rows = []; component_rows = []; timing_rows = []
    for record in inv['records']:
        cfg, iid = record['config'], record['image_id']; out = read(record['output_path'])
        assert (cfg, iid) not in outputs; outputs[cfg, iid] = out
        run = read(record['run_path']); raw_units = out['units']
        row = dict(config=cfg, label=LABELS[cfg], image_id=iid, output_path=record['output_path'], status=out['status'],
            errors=out['errors'], units=len(raw_units), minimal_units=len(atoms(out)),
            unlocated_units=sum(u['location_status'] != 'valid' for u in raw_units),
            unlocated_codepoints=sum(len(u['text']) for u in raw_units if u['location_status'] != 'valid'),
            unlocated=[dict(id=u['id'], text=u['text']) for u in raw_units if u['location_status'] != 'valid'])
        if iid in golds:
            row['dev'] = score_dev(out, golds[iid])
            for o in row['dev']['occurrences']:
                occurrence_rows.append(dict(config=cfg, image_id=iid, occurrence_id=o['id'], type=o['type'],
                    gold=' | '.join(o['gold_segments']), predicted=' | '.join(e['predicted'] for e in o['segments']),
                    exact_no_whitespace=o['exact_no_whitespace'], strict=o['exact_strict'],
                    location_link=o['text_location_link'], joint=o['joint_exact_link_full'], **o['geometry'],
                    selected_unit_ids=' | '.join(o['selected_unit_ids'])))
        elif iid == 'impl01-smoke':
            row['smoke'] = smoke_score(out)
        elif iid == 'check-01':
            row['blank_generated_text'] = [u['text'] for u in raw_units if u['text']]
        for c in row.get('dev', row.get('smoke', {})).get('components', []):
            component_rows.append(dict(config=cfg, image_id=iid, component=c['id'], block_ids=' | '.join(c['block_ids']),
                unit_ids=' | '.join(c['unit_ids']), gold=c['gold_raw'], predicted=c['predicted_raw'],
                **c['compact_metrics'], raw_edits=c['raw_metrics']['edits']))
        rr = next(r for r in run['records'] if r['image_id'] == iid)
        timing_rows.append(dict(config=cfg, image_id=iid, run_id=run['run_id'],
            request_index=run['records'].index(rr), first_request=run['records'].index(rr) == 0,
            inference_seconds=rr.get('inference_seconds'), load_seconds=run.get('load_seconds'),
            device=json.dumps(out['engine']['device'], ensure_ascii=False),
            width=out['input']['width'], height=out['input']['height'], units=len(raw_units),
            memory_json=json.dumps(out.get('metrics', {}), ensure_ascii=False)))
        results.append(row)
    summary = aggregate(results)
    write(HERE/'results.json', dict(schema='ocr-comparison-v0.1', method_sha256=sha(HERE/'METHOD.md'),
        analysis_sha256=sha(__file__), inventory_sha256=sha(HERE/'inventory.json'),
        scope='posthoc development comparison, not PII detection/masking success', summary=summary, results=results))
    csv_write(HERE/'occurrences.csv', occurrence_rows); csv_write(HERE/'components.csv', component_rows)
    csv_write(HERE/'timing-records.csv', timing_rows)
    csv_write(HERE/'summary.csv', [dict(config=s['config'], label=s['label'], group=s['group'],
        exact=s['exact_no_whitespace'], strict=s['exact_strict'], denominator=s['occurrences'], joint=s['joint_exact_link_full'],
        **s['text_metrics'], **s['unique_geometry']) for s in summary])
    if not args.no_visuals:
        create_visuals(inv, outputs, golds)
    print(json.dumps([dict(config=s['config'], exact=s['exact_no_whitespace'], strict=s['exact_strict'], joint=s['joint_exact_link_full'],
        edits=s['text_metrics']['edits'], missed=s['unique_geometry']['missed_target_pixels']) for s in summary]))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8'); main()
