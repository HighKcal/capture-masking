"""Repair verified RapidOCR crop-frame mismatches without recognizing text."""

import copy
import hashlib
import json
import math

import cv2
import numpy as np


VERSION = 'detail-geometry-repair-v0.1'
FRAME_MISMATCH = 'native_inverse_frame_differs_from_actual_restored_crop'
GEOMETRY_SOURCE = 'RapidOCR CTC crop boxes projected by the verified forward crop inverse'
IDENTITY_KEYS = ('image_id', 'ocr_run_id', 'file_sha256', 'rgb_sha256', 'width', 'height', 'ocr_sha256')


class UnverifiedGeometry(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise UnverifiedGeometry(reason)


def digest(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def finite_number(value):
    return type(value) in (float, int) and math.isfinite(value)


def same_quad(polygon, size):
    require(isinstance(polygon, list) and len(polygon) == 4 and all(
        isinstance(point, list) and len(point) == 2 and all(finite_number(v) for v in point)
        and 0 <= point[0] <= size[0] and 0 <= point[1] <= size[1] for point in polygon), 'polygon_bounds')
    quad = np.asarray(polygon, dtype=np.float64)
    edges = np.roll(quad, -1, axis=0) - quad
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    require(np.all(cross > 0), 'polygon_not_clockwise_convex')
    return quad


def bind_identity(rgb, ocr, identity, details):
    source = ocr['input']
    require(set(identity) == set(IDENTITY_KEYS) and identity == details['identity']
            and ocr['image_id'] == identity['image_id'] and ocr['run_id'] == identity['ocr_run_id']
            and identity['file_sha256'] == source['sha256'] and identity['ocr_sha256'] == digest(ocr),
            'detail_geometry_identity')
    size = (identity['width'], identity['height'])
    require(all(type(value) is int and value > 0 for value in size)
            and rgb.mode == 'RGB' and rgb.size == size == (source['width'], source['height'])
            and hashlib.sha256(rgb.tobytes()).hexdigest() == identity['rgb_sha256'], 'detail_geometry_rgb')
    prefix = f"{identity['ocr_run_id']}:{identity['image_id']}:u"
    require(all(row['id'] == f'{prefix}{i:04d}' for i, row in enumerate(ocr['units'])), 'ocr_unit_order')
    return size


def trace_routes(ocr, detail, size):
    """Bind the observed native return rows back to their occurrence and crop."""
    require(detail['schema_version'] == 'ocr-coordinate-details-v0' and ocr['status'] == 'ok'
            and detail['settings'] == dict(return_word_box=True, return_single_char_box=True, use_cls=False,
                                           baseline_model_settings_digest=digest(ocr['engine'])),
            'detail_settings')
    evidence = detail['evidence']
    require(evidence['options_after_call'] == dict(return_word_box=False, return_single_char_box=False, use_cls=False),
            'reset_detail_options')
    traces, public = evidence['trace'], evidence['public']
    recognition, calc, before = (traces[key] for key in ('recognition', 'calculation', 'before_score_filter'))
    adapter = detail['transforms']['adapter']
    require(adapter['op_record'] == dict(padding_1={'left': 0, 'top': 0},
                                       preprocess={'ratio_h': 1.0, 'ratio_w': 1.0})
            and adapter['preprocess_shape'][:2] == [size[1], size[0]], 'nonidentity_preprocessing')
    texts, scores = recognition['texts'], recognition['scores']
    pre_filter = adapter['pre_filter']
    n = len(texts)
    require(recognition['texts'] == pre_filter['texts'] and scores == pre_filter['scores']
            and n == len(scores) == len(pre_filter['boxes']) == len(adapter['crop_shapes'])
            == len(recognition['ctc']) == len(recognition['word_info'])
            and all(isinstance(text, str) for text in texts)
            and all(finite_number(s) and 0 <= s <= 1 for s in scores), 'native_recognition_stream')
    nonblank = [i for i, text in enumerate(texts) if text.strip()]
    keep = [j for j, i in enumerate(nonblank) if scores[i] >= 0.5]
    require(calc == detail['transforms']['calculation']
            and calc['texts'] == before['texts'] == [texts[i] for i in nonblank]
            and calc['boxes'] == before['boxes'] == [pre_filter['boxes'][i] for i in nonblank]
            and calc['crop_shapes'] == [adapter['crop_shapes'][i] for i in nonblank]
            and calc['word_info'] == [recognition['word_info'][i] for i in nonblank]
            and before['scores'] == [scores[i] for i in nonblank]
            and calc['raw_rows'] == calc['returned_rows'] == before['word_results']
            and len(calc['returned_rows']) == len(nonblank)
            and all(isinstance(row, list) and row for row in calc['returned_rows']), 'native_calc_stream')
    require(all(public[key] == [before[key][j] for j in keep] for key in
                ('texts', 'scores', 'boxes', 'word_results')) and len(keep) == len(ocr['units']),
            'native_filtered_stream')
    require(len(detail['units']) == len(keep) and all(
        unit['text'] == public['texts'][i] == row['text']
        and unit['polygons'] == row['polygons'] == [public['boxes'][i]]
        and unit['confidence'] == public['scores'][i]
        and row['unit_id'] == unit['id'] and row['original_order_index'] == nonblank[keep[i]]
        for i, (row, unit) in enumerate(zip(detail['units'], ocr['units']))), 'detail_occurrence_stream')
    return recognition, calc, adapter, [(nonblank[j], j) for j in keep]


def frame_sizes(row, raw_quad, shape, size):
    transform = row['transform']
    quad = same_quad(raw_quad, size)
    edges = [float(np.linalg.norm(edge)) for edge in np.roll(quad, -1, axis=0) - quad]
    width = int(max(edges[0], edges[2]))
    height = int(max(edges[1], edges[3]))
    old_width, old_height = int(edges[0]), int(edges[3])
    require(width > 0 and height > 0 and old_width > 0 and old_height > 0
            and height / width < 1.5 and round(max(edges[1], edges[3]) / max(edges[0], edges[2]), 2) < 1.5
            and all(0 < x < size[0] and 0 < y < size[1] for x, y in quad),
            'rotated_clipped_or_degenerate_frame')
    require([width, height] != [old_width, old_height] and shape == [height, width, 3]
            and transform['status'] == 'uncertain' and transform['reasons'] == [FRAME_MISMATCH]
            and transform['forward_rot90'] is False and transform['inverse_rot90'] is False
            and transform['parent_was_clipped'] is False
            and transform['raw_crop_shape'] == transform['mapped_crop_shape'] == shape
            and transform['forward_quad'] == transform['restored_quad'] == raw_quad
            and transform['forward_crop_dimensions'] == [width, height]
            and transform['inverse_frame_dimensions'] == [old_width, old_height]
            and transform['inverse_integerization'] == 'native int() truncation toward zero, retained unchanged',
            'actual_crop_frame_not_proven')
    return quad.astype(np.float32), (width, height), (old_width, old_height)


def crop_rows(text, score, ctc, word_info, size):
    """Recreate RapidOCR's crop-local CTC boxes, before native inverse rounding."""
    width, height = size
    chars, columns, scores = (ctc[key] for key in ('tokens', 'columns', 'confidences'))
    count, ratio, max_ratio = (ctc[key] for key in ('timestep_count', 'wh_ratio', 'max_wh_ratio'))
    require(ctc['text'] == text and isinstance(chars, list) and chars
            and all(isinstance(c, str) and len(c) == 1 for c in chars)
            and ''.join(chars) == text and len(chars) == len(columns) == len(scores)
            and type(count) is int and count > 0
            and all(type(col) is int and 0 <= col < count for col in columns)
            and all(a < b for a, b in zip(columns, columns[1:]))
            and all(finite_number(s) and 0 <= s <= 1 for s in scores)
            and np.mean(scores).round(5).tolist() == score and word_info['confs'] == scores,
            'ctc_transcript_columns_or_scores')
    require(finite_number(ratio) and finite_number(max_ratio) and 0 < ratio <= max_ratio
            and ratio == width / height and word_info['line_txt_len'] == count * ratio / max_ratio,
            'ctc_column_frame')
    groups, group_cols, kinds = (word_info[key] for key in ('words', 'word_cols', 'word_types'))
    offsets = [i for i, char in enumerate(text) if not char.isspace()]
    require(groups and len(groups) == len(group_cols) == len(kinds)
            and all(group and len(group) == len(cols) and kind in ('en&num', 'cn')
                    for group, cols, kind in zip(groups, group_cols, kinds))
            and [char for group in groups for char in group] == [chars[i] for i in offsets]
            and [col for group in group_cols for col in group] == [columns[i] for i in offsets],
            'ctc_word_info_correspondence')
    column_width = width / word_info['line_txt_len']
    # CalRecBoxes' fallback width / len(text) lacks observed inter-column width.
    # Require at least one observed multi-character group in this repair path.
    group_widths = [(cols[-1] - cols[0]) * column_width / (len(cols) - 1)
                    for cols in group_cols if len(cols) > 1]
    require(group_widths, 'no_observed_intercolumn_width')
    half_width = sum(group_widths) / len(group_widths) / 2
    cells = []
    for offset in offsets:
        center = (columns[offset] + 0.5) * column_width
        cells.append([max(0, int(center - half_width)), min(width, int(center + half_width))])
    for left, right in zip(cells, cells[1:]):
        if left[1] > right[0]:
            overlap = left[1] - right[0]
            left[1] -= overlap / 2
            right[0] += overlap - overlap / 2
    require(all(0 <= left < right <= width for left, right in cells), 'ctc_crop_boxes_outside_frame')
    return offsets, [[[left, 0], [right, 0], [right, height], [left, height]] for left, right in cells]


def inverse(quad, size):
    width, height = size
    rectangle = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    forward = cv2.getPerspectiveTransform(quad, rectangle)
    success, reverse = cv2.invert(forward)
    require(bool(success) and np.isfinite(reverse).all(), 'perspective_not_invertible')
    return reverse


def project(poly, reverse):
    points = [np.dot(reverse, [x, y, 1]) for x, y in poly]
    require(all(np.isfinite(point).all() and point[2] > 0 for point in points), 'perspective_denominator')
    return [[float(x / z), float(y / z)] for x, y, z in points]


def legacy_order(points):
    # Narrow horizontal subset where CalRecBoxes sorts the two sides by y.
    center_x = np.mean(points, axis=0)[0]
    require(all(point[0] != center_x for point in points), 'unsupported_native_corner_order')
    left = sorted([p for p in points if p[0] < center_x], key=lambda p: p[1])
    right = sorted([p for p in points if p[0] > center_x], key=lambda p: p[1])
    require(len(left) == len(right) == 2 and left[0][1] < left[1][1] and right[0][1] < right[1][1],
            'unsupported_native_corner_order')
    return [left[0], right[0], right[1], left[1]]


def repaired_unit(row, pre_index, calc_index, trace, size):
    recognition, calc, adapter = trace
    raw_quad = adapter['pre_filter']['boxes'][pre_index]
    shape = calc['crop_shapes'][calc_index]
    quad, actual, legacy = frame_sizes(row, raw_quad, shape, size)
    ctc = recognition['ctc'][pre_index]
    word_info = calc['word_info'][calc_index]
    text, score = row['text'], recognition['scores'][pre_index]
    offsets, cells = crop_rows(text, score, ctc, word_info, actual)
    native = calc['raw_rows'][calc_index]
    require(len(native) == len(offsets), 'native_ctc_row_count')
    left, top = int(quad[:, 0].min()), int(quad[:, 1].min())
    legacy_inverse = inverse(quad - np.asarray([left, top], dtype=np.float32), legacy)
    current_inverse = inverse(quad, actual)
    width, height = actual
    reference = project([[0, 0], [width, 0], [width, height], [0, height]], current_inverse)
    require(np.allclose(reference, quad, rtol=0.0, atol=1e-7), 'forward_inverse_corner_reprojection')
    tokens = []
    for i, (offset, cell) in enumerate(zip(offsets, cells)):
        old_polygon = [[int(x + left), int(y + top)] for x, y in project(cell, legacy_inverse)]
        old_polygon = legacy_order(old_polygon)
        require(native[i] == [text[offset], word_info['confs'][i], old_polygon], 'native_polygon_replay_mismatch')
        polygon = [[round(x, 9), round(y, 9)] for x, y in project(cell, current_inverse)]
        same_quad(polygon, size)
        tokens.append(dict(text=text[offset], start=offset, end=offset + 1, polygon=polygon,
                           score=ctc['confidences'][offset], native_score=native[i][1],
                           score_status='native_aligned' if i == offset else 'raw_ctc_reassociated_after_whitespace',
                           ctc_token_index=offset, ctc_column=ctc['columns'][offset], geometry_source=GEOMETRY_SOURCE,
                           alignment='whole nonspace stream and native crop box replay verified',
                           score_meaning='recognition confidence; not location accuracy probability'))
    transform = dict(row['transform'], status='ok', reasons=[], inverse_frame_dimensions=list(actual),
                     legacy_inverse_frame_dimensions=list(legacy), legacy_reasons=[FRAME_MISMATCH],
                     repair_method=VERSION)
    evidence = dict(ctc=copy.deepcopy(ctc), original_text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                    omitted_whitespace_offsets=[i for i, char in enumerate(text) if char.isspace()])
    return dict(row, status='ok', reasons=[], tokens=tokens, transform=transform, evidence=evidence)


def repair(rgb, ocr, identity, raw_details):
    """Return a copied detail artifact and deterministic evidence of any repair."""
    size = bind_identity(rgb, ocr, identity, raw_details)
    raw_hash = digest(raw_details)
    effective = copy.deepcopy(raw_details)
    actions, retained = [], []
    rows = [i for i, row in enumerate(raw_details['units'])
            if row.get('status') == 'uncertain' and row.get('tokens') == []
            and row.get('reasons') in (['ValueError'], [f'ValueError: {FRAME_MISMATCH}'])
            and row.get('transform', {}).get('reasons') == [FRAME_MISMATCH]]
    if rows and raw_details['status'] == 'partial_failure':
        try:
            recognition, calc, adapter, mapping = trace_routes(ocr, raw_details, size)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            retained.extend(dict(unit_id=raw_details['units'][i]['unit_id'], reason='unverified_native_trace') for i in rows)
        else:
            for i in rows:
                row = raw_details['units'][i]
                try:
                    fixed = repaired_unit(row, *mapping[i], (recognition, calc, adapter), size)
                except UnverifiedGeometry as error:
                    retained.append(dict(unit_id=row['unit_id'], reason=str(error)))
                    continue
                except (KeyError, IndexError, TypeError, ValueError, AttributeError, cv2.error):
                    retained.append(dict(unit_id=row['unit_id'], reason='malformed_native_unit'))
                    continue
                effective['units'][i] = fixed
                transform = fixed['transform']
                actions.append(dict(unit_id=row['unit_id'], order_index=i, reason=FRAME_MISMATCH,
                                    token_count=len(fixed['tokens']),
                                    forward_size=transform['forward_crop_dimensions'],
                                    legacy_inverse_size=transform['legacy_inverse_frame_dimensions']))
            if actions and not effective['reasons'] and not effective['evidence']['warnings'] and all(
                    row['status'] == 'ok' for row in effective['units']):
                effective['status'] = 'ok'
    receipt = dict(version=VERSION, raw_sha256=raw_hash, effective_sha256=digest(effective),
                   repaired_count=len(actions), repaired_units=actions, retained_uncertain=retained)
    return effective, receipt
