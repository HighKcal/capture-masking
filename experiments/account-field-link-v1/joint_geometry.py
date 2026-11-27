"""Bounded field geometry from the unchanged original RGB raster and OCR boxes."""

import cv2
import numpy as np


MAX_PIXELS = 16_000_000
MAX_UNITS = 300
MAX_LINES = 2400
MAX_SIDES = 5
MAX_TABLE_NEIGHBORS = 8
EDGE_CONTRAST = 18


def _arr(rgb):
    return np.asarray(rgb)[..., :3]


def _box(unit):
    return unit['_g']['box']


def _height(unit):
    return unit['_g']['h']


def _center(box, axis):
    return (box[axis] + box[axis + 2]) / 2


def _axis_lines(image, units, axis, minimum):
    # Axis derivatives keep a rectangle side even when a connector touches it.
    if axis == 'h':
        delta = cv2.absdiff(image[1:], image[:-1])
        mask = np.pad((delta.max(axis=2) >= EDGE_CONTRAST).astype('uint8'), ((1, 0), (0, 0)))
    else:
        delta = cv2.absdiff(image[:, 1:], image[:, :-1])
        mask = np.pad((delta.max(axis=2) >= EDGE_CONTRAST).T.astype('uint8'), ((1, 0), (0, 0)))
    # Text interiors are not independent evidence of a UI border or connector.
    for unit in units:
        left, top, right, bottom = _box(unit)
        l, t = max(0, int(left)), max(0, int(top))
        r, b = int(np.ceil(right)), int(np.ceil(bottom))
        if axis == 'h':
            mask[t:b, l:r] = 0
        else:
            mask[l:r, t:b] = 0
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    opened = cv2.getStructuringElement(cv2.MORPH_RECT, (minimum, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close, borderType=cv2.BORDER_REPLICATE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opened, anchor=((minimum - 1) // 2, 0),
                            borderType=cv2.BORDER_REPLICATE)
    positions, changes = np.nonzero(np.diff(np.pad(mask, ((0, 0), (1, 1))), axis=1))
    if len(changes) > MAX_LINES * 2:
        raise ValueError('Account geometry line limit exceeded')
    lines = []
    for index in range(0, len(changes), 2):
        at, lo, hi = int(positions[index]), int(changes[index]), int(changes[index + 1])
        if hi - lo >= minimum:
            lines.append(dict(axis=axis, at=at, lo=lo, hi=hi))
    return lines


def _covers(line, low, high, tolerance):
    return line['lo'] <= low + tolerance and line['hi'] >= high - tolerance


def _independent(lines, tolerance):
    """Drop duplicate gradient sides from the same thin stroke."""
    result, seen = [], set()
    for side in lines:
        at = side['at']
        if not any(pos in seen for pos in range(at - tolerance, at + tolerance + 1)):
            result.append(side)
            seen.add(at)
    return result


def _cell(unit, units, horizontal, vertical):
    left, top, right, bottom = _box(unit)
    h = _height(unit)
    margin = max(2, .08 * h)
    wide = [side for side in horizontal if side['lo'] <= left - margin and side['hi'] >= right + margin]
    tops = [side for side in wide if margin <= top - side['at'] <= 1.45 * h]
    bottoms = [side for side in wide if margin <= side['at'] - bottom <= 1.45 * h]
    tops = _independent(sorted(tops, key=lambda side: -side['at']), 1)[:MAX_SIDES]
    bottoms = _independent(sorted(bottoms, key=lambda side: side['at']), 1)[:MAX_SIDES]
    plausible = []
    for upper in tops:
        for lower in bottoms:
            y0, y1 = upper['at'], lower['at']
            if not 1.15 * h <= y1 - y0 <= 3.6 * h:
                continue
            corner = min(.56 * h, .24 * (y1 - y0))
            ls = (upper['lo'] + lower['lo']) / 2
            rs = (upper['hi'] + lower['hi']) / 2
            eligible = [side for side in vertical if _covers(side, y0, y1, corner)
                        and abs(side['lo'] - y0) <= corner and abs(side['hi'] - y1) <= corner]
            lhs = [side for side in eligible if side['at'] <= left - margin and abs(side['at'] - ls) <= corner]
            rhs = [side for side in eligible if side['at'] >= right + margin and abs(side['at'] - rs) <= corner]
            lhs = _independent(sorted(lhs, key=lambda side: abs(side['at'] - ls)), 1)[:MAX_SIDES]
            rhs = _independent(sorted(rhs, key=lambda side: abs(side['at'] - rs)), 1)[:MAX_SIDES]
            for west in lhs:
                for east in rhs:
                    x0, x1 = west['at'], east['at']
                    if any(abs(side['lo'] - x0) > corner or abs(side['hi'] - x1) > corner for side in (upper, lower)):
                        continue
                    box = [x0, y0, x1, y1]
                    area = (x1 - x0) * (y1 - y0)
                    plausible.append((area, box, [upper, lower, west, east]))
    for _, box, sides in sorted(plausible, key=lambda entry: entry[0]):
        if not any(other['id'] != unit['id'] and box[0] <= _center(_box(other), 0) <= box[2]
                   and box[1] <= _center(_box(other), 1) <= box[3] for other in units):
            return dict(box=box, unit_id=unit['id'], evidence='four_original_rgb_sides_one_ocr_unit', sides=sides)
    return None


def analyze(rgb, units):
    if rgb is None or not units:
        return {}, []
    image = _arr(rgb)
    height, width = image.shape[:2]
    if height * width > MAX_PIXELS or len(units) > MAX_UNITS:
        raise ValueError('Account geometry input limit exceeded')
    if min(height, width) < 3:
        return {}, []
    scale = float(np.median([_height(unit) for unit in units]))
    minimum = max(7, min(18, round(.34 * scale)))
    horizontal = _axis_lines(image, units, 'h', minimum)
    vertical = _axis_lines(image, units, 'v', minimum)
    if len(horizontal) + len(vertical) > MAX_LINES:
        raise ValueError('Account geometry line limit exceeded')
    found = {unit['id']: cell for unit in units if (cell := _cell(unit, units, horizontal, vertical))}
    return found, horizontal + vertical


def _visible_stem(image, x, low, high, scale):
    radius = max(4, round(.18 * scale))
    left, right = x - radius, x + radius
    if left < 0 or right >= image.shape[1] or high - low < 3:
        return False
    lane = image[low:high, left:right + 1].astype('int16')
    left_bg, right_bg = lane[:, 0], lane[:, -1]
    if (np.abs(left_bg - right_bg).max(axis=1) >= EDGE_CONTRAST).mean() > .22:
        return False
    both = (np.abs(lane[:, 1:-1] - left_bg[:, None]).max(axis=2) >= EDGE_CONTRAST) & (
        np.abs(lane[:, 1:-1] - right_bg[:, None]).max(axis=2) >= EDGE_CONTRAST)
    # A single narrow x lane must carry the stem along the entire gap.
    central = max(1, min(radius - 1, round(.09 * scale)))
    for offset in range(radius - 1 - central, radius + central):
        occupied = both[:, offset]
        if occupied.mean() >= .83 and occupied[:3].any() and occupied[-3:].any():
            holes = np.flatnonzero(occupied)
            if len(holes) < 2 or np.diff(holes).max() <= 4:
                return True
    return False


def connector(rgb, source_cell, target_cell):
    if rgb is None or not source_cell or not target_cell:
        return False
    first, second = sorted((source_cell['box'], target_cell['box']), key=lambda box: box[1])
    scale = min(first[3] - first[1], second[3] - second[1])
    a, b = _center(first, 0), _center(second, 0)
    gap = second[1] - first[3]
    if abs(a - b) > .20 * scale or not 4 <= gap <= 2.4 * scale:
        return False
    image = _arr(rgb)
    center = round((a + b) / 2)
    radius = max(8, round(.60 * scale))
    lo = max(first[0], second[0], center - radius)
    hi = min(first[2], second[2], center + radius)
    exterior = image[first[3] + 2:second[1] - 2, lo:hi]
    if len(exterior) >= 2 and exterior.shape[1] >= 4:
        across = cv2.absdiff(exterior[1:], exterior[:-1]).max(axis=2) >= EDGE_CONTRAST
        if np.any(across.mean(axis=1) >= .65):
            return False
    return _visible_stem(image, center, first[3] + 1, second[1] - 1, scale)


def _separate_stroke(lines, axis, low, high, cross_low, cross_high):
    cross_len = cross_high - cross_low
    return any(line['axis'] == axis and low + 2 < line['at'] < high - 2
               and min(line['hi'], cross_high) - max(line['lo'], cross_low) >= .65 * cross_len for line in lines)


def _same_exterior(image, direction, low, high, cross_low, cross_high, source_background):
    if high - low < 5 or cross_high <= cross_low:
        return False
    if direction == 'below':
        lane = image[low:high, cross_low:cross_high]
    else:
        lane = image[cross_low:cross_high, low:high].transpose(1, 0, 2)
    if not lane.size:
        return False
    # Median across the approach removes thin connector strokes and preserves a panel fill change.
    profile = np.median(lane, axis=1)
    endpoint = max(2, len(profile) // 5)
    colors = [np.median(profile[start:start + endpoint], axis=0)
              for start in range(0, len(profile) - endpoint + 1, endpoint)]
    if source_background is not None:
        colors.append(source_background)
    return bool(np.max(np.ptp(colors, axis=0)) < 32)


def _table(a, b, units, lines):
    al, at, ar, ab = _box(a)
    bl, bt, br, bb = _box(b)
    h = max(_height(a), _height(b))
    if abs(al - bl) > .8 * h or min(ar, br) <= max(al, bl):
        return None
    horizontal = [line for line in lines if line['axis'] == 'h' and ab < line['at'] < bt
                  and _covers(line, min(al, bl), max(ar, br), .18 * h)]
    if len(_independent(horizontal, 2)) != 1:
        return None
    outer = (min(al, bl), max(ar, br))
    columns = [line for line in lines if line['axis'] == 'v' and _covers(line, at, bb, .25 * h)]
    if any(outer[0] < line['at'] < outer[1] for line in columns):
        return None
    exclusions = {a['id'], b['id']}

    def neighbors(unit):
        others = [other for other in units if other['id'] not in exclusions and
                  abs(_center(_box(other), 1) - _center(_box(unit), 1)) <= .40 * h]
        return sorted(others, key=lambda other: abs(_box(other)[0] - outer[0]))[:MAX_TABLE_NEIGHBORS]

    near_values = neighbors(b)
    for other in neighbors(a):
        ol, ot, oright, ob = _box(other)
        for value in near_values:
            vl, vt, vr, vb = _box(value)
            if abs(ol - vl) > .8 * h or min(oright, vr) <= max(ol, vl):
                continue
            other_outer = (min(ol, vl), max(oright, vr))
            left, right = sorted((outer, other_outer))
            if left[1] >= right[0] or any(other_outer[0] < line['at'] < other_outer[1] for line in columns):
                continue
            vertical = next((line for line in columns if left[1] < line['at'] < right[0]
                             and _covers(line, min(at, ot), max(bb, vb), .25 * h)), None)
            row = next((line for line in horizontal if max(ab, ob) < line['at'] < min(bt, vt)
                        and _covers(line, left[0], right[1], .25 * h)), None)
            if vertical and row:
                return dict(kind='same_column_grid', other_label_id=other['id'], other_value_id=value['id'],
                            column_separator=vertical, row_separator=row)
    return None


def boundary_evidence(a, b, units, rgb, cells, lines, direction, blocked):
    if rgb is None or direction not in ('right', 'below'):
        return None
    if direction == 'below' and blocked == 'horizontal_visual_boundary':
        grid = _table(a, b, units, lines)
        if grid:
            return grid
    if blocked not in ('vertical_visual_boundary', 'horizontal_visual_boundary', 'different_background_panels'):
        return None
    target = cells.get(b['id'])
    source = cells.get(a['id'])
    if not target:
        return None
    linked = bool(source and direction == 'below' and connector(rgb, source, target))
    if source and not linked:
        return None
    image = _arr(rgb)
    axis = 1 if direction == 'below' else 0
    cross = 1 - axis
    a_box, b_box, cell_box = _box(a), _box(b), target['box']
    begin = source['box'][axis + 2] if source else a_box[axis + 2]
    entry = cell_box[axis]
    c0, c1 = max(a_box[cross], b_box[cross]), min(a_box[cross + 2], b_box[cross + 2])
    scale = min(_height(a), _height(b))
    if begin >= entry or c1 - c0 < .28 * scale:
        return None
    entry_side = target['sides'][0 if axis == 1 else 2]
    if not _covers(entry_side, c0, c1, 1):
        return None
    line_axis = 'h' if axis == 1 else 'v'
    if _separate_stroke(lines, line_axis, begin, entry, c0, c1):
        return None
    source_background = None
    if not source:
        left, top, _, bottom = a_box
        x = int(left - .25 * _height(a))
        if x >= 0:
            source_background = np.median(image[int(top):int(np.ceil(bottom)), x], axis=0)
    if not _same_exterior(image, direction, int(np.ceil(begin)) + 2, int(entry) - 2,
                          max(0, int(np.ceil(c0))), int(c1), source_background):
        return None
    return dict(kind='target_cell_entry', target=target, source=source, connector=linked,
                entry_side=entry_side, exterior_gap=[begin, entry])


def choice_group(label, values, units, rgb, cells, lines):
    if rgb is None or len(values) != 2 or len({value['id'] for value in values}) != 2:
        return None
    if any(value['id'] not in cells for value in values):
        return None
    a, b = sorted((cells[value['id']] for value in values), key=lambda cell: cell['box'][0])
    left, right = a['box'], b['box']
    label_box = _box(label)
    h = max(_height(label), *(_height(value) for value in values))
    x0, x1, center = _center(left, 0), _center(right, 0), _center(label_box, 0)
    gap_lo, gap_hi = label_box[3], min(left[1], right[1])
    if (not x0 < center < x1 or left[2] >= right[0] or abs(center - (x0 + x1) / 2) > .6 * h
            or abs(left[1] - right[1]) > .30 * h or not .45 * h <= gap_hi - gap_lo <= 3.6 * h):
        return None
    # A vertical branch must meet the horizontal rail and the label or input edge.
    tolerance = max(3, .14 * h)
    arms = [line for line in lines if line['axis'] == 'v']
    for rail in lines:
        y = rail['at']
        if rail['axis'] != 'h' or not gap_lo + 3 <= y <= gap_hi - 3 or not _covers(rail, x0, x1, tolerance):
            continue
        if abs(rail['lo'] - x0) > tolerance or abs(rail['hi'] - x1) > tolerance:
            continue
        endpoints = ((center, gap_lo, y), (x0, y, left[1]), (x1, y, right[1]))
        stems = [next((line for line in arms if abs(line['at'] - x) <= tolerance and
                       abs(line['lo'] - low) <= tolerance and abs(line['hi'] - high) <= tolerance), None)
                 for x, low, high in endpoints]
        if all(stems):
            if any(_separate_stroke(lines, 'h', low, high, x - .45 * h, x + .45 * h)
                   for x, low, high in endpoints):
                return None
            ignored = {label['id'], a['unit_id'], b['unit_id']}
            if any(unit['id'] not in ignored and gap_lo < _center(_box(unit), 1) < gap_hi
                   and x0 < _center(_box(unit), 0) < x1 for unit in units):
                return None
            return dict(kind='visible_branch_connector', cells=[a, b], rail=rail, stems=stems)
    return None
