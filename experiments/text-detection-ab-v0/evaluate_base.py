"""Validate and score TXT-02 JSONL using the frozen source-span contract."""

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import json
from pathlib import Path


TYPES = ("NAME", "PHONE", "EMAIL", "ACCOUNT_ID")
NAME_TYPES = TYPES[:1]
CONTACT_TYPES = TYPES[1:]
CONFIGS = {
    "N0": NAME_TYPES, "N1": NAME_TYPES, "N2": NAME_TYPES,
    "C0": CONTACT_TYPES, "C1": CONTACT_TYPES,
    "BASE": TYPES, "FIXED": TYPES, "FIELD": TYPES,
}
COMPARISONS = (
    ("N0", "N1"), ("N1", "N2"), ("C0", "C1"), ("BASE", "FIXED"), ("FIXED", "FIELD"),
    ("raw_model", "N0"), ("raw_model", "N1"), ("raw_model", "N2"),
)
AUDIT_STAGES = {
    "N0": ("N0",), "N1": ("N1",), "N2": ("N1", "field"),
    "BASE": ("N0",), "FIXED": ("N1",), "FIELD": ("N1", "field"),
}
SPLITS = ("contract", "dev", "eval", "holdout_policy")
STATUSES = ("ok", "partial_failure", "failure")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonempty_string(value):
    return isinstance(value, str) and bool(value)


def _segment_key(segments, blocks, context):
    _require(isinstance(segments, list) and segments, f"{context}: missing segments")
    order = {block_id: index for index, block_id in enumerate(blocks)}
    previous = (-1, -1)
    key = []
    for segment in segments:
        _require(isinstance(segment, dict), f"{context}: invalid segment")
        block_id = segment.get("block_id")
        _require(isinstance(block_id, str) and block_id in blocks,
                 f"{context}: unknown block {block_id!r}")
        start, end = segment.get("start"), segment.get("end")
        _require(type(start) is int and type(end) is int and
                 0 <= start < end <= len(blocks[block_id]),
                 f"{context}: invalid code-point interval in {block_id!r}")
        original = blocks[block_id][start:end]
        _require(segment.get("text") == original and "\n" not in original and "\r" not in original,
                 f"{context}: segment text must match the original slice and omit newlines")
        _require((order[block_id], start) > previous,
                 f"{context}: segments must be ordered, disjoint, and merged when adjacent")
        key.append((block_id, start, end))
        previous = (order[block_id], end)
    return tuple(key)


def _check_split(seen, value, split, description):
    previous_split = seen.setdefault(value, split)
    _require(previous_split == split, f"dev/eval leakage: {description} appears in both splits")


def validate_cases(cases):
    """Raise ValueError for malformed cases or identifiable dev/eval leakage."""
    case_ids = set()
    families, originals, names = {}, {}, {}
    for case in cases:
        _require(isinstance(case, dict), "case must be an object")
        case_id, split, family = case.get("id"), case.get("split"), case.get("family")
        _require(_nonempty_string(case_id) and case_id not in case_ids,
                 f"invalid or duplicate case id: {case_id!r}")
        case_ids.add(case_id)
        _require(split in SPLITS and _nonempty_string(family), f"{case_id}: invalid split/family")
        tags = case.get("tags")
        _require(isinstance(tags, list) and all(isinstance(tag, str) for tag in tags),
                 f"{case_id}: tags must be strings")
        block_list = case.get("blocks")
        _require(isinstance(block_list, list) and block_list, f"{case_id}: missing blocks")
        blocks = {}
        for block in block_list:
            _require(isinstance(block, dict), f"{case_id}: invalid block")
            block_id = block.get("id")
            _require(_nonempty_string(block_id) and block_id not in blocks and
                     isinstance(block.get("text"), str), f"{case_id}: invalid or duplicate block")
            blocks[block_id] = block["text"]
        if split in ("dev", "eval"):
            _check_split(families, family, split, f"family {family!r}")
            _check_split(originals, tuple(blocks.values()), split, "identical block texts")

        gold = case.get("gold")
        _require(isinstance(gold, list), f"{case_id}: gold must be a list")
        occurrence_ids, gold_keys = set(), set()
        for occurrence in gold:
            _require(isinstance(occurrence, dict), f"{case_id}: invalid gold occurrence")
            occurrence_id, entity_type = occurrence.get("id"), occurrence.get("type")
            _require(_nonempty_string(occurrence_id) and occurrence_id not in occurrence_ids,
                     f"{case_id}: invalid or duplicate gold occurrence id {occurrence_id!r}")
            occurrence_ids.add(occurrence_id)
            _require(entity_type in TYPES, f"{case_id}/{occurrence_id}: invalid gold type")
            key = (entity_type, _segment_key(occurrence.get("segments"), blocks,
                                            f"{case_id}/{occurrence_id}"))
            _require(key not in gold_keys, f"{case_id}/{occurrence_id}: duplicate gold location")
            gold_keys.add(key)
            if split in ("dev", "eval") and entity_type == "NAME":
                _check_split(names, _surface(occurrence), split, "annotated NAME text")

        excluded = case.get("excluded", [])
        _require(isinstance(excluded, list), f"{case_id}: excluded must be a list")
        for entry in excluded:
            _require(isinstance(entry, dict) and _nonempty_string(entry.get("reason")),
                     f"{case_id}: excluded entry must have a reason")
            _segment_key(entry.get("segments"), blocks, f"{case_id}/excluded")
        links = case.get("links", [])
        _require(isinstance(links, list), f"{case_id}: links must be a list")
        for link in links:
            _require(isinstance(link, dict) and link.get("relation") == "same_field" and
                     link.get("joiner") == "", f"{case_id}: invalid link relationship")
            linked = link.get("blocks")
            _require(isinstance(linked, list) and len(linked) >= 2 and
                     all(isinstance(block, str) and block in blocks for block in linked) and
                     len(set(linked)) == len(linked), f"{case_id}: invalid linked blocks")


def _surface(occurrence):
    return "".join(segment["text"] for segment in occurrence["segments"])


def _position(occurrence):
    return tuple((s["block_id"], s["start"], s["end"]) for s in occurrence["segments"])


def _key(occurrence):
    return occurrence["type"], _position(occurrence)


def _predictions(items, blocks, entity_types, context):
    _require(isinstance(items, list), f"{context}: predictions must be a list")
    result, ids = {}, {}
    for item in items:
        _require(isinstance(item, dict), f"{context}: invalid prediction")
        item_id, entity_type, sources = item.get("id"), item.get("type"), item.get("sources")
        _require(_nonempty_string(item_id) and entity_type in entity_types,
                 f"{context}: invalid prediction id/type")
        _require(isinstance(sources, list) and bool(sources) and all(map(_nonempty_string, sources)),
                 f"{context}/{item_id}: invalid sources")
        key = (entity_type, _segment_key(item.get("segments"), blocks, f"{context}/{item_id}"))
        _require(ids.setdefault(item_id, key) == key, f"{context}: reused prediction id {item_id!r}")
        merged = result.setdefault(key, {**item, "sources": []})
        merged["sources"] = list(dict.fromkeys(merged["sources"] + sources))
        if merged["id"] != item_id and item_id not in merged.get("duplicate_ids", []):
            merged["duplicate_ids"] = [*merged.get("duplicate_ids", []), item_id]
    return result


def _intersection(first, second):
    return sum(max(0, min(end_a, end_b) - max(start_a, start_b))
               for block_a, start_a, end_a in first
               for block_b, start_b, end_b in second if block_a == block_b)


def _partial_matches(missed, extras):
    candidates = []
    for gold_key in missed:
        for pred_key in extras:
            if gold_key[0] != pred_key[0]:
                continue
            gold_segments, pred_segments = gold_key[1], pred_key[1]
            overlap = _intersection(gold_segments, pred_segments)
            if overlap:
                union = sum(end - start for _, start, end in gold_segments + pred_segments) - overlap
                candidates.append((Fraction(-overlap, union), gold_key, pred_key))
    gold_pairs, pred_pairs = {}, {}
    for negative_iou, gold_key, pred_key in sorted(candidates):
        if gold_key in gold_pairs or pred_key in pred_pairs:
            continue
        pair = {
            "gold_id": missed[gold_key]["id"], "prediction_id": extras[pred_key]["id"],
            "iou": float(-negative_iou),
            "kind": "boundary" if len(gold_key[1]) == len(pred_key[1]) == 1 else "grouping",
        }
        gold_pairs[gold_key] = pred_pairs[pred_key] = pair
    return gold_pairs, pred_pairs


def _score(gold, predictions, excluded, status, duplicate_count=0, rejections=None):
    missed = {key: gold[key] for key in sorted(gold.keys() - predictions.keys())}
    extras = {key: predictions[key] for key in sorted(predictions.keys() - gold.keys())}
    gold_pairs, pred_pairs = _partial_matches(missed, extras)
    repeated = Counter((item["type"], _surface(item)) for item in gold.values())

    def gold_record(item):
        return {"gold": item, "repeated": repeated[(item["type"], _surface(item))] > 1,
                "multi_segment": len(item["segments"]) > 1}

    result = {
        "status": status, "duplicate_predictions": duplicate_count,
        "prediction_count": len(predictions),
        "tp": [{**gold_record(gold[key]), "prediction": predictions[key]}
               for key in sorted(gold.keys() & predictions.keys())],
        "fp": [], "fn": [],
    }
    for key, prediction in extras.items():
        exclusions = [item for item in excluded if _intersection(key[1], _position(item))]
        pair = pred_pairs.get(key)
        result["fp"].append({
            "prediction": prediction,
            "kind": "policy" if exclusions else pair["kind"] if pair else "spurious",
            "partial": pair, "excluded_overlaps": exclusions,
            "wrong_type_gold_ids": [item["id"] for other, item in gold.items()
                                    if other[0] != key[0] and _intersection(key[1], other[1])],
        })
    for key, item in missed.items():
        pair = gold_pairs.get(key)
        reasons = rejections.get(key, []) if rejections else []
        result["fn"].append({
            **gold_record(item), "partial": pair,
            "kind": pair["kind"] if pair else "failure" if status == "failure" else "missed",
            "policy_rejections": reasons,
        })
    return result


def _counts(tp, fp, fn):
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {"tp": tp, "fp": fp, "fn": fn, "gold": tp + fn, "predicted": tp + fp,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def _summarize(scores, entity_types):
    outcomes = {outcome: [record for score in scores for record in score[outcome]]
                for outcome in ("tp", "fp", "fn")}
    outcome_counts = {outcome: Counter((record.get("gold") or record["prediction"])["type"]
                                      for record in records) for outcome, records in outcomes.items()}
    by_type = {entity_type: _counts(*(counts[entity_type] for counts in outcome_counts.values()))
               for entity_type in entity_types}
    diagnostic_counts = Counter(record["kind"] for record in outcomes["fp"])
    status_counts = Counter(score["status"] for score in scores)
    return {
        "cases": len(scores), "by_type": by_type,
        "micro": _counts(*(len(records) for records in outcomes.values())),
        "statuses": {status: status_counts[status] for status in STATUSES},
        "ok_zero_candidates": sum(score["status"] == "ok" and score["prediction_count"] == 0
                                  for score in scores),
        "missing_outputs": sum(score.get("output_missing", False) for score in scores),
        "duplicate_predictions": sum(score["duplicate_predictions"] for score in scores),
        "diagnostics": {
            "boundary_pairs": sum(bool(record["partial"] and record["partial"]["kind"] == "boundary")
                                  for record in outcomes["fp"]),
            "grouping_pairs": sum(bool(record["partial"] and record["partial"]["kind"] == "grouping")
                                  for record in outcomes["fp"]),
            "policy_fp": diagnostic_counts["policy"], "spurious_fp": diagnostic_counts["spurious"],
            "policy_fn": sum(bool(record["policy_rejections"]) for record in outcomes["fn"]),
            "wrong_type_overlap_fp": sum(bool(record["wrong_type_gold_ids"]) for record in outcomes["fp"]),
        },
        "repeated_positions": {outcome: sum(record["repeated"] for record in outcomes[outcome])
                               for outcome in ("tp", "fn")},
        "multi_segment": {
            "tp": sum(record["multi_segment"] for record in outcomes["tp"]),
            "fn": sum(record["multi_segment"] for record in outcomes["fn"]),
            "fp": sum(len(record["prediction"]["segments"]) > 1 for record in outcomes["fp"]),
        },
    }


def _delta(baseline, comparison, gold):
    return {
        "added": [{"prediction": comparison[key], "exact_gold": gold.get(key)}
                  for key in sorted(comparison.keys() - baseline.keys())],
        "removed": [{"prediction": baseline[key], "exact_gold": gold.get(key)}
                    for key in sorted(baseline.keys() - comparison.keys())],
    }


def _delta_counts(by_direction):
    result = {}
    for direction, records in by_direction.items():
        exact_count = sum(record["exact_gold"] is not None for record in records)
        result[f"{direction}_tp"] = exact_count
        result[f"{direction}_fp"] = len(records) - exact_count
    return result


def _summarize_delta(rows, comparison, entity_types):
    deltas = [detail["comparisons"][comparison] for detail in rows if comparison in detail["comparisons"]]
    by_direction = {direction: [record for delta in deltas for record in delta[direction]]
                    for direction in ("added", "removed")}
    result = {
        "cases": len(deltas), "skipped_cases": len(rows) - len(deltas),
        **_delta_counts(by_direction),
        "by_type": {entity_type: _delta_counts({
            direction: [record for record in records if record["prediction"]["type"] == entity_type]
            for direction, records in by_direction.items()}) for entity_type in entity_types},
    }
    if comparison == "N1_to_N2":
        field = [record for record in by_direction["added"] if record["field_supplement"]]
        result["field_supplements"] = _delta_counts({"added": field})
        result["without_raw_name_overlap"] = _delta_counts({
            "added": [record for record in field if record["raw_name_overlap"] is False]})
        result["unknown_raw_name_overlap"] = _delta_counts({
            "added": [record for record in field if record["raw_name_overlap"] is None]})
    return result


def _token_records(gold, boundaries, blocks, case_id):
    _require(isinstance(boundaries, dict), f"{case_id}: token_boundaries must be an object")
    offsets = {}
    for block_id, entries in boundaries.items():
        _require(block_id in blocks and isinstance(entries, dict),
                 f"{case_id}: invalid token boundary block {block_id!r}")
        offsets[block_id] = {}
        for edge in ("starts", "ends"):
            positions = entries.get(edge)
            _require(isinstance(positions, list) and all(type(pos) is int and
                     0 <= pos <= len(blocks[block_id]) for pos in positions),
                     f"{case_id}/{block_id}: invalid token {edge}")
            offsets[block_id][edge] = set(positions)
    records = []
    for occurrence in gold.values():
        segments = []
        for block, start, end in _position(occurrence):
            edges = offsets.get(block)
            segments.append({"start": start in edges["starts"] if edges is not None else None,
                             "end": end in edges["ends"] if edges is not None else None})
        known = all(edge is not None for segment in segments for edge in segment.values())
        records.append({"gold_id": occurrence["id"], "type": occurrence["type"],
                        "representable": all(all(segment.values()) for segment in segments) if known else None,
                        "cross_block": len({segment["block_id"] for segment in occurrence["segments"]}) > 1,
                        "segment_edges": segments})
    return records


def _token_summary(records):
    counts = Counter(record["representable"] for record in records)
    known = counts[True] + counts[False]
    return {"gold": len(records), "representable": counts[True], "not_representable": counts[False],
            "unknown": counts[None], "rate_among_known": counts[True] / known if known else None,
            "cross_block": sum(record["cross_block"] for record in records)}


def _policy_rejections(audit, config):
    rejected = defaultdict(list)
    for stage in AUDIT_STAGES.get(config, ()):
        for entry in audit.get(stage, []):
            if entry["action"] == "reject":
                rejected[_key(entry["candidate"])].append({"source": stage, "reason": entry["reason"]})
    return rejected


def _case_detail(case, output):
    case_id = case["id"]
    blocks = {block["id"]: block["text"] for block in case["blocks"]}
    gold = {_key(item): item for item in case["gold"]}
    holdout = case["split"] == "holdout_policy"
    output_missing = output is None
    if output_missing:
        output = {"status": "failure", "errors": ["missing case output"]}
    status = output.get("status")
    _require(status in STATUSES, f"{case_id}: invalid output status")
    errors = output.get("errors", [])
    _require(isinstance(errors, list), f"{case_id}: errors must be a list")
    configs = output.get("configs", {})
    config_statuses = output.get("config_statuses", {})
    _require(isinstance(configs, dict) and isinstance(config_statuses, dict),
             f"{case_id}: invalid configs/config_statuses")
    _require(configs.keys() <= CONFIGS.keys() and config_statuses.keys() <= CONFIGS.keys(),
             f"{case_id}: unknown configuration")
    excluded = case.get("excluded", [])
    audit = output.get("policy_audit", {})
    _require(isinstance(audit, dict), f"{case_id}: policy_audit must be an object")
    raw_present = "raw_names" in output
    raw_items = output.get("raw_names", [])
    raw_status = output.get("raw_names_status", "ok") if raw_present else "failure"
    _require(raw_status in STATUSES, f"{case_id}: invalid raw_names_status")
    raw = _predictions(raw_items, blocks, NAME_TYPES, f"{case_id}/raw_names")
    raw_duplicate_count = len(raw_items) - len(raw)
    if raw_status == "failure":
        raw = {}
    detail = {"id": case_id, "split": case["split"], "family": case["family"], "tags": case["tags"],
              "layout": "metadata" if case.get("links") else "plain_text", "status": status,
              "errors": errors, "blocks": case["blocks"], "gold": case["gold"],
              "excluded": excluded, "links": case.get("links", []), "configs": {},
              "comparisons": {}, "token_boundaries": [], "policy_audit": audit,
              "output_missing": output_missing, "raw_name_predictions": raw_items if raw_present else None,
              "raw_names_available": raw_present and raw_status != "failure", "raw_names_status": raw_status}
    evaluated = {"raw_model": raw}
    statuses = {"raw_model": raw_status}
    for config, entity_types in CONFIGS.items():
        items = configs.get(config, [])
        config_status = config_statuses.get(config, status) if config in configs else "failure"
        _require(config_status in STATUSES, f"{case_id}/{config}: invalid config status")
        predictions = _predictions(items, blocks, entity_types, f"{case_id}/{config}")
        duplicate_count = len(items) - len(predictions)
        if config_status == "failure":
            predictions = {}
        evaluated[config] = predictions
        statuses[config] = config_status
        if holdout:
            score = {"status": config_status, "predictions": list(predictions.values())}
        else:
            target = {key: item for key, item in gold.items() if key[0] in entity_types}
            rejections = _policy_rejections(audit, config)
            score = _score(target, predictions, excluded, config_status, duplicate_count, rejections)
        score["output_missing"] = config not in configs
        detail["configs"][config] = score
    if holdout:
        return detail

    names = {key: item for key, item in gold.items() if key[0] == "NAME"}
    detail["raw_names"] = _score(names, raw, excluded, raw_status, raw_duplicate_count)
    detail["raw_names"]["output_missing"] = not raw_present
    for before, after in COMPARISONS:
        if statuses[before] != "ok" or statuses[after] != "ok":
            continue
        delta = _delta(evaluated[before], evaluated[after], gold)
        if (before, after) == ("N1", "N2"):
            for record in delta["added"]:
                prediction = record["prediction"]
                record["field_supplement"] = "name_field_v1" in prediction["sources"]
                overlaps = any(_intersection(_position(prediction), key[1]) for key in raw)
                record["raw_name_overlap"] = True if overlaps else False if raw_status == "ok" else None
        detail["comparisons"][f"{before}_to_{after}"] = delta
    detail["token_boundaries"] = _token_records(gold, output.get("token_boundaries", {}), blocks, case_id)
    return detail


def evaluate(cases, outputs):
    """Return aggregate and case-level exact scores for iterable JSONL records."""
    cases = list(cases)
    validate_cases(cases)
    case_ids = {case["id"] for case in cases}
    by_id = {}
    for output in outputs:
        _require(isinstance(output, dict), "output must be an object")
        case_id = output.get("id")
        _require(_nonempty_string(case_id) and case_id in case_ids and case_id not in by_id,
                 f"invalid, unknown, or duplicate output id: {case_id!r}")
        by_id[case_id] = output
    details = [_case_detail(case, by_id.get(case["id"])) for case in cases]
    scored = [detail for detail in details if detail["split"] != "holdout_policy"]
    strata = {field: defaultdict(list) for field in ("split", "layout", "tag")}
    for detail in scored:
        strata["split"][detail["split"]].append(detail)
        strata["layout"][detail["layout"]].append(detail)
        for tag in dict.fromkeys(detail["tags"]):
            strata["tag"][tag].append(detail)

    def summarize_config(rows, config):
        return _summarize([detail["configs"][config] for detail in rows], CONFIGS[config])

    configs = {}
    for config in CONFIGS:
        configs[config] = summarize_config(scored, config)
        configs[config]["strata"] = {
            field: {value: summarize_config(rows, config) for value, rows in groups.items()}
            for field, groups in strata.items()
        }
    comparisons = {}
    for before, after in COMPARISONS:
        comparison = f"{before}_to_{after}"
        types = CONFIGS[after]
        comparisons[comparison] = _summarize_delta(scored, comparison, types)
        comparisons[comparison]["by_split"] = {
            split: _summarize_delta(rows, comparison, types) for split, rows in strata["split"].items()
        }
    raw_available = sum(detail["raw_names_available"] for detail in scored)
    tokens = [record for detail in scored for record in detail["token_boundaries"]]
    statuses = Counter(detail["status"] for detail in scored)
    return {
        "case_count": len(cases), "scored_case_count": len(scored),
        "policy_holdout_count": len(cases) - len(scored),
        "case_statuses": {status: statuses[status] for status in STATUSES},
        "case_missing_outputs": sum(detail["output_missing"] for detail in scored),
        "configs": configs, "comparisons": comparisons,
        "raw_names": {
            "available_cases": raw_available,
            "scores": _summarize([detail["raw_names"] for detail in scored], ("NAME",)) if raw_available else None,
            "by_split": {split: _summarize([detail["raw_names"] for detail in rows], ("NAME",))
                         for split, rows in strata["split"].items()
                         if any(detail["raw_names_available"] for detail in rows)},
        },
        "token_boundaries": {"all": _token_summary(tokens), "by_type": {
            entity_type: _token_summary([record for record in tokens if record["type"] == entity_type])
            for entity_type in TYPES},
            "by_split": {split: {"all": _token_summary([record for row in rows for record in row["token_boundaries"]]),
                                 "NAME": _token_summary([record for row in rows for record in row["token_boundaries"]
                                                        if record["type"] == "NAME"])}
                         for split, rows in strata["split"].items()}},
        "details": details,
    }


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        records = []
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error.msg}") from error
    return records


def write_detail_csv(path, summary):
    columns = ("id", "split", "family", "layout", "tags", "case_status", "errors", "blocks", "config",
               "config_status", "case_output_missing", "output_missing", "type", "outcome", "gold", "raw",
               "raw_status", "prediction", "policy_audit", "diagnostic")

    def encode(value):
        return json.dumps(value, ensure_ascii=False)

    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in summary["details"]:
            common = {key: case[key] for key in ("id", "split", "family", "layout")}
            common.update({"case_status": case["status"], "case_output_missing": case["output_missing"],
                           "raw": encode(case["raw_name_predictions"]), "raw_status": case["raw_names_status"],
                           **{key: encode(case[key]) for key in ("tags", "errors", "blocks")}})
            for config, score in case["configs"].items():
                audit = {stage: case["policy_audit"][stage] for stage in AUDIT_STAGES.get(config, ())
                         if stage in case["policy_audit"]}
                context = {**common, "config": config, "config_status": score["status"],
                           "output_missing": score["output_missing"], "policy_audit": encode(audit)}
                if case["split"] == "holdout_policy":
                    for prediction in score["predictions"] or [None]:
                        writer.writerow({**context, "outcome": "unscored", "prediction": encode(prediction),
                                         "type": prediction["type"] if prediction else ""})
                    continue
                for outcome in ("tp", "fp", "fn"):
                    for record in score[outcome]:
                        gold, prediction = record.get("gold"), record.get("prediction")
                        diagnostic = {key: value for key, value in record.items() if key not in ("gold", "prediction")}
                        writer.writerow({**context, "type": (gold or prediction)["type"], "outcome": outcome,
                                         "gold": encode(gold), "prediction": encode(prediction),
                                         "diagnostic": encode(diagnostic)})
                if not any(score[outcome] for outcome in ("tp", "fp", "fn")):
                    writer.writerow({**context, "outcome": "empty"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="extend", nargs="+", required=True)
    parser.add_argument("--predictions", type=Path, action="extend", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    try:
        summary = evaluate((case for path in args.data for case in read_jsonl(path)),
                           (output for path in args.predictions for output in read_jsonl(path)))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.csv:
        write_detail_csv(args.csv, summary)


if __name__ == "__main__":
    main()
