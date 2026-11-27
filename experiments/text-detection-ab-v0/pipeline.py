"""TXT-AB01 wiring only; frozen TXT02 policy functions are unchanged."""
import copy
import time
from detectors import dedup
from improved import improved_names, improved_contacts, field_names
from evaluate_base import _predictions, TYPES


def combined_status(*statuses):
    if all(s == 'ok' for s in statuses):
        return 'ok'
    if all(s == 'failure' for s in statuses):
        return 'failure'
    return 'partial_failure'


def block_status(statuses):
    return combined_status(*statuses.values()) if statuses else 'ok'


def error_record(stage, error, **extra):
    return dict(stage=stage, exception=type(error).__name__, message=str(error), **extra)


def valid_candidates(items, blocks, types, context):
    texts = {b['id']: b['text'] for b in blocks}
    _predictions(items, texts, types, context)
    return items


def accept_raw(raw, blocks):
    """Quarantine invalid candidates before the evaluator sees them; block atomicity."""
    bad = set()
    raw.setdefault('errors', [])
    raw.setdefault('invalid_raw_candidates', [])
    valid = []
    for p in raw['raw_names']:
        try:
            valid_candidates([p], blocks, ('NAME',), 'raw candidate')
            if len(p['segments']) != 1:
                raise ValueError('Model NAME must be one within-block segment')
            valid.append(p)
        except Exception as error:
            raw['invalid_raw_candidates'].append(p)
            segments = p.get('segments') if isinstance(p, dict) else None
            ids = {s['block_id'] for s in segments
                   if isinstance(s, dict) and isinstance(s.get('block_id'), str)} if isinstance(segments, list) else set()
            bad.update(ids if ids and ids <= raw['block_statuses'].keys() else raw['block_statuses'])
            raw['errors'].append(error_record('raw_validation', error, block_ids=sorted(bad)))
    for bid in bad:
        raw['block_statuses'][bid] = 'failure'
    raw['raw_names'] = [p for p in valid
                        if all(raw['block_statuses'].get(s['block_id']) == 'ok' for s in p['segments'])]
    raw['status'] = block_status(raw['block_statuses'])
    return raw


def single_path(model, case):
    """One timed RAW -> N1 + independent C1 -> FULL path, including policy audits.

    Excludes N2, result hashing, diagnostic copies, serialization and disk I/O.
    Detailed per-token evidence construction remains inside raw generation.
    """
    blocks = [dict(id=b['id'], text=b['text']) for b in case['blocks']]
    rule_case = dict(id=case['id'], blocks=blocks, links=case.get('links', []))
    model.sync()
    started = time.perf_counter()
    try:
        raw = model.infer(blocks)
    except Exception as error:
        raw = dict(raw_names=[], raw_model={}, raw_windows={}, token_boundaries={},
                   model_issues=[], block_diagnostics={}, tokenization_evidence={},
                   block_statuses={b['id']: 'failure' for b in blocks}, status='failure',
                   errors=[error_record('model_case', error)], timing={})
    raw = accept_raw(raw, blocks)
    raw_done = time.perf_counter()
    errors = list(raw['errors'])
    statuses = {}
    try:
        n1, audit = improved_names(raw['raw_names'], blocks)
        valid_candidates(n1, blocks, ('NAME',), 'N1')
        statuses['N1'] = raw['status']
    except Exception as error:
        n1, audit = [], []
        statuses['N1'] = 'failure'
        errors.append(error_record('N1', error))
    n1_done = time.perf_counter()
    try:
        c1, contact_audit = improved_contacts(rule_case)
        valid_candidates(c1, blocks, TYPES[1:], 'C1')
        statuses['C1'] = 'ok'
    except Exception as error:
        c1, contact_audit = [], []
        statuses['C1'] = 'failure'
        errors.append(error_record('C1', error))
    c1_done = time.perf_counter()
    full = dedup(n1 + c1)
    statuses['FULL'] = combined_status(statuses['N1'], statuses['C1'])
    model.sync()
    end = time.perf_counter()
    timing = dict(raw.get('timing', {}), raw_generation_ms=(raw_done - started) * 1000,
                  N1_ms=(n1_done - raw_done) * 1000, C1_ms=(c1_done - n1_done) * 1000,
                  single_path_ms=(end - started) * 1000)
    result = dict(raw, id=case['id'], raw_names_status=raw['status'], errors=errors,
                  status=statuses['FULL'], config_statuses=statuses,
                  configs=dict(N1=n1, C1=c1, FULL=full), policy_audit=dict(N1=audit),
                  contact_audit=contact_audit, timing=timing)
    return result


def add_field_diagnostic(result, case):
    """Untimed auxiliary N2. A field candidate never erases a model failure."""
    try:
        fields, audit = field_names(case['blocks'])
        valid_candidates(fields, case['blocks'], ('NAME',), 'field')
        result['field_candidates'] = fields
        result['policy_audit']['field'] = audit
        result['configs']['N2'] = dedup(copy.deepcopy(result['configs']['N1']) + fields)
        result['config_statuses']['N2'] = combined_status(result['config_statuses']['N1'], 'ok')
    except Exception as error:
        result['field_candidates'] = []
        result['policy_audit']['field'] = []
        result['configs']['N2'] = copy.deepcopy(result['configs']['N1'])
        result['config_statuses']['N2'] = combined_status(result['config_statuses']['N1'], 'failure')
        result['errors'].append(error_record('field', error))
    return result
