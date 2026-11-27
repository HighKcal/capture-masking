"""Compare the frozen ACCOUNT-01 path with ACCOUNT-02 on local images."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import account2_inputs as inputs
import account2_output as output
from account2_output import (ACCOUNT01, FALLBACKS, HERE, METHOD, PRODUCER, RECALL, SCHEMA, TASK,
                             api, digest, failure, load, old, read, sha)


def previous_sources():
    return output.PREVIOUS_SOURCES


def method_proof():
    return load('_account02_freeze', TASK / 'freeze_guard.py').verify_method()


def _without_timing(relations):
    body = {key: value for key, value in relations.items() if key != 'seconds'}
    if 'account_stage' in body:
        body['account_stage'] = {key: value for key, value in body['account_stage'].items() if key != 'seconds'}
    return body


def _check_replay(current, expected, name):
    if (digest(current[0]) != digest(expected[0])
            or digest(_without_timing(current[1])) != digest(_without_timing(expected[1]))):
        raise ValueError(name + ' replay differs from its saved analysis')


def build_baseline(item, recall, account01):
    """The current frozen baseline is FIELD, recall, and ACCOUNT-01 in that order."""
    ocr, case, rgb, raw = (item[key] for key in ('ocr', 'case', 'rgb', 'baseline'))
    api.verify_detection(case, raw)
    start = time.perf_counter()
    detection, relations = api.field.supplement(ocr, raw, rgb)
    item['timing']['FIELD_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    detection, relations = recall(ocr, case, raw, detection, relations, rgb)
    item['timing']['PII_RECALL_seconds'] = time.perf_counter() - start
    if 'expected_recall' in item:
        _check_replay((detection, relations), item['expected_recall'], 'PII-RECALL01')
        item['execution']['reused_analysis']['pii_recall_replay_matches'] = True
    start = time.perf_counter()
    detection, relations = account01(ocr, detection, relations, rgb)
    item['timing']['ACCOUNT01_seconds'] = time.perf_counter() - start
    if 'expected_account' in item:
        _check_replay((detection, relations), item['expected_account'], 'ACCOUNT-01')
        item['execution']['reused_analysis']['account01_replay_matches'] = True
    return detection, relations


def validate_joint(item, baseline, before_relations, detection, relations):
    api.verify_detection(item['case'], detection)
    old.verify_relation_binding(item['identity'], relations)
    before, after = (record['configs']['FULL'] for record in (baseline, detection))
    non_account = lambda records: [p for p in records if p['type'] != 'ACCOUNT_ID']
    if non_account(before) != non_account(after) or non_account(before_relations['relations']) != non_account(relations['relations']):
        raise ValueError('ACCOUNT-02 modified non-account candidates or relationships')
    if (any(detection['configs'].get(key) != val for key, val in baseline['configs'].items() if key != 'FULL')
            or any(detection['config_statuses'].get(key) != val for key, val in baseline['config_statuses'].items())
            or detection['status'] != baseline['status'] or relations['status'] != before_relations['status']):
        raise ValueError('ACCOUNT-02 changed upstream data or status')
    stage = relations['account_joint']
    if detection['config_statuses']['ACCOUNT2'] != stage['status']:
        raise ValueError('ACCOUNT-02 status differs between detection and relations')
    if stage['status'] != 'ok' and (before != after or before_relations['relations'] != relations['relations']):
        raise ValueError('An incomplete stage changed the baseline candidates or relationships')
    known = {candidate['id']: candidate for candidate in before}
    texts = {block['id']: block['text'] for block in item['case']['blocks']}
    candidates = {}
    for candidate in after:
        cid = candidate['id']
        if not isinstance(cid, str) or not cid or cid in candidates or not candidate['segments']:
            raise ValueError('Invalid candidate identity or empty segments')
        if cid in known and known[cid] != candidate:
            raise ValueError('A changed candidate kept an obsolete identity')
        candidates[cid] = candidate
        for segment in candidate['segments']:
            text, start, end = texts[segment['block_id']], segment['start'], segment['end']
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                    or segment['text'] != text[start:end]):
                raise ValueError('Candidate does not use an exact original code-point slice')
    for relation in relations['relations']:
        active, supported = relation['candidate_ids'], relation.get('supported_candidate_ids', [])
        if set(active + supported) - candidates.keys():
            raise ValueError('A current relationship references a removed candidate')
        if ((active and relation['status'] != 'accepted') or (supported and relation['status'] != 'deferred')
                or len(set(active + supported)) != len(active + supported)
                or any(candidates[cid]['type'] != relation['type'] for cid in active + supported)):
            raise ValueError('A current relationship has an invalid candidate consumer or status')
        if any(candidates[cid]['segments'] != relation['value_segments'] for cid in supported):
            raise ValueError('An independently supported candidate differs from its exact value segments')


def apply_joint(item, baseline, relations, joint):
    args = copy.deepcopy((item['ocr'], baseline, relations))
    rgb = item['rgb'].copy()
    start = time.perf_counter()
    try:
        detection, updated = joint(*args, rgb)
        item['timing']['ACCOUNT02_seconds'] = time.perf_counter() - start
        validate_joint(item, baseline, relations, detection, updated)
    except Exception as error:
        item['timing'].setdefault('ACCOUNT02_seconds', time.perf_counter() - start)
        detection, updated = copy.deepcopy((baseline, relations))
        detection['config_statuses']['ACCOUNT2'] = 'failure'
        updated['account_joint'] = dict(failure('ACCOUNT-02', error), actions=[], relations=[])
    return detection, updated


def save_arm(root, case_dir, arm, item, detection, relations, versions, environment):
    directory = case_dir / arm
    record = dict(renders={})
    try:
        analysis, blobs = output.save_analysis(directory, item, detection, relations, arm, versions, environment)
    except Exception as error:
        return dict(record, status='failure', failure=failure('analysis', error))
    record.update(analysis_directory=directory.relative_to(root).as_posix(), analysis_sha256=sha(directory / 'analysis.json'),
                  **{key: analysis[key] for key in ('analysis_id', 'analysis_digest', 'candidate_version', 'statuses')})
    if 'account_joint' in relations:
        stage = relations['account_joint']
        record['account_joint_status'] = stage['status']
        if stage['status'] == 'failure':
            record['account_joint_failure'] = {key: stage[key] for key in ('stage', 'type', 'message', 'errors') if key in stage}
    candidates = blobs['refined']['candidates']
    selection = api.make_selection(analysis, candidates, [candidate['id'] for candidate in candidates])
    errors = []
    for label, fallback in FALLBACKS.items():
        target = directory / 'renders' / label
        try:
            rendered = output.render(target, analysis, candidates, item['rgb'], selection, fallback)
        except Exception as error:
            errors.append(failure('render/' + label, error))
            continue
        record['renders'][label] = dict(directory=target.relative_to(root).as_posix(),
                                        sha256=sha(target / 'render.json'), render_id=rendered['render_id'])
    record['status'] = 'failure' if errors else analysis['statuses']['status']
    if errors:
        record['failures'] = errors
    return record


def process_item(root, item, recall, account01, joint, versions, environment):
    row = dict(item['metadata'], timing=item['timing'], arms={})
    directory = root / 'cases' / row['image_id']
    row['shared'] = output.save_shared(root, directory / 'shared', item)
    if 'failure' in item:
        return dict(row, status='failure', failure=item['failure'])
    row['reused_analysis'] = item['execution']['reused_analysis']
    try:
        baseline, baseline_relations = build_baseline(item, recall, account01)
    except Exception as error:
        return dict(row, status='failure', failure=failure('ACCOUNT-01 baseline replay', error))
    improved, improved_relations = apply_joint(item, baseline, baseline_relations, joint)
    for arm, detection, relations in (('baseline', baseline, baseline_relations), ('improved', improved, improved_relations)):
        row['arms'][arm] = save_arm(root, directory, arm, item, detection, relations, versions, environment)
    statuses = {arm['status'] for arm in row['arms'].values()}
    row['status'] = statuses.pop() if len(statuses) == 1 else 'partial_failure'
    return row


def execute(target, *, images=None, reuse=None, manifest=None, joint=None):
    if sum(value is not None for value in (images, reuse, manifest)) != 1 or images == [] or reuse == []:
        raise ValueError('Choose one input mode with at least one input')
    target = output.confined(target)
    if target.exists():
        raise ValueError('Choose a new output directory')
    manifest_record, proof = None, None
    injected = joint is not None
    if manifest is not None:
        if injected:
            raise ValueError('Confirmation requires the frozen ACCOUNT-02 implementation')
        images, manifest_record, proof = inputs.confirmation_inputs(manifest, method_proof)
    mode = 'manifest' if manifest is not None else 'images' if images is not None else 'reuse'
    if not injected:
        joint = load('account_joint', TASK / 'account_joint.py').supplement
    prior = previous_sources()
    proof01 = inputs.prior_method([*prior, *map(Path, api.source_hashes())])
    recall = load('_account02_previous_recall', RECALL / 'improvement.py').improve
    account01 = load('_account02_previous_account', ACCOUNT01 / 'account_stage.py').supplement
    producer = dict(name=PRODUCER, files={str(path): sha(path) for path in output.PRODUCER_SOURCES},
                    account01_method_freeze=proof01, method_freeze=proof, mask_method=METHOD,
                    injected_test_stage=injected)
    versions = old.previous.runtime_versions(producer)
    environment = old.previous.execution_environment()
    target.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    records, timing, ocr_run_id, run_error = [], {}, None, None
    try:
        with inputs.local_caches(target / 'temp'), api.support.ocr_runner.python_network_guard():
            if reuse is not None:
                items = inputs.prepare_reused(reuse)
            else:
                items, ocr_run_id, timing = inputs.prepare_fresh(images, target, manifest is not None)
            for item in items:
                try:
                    records.append(process_item(target, item, recall, account01, joint, versions, environment))
                except Exception as error:
                    records.append(dict(item['metadata'], status='failure', arms={},
                                        failure=failure('save shared evidence', error)))
        if proof is not None and method_proof() != proof:
            raise ValueError('Method freeze changed during execution')
    except Exception as error:
        run_error = failure('run', error)
    timing['total_seconds'] = time.perf_counter() - start
    complete = run_error is None and bool(records) and all(
        set(row['arms']) == {'baseline', 'improved'} and all(
            set(arm['renders']) == set(FALLBACKS) and arm.get('account_joint_status') != 'failure'
            for arm in row['arms'].values()) for row in records)
    if complete and all(row['status'] == 'ok' for row in records):
        status = 'ok'
    elif any(arm['renders'] for row in records for arm in row['arms'].values()):
        status = 'partial_failure'
    else:
        status = 'failure'
    result = dict(schema=SCHEMA, status=status, complete=complete, mode=mode, manifest=manifest_record,
                  manifests=[manifest_record] if manifest_record else [], method_freeze=proof, producer=producer,
                  environment=environment, ocr_run_id=ocr_run_id, timing=timing, records=records)
    if run_error is not None:
        result['failure'] = run_error
    worker = target / 'ner-baseline/ner/run.json'
    if worker.is_file():
        result['NER_worker_run'] = output.file_entry(target, worker)
    output.write(target / 'run.json', result)
    return result


def selection_command(arguments):
    parser = argparse.ArgumentParser(description='Review or change a saved analysis selection without inference.')
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list')
    selecting = commands.add_parser('select')
    rendering = commands.add_parser('render')
    for command in (listing, selecting, rendering):
        command.add_argument('--analysis', type=Path, required=True)
    for command in (selecting, rendering):
        command.add_argument('--output', type=Path, required=True)
    choices = selecting.add_mutually_exclusive_group(required=True)
    choices.add_argument('--all', action='store_true')
    choices.add_argument('--none', action='store_true')
    choices.add_argument('--ids', nargs='+')
    rendering.add_argument('--selection', type=Path, required=True)
    rendering.add_argument('--fallback', choices=FALLBACKS, required=True)
    args = parser.parse_args(arguments)
    analysis, blobs, rgb = output.load_analysis(args.analysis)
    candidates = blobs['refined']['candidates']
    if args.command == 'list':
        result = dict(analysis_id=analysis['analysis_id'], candidate_version=analysis['candidate_version'],
                      statuses=analysis['statuses'], candidates=candidates)
    elif args.command == 'select':
        selected = [candidate['id'] for candidate in candidates] if args.all else (args.ids or [])
        result = api.make_selection(analysis, candidates, selected)
        output.write(args.output, result)
    else:
        result = output.render(args.output, analysis, candidates, rgb, read(inputs.ordinary(args.selection)),
                               FALLBACKS[args.fallback])
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    if arguments and arguments[0] in ('list', 'select', 'render'):
        return selection_command(arguments)
    if arguments and arguments[0] == 'gallery':
        from account2_gallery import main as gallery_main
        return gallery_main(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__, epilog='Use list, select, render, or gallery on saved results.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--images', nargs='+', type=Path)
    group.add_argument('--reuse', nargs='+', type=Path)
    group.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(arguments)
    result = execute(args.output, images=args.images, reuse=args.reuse, manifest=args.manifest)
    print(json.dumps(dict(status=result['status'], complete=result['complete'], records=len(result['records']))))
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
