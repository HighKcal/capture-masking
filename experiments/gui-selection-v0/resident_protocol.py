"""Bounded, text-free controls for the GUI-06 CPU NER epoch."""

import hashlib
from pathlib import Path
import re
import stat

from fresh_protocol import (
    MAX_EVENT_BYTES, MAX_RESULT_BYTES, MAX_TEXT, MAX_UNITS, ProtocolError,
    atomic_json, digest, hex_value, parse_json, read_bound_json, read_json, sha, validate_request,
)


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
METHOD = EXPERIMENTS / 'account-field-link-v1/METHOD_FREEZE.json'
METHOD_SHA = 'e344179f05c398e289310627b5ea5290a046560b0361e79ac781277c1f708d11'
ACCOUNT_JOINT_SOURCE = EXPERIMENTS / 'account2-block-fix-v0/account_joint.py'
ACCOUNT_JOINT_SHA = '013adc295d3deea37bd41d30129cba46afdd95c4d56fbb7e2718d8f0b0fea0ba'
ACCOUNT_JOINT_VERSION = 'account-field-joint-v1.0.1'
NAME_ACCOUNT_SOURCE = EXPERIMENTS / 'name-account-recall-v0/improvement.py'
NAME_ACCOUNT_SHA = 'aa84b7d5ee0ae90eff4b737e9ab8bac50c762790a96c8967e82fed8d8d6f2460'
NAME_ACCOUNT_VERSION = 'name-account-recall-v0.1.2'
NAME_ACCOUNT_FEATURES = ('mention', 'account_boundary', 'account_context')
EMAIL_TAIL_SOURCE = EXPERIMENTS / 'email-tail-fix-v0/email_tail.py'
EMAIL_TAIL_SHA = '99c7449c88165e7c4333a7a78bb0e7b26b92bfeda933bc49ba03207a82d25571'
EMAIL_TAIL_VERSION = 'email-tail-v0.1'
CONTACT_COORDINATE = EXPERIMENTS / 'contact-coordinate-fix-v0'
CONTACT_CANDIDATES_SOURCE = CONTACT_COORDINATE / 'contact_candidates.py'
CONTACT_CANDIDATES_SHA = 'b24391ed04473626a8f67513392c39ee7484bc46432ded88924183214aadb958'
CONTACT_CANDIDATES_VERSION = 'contact-candidates-v0.1'
DETAIL_GEOMETRY_SOURCE = CONTACT_COORDINATE / 'detail_geometry.py'
DETAIL_GEOMETRY_SHA = '9e9b31437314cc9294b291f5f3ad56677c351fbfd533a21180a4d5f5dbe56f9b'
DETAIL_GEOMETRY_VERSION = 'detail-geometry-repair-v0.1'
TAIL_BOUNDARY_SOURCE = CONTACT_COORDINATE / 'tail_boundary.py'
TAIL_BOUNDARY_SHA = 'ec1b581446ccac34f6e6cd5ac5293d72437eebf80aef8cb13588afc5f80dc402'
REAL_FIX_DIR = EXPERIMENTS / 'real-fix-v0'
REAL_FIX_VERSION = 'real-fix-v0.1'
REAL_FIX_HASHES = {'layout.py': 'c34189dec0f7dc83fd7095e51c833a76cfdb9c29623c99541b85a64cdf58599e', 'phone.py': '555eff9c053824029386648bc872f8cd1bc9b8fe07c74eccea73f6923481a8a1', 'names.py': '720121cc74bb7e557da3c70bb6b0e470afa3d2a3399498277f21d667f12dc02e'}
SPACED_EMAIL_SOURCE = EXPERIMENTS / 'spaced-email-v0/email_rule.py'
SPACED_EMAIL_SHA = 'a393374e43dc449dfc41bea058bca868734321517ae48b7af73286b596824399'
SPACED_EMAIL_VERSION = 'spaced-email-v0.1'
MODEL_DIR = EXPERIMENTS / 'text-detection-ab-v0'
MODEL_SPEC = MODEL_DIR / 'models.json'
MODEL_PIN = dict(model='A', revision='95a3af685dccf4bb3f7fff6fc81146fed71500bb',
                 weights_sha256='ca27622d097f3101f00ddee725959bdb49a68303c76c58ae579461bb54a20f35',
                 device='cpu', dtype='float32', attention='eager', batch=1,
                 argmax_basis='full_head_logits', window=256, overlap=64,
                 cpu_threads=6, interop_threads=1, deterministic_algorithms=True)
PRODUCER_SOURCES = ('fresh_engine.py', 'fresh_worker.py', 'ner_bootstrap.py', 'fresh_protocol.py',
                    'resident_protocol.py', 'resident_ner.py', 'resident_broker.py', 'cpu_model_adapter.py')
GUI_SOURCES = PRODUCER_SOURCES + ('fresh_consumer.py', 'fresh_controller.py', 'fresh_storage.py',
                                 'verified_input.py', 'win_job.py', 'core.py', 'app.py', 'type_labels.py')
MAX_CONTROL_BYTES = 128 * 1024
MAX_SEQUENCE = (1 << 63) - 1
MAX_SECONDS = 86_400
MAX_MEMORY_BYTES = 1 << 50
MAX_PINS = 512
EPOCH_SCHEMA = 'gui06-ner-epoch-v1'
COMMAND_SCHEMA = 'gui06-ner-command-v1'
STATE_SCHEMA = 'gui06-ner-state-v1'
ARTIFACTS = ('ner/run.json', 'ner/000.json')
TICKET_FIELDS = {'dispatch_seq', 'request_id', 'image_id', 'generation',
                 'request_sha256', 'input_sha256', 'case_sha256'}
CUDA_MEMORY_FIELDS = {'allocated_bytes', 'reserved_bytes', 'peak_allocated_bytes', 'peak_reserved_bytes'}
OS_MEMORY_FIELDS = {'working_set_bytes', 'peak_working_set_bytes', 'private_commit_bytes'}
MEMORY_FIELDS = CUDA_MEMORY_FIELDS | OS_MEMORY_FIELDS
TIMING_FIELDS = {'library_import_seconds', 'constructor_seconds',
                 'initial_library_import_seconds', 'initial_constructor_seconds',
                 'inference_seconds', 'single_path_ms', 'forward_wall_ms', 'forward_cuda_event_ms'}
STATE_FIELDS = {'schema', 'session_id', 'epoch', 'state_seq', 'command_seq', 'stage',
                'ticket', 'pins_sha256'}
STARTUP_STAGES = {'ready': 1, 'model_prepare': 2, 'model_ready': 3}
STAGE_FIELDS = {
    **{stage: set() for stage in STARTUP_STAGES},
    'request_prepare': set(), 'infer': set(), 'idle': set(),
    'complete': {'artifacts', 'timing', 'memory', 'model_reused', 'model_forward'},
    'released': {'complete_sha256', 'artifacts', 'memory_idle'},
    'failure': {'code'},
}
REQUEST_STAGE_OFFSETS = {'request_prepare': (4, 1), 'infer': (5, 1), 'complete': (6, 1), 'released': (7, 2)}


def _fields(value, expected):
    return isinstance(value, dict) and value.keys() == expected


def _integer(value, minimum=1, maximum=MAX_SEQUENCE):
    return type(value) is int and minimum <= value <= maximum


def plain_path(path, *, directory=False, missing_ok=False):
    """Check the exact local path and each ancestor, including Windows junctions."""
    path = Path(path)
    if (not path.is_absolute() or '..' in path.parts
            or (path.drive and (len(path.drive) != 2 or path.drive[1] != ':'))
            or any(':' in part or part.endswith((' ', '.')) or Path(part).is_reserved()
                   for part in path.parts[1:])):
        raise ProtocolError('ner_path_invalid')
    for item in (*reversed(path.parents), path):
        try:
            info = item.lstat()
        except FileNotFoundError:
            if missing_ok:
                return path
            raise
        if (stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                or (not stat.S_ISDIR(info.st_mode) if item != path or directory
                    else not stat.S_ISREG(info.st_mode) or info.st_nlink != 1)):
            raise ProtocolError('ner_path_redirected')
    return path


def _session_id(session_dir):
    name = Path(session_dir).name
    session_id = name.removeprefix('session-')
    if name != 'session-' + session_id or not hex_value(session_id, 32):
        raise ProtocolError('ner_session_invalid')
    return session_id


def runtime_dir(session_dir, epoch_hex):
    _session_id(session_dir)
    if not hex_value(epoch_hex, 32):
        raise ProtocolError('ner_epoch_invalid')
    return Path(session_dir) / ('runtime-ner-' + epoch_hex)


def _read_pinned_source(path, digest, code):
    try:
        source = plain_path(path).read_bytes()
    except (OSError, ProtocolError):
        raise ProtocolError(code) from None
    if hashlib.sha256(source).hexdigest() != digest:
        raise ProtocolError(code)
    return source


def real_fix_pins(*, verify=False):
    pins = {REAL_FIX_DIR / name: value for name, value in REAL_FIX_HASHES.items()}
    if verify:
        for path, value in pins.items():
            _read_pinned_source(path, value, 'real_fix_source_changed')
    return pins


def read_real_fix_source(name):
    if name not in REAL_FIX_HASHES:
        raise ProtocolError('real_fix_source_unknown')
    return _read_pinned_source(REAL_FIX_DIR / name, REAL_FIX_HASHES[name], 'real_fix_source_changed')


def read_spaced_email_source():
    return _read_pinned_source(SPACED_EMAIL_SOURCE, SPACED_EMAIL_SHA, 'spaced_email_source_changed')


def read_account_joint_source():
    return _read_pinned_source(ACCOUNT_JOINT_SOURCE, ACCOUNT_JOINT_SHA, 'account2_source_changed')


def read_name_account_source():
    return _read_pinned_source(NAME_ACCOUNT_SOURCE, NAME_ACCOUNT_SHA, 'name_account_source_changed')


def read_email_tail_source():
    return _read_pinned_source(EMAIL_TAIL_SOURCE, EMAIL_TAIL_SHA, 'email_tail_source_changed')


def read_contact_candidates_source():
    return _read_pinned_source(CONTACT_CANDIDATES_SOURCE, CONTACT_CANDIDATES_SHA, 'contact_candidates_source_changed')


def read_detail_geometry_source():
    return _read_pinned_source(DETAIL_GEOMETRY_SOURCE, DETAIL_GEOMETRY_SHA, 'detail_geometry_source_changed')


def contact_coordinate_pins(*, verify=False):
    pins = {CONTACT_CANDIDATES_SOURCE: CONTACT_CANDIDATES_SHA, DETAIL_GEOMETRY_SOURCE: DETAIL_GEOMETRY_SHA,
            TAIL_BOUNDARY_SOURCE: TAIL_BOUNDARY_SHA}
    if verify:
        read_contact_candidates_source()
        read_detail_geometry_source()
        _read_pinned_source(TAIL_BOUNDARY_SOURCE, TAIL_BOUNDARY_SHA, 'tail_boundary_source_changed')
    return pins


def _read_frozen_pins():
    frozen = read_bound_json(plain_path(METHOD), METHOD_SHA)['files']
    pins = {name: value for name, value in frozen.items() if Path(name).suffix == '.py'}
    for path in (MODEL_SPEC, EXPERIMENTS / 'ocr-ner-link-v0/results/evaluation/ocr-run.json'):
        pins[str(path)] = frozen[str(path)]
    spec = read_bound_json(plain_path(MODEL_SPEC), pins[str(MODEL_SPEC)])['A']
    model_root = plain_path(spec['path'], directory=True)
    if spec['revision'] != MODEL_PIN['revision']:
        raise ProtocolError('ner_model_pin_changed')
    model_files = {}
    for item in spec['files']:
        name = item['name']
        if not isinstance(name, str) or Path(name).name != name:
            raise ProtocolError('ner_model_pin_changed')
        path = str(model_root / name)
        if path in model_files or item['sha256'] != frozen.get(path):
            raise ProtocolError('ner_model_pin_changed')
        model_files[path] = item['sha256']
    if (model_files.get(str(model_root / 'model.safetensors')) != MODEL_PIN['weights_sha256']
            or {str(entry) for entry in model_root.iterdir()} != set(model_files)):
        raise ProtocolError('ner_model_pin_changed')
    pins.update(model_files)
    pins[str(METHOD)] = METHOD_SHA
    pins[str(ACCOUNT_JOINT_SOURCE)] = ACCOUNT_JOINT_SHA
    pins[str(NAME_ACCOUNT_SOURCE)] = NAME_ACCOUNT_SHA
    pins[str(EMAIL_TAIL_SOURCE)] = EMAIL_TAIL_SHA
    pins.update({str(path): value for path, value in contact_coordinate_pins().items()})
    pins.update({str(path): value for path, value in real_fix_pins().items()})
    pins[str(SPACED_EMAIL_SOURCE)] = SPACED_EMAIL_SHA
    return pins


def make_epoch(session_dir, epoch_hex):
    """Compute GUI source digests and copy the independent frozen expected digests."""
    pins = _read_frozen_pins()
    pins.update({str(HERE / name): sha(plain_path(HERE / name)) for name in GUI_SOURCES})
    return validate_epoch(dict(schema=EPOCH_SCHEMA, session_id=_session_id(session_dir),
                               epoch=epoch_hex, pins=pins, pins_sha256=digest(pins)))


def validate_epoch(value):
    if (not _fields(value, {'schema', 'session_id', 'epoch', 'pins', 'pins_sha256'})
            or value['schema'] != EPOCH_SCHEMA
            or not all(hex_value(value[key], 32) for key in ('session_id', 'epoch'))
            or not isinstance(value['pins'], dict) or not 0 < len(value['pins']) <= MAX_PINS
            or not all(isinstance(name, str) and 0 < len(name) <= 1024 and Path(name).is_absolute()
                       and hex_value(hash_value, 64) for name, hash_value in value['pins'].items())
            or len({Path(name) for name in value['pins']}) != len(value['pins'])
            or not hex_value(value['pins_sha256'], 64)
            or value['pins_sha256'] != digest(value['pins'])):
        raise ProtocolError('ner_epoch_invalid')
    return value


def verify_epoch_sources(epoch, *, rehash_frozen_code=True):
    validate_epoch(epoch)
    frozen = _read_frozen_pins()
    expected = dict(frozen)
    expected.update({str(HERE / name): sha(plain_path(HERE / name)) for name in GUI_SOURCES})
    if expected != epoch['pins']:
        raise ProtocolError('ner_epoch_pins_changed')
    if rehash_frozen_code:
        for name, value in frozen.items():
            path = Path(name)
            if (path.suffix == '.py' or path.name in {'models.json', 'ocr-run.json'}) and sha(plain_path(path)) != value:
                raise ProtocolError('ner_epoch_pins_changed')
    return epoch


def isolated_pins(producer_sha256):
    pins = _read_frozen_pins()
    pins.update({str(HERE / name): sha(plain_path(HERE / name)) for name in PRODUCER_SOURCES})
    producer = {Path(name).relative_to(EXPERIMENTS.parent).as_posix(): value
                for name, value in pins.items()
                if Path(name).suffix == '.py' or Path(name) in (METHOD, MODEL_SPEC)
                or Path(name).name == 'ocr-run.json'}
    if digest(producer) != producer_sha256:
        raise ProtocolError('ner_isolated_sources_changed')
    return pins


def validate_ticket(ticket):
    if (not _fields(ticket, TICKET_FIELDS)
            or not all(hex_value(ticket[key], 32) for key in ('request_id', 'image_id'))
            or not all(hex_value(ticket[key], 64)
                       for key in ('request_sha256', 'input_sha256', 'case_sha256'))
            or not _integer(ticket['dispatch_seq']) or not _integer(ticket['generation'])):
        raise ProtocolError('ner_ticket_invalid')
    return ticket


def request_dir(session_dir, ticket):
    _session_id(session_dir)
    validate_ticket(ticket)
    return Path(session_dir) / ('image-' + ticket['image_id']) / ('request-' + ticket['request_id'])


def _case(job):
    if (not _fields(job, {'schema', 'cases'}) or job['schema'] != 'ocr-link-ner-input-v1'
            or not isinstance(job['cases'], list) or len(job['cases']) != 1):
        raise ProtocolError('ner_input_invalid')
    case = job['cases'][0]
    if (not _fields(case, {'id', 'blocks', 'links'}) or case['links'] != []
            or not isinstance(case['id'], str) or not 0 < len(case['id']) <= 256
            or not isinstance(case['blocks'], list) or len(case['blocks']) > MAX_UNITS):
        raise ProtocolError('ner_case_invalid')
    seen = set()
    text_count = 0
    for block in case['blocks']:
        if (not _fields(block, {'id', 'text'}) or not isinstance(block['id'], str)
                or not 0 < len(block['id']) <= 256 or block['id'] in seen
                or not isinstance(block['text'], str)):
            raise ProtocolError('ner_block_invalid')
        seen.add(block['id'])
        text_count += len(block['text'])
    if text_count > MAX_TEXT:
        raise ProtocolError('ner_text_limit')
    return case


def read_isolated_case(directory, input_sha256):
    directory = plain_path(directory, directory=True)
    request = validate_request(read_json(plain_path(directory / 'request.json'), MAX_EVENT_BYTES))
    if (request['ner_session'] is not None or directory.name != 'request-' + request['request_id']
            or directory.parent.name != 'image-' + request['image_id']):
        raise ProtocolError('ner_isolated_request_identity')
    return _case(read_bound_json(plain_path(directory / 'ner-input.json'), input_sha256))


def make_ticket(directory, dispatch_seq):
    directory = plain_path(directory, directory=True)
    request, request_sha256 = _read_with_hash(directory / 'request.json', MAX_EVENT_BYTES)
    job, input_sha256 = _read_with_hash(directory / 'ner-input.json', MAX_RESULT_BYTES)
    validate_request(request)
    case = _case(job)
    ticket = dict(dispatch_seq=dispatch_seq, request_id=request['request_id'],
                  image_id=request['image_id'], generation=request['generation'],
                  request_sha256=request_sha256, input_sha256=input_sha256, case_sha256=digest(case))
    if directory != request_dir(directory.parent.parent, ticket):
        raise ProtocolError('ner_request_path_invalid')
    descriptor = request.get('ner_session')
    if descriptor is not None and (descriptor['session_id'] != _session_id(directory.parent.parent)
                                   or descriptor['dispatch_seq'] != dispatch_seq):
        raise ProtocolError('ner_ticket_mismatch')
    return ticket


def _read_with_hash(path, limit):
    with plain_path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ProtocolError('ner_input_limit')
    return parse_json(raw), hashlib.sha256(raw).hexdigest()


def read_ticket_case(session_dir, ticket, *, epoch):
    directory = plain_path(request_dir(session_dir, ticket), directory=True)
    request = validate_request(read_bound_json(plain_path(directory / 'request.json'),
                                              ticket['request_sha256'], max_bytes=MAX_EVENT_BYTES))
    case = _case(read_bound_json(plain_path(directory / 'ner-input.json'), ticket['input_sha256']))
    descriptor = {key: epoch[key] for key in ('session_id', 'epoch', 'pins_sha256')}
    descriptor['dispatch_seq'] = ticket['dispatch_seq']
    if (_session_id(session_dir) != epoch['session_id'] or request.get('ner_session') != descriptor
            or any(request[key] != ticket[key] for key in ('request_id', 'image_id', 'generation'))
            or digest(case) != ticket['case_sha256']):
        raise ProtocolError('ner_ticket_mismatch')
    return case


def make_command(epoch, command_seq, action, ticket, complete_sha256=None):
    return validate_command(dict(schema=COMMAND_SCHEMA, session_id=epoch['session_id'],
                                 epoch=epoch['epoch'], command_seq=command_seq, action=action,
                                 ticket=ticket, complete_sha256=complete_sha256), epoch)


def validate_command(value, epoch):
    if (not _fields(value, {'schema', 'session_id', 'epoch', 'command_seq', 'action',
                            'ticket', 'complete_sha256'})
            or value['schema'] != COMMAND_SCHEMA
            or any(value[key] != epoch[key] for key in ('session_id', 'epoch'))
            or not _integer(value['command_seq']) or value['action'] not in ('analyze', 'release', 'ack')
            or (value['complete_sha256'] is not None if value['action'] == 'analyze'
                else not hex_value(value['complete_sha256'], 64))):
        raise ProtocolError('ner_command_invalid')
    validate_ticket(value['ticket'])
    return value


def _validate_artifacts(artifacts):
    if (not _fields(artifacts, set(ARTIFACTS))
            or not all(_fields(item, {'sha256', 'bytes'}) and hex_value(item['sha256'], 64)
                       and _integer(item['bytes'], maximum=MAX_RESULT_BYTES) for item in artifacts.values())):
        raise ProtocolError('ner_artifacts_invalid')


def _valid_memory(memory):
    return (_fields(memory, MEMORY_FIELDS)
            and all(_integer(memory[key], 1, MAX_MEMORY_BYTES) for key in OS_MEMORY_FIELDS)
            and memory['working_set_bytes'] <= memory['peak_working_set_bytes']
            and all(memory[key] is None for key in CUDA_MEMORY_FIELDS))


def _valid_memory_pair(memory):
    return _fields(memory, {'load', 'finish'}) and all(_valid_memory(entry) for entry in memory.values())


def has_model_error(result):
    return any(error['stage'] in ('model_case', 'model_block') for error in result.get('errors', ()))


def explicit_model_failure(result):
    failures = ('failure', 'partial_failure')
    return (result.get('status') in failures and result.get('raw_names_status') in failures
            and has_model_error(result))


def model_forward_count(result):
    """Every published batch-1 window requires a returned model forward."""
    return sum(len(windows) for windows in result['raw_windows'].values())


def valid_model_metadata(metadata):
    return (isinstance(metadata, dict) and all(
        type(metadata.get(key)) is type(expected) and metadata[key] == expected
        for key, expected in MODEL_PIN.items()))


def valid_run_metadata(run):
    return (isinstance(run, dict) and 'cuda_device' in run and run['cuda_device'] is None
            and valid_model_metadata(run.get('metadata')))


def _valid_duration(value, *, milliseconds=False):
    return type(value) in (int, float) and 0 <= value <= MAX_SECONDS * (1000 if milliseconds else 1)


def _valid_forward(forward):
    return (_fields(forward, {'count', 'first_in_epoch'})
            and _integer(forward['count'], 0, 4 * MAX_TEXT + MAX_UNITS)
            and type(forward['first_in_epoch']) is bool
            and (not forward['first_in_epoch'] or bool(forward['count'])))


def _valid_forward_timing(timing, count):
    return (isinstance(timing, dict) and 'forward_cuda_event_ms' in timing
            and timing['forward_cuda_event_ms'] is None
            and _valid_duration(timing.get('forward_wall_ms'), milliseconds=True)
            and bool(timing['forward_wall_ms']) == bool(count))


def valid_forward_result(result, count, *, isolated=False):
    if (not _integer(count, 0, 4 * MAX_TEXT + MAX_UNITS)
            or not _valid_forward_timing(result.get('timing'), count)):
        return False
    windows = model_forward_count(result)
    if has_model_error(result):
        return isolated and explicit_model_failure(result) and windows <= count
    return windows == count


def valid_model_result(result, run, complete=None):
    if not valid_run_metadata(run) or result.get('model_metadata') != run['metadata']:
        return False
    if complete is not None:
        return (valid_forward_result(result, complete['model_forward']['count'])
                and all(result['timing'].get(key) == complete['timing'][key]
                        for key in ('single_path_ms', 'forward_wall_ms', 'forward_cuda_event_ms')))
    isolated = run.get('isolated')
    if ('resident' in run or not _fields(isolated, {'model_reused', 'memory', 'model_forward', 'inference_seconds'})
            or isolated['model_reused'] is not False or not _valid_memory_pair(isolated['memory'])
            or not _valid_duration(isolated['inference_seconds'])):
        return False
    forward = isolated['model_forward']
    return (_valid_forward(forward) and forward['first_in_epoch'] == bool(forward['count'])
            and valid_forward_result(result, forward['count'], isolated=True))


def validate_state(value, epoch):
    stage = value.get('stage') if isinstance(value, dict) else None
    if (not isinstance(stage, str) or stage not in STAGE_FIELDS
            or not _fields(value, STATE_FIELDS | STAGE_FIELDS[stage])
            or value['schema'] != STATE_SCHEMA
            or any(value[key] != epoch[key] for key in ('session_id', 'epoch', 'pins_sha256'))
            or not _integer(value['state_seq'])
            or not _integer(value['command_seq'], 0 if stage in STARTUP_STAGES or stage == 'failure' else 1)):
        raise ProtocolError('ner_state_invalid')
    ticket = value['ticket']
    if stage in STARTUP_STAGES or stage == 'idle':
        if ticket is not None:
            raise ProtocolError('ner_state_invalid')
    elif stage != 'failure' or ticket is not None:
        validate_ticket(ticket)
    sequences = value['state_seq'], value['command_seq']
    if stage in STARTUP_STAGES and sequences != (STARTUP_STAGES[stage], 0):
        raise ProtocolError('ner_stage_sequence')
    if stage == 'idle':
        dispatch, remainder = divmod(value['command_seq'], 3)
        if remainder or value['state_seq'] != 5 * dispatch + 3:
            raise ProtocolError('ner_stage_sequence')
    if stage in REQUEST_STAGE_OFFSETS:
        dispatch = ticket['dispatch_seq'] - 1
        state_offset, command_offset = REQUEST_STAGE_OFFSETS[stage]
        if sequences != (5 * dispatch + state_offset, 3 * dispatch + command_offset):
            raise ProtocolError('ner_stage_sequence')
    if stage in ('complete', 'released'):
        _validate_artifacts(value['artifacts'])
    if stage == 'complete':
        timing, memory, reused = value['timing'], value['memory'], value['model_reused']
        if (type(reused) is not bool or not _fields(timing, TIMING_FIELDS)
                or reused != (ticket['dispatch_seq'] > 1)
                or not all(_valid_duration(seconds, milliseconds=key.endswith('_ms'))
                           for key, seconds in timing.items() if key != 'forward_cuda_event_ms')
                or not _valid_memory_pair(memory)
                or any(timing[key] != (0 if reused else timing['initial_' + key])
                       for key in ('library_import_seconds', 'constructor_seconds'))):
            raise ProtocolError('ner_metrics_invalid')
        forward = value['model_forward']
        if (not _valid_forward(forward) or not _valid_forward_timing(timing, forward['count'])
                or (ticket['dispatch_seq'] == 1 and forward['first_in_epoch'] != bool(forward['count']))):
            raise ProtocolError('ner_metrics_invalid')
    if stage == 'released' and (not hex_value(value['complete_sha256'], 64)
                                or not _valid_memory(value['memory_idle'])):
        raise ProtocolError('ner_release_invalid')
    if stage == 'failure' and (not isinstance(value['code'], str)
                               or not re.fullmatch('[a-z][a-z0-9_]{0,79}', value['code'])):
        raise ProtocolError('ner_failure_invalid')
    return value


def read_control(path):
    return read_json(plain_path(path), MAX_CONTROL_BYTES)


def write_control(path, value):
    atomic_json(path, value, MAX_CONTROL_BYTES)


def artifact_manifest(directory):
    manifest = {}
    for name in ARTIFACTS:
        path = plain_path(directory / name)
        size = path.stat().st_size
        if not _integer(size, maximum=MAX_RESULT_BYTES):
            raise ProtocolError('ner_artifacts_invalid')
        manifest[name] = dict(sha256=sha(path), bytes=size)
    return manifest


def verify_artifacts(directory, artifacts):
    _validate_artifacts(artifacts)
    if artifact_manifest(directory) != artifacts:
        raise ProtocolError('ner_artifacts_changed')
