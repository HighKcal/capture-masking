"""Bounded, file-based GUI-07 protocol. No inference or private-text logging."""
import hashlib
import json
import os
from pathlib import Path
import re
import time

ENGINE_VERSION = 'GUI-07-v0.7'
REQUEST_SCHEMA = 'gui06-request-v1'
EVENT_SCHEMA = 'gui06-event-v1'
RESULT_SCHEMA = 'gui06-result-v1'
ANALYSIS_SCHEMA = 'gui06-fresh-analysis-v1'
MAX_PNG_BYTES = 32 * 1024 * 1024
MAX_SIDE = 8192
MAX_PIXELS = 8 * 1024 * 1024
MAX_UNITS = 1024
MAX_TEXT = 32768
MAX_CANDIDATES = 256
MAX_SEGMENTS = 2048
MAX_DETAIL_TOKENS = 32768
MAX_MASK_BYTES = 256 * 1024 * 1024
MAX_RESULT_BYTES = 32 * 1024 * 1024
MAX_EVENT_BYTES = 4096
STAGES = {'startup', 'prepare', 'ocr', 'ner_prepare', 'ner', 'detail',
          'postprocess', 'complete', 'failure'}


class ProtocolError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value, max_bytes=MAX_RESULT_BYTES):
    data = canonical(value)
    if len(data) > max_bytes:
        raise ProtocolError('result_limit')
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(21):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in {5, 32, 33} or attempt == 20:
                raise
            time.sleep(.01)


def read_json(path, max_bytes=MAX_RESULT_BYTES):
    with Path(path).open('rb') as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ProtocolError('result_limit')
    return parse_json(raw)


def read_bound_json(path, expected_sha, expected_bytes=None, max_bytes=MAX_RESULT_BYTES):
    with Path(path).open('rb') as stream:
        raw = stream.read(max_bytes + 1)
    if (len(raw) > max_bytes or (expected_bytes is not None and len(raw) != expected_bytes)
            or hashlib.sha256(raw).hexdigest() != expected_sha):
        raise ProtocolError('artifact_changed')
    return parse_json(raw)


def parse_json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ProtocolError('duplicate_json_key')
            out[key] = value
        return out
    try:
        return json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError('invalid_number')))
    except (ValueError, UnicodeError, RecursionError):
        raise ProtocolError('invalid_json') from None


def hex_value(value, length):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None


def validate_request(request):
    if (not isinstance(request, dict) or set(request) != {'schema', 'request_id', 'image_id',
            'generation', 'engine_version', 'snapshot', 'ner_session'}
            or request['schema'] != REQUEST_SCHEMA
            or request['engine_version'] != ENGINE_VERSION
            or not hex_value(request['request_id'], 32) or not hex_value(request['image_id'], 32)
            or type(request['generation']) is not int or request['generation'] < 1):
        raise ProtocolError('request_identity')
    snapshot = request['snapshot']
    if (not isinstance(snapshot, dict) or set(snapshot) != {'file', 'file_sha256', 'rgb_sha256', 'width', 'height'}
            or snapshot['file'] != '../input.png'
            or not hex_value(snapshot['file_sha256'], 64) or not hex_value(snapshot['rgb_sha256'], 64)):
        raise ProtocolError('snapshot_identity')
    w, h = snapshot['width'], snapshot['height']
    if (type(w) is not int or type(h) is not int or not 0 < w <= MAX_SIDE
            or not 0 < h <= MAX_SIDE or w * h > MAX_PIXELS):
        raise ProtocolError('image_limit')
    resident = request['ner_session']
    if resident is not None and (
            not isinstance(resident, dict)
            or set(resident) != {'session_id', 'epoch', 'dispatch_seq', 'pins_sha256'}
            or not all(hex_value(resident[key], 32) for key in ('session_id', 'epoch'))
            or not hex_value(resident['pins_sha256'], 64)
            or type(resident['dispatch_seq']) is not int or not 1 <= resident['dispatch_seq'] < 2**63):
        raise ProtocolError('ner_session_identity')
    return request


def validate_snapshot(request_dir, request):
    validate_request(request)
    directory = Path(request_dir).resolve(strict=True)
    if directory.name != 'request-' + request['request_id'] or directory.parent.name != 'image-' + request['image_id']:
        raise ProtocolError('request_path')
    if request['ner_session'] is not None and directory.parent.parent.name != 'session-' + request['ner_session']['session_id']:
        raise ProtocolError('ner_session_path')
    path = directory.parent / 'input.png'
    if (path.is_symlink() or path.resolve(strict=True) != path or not path.is_file()
            or path.stat().st_size > MAX_PNG_BYTES or sha(path) != request['snapshot']['file_sha256']):
        raise ProtocolError('snapshot_changed')
    return path


def validate_event(event, request):
    if not isinstance(event, dict) or event.get('schema') != EVENT_SCHEMA:
        raise ProtocolError('event_schema')
    for key in ('request_id', 'image_id', 'generation'):
        if event.get(key) != request[key]:
            raise ProtocolError('event_identity')
    if (type(event.get('seq')) is not int or event['seq'] < 1 or event.get('stage') not in STAGES
            or event.get('status') not in {'running', 'complete', 'failure'}):
        raise ProtocolError('event_state')
    if 'code' in event and (not isinstance(event['code'], str)
            or not re.fullmatch('[a-z][a-z0-9_]{0,79}', event['code'])):
        raise ProtocolError('event_code')
    if event['status'] == 'complete':
        if (event['stage'] != 'complete' or not hex_value(event.get('result_sha256'), 64)
                or type(event.get('result_bytes')) is not int
                or not 0 < event['result_bytes'] <= MAX_RESULT_BYTES):
            raise ProtocolError('event_result')
    elif event['stage'] == 'complete':
        raise ProtocolError('event_state')
    if event['status'] == 'failure' and (event['stage'] != 'failure' or 'code' not in event):
        raise ProtocolError('event_failure')
    return event


def enforce_limits(blobs, width, height):
    units = blobs['ocr'].get('units', [])
    candidates = blobs['detection']['configs']['FULL']
    if len(units) > MAX_UNITS or sum(len(u['text']) for u in units) > MAX_TEXT:
        raise ProtocolError('ocr_limit')
    if len(candidates) > MAX_CANDIDATES or sum(len(c['segments']) for c in candidates) > MAX_SEGMENTS:
        raise ProtocolError('candidate_limit')
    if any(sum(len(u.get('tokens', [])) for u in blobs[key]['units']) > MAX_DETAIL_TOKENS
           for key in ('detail', 'detail_geometry')):
        raise ProtocolError('detail_limit')
    if (len(candidates) + 4) * width * height > MAX_MASK_BYTES:
        raise ProtocolError('mask_memory_limit')
