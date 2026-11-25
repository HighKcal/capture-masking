"""Job-owned GUI-06 CPU NER process. Only the pinned model outlives a request."""

import ctypes
from ctypes import wintypes
from contextlib import ExitStack, contextmanager
import errno
from functools import lru_cache
import gc
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
if __name__ != '__main__':
    import resident_protocol as protocol

POLL_SECONDS = 0.025


class IsolationViolation(PermissionError):
    pass


class RequestGuard:
    """One sticky CPython audit hook with a changing, narrow write scope.

    This hook covers audited Python operations. Native library I/O is outside
    the audit hook. The parent owns the OS Job and the lifetime of the epoch.
    """

    _write_flags = (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
                    | getattr(os, 'O_TEMPORARY', 0))
    _mutations = {'os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.chown',
                  'os.utime', 'os.truncate', 'shutil.rmtree', 'os.chflags'}
    _external = {'os.link', 'os.symlink', 'subprocess.Popen', 'os.system',
                 'os.spawn', 'os.exec', 'os.startfile', 'os.startfile/2',
                 'winreg.CreateKey', 'winreg.DeleteKey', 'winreg.DeleteValue',
                 'winreg.SetValue', 'winreg.SaveKey', 'winreg.LoadKey'}

    def __init__(self, root):
        self.root = protocol.plain_path(root, directory=True)
        self.cache = self.root / 'cache'
        self.state_paths = {self.root / 'state.json', self.root / 'state.json.part'}
        self.initializing = False
        self.output = None
        self.request_cache = None
        self.denied = False
        sys.addaudithook(self._audit)

    def _block(self):
        self.denied = True
        raise IsolationViolation('GUI-04 NER isolation blocked an operation')

    def _check_path(self, value):
        try:
            raw = Path(os.fsdecode(value))
            if '..' in raw.parts or (raw.drive and not raw.is_absolute()):
                raise ValueError
            path = raw.absolute()
            scope = (self.cache,) if self.initializing else (self.output, self.request_cache)
            if path not in self.state_paths and not any(
                    root is not None and path.is_relative_to(root) for root in scope):
                raise ValueError
            # For a create operation, the target may not exist. Existing
            # parents and existing targets must always be ordinary entries.
            protocol.plain_path(path, directory=path.is_dir(), missing_ok=True)
        except (OSError, ValueError, TypeError, protocol.ProtocolError):
            self._block()

    def _audit(self, event, args):
        if event.startswith('socket.') and not self._local_socket_operation(event, args):
            self._block()
        if event in self._external:
            self._block()
        if event == 'open' and args[2] & self._write_flags:
            self._check_path(args[0])
        if event in self._mutations:
            self._check_path(args[0])
        if event == 'os.rename':
            self._check_path(args[0])
            self._check_path(args[1])

    def _local_socket_operation(self, event, args):
        if event == 'socket.gethostname':
            return True
        if not self.initializing:
            return False
        # urllib3 checks IPv6 availability during import by binding an
        # unconnected stream socket to an ephemeral port on literal loopback.
        if event == 'socket.__new__':
            return args[1:] == (socket.AF_INET6, socket.SOCK_STREAM, 0)
        if event == 'socket.bind':
            stream, address = args
            return ((stream.family, stream.type, stream.proto) == (socket.AF_INET6, socket.SOCK_STREAM, 0)
                    and address == ('::1', 0))
        return False

    def verify(self):
        if self.denied:
            self._block()

    def activate(self, directory):
        self.verify()
        self.output = directory / 'ner'
        self.request_cache = directory / 'ner-cache'
        self.request_cache.mkdir()
        self._temp(self.request_cache)

    def revoke(self):
        self.output = self.request_cache = None
        self.initializing = False
        self._temp(self.cache)

    @staticmethod
    def _temp(directory):
        os.environ.update({key: str(directory) for key in ('TEMP', 'TMP', 'TMPDIR')})
        tempfile.tempdir = str(directory)


class _FileInfo(ctypes.Structure):
    _fields_ = [
        ('attributes', wintypes.DWORD), ('creation', wintypes.FILETIME),
        ('access', wintypes.FILETIME), ('write', wintypes.FILETIME),
        ('volume', wintypes.DWORD), ('size_high', wintypes.DWORD),
        ('size_low', wintypes.DWORD), ('links', wintypes.DWORD),
        ('index_high', wintypes.DWORD), ('index_low', wintypes.DWORD),
    ]


class _ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('page_fault_count', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            'peak_working_set_bytes', 'working_set_bytes', 'quota_peak_paged_pool_usage',
            'quota_paged_pool_usage', 'quota_peak_nonpaged_pool_usage', 'quota_nonpaged_pool_usage',
            'pagefile_usage', 'peak_pagefile_usage', 'private_commit_bytes')]


@lru_cache(maxsize=1)
def _kernel():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    for name, result, args in (
        ('CreateFileW', wintypes.HANDLE, (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
         ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)),
        ('CloseHandle', wintypes.BOOL, (wintypes.HANDLE,)),
        ('GetFileType', wintypes.DWORD, (wintypes.HANDLE,)),
        ('GetFileInformationByHandle', wintypes.BOOL, (wintypes.HANDLE, ctypes.POINTER(_FileInfo))),
        ('GetCurrentProcess', wintypes.HANDLE, ()),
        ('K32GetProcessMemoryInfo', wintypes.BOOL,
         (wintypes.HANDLE, ctypes.POINTER(_ProcessMemory), wintypes.DWORD)),
    ):
        function = getattr(api, name)
        function.restype, function.argtypes = result, args
    return api


def process_memory():
    api = _kernel()
    info = _ProcessMemory()
    info.cb = ctypes.sizeof(info)
    if not api.K32GetProcessMemoryInfo(api.GetCurrentProcess(), ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {key: getattr(info, key) for key in protocol.OS_MEMORY_FIELDS}


def _open_pin(path, *, directory):
    api = _kernel()
    share_read, share_write = 1, 2
    read_attributes, generic_read = 0x80, 0x80000000
    open_existing, open_reparse_point, backup_semantics = 3, 0x00200000, 0x02000000
    handle = api.CreateFileW(
        str(path), read_attributes if directory else generic_read,
        share_read | share_write if directory else share_read,
        None, open_existing, open_reparse_point | backup_semantics, None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    info = _FileInfo()
    try:
        if (api.GetFileType(handle) != 1 or not api.GetFileInformationByHandle(handle, ctypes.byref(info))
                or info.attributes & 0x400 or bool(info.attributes & 0x10) != directory
                or (not directory and info.links != 1)):
            raise protocol.ProtocolError('ner_pin_redirected')
    except BaseException:
        api.CloseHandle(handle)
        raise
    return handle


def _locked_stream(path):
    import msvcrt

    handle = _open_pin(path, directory=False)
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        _kernel().CloseHandle(handle)
        raise
    try:
        return os.fdopen(fd, 'rb')
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def lock_pins(pins, *, epoch_path=None, epoch=None):
    """Hash through the native read handles and keep them locked for the epoch."""
    paths = [Path(name) for name in pins]
    if epoch_path is not None:
        paths.append(epoch_path)
    with ExitStack() as stack:
        directories = set()
        for path in paths:
            protocol.plain_path(path)
            for directory in reversed(path.parents):
                if directory not in directories:
                    handle = _open_pin(directory, directory=True)
                    stack.callback(_kernel().CloseHandle, handle)
                    directories.add(directory)
            stream = stack.enter_context(_locked_stream(path))
            if path == epoch_path:
                raw = stream.read(protocol.MAX_CONTROL_BYTES + 1)
                if len(raw) > protocol.MAX_CONTROL_BYTES or protocol.parse_json(raw) != epoch:
                    raise protocol.ProtocolError('ner_epoch_pins_changed')
            elif hashlib.file_digest(stream, 'sha256').hexdigest() != pins[str(path)]:
                raise protocol.ProtocolError('ner_epoch_pins_changed')
        yield


def _load_source(name, path, expected=None):
    """Load the exact source bytes, including when an older pyc exists."""
    if name in sys.modules:
        raise RuntimeError('Unexpected already loaded NER module')
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if expected is not None and digest != expected:
        raise protocol.ProtocolError('ner_epoch_pins_changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), 'exec'), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module, digest


def require_model_success(result):
    # single_path recomputes RAW status from block statuses. An exception on
    # empty input can therefore return status=ok with a model_case error.
    if protocol.has_model_error(result):
        raise protocol.ProtocolError('ner_model_inference_failed')


class NameBackend:
    def __init__(self, root, epoch):
        self.root = root
        self.pins = epoch['pins']
        self.model = None

    def _load(self, directory, name):
        path = directory / (name + '.py')
        return _load_source(name, path, self.pins[str(path)])[0]

    def prepare(self, *, configure_caches=None):
        if self.model is not None:
            return
        started = time.perf_counter()
        if configure_caches is None:
            configure_caches = self._load(HERE, 'ner_bootstrap').configure_caches
        configure_caches(self.root)
        if sys.flags.no_site:
            import site

            site.main()
        if any(name in sys.modules for name in ('torch', 'transformers', 'huggingface_hub')):
            raise RuntimeError('Model libraries loaded before pinned NER preparation')
        adapter = self._load(HERE, 'cpu_model_adapter')
        for name in ('detectors', 'improved', 'evaluate_base'):
            self._load(protocol.MODEL_DIR, name)
        self.single_path = self._load(protocol.MODEL_DIR, 'pipeline').single_path
        imported = time.perf_counter()
        self.model = adapter.NameModel()
        self.initial_library_import_seconds = imported - started
        self.initial_constructor_seconds = time.perf_counter() - imported
        if not protocol.valid_model_metadata(self.model.metadata):
            raise protocol.ProtocolError('ner_model_pin_changed')
        self.load_memory = self.memory()
        self.metadata = dict(
            python=sys.version, executable=sys.executable, platform=platform.platform(),
            packages={name: importlib.metadata.version(name)
                      for name in ('torch', 'transformers', 'tokenizers', 'safetensors')},
            cuda_device=None, metadata=self.model.metadata,
        )

    def memory(self):
        return process_memory() | dict.fromkeys(protocol.CUDA_MEMORY_FIELDS)

    def infer(self, case):
        result = self.single_path(self.model, case)
        if protocol.has_model_error(result):
            # The shared pipeline's case-exception record drops adapter evidence.
            # Retain the real CPU metadata and any forwards already returned.
            result.setdefault('model_metadata', self.model.metadata)
            result['timing'].setdefault('forward_wall_ms', self.model.forward_ms)
            result['timing'].setdefault('forward_cuda_event_ms', None)
        return result

    def forward_count(self):
        return self.model.forward_count


def infer_request(backend, case, *, isolated=False):
    started = time.perf_counter()
    result = backend.infer(case)
    inference_seconds = time.perf_counter() - started
    if not isolated or not protocol.explicit_model_failure(result):
        require_model_success(result)
    count = backend.forward_count()
    if not protocol.valid_forward_result(result, count, isolated=isolated):
        raise protocol.ProtocolError('ner_forward_count_mismatch')
    return result, inference_seconds, count


def write_result(path, case, result, case_sha256):
    result['input_case_sha256'] = case_sha256
    started = time.perf_counter()
    protocol.atomic_json(path, result)
    return dict(id=case['id'], output=path.name, sha256=protocol.sha(path), status=result['status'],
                candidates=len(result['configs']['FULL']), detection_ms=result['timing']['single_path_ms'],
                record_seconds=time.perf_counter() - started)


class Runtime:
    """Poll one command slot. Persist only binding and scalar runtime metadata."""

    def __init__(self, root, epoch, backend, guard):
        self.root = root
        self.epoch = epoch
        self.backend = backend
        self.guard = guard
        self.command_seq = self.state_seq = self.dispatch_seq = 0
        self.command_sha256 = self.ticket = self.complete = None
        self.model_used = self.forward_used = False
        self.stage = 'ready'

    def emit(self, stage, **fields):
        self.state_seq += 1
        state = dict(schema=protocol.STATE_SCHEMA, state_seq=self.state_seq,
                     command_seq=self.command_seq, stage=stage, ticket=self.ticket,
                     **{key: self.epoch[key] for key in ('session_id', 'epoch', 'pins_sha256')},
                     **fields)
        protocol.write_control(self.root / 'state.json', protocol.validate_state(state, self.epoch))
        self.stage = stage
        return state

    def next_command(self):
        try:
            command = protocol.validate_command(protocol.read_control(self.root / 'command.json'), self.epoch)
        except FileNotFoundError:
            if self.command_seq:
                raise protocol.ProtocolError('ner_command_missing') from None
            return None
        except PermissionError as error:
            winerror = getattr(error, 'winerror', None)
            # Path.open can surface a CRT EACCES without the Win32 share code.
            if winerror in {5, 32, 33} or (winerror is None and error.errno == errno.EACCES):
                return None
            raise
        command_sha256 = protocol.digest(command)
        if command_sha256 == self.command_sha256:
            return None
        if command['command_seq'] != self.command_seq + 1:
            raise protocol.ProtocolError('ner_command_order')
        self.command_seq = command['command_seq']
        self.command_sha256 = command_sha256
        return command

    def prepare_model(self):
        self.emit('model_prepare')
        self.guard.initializing = True
        try:
            self.backend.prepare()
        finally:
            self.guard.revoke()
        self.guard.verify()
        self.emit('model_ready')

    def analyze(self, ticket):
        self.emit('request_prepare')
        session = self.root.parent
        directory = protocol.plain_path(protocol.request_dir(session, ticket), directory=True)
        case = protocol.read_ticket_case(session, ticket, epoch=self.epoch)
        self.guard.activate(directory)
        (directory / 'ner').mkdir()
        self.guard.verify()
        self.emit('infer')
        result, inference_seconds, count = infer_request(self.backend, case)
        self.guard.verify()
        forward = dict(count=count, first_in_epoch=bool(count and not self.forward_used))
        timing = dict(
            library_import_seconds=0 if self.model_used else self.backend.initial_library_import_seconds,
            constructor_seconds=0 if self.model_used else self.backend.initial_constructor_seconds,
            initial_library_import_seconds=self.backend.initial_library_import_seconds,
            initial_constructor_seconds=self.backend.initial_constructor_seconds,
            inference_seconds=inference_seconds, single_path_ms=result['timing']['single_path_ms'],
            forward_wall_ms=result['timing']['forward_wall_ms'],
            forward_cuda_event_ms=result['timing']['forward_cuda_event_ms'],
        )
        memory = dict(load=self.backend.load_memory, finish=self.backend.memory())
        record = write_result(directory / 'ner/000.json', case, result, ticket['case_sha256'])
        resident = dict(ticket=ticket, model_reused=self.model_used, memory=memory, model_forward=forward,
                        **{key: self.epoch[key] for key in ('session_id', 'epoch', 'pins_sha256')},
                        **{key: timing[key] for key in (
                            'initial_library_import_seconds', 'initial_constructor_seconds', 'inference_seconds')})
        protocol.atomic_json(directory / 'ner/run.json', dict(
            **self.backend.metadata, input_sha256=ticket['input_sha256'],
            library_import_seconds=timing['library_import_seconds'],
            constructor_seconds=timing['constructor_seconds'], records=[record], resident=resident,
        ))
        self.guard.verify()
        return dict(artifacts=protocol.artifact_manifest(directory), timing=timing,
                    memory=memory, model_reused=self.model_used, model_forward=forward)

    def advance(self, command):
        action, ticket = command['action'], command['ticket']
        if action == 'analyze' and self.stage in ('model_ready', 'idle'):
            if ticket['dispatch_seq'] != self.dispatch_seq + 1:
                raise protocol.ProtocolError('ner_dispatch_order')
            self.ticket = ticket
            self.dispatch_seq = ticket['dispatch_seq']
            metrics = self.analyze(ticket)
            # analyze has returned: no case, result, or file handles are retained.
            self.complete = self.emit('complete', **metrics)
            self.model_used = True
            self.forward_used |= bool(metrics['model_forward']['count'])
            return
        if (ticket != self.ticket or self.complete is None
                or command['complete_sha256'] != protocol.digest(self.complete)):
            raise protocol.ProtocolError('ner_command_binding')
        if action == 'release' and self.stage == 'complete':
            gc.collect()
            self.guard.revoke()
            self.guard.verify()
            directory = protocol.request_dir(self.root.parent, self.ticket)
            protocol.verify_artifacts(directory, self.complete['artifacts'])
            self.emit('released', complete_sha256=command['complete_sha256'],
                      artifacts=self.complete['artifacts'], memory_idle=self.backend.memory())
        elif action == 'ack' and self.stage == 'released':
            self.guard.verify()
            self.ticket = self.complete = None
            self.emit('idle')
        else:
            raise protocol.ProtocolError('ner_command_order')

    def run(self):
        try:
            self.emit('ready')
            self.prepare_model()
            while True:
                self.guard.verify()
                command = self.next_command()
                if command is None:
                    time.sleep(POLL_SECONDS)
                else:
                    self.advance(command)
        except IsolationViolation:
            code = 'ner_isolation_blocked'
        except protocol.ProtocolError as error:
            code = error.code
        except Exception:
            code = 'ner_model_prepare_failed' if self.stage == 'model_prepare' else 'ner_runtime_failed'
        # Leave the exception scope before collection so its traceback cannot
        # retain private request locals. No further inference follows a fault.
        self.guard.revoke()
        self.complete = None
        gc.collect()
        try:
            self.guard.verify()
        except IsolationViolation:
            code = 'ner_isolation_blocked'
        try:
            self.emit('failure', code=code)
        finally:
            self.ticket = None
        return 1


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime-dir', required=True, type=Path)
    args = parser.parse_args()
    if sys.stdin.buffer.readline(4) != b'GO\n':
        return 1

    global protocol
    sys.dont_write_bytecode = True
    loaded_pins = {}
    for name in ('fresh_protocol', 'resident_protocol'):
        path = HERE / (name + '.py')
        module, loaded_pins[str(path)] = _load_source(name, path)
    protocol = module
    root = protocol.plain_path(args.runtime_dir, directory=True)
    epoch_path = root / 'epoch.json'
    epoch = protocol.validate_epoch(protocol.read_control(epoch_path))
    if (root != protocol.runtime_dir(root.parent, epoch['epoch'])
            or root.parent.name != 'session-' + epoch['session_id']
            or any(epoch['pins'].get(name) != value for name, value in loaded_pins.items())):
        raise protocol.ProtocolError('ner_epoch_pins_changed')
    protocol.verify_epoch_sources(epoch, rehash_frozen_code=False)
    with lock_pins(epoch['pins'], epoch_path=epoch_path, epoch=epoch):
        protocol.verify_epoch_sources(epoch, rehash_frozen_code=False)
        return Runtime(root, epoch, NameBackend(root, epoch), RequestGuard(root)).run()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        sys.exit(1)
