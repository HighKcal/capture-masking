"""Offline cache setup and the Job-contained isolated CPU NER entry point."""

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile


HERE = Path(__file__).resolve().parent
CACHE_NAMES = (
    'HF_HOME', 'HF_HUB_CACHE', 'HF_ASSETS_CACHE', 'HF_XET_CACHE', 'HF_MODULES_CACHE',
    'TRANSFORMERS_CACHE', 'TORCH_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR',
    'CUDA_CACHE_PATH', 'XDG_CACHE_HOME', 'NUMBA_CACHE_DIR',
)


class IsolationViolation(PermissionError):
    pass


def configure_caches(request_dir):
    cache = request_dir / 'cache'
    cache.mkdir(exist_ok=True)
    env = {key: str(cache / key.lower()) for key in CACHE_NAMES}
    env.update(TEMP=str(cache), TMP=str(cache), TMPDIR=str(cache),
               PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               HF_HUB_DISABLE_XET='1', DO_NOT_TRACK='1')
    os.environ.update(env)
    tempfile.tempdir = str(cache)
    sys.dont_write_bytecode = True
    return env


def install_guard(request_dir):
    """Fail on audited Python writes outside the request and Python network calls."""
    root = request_dir.resolve(strict=True)
    denied = False
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
    write_flags |= getattr(os, 'O_TEMPORARY', 0)
    mutation = {'os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.chown',
                'os.utime', 'os.truncate', 'shutil.rmtree'}
    network = {'socket.connect', 'socket.getaddrinfo', 'socket.gethostbyname',
               'socket.gethostbyaddr', 'socket.getnameinfo', 'socket.sendto'}

    def block():
        nonlocal denied
        denied = True
        raise IsolationViolation('GUI-02 request isolation blocked an operation')

    def inside(path):
        if isinstance(path, int):
            block()
        try:
            return Path(os.fsdecode(path)).resolve().is_relative_to(root)
        except (OSError, TypeError, ValueError):
            block()

    def audit(event, args):
        if event in network or event in {'os.link', 'os.symlink'}:
            block()
        if (event == 'open' and not isinstance(args[0], int)
                and os.fsdecode(args[0]) != os.devnull):
            if args[2] & write_flags and not inside(args[0]):
                block()
        if event in mutation and not inside(args[0]):
            block()
        if event == 'os.rename' and not all(inside(path) for path in args[:2]):
            block()

    def verify():
        if denied:
            block()

    sys.addaudithook(audit)
    return verify


def _load_local(name, loaded):
    if name in sys.modules:
        raise ImportError('Unexpected already loaded isolated NER module')
    path = HERE / (name + '.py')
    source = path.read_bytes()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    loaded[str(path)] = hashlib.sha256(source).hexdigest()
    return module


def run_isolated(directory, input_sha256, protocol, runtime, pins):
    case = protocol.read_isolated_case(directory, input_sha256)
    guard = runtime.RequestGuard(directory)
    backend = runtime.NameBackend(directory, {'pins': pins})
    try:
        guard.initializing = True
        try:
            backend.prepare(configure_caches=configure_caches)
        finally:
            guard.revoke()
        guard.verify()
        guard.activate(directory)
        output = directory / 'ner'
        output.mkdir()
        run_path = output / 'run.json'
        run = dict(**backend.metadata, input_sha256=input_sha256,
                   library_import_seconds=backend.initial_library_import_seconds,
                   constructor_seconds=backend.initial_constructor_seconds, records=[])
        protocol.atomic_json(run_path, run)
        result, inference_seconds, count = runtime.infer_request(backend, case, isolated=True)
        guard.verify()
        run['isolated'] = dict(model_reused=False, inference_seconds=inference_seconds,
                               memory=dict(load=backend.load_memory, finish=backend.memory()),
                               model_forward=dict(count=count, first_in_epoch=bool(count)))
        run['records'] = [runtime.write_result(output / '000.json', case, result, protocol.digest(case))]
        protocol.atomic_json(run_path, run)
    finally:
        guard.revoke()
        guard.verify()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--request-dir', required=True, type=Path)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--producer-sha256', required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    loaded = {}
    for name in ('fresh_protocol', 'resident_protocol'):
        protocol = _load_local(name, loaded)
    runtime = _load_local('resident_ner', loaded)
    pins = protocol.isolated_pins(args.producer_sha256)
    if any(pins.get(name) != digest for name, digest in loaded.items()):
        raise protocol.ProtocolError('ner_isolated_sources_changed')
    try:
        with runtime.lock_pins(pins):
            run_isolated(args.request_dir, args.input_sha256, protocol, runtime, pins)
    except runtime.IsolationViolation:
        raise IsolationViolation('Isolated NER request scope blocked an operation') from None


if __name__ == '__main__':
    try:
        main()
    except IsolationViolation:
        sys.exit(2)
    except Exception:
        # The launcher sees the nonzero exit and never treats partial files as results.
        sys.exit(1)
