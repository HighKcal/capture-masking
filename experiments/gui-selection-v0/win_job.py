"""Own a trusted worker's Windows process tree and poll it from the GUI."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from ctypes import wintypes as wt
from functools import lru_cache


_JOB_BASIC_ACCOUNTING = 1
_JOB_EXTENDED_LIMIT = 9
_KILL_ON_JOB_CLOSE = 0x2000
_PROC_HANDLE_LIST = 0x00020002
_PROC_JOB_LIST = 0x0002000D
_CREATE_SUSPENDED = 0x00000004
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_STARTF_USESTDHANDLES = 0x00000100
_HANDLE_FLAG_INHERIT = 0x00000001
_WAIT_TIMEOUT = 258
_CANCEL_EXIT_CODE = 1
_GO = b"GO\n"


class ProcessError(RuntimeError):
    """A sanitized process error with an optional owned tree to drain."""

    def __init__(self, code: str, pending: ManagedProcess | None = None):
        self.code = code
        self.pending = pending
        super().__init__(code)


class _BasicLimit(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wt.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wt.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wt.DWORD),
        ("SchedulingClass", wt.DWORD),
    ]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimit),
        ("IoInfo", ctypes.c_uint64 * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BasicAccounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", wt.DWORD),
        ("TotalProcesses", wt.DWORD),
        ("ActiveProcesses", wt.DWORD),
        ("TotalTerminatedProcesses", wt.DWORD),
    ]


class _StartupInfo(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("lpReserved", wt.LPWSTR),
        ("lpDesktop", wt.LPWSTR),
        ("lpTitle", wt.LPWSTR),
        ("dwX", wt.DWORD),
        ("dwY", wt.DWORD),
        ("dwXSize", wt.DWORD),
        ("dwYSize", wt.DWORD),
        ("dwXCountChars", wt.DWORD),
        ("dwYCountChars", wt.DWORD),
        ("dwFillAttribute", wt.DWORD),
        ("dwFlags", wt.DWORD),
        ("wShowWindow", wt.WORD),
        ("cbReserved2", wt.WORD),
        ("lpReserved2", wt.LPBYTE),
        ("hStdInput", wt.HANDLE),
        ("hStdOutput", wt.HANDLE),
        ("hStdError", wt.HANDLE),
    ]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p)]


class _ProcessInfo(ctypes.Structure):
    _fields_ = [
        ("hProcess", wt.HANDLE), ("hThread", wt.HANDLE),
        ("dwProcessId", wt.DWORD), ("dwThreadId", wt.DWORD),
    ]


@lru_cache(maxsize=1)
def _kernel():
    if sys.platform != "win32":
        raise ProcessError("job_setup_failed")
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    ptr = ctypes.c_void_p
    size_ptr = ctypes.POINTER(ctypes.c_size_t)
    signatures = (
        ("CreateJobObjectW", wt.HANDLE, ptr, wt.LPCWSTR),
        ("GetHandleInformation", wt.BOOL, wt.HANDLE, wt.LPDWORD),
        ("SetInformationJobObject", wt.BOOL, wt.HANDLE, ctypes.c_int, ptr, wt.DWORD),
        ("QueryInformationJobObject", wt.BOOL, wt.HANDLE, ctypes.c_int, ptr, wt.DWORD, wt.LPDWORD),
        ("InitializeProcThreadAttributeList", wt.BOOL, ptr, wt.DWORD, wt.DWORD, size_ptr),
        ("UpdateProcThreadAttribute", wt.BOOL, ptr, wt.DWORD, ctypes.c_size_t,
         ptr, ctypes.c_size_t, ptr, size_ptr),
        ("DeleteProcThreadAttributeList", None, ptr),
        ("CreateProcessW", wt.BOOL, wt.LPCWSTR, wt.LPWSTR, ptr, ptr, wt.BOOL,
         wt.DWORD, ptr, wt.LPCWSTR, ctypes.POINTER(_StartupInfoEx), ctypes.POINTER(_ProcessInfo)),
        ("IsProcessInJob", wt.BOOL, wt.HANDLE, wt.HANDLE, ctypes.POINTER(wt.BOOL)),
        ("ResumeThread", wt.DWORD, wt.HANDLE),
        ("WaitForSingleObject", wt.DWORD, wt.HANDLE, wt.DWORD),
        ("GetExitCodeProcess", wt.BOOL, wt.HANDLE, wt.LPDWORD),
        ("TerminateJobObject", wt.BOOL, wt.HANDLE, wt.UINT),
        ("CloseHandle", wt.BOOL, wt.HANDLE),
    )
    for name, result, *args in signatures:
        function = getattr(api, name)
        function.argtypes = args
        function.restype = result
    return api


@contextmanager
def _startup_attributes(api, job: int, read_handle: int, null_handle: int):
    size = ctypes.c_size_t()
    api.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
    if not size.value:
        raise ProcessError("job_setup_failed")
    attributes = ctypes.create_string_buffer(size.value)
    if not api.InitializeProcThreadAttributeList(attributes, 2, 0, ctypes.byref(size)):
        raise ProcessError("job_setup_failed")
    handles = (wt.HANDLE * 2)(read_handle, null_handle)
    jobs = (wt.HANDLE * 1)(job)
    try:
        for key, values in ((_PROC_HANDLE_LIST, handles), (_PROC_JOB_LIST, jobs)):
            if not api.UpdateProcThreadAttribute(
                attributes, 0, key, values, ctypes.sizeof(values), None, None,
            ):
                raise ProcessError("job_setup_failed")
        yield ctypes.cast(attributes, ctypes.c_void_p)
    finally:
        api.DeleteProcThreadAttributeList(attributes)


def _create_process(api, job, command, cwd, env, read_fd, null_fd) -> _ProcessInfo:
    import msvcrt

    if not command or not os.path.isabs(command[0]) or any("\0" in arg for arg in command):
        raise ProcessError("worker_start_failed")
    entries = []
    for name, value in sorted(env.items(), key=lambda item: item[0].upper()):
        if not name or "=" in name or "\0" in name or "\0" in value:
            raise ProcessError("worker_start_failed")
        entries.append(f"{name}={value}")
    if len({name.upper() for name in env}) != len(env):
        raise ProcessError("worker_start_failed")
    env_block = ctypes.create_unicode_buffer("\0".join(entries) + "\0")
    command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(command))
    read_handle = msvcrt.get_osfhandle(read_fd)
    null_handle = msvcrt.get_osfhandle(null_fd)
    os.set_handle_inheritable(read_handle, True)
    os.set_handle_inheritable(null_handle, True)
    process = _ProcessInfo()
    # The venv redirector can spawn before the Python gate. Assign at creation
    # and keep the first thread suspended until the parent verifies ownership.
    with _startup_attributes(api, job, read_handle, null_handle) as attributes:
        startup = _StartupInfoEx()
        startup.StartupInfo.cb = ctypes.sizeof(startup)
        startup.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
        startup.StartupInfo.hStdInput = read_handle
        startup.StartupInfo.hStdOutput = null_handle
        startup.StartupInfo.hStdError = null_handle
        startup.lpAttributeList = attributes
        flags = (_CREATE_SUSPENDED | _EXTENDED_STARTUPINFO_PRESENT
                 | _CREATE_UNICODE_ENVIRONMENT | subprocess.CREATE_NO_WINDOW)
        if not api.CreateProcessW(
            command[0], command_line, None, None, True, flags, env_block, os.fspath(cwd),
            ctypes.byref(startup), ctypes.byref(process),
        ):
            raise ProcessError("worker_start_failed")
    return process


def _verify_ownership(api, job: int, process: int) -> None:
    member = wt.BOOL()
    handle_flags = wt.DWORD()
    limits = _ExtendedLimit()
    if (
        not api.IsProcessInJob(process, job, ctypes.byref(member)) or not member.value
        or not api.GetHandleInformation(job, ctypes.byref(handle_flags))
        or handle_flags.value & _HANDLE_FLAG_INHERIT
        or not api.QueryInformationJobObject(
            job, _JOB_EXTENDED_LIMIT, ctypes.byref(limits), ctypes.sizeof(limits), None,
        )
        or limits.BasicLimitInformation.LimitFlags != _KILL_ON_JOB_CLOSE
    ):
        raise ProcessError("job_ownership_failed")


class ManagedProcess:
    """Start a trusted bootstrap with a one-shot GO pipe, using Windows 10+.

    Pass an absolute interpreter path and a bootstrap that waits for GO before
    model imports or work. Launch Python with -B -I -S to suppress startup hooks.
    Poll on the owning thread. Keep this object alive until finished and closed.
    """

    def __init__(self, command, cwd, env):
        self._api = _kernel()
        self._process = None
        self._returncode = None
        self._termination_requested = False
        self._job = self._api.CreateJobObjectW(None, None)
        if not self._job:
            raise ProcessError("job_setup_failed")
        try:
            limits = _ExtendedLimit()
            limits.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
            if not self._api.SetInformationJobObject(
                self._job, _JOB_EXTENDED_LIMIT, ctypes.byref(limits), ctypes.sizeof(limits),
            ):
                raise ProcessError("job_setup_failed")
            with ExitStack() as cleanup:
                read_fd, write_fd = os.pipe()
                cleanup.callback(os.close, write_fd)
                with ExitStack() as child_handles:
                    child_handles.callback(os.close, read_fd)
                    null_fd = os.open(os.devnull, os.O_WRONLY | os.O_BINARY)
                    child_handles.callback(os.close, null_fd)
                    process = _create_process(self._api, self._job, command, cwd, env, read_fd, null_fd)
                    self._process = process.hProcess
                    self.pid = process.dwProcessId
                    cleanup.callback(self._api.CloseHandle, process.hThread)
                _verify_ownership(self._api, self._job, self._process)
                if self._api.ResumeThread(process.hThread) != 1:
                    raise ProcessError("bootstrap_gate_failed")
                if os.write(write_fd, _GO) != len(_GO):
                    raise ProcessError("bootstrap_gate_failed")
        except (ProcessError, OSError, ValueError) as error:
            code = error.code if isinstance(error, ProcessError) else (
                "bootstrap_gate_failed" if self._process else "worker_start_failed"
            )
            if self._process:
                try:
                    self.cancel()
                except ProcessError as stop_error:
                    raise ProcessError(stop_error.code, self) from None
                raise ProcessError(code, self) from None
            self._api.CloseHandle(self._job)
            self._job = None
            raise ProcessError(code) from None

    def poll(self) -> int | None:
        """Report the root exit code once the root process handle is signaled."""
        if self._returncode is None:
            state = self._api.WaitForSingleObject(self._process, 0)
            if state == _WAIT_TIMEOUT:
                return None
            exit_code = wt.DWORD()
            if state != 0 or not self._api.GetExitCodeProcess(self._process, ctypes.byref(exit_code)):
                raise ProcessError("process_status_failed")
            self._returncode = exit_code.value
        return self._returncode

    def active_count(self) -> int:
        """Return the Job's active process count, including nested descendants."""
        if not self._job:
            return 0
        counts = _BasicAccounting()
        if not self._api.QueryInformationJobObject(
            self._job, _JOB_BASIC_ACCOUNTING, ctypes.byref(counts), ctypes.sizeof(counts), None,
        ):
            raise ProcessError("process_status_failed")
        return counts.ActiveProcesses

    @property
    def finished(self) -> bool:
        return self.poll() is not None and self.active_count() == 0

    def cancel(self) -> None:
        """Request termination of the owned Job without waiting for completion."""
        if self._job and not self._termination_requested:
            if not self._api.TerminateJobObject(self._job, _CANCEL_EXIT_CODE):
                raise ProcessError("job_termination_failed")
            self._termination_requested = True

    def close(self) -> None:
        """Release parent handles once no active processes remain."""
        if not self.finished:
            raise ProcessError("process_running")
        for attr in ("_process", "_job"):
            handle = getattr(self, attr)
            if handle:
                if not self._api.CloseHandle(handle):
                    raise ProcessError("process_cleanup_failed")
                setattr(self, attr, None)

    def __del__(self):
        # The Job's last handle kills its members if a caller drops ownership.
        for attr in ("_job", "_process"):
            handle = getattr(self, attr, None)
            if handle:
                self._api.CloseHandle(handle)
