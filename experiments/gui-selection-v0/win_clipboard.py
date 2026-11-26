"""Publish and verify a rendered RGB image on the Windows clipboard."""

from __future__ import annotations

import ctypes
import os
import struct
import sys
from contextlib import contextmanager
from ctypes import wintypes
from dataclasses import dataclass, field, replace
from functools import lru_cache
from hashlib import sha256

import numpy as np


_CF_DIB = 8
_GMEM_MOVEABLE = 0x0002
_BI_RGB = 0
_DIB_HEADER = struct.Struct("<IiiHHIIiiII")
_EXCLUDE_VALUE = struct.pack("<I", 0)
_EXCLUDE_NAMES = ("CanIncludeInClipboardHistory", "CanUploadToCloudClipboard")

_ERRORS = {
    "allocation": "Could not prepare the clipboard data.",
    "busy": "Windows clipboard is busy or unavailable.",
    "changed": "The clipboard changed before verification completed.",
    "superseded": "The clipboard changed while this automatic copy was waiting; no data was replaced.",
    "close": "Could not release the Windows clipboard.",
    "empty": "Could not replace the clipboard contents.",
    "exclusion": "Could not verify clipboard history and sync exclusions.",
    "image": "Could not verify the copied image.",
    "input": "The clipboard image must be a nonempty uint8 RGB array.",
    "owner": "A valid window on the calling thread is required.",
    "platform": "Windows clipboard is unavailable on this platform.",
    "rollback": "Could not remove the incomplete clipboard item.",
}


class ClipboardError(RuntimeError):
    """A clipboard failure with a stable code and a sanitized message."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(_ERRORS[code])


@dataclass(frozen=True)
class ClipboardReceipt:
    owner_hwnd: int
    sequence: int
    width: int
    height: int
    _dib_digest: bytes = field(repr=False)


@lru_cache(maxsize=1)
def _win32():
    if sys.platform != "win32":
        raise ClipboardError("platform")
    user = ctypes.WinDLL("user32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = (
        (user, "OpenClipboard", wintypes.BOOL, wintypes.HWND),
        (user, "CloseClipboard", wintypes.BOOL),
        (user, "EmptyClipboard", wintypes.BOOL),
        (user, "GetClipboardOwner", wintypes.HWND),
        (user, "GetClipboardSequenceNumber", wintypes.DWORD),
        (user, "RegisterClipboardFormatW", wintypes.UINT, wintypes.LPCWSTR),
        (user, "SetClipboardData", wintypes.HANDLE, wintypes.UINT, wintypes.HANDLE),
        (user, "GetClipboardData", wintypes.HANDLE, wintypes.UINT),
        (user, "GetWindowThreadProcessId", wintypes.DWORD, wintypes.HWND,
         ctypes.POINTER(wintypes.DWORD)),
        (kernel, "GetCurrentThreadId", wintypes.DWORD),
        (kernel, "GlobalAlloc", wintypes.HGLOBAL, wintypes.UINT, ctypes.c_size_t),
        (kernel, "GlobalLock", ctypes.c_void_p, wintypes.HGLOBAL),
        (kernel, "GlobalUnlock", wintypes.BOOL, wintypes.HGLOBAL),
        (kernel, "GlobalSize", ctypes.c_size_t, wintypes.HGLOBAL),
        (kernel, "GlobalFree", wintypes.HGLOBAL, wintypes.HGLOBAL),
    )
    for dll, name, result, *args in signatures:
        function = getattr(dll, name)
        function.argtypes = args
        function.restype = result
    return user, kernel


def _row_stride(width: int, height: int) -> int:
    if not (0 < width <= 0x7FFFFFFF and 0 < height <= 0x7FFFFFFF):
        raise ClipboardError("input")
    stride = (width * 3 + 3) & ~3
    if stride * height > min(0xFFFFFFFF, sys.maxsize - _DIB_HEADER.size):
        raise ClipboardError("input")
    return stride


def _pack_dib(rgb: np.ndarray) -> tuple[bytes, int, int]:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or (
        rgb.ndim != 3 or rgb.shape[2] != 3
    ):
        raise ClipboardError("input")
    height, width, _ = rgb.shape
    stride = _row_stride(width, height)
    # Positive height gives a bottom-up DIB. Each pixel is BGR and each row
    # is aligned to a DWORD with zero padding.
    packed = bytearray(_DIB_HEADER.size + stride * height)
    _DIB_HEADER.pack_into(
        packed, 0, _DIB_HEADER.size, width, height, 1, 24, _BI_RGB,
        stride * height, 0, 0, 0, 0,
    )
    rows = np.ndarray((height, width, 3), dtype=np.uint8, buffer=packed,
                      offset=_DIB_HEADER.size, strides=(stride, 3, 1))
    rows[:] = rgb[::-1, :, ::-1]
    return bytes(packed), width, height


@contextmanager
def _opened(user, hwnd: int | None):
    if not user.OpenClipboard(hwnd):
        raise ClipboardError("busy")
    try:
        yield
    finally:
        if not user.CloseClipboard():
            raise ClipboardError("close") from None


@contextmanager
def _locked(kernel, handle: int, error: str):
    pointer = kernel.GlobalLock(handle)
    if not pointer:
        raise ClipboardError(error)
    try:
        yield pointer
    finally:
        # GlobalUnlock returns zero after the last successful unlock, too.
        ctypes.set_last_error(0)
        if not kernel.GlobalUnlock(handle) and ctypes.get_last_error():
            raise ClipboardError(error) from None


def _read_bytes(user, kernel, fmt: int, length: int, error: str) -> bytes:
    handle = user.GetClipboardData(fmt)
    if not handle or kernel.GlobalSize(handle) < length:
        raise ClipboardError(error)
    with _locked(kernel, handle, error) as pointer:
        return ctypes.string_at(pointer, length)


def _exclusion_formats(user) -> tuple[int, int]:
    history, cloud = (user.RegisterClipboardFormatW(name) for name in _EXCLUDE_NAMES)
    if not history or not cloud or history == cloud:
        raise ClipboardError("exclusion")
    return history, cloud


def _check_exclusions(user, kernel, formats: tuple[int, int]) -> None:
    for fmt in formats:
        if _read_bytes(user, kernel, fmt, len(_EXCLUDE_VALUE), "exclusion") != _EXCLUDE_VALUE:
            raise ClipboardError("exclusion")


def _check_stamp(user, receipt: ClipboardReceipt) -> None:
    if user.GetClipboardOwner() != receipt.owner_hwnd or (
        not receipt.sequence or user.GetClipboardSequenceNumber() != receipt.sequence
    ):
        raise ClipboardError("changed")


def current_sequence() -> int:
    """Read only the clipboard change number, without inspecting its contents."""
    user, _ = _win32()
    sequence = user.GetClipboardSequenceNumber()
    if not sequence:
        raise ClipboardError("changed")
    return sequence


def publish_rgb(rgb: np.ndarray, owner_hwnd: int, *, expected_sequence: int | None = None) -> ClipboardReceipt:
    """Replace and verify RGB; optionally preserve an intervening clipboard item.

    The optional comparison runs under the Windows clipboard lock before the
    first mutation. Explicit callers retain the existing replacement behavior.
    """
    user, kernel = _win32()
    if expected_sequence is not None and (type(expected_sequence) is not int
                                          or not 0 < expected_sequence <= 0xFFFFFFFF):
        raise ClipboardError("changed")
    owner_pid = wintypes.DWORD()
    if type(owner_hwnd) is not int or not 0 < owner_hwnd < 1 << (ctypes.sizeof(ctypes.c_void_p) * 8):
        raise ClipboardError("owner")
    owner_tid = user.GetWindowThreadProcessId(owner_hwnd, ctypes.byref(owner_pid))
    if owner_tid != kernel.GetCurrentThreadId() or owner_pid.value != os.getpid():
        raise ClipboardError("owner")

    try:
        dib, width, height = _pack_dib(rgb)
        digest = sha256(dib).digest()
        formats = _exclusion_formats(user)
        payloads = dict.fromkeys(formats, _EXCLUDE_VALUE)
        payloads[_CF_DIB] = dib
        pending = dict.fromkeys(payloads)
        try:
            for fmt, data in payloads.items():
                handle = kernel.GlobalAlloc(_GMEM_MOVEABLE, len(data))
                if not handle:
                    raise ClipboardError("allocation")
                pending[fmt] = handle
                with _locked(kernel, handle, "allocation") as pointer:
                    ctypes.memmove(pointer, data, len(data))

            with _opened(user, owner_hwnd):
                if expected_sequence is not None and user.GetClipboardSequenceNumber() != expected_sequence:
                    raise ClipboardError("superseded")
                if not user.EmptyClipboard():
                    raise ClipboardError("empty")
                try:
                    if user.GetClipboardOwner() != owner_hwnd:
                        raise ClipboardError("changed")
                    for fmt in formats:
                        if not user.SetClipboardData(fmt, pending[fmt]):
                            raise ClipboardError("exclusion")
                        pending[fmt] = None  # Ownership has passed to Windows.
                    _check_exclusions(user, kernel, formats)

                    if not user.SetClipboardData(_CF_DIB, pending[_CF_DIB]):
                        raise ClipboardError("image")
                    pending[_CF_DIB] = None
                    receipt = ClipboardReceipt(
                        owner_hwnd, user.GetClipboardSequenceNumber(), width, height, digest,
                    )
                    _check_stamp(user, receipt)
                    _check_exclusions(user, kernel, formats)
                    if _read_bytes(user, kernel, _CF_DIB, len(dib), "image") != dib:
                        raise ClipboardError("image")
                    _check_stamp(user, receipt)
                except BaseException:
                    # Remove only the partial item from this explicit copy, while
                    # our write lock still prevents another app from replacing it.
                    if user.GetClipboardOwner() == owner_hwnd and not user.EmptyClipboard():
                        raise ClipboardError("rollback") from None
                    raise

            # Windows can increment the sequence as the write is closed. Pin
            # the committed sequence and verify the owned item again read-only.
            receipt = replace(receipt, sequence=user.GetClipboardSequenceNumber())
            verify_current(receipt)
            return receipt
        finally:
            # The system frees transferred HGLOBALs. Free only untransferred ones.
            free_failed = False
            for handle in pending.values():
                if handle and kernel.GlobalFree(handle):
                    free_failed = True
            if free_failed:
                raise ClipboardError("allocation") from None
    except MemoryError:
        raise ClipboardError("allocation") from None


def _verified_dib(receipt: ClipboardReceipt) -> bytes:
    user, kernel = _win32()
    _check_stamp(user, receipt)
    formats = _exclusion_formats(user)
    stride = _row_stride(receipt.width, receipt.height)
    length = _DIB_HEADER.size + stride * receipt.height
    try:
        with _opened(user, None):
            _check_stamp(user, receipt)
            _check_exclusions(user, kernel, formats)
            data = _read_bytes(user, kernel, _CF_DIB, length, "image")
            if sha256(data).digest() != receipt._dib_digest:
                raise ClipboardError("image")
            _check_stamp(user, receipt)
    except MemoryError:
        raise ClipboardError("allocation") from None
    _check_stamp(user, receipt)
    return data


def verify_current(receipt: ClipboardReceipt) -> None:
    """Raise ClipboardError unless the same copy is still on the clipboard."""
    _verified_dib(receipt)


def read_own_rgb(receipt: ClipboardReceipt) -> np.ndarray:
    """Read a verified item produced by this process, for pixel comparison."""
    dib = _verified_dib(receipt)
    stride = _row_stride(receipt.width, receipt.height)
    bgr = np.ndarray((receipt.height, receipt.width, 3), dtype=np.uint8, buffer=dib,
                     offset=_DIB_HEADER.size, strides=(stride, 3, 1))
    try:
        return bgr[::-1, :, ::-1].copy()
    except MemoryError:
        raise ClipboardError("allocation") from None
