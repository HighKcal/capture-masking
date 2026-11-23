"""Single native primary-monitor capture and overlay-local raster mapping."""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes as win
from dataclasses import dataclass, field
from functools import lru_cache
import os
import sys
import threading

import numpy as np


class CaptureError(RuntimeError):
    pass


class SelectionError(ValueError):
    pass


@dataclass(frozen=True)
class MonitorSnapshot:
    handle: int
    device: str
    rect: tuple[int, int, int, int]
    orientation: int
    bits_per_pixel: int
    refresh_hz: int
    display_flags: int
    scale_percent: int
    thread_awareness: str = field(compare=False)

    @property
    def width(self):
        return self.rect[2] - self.rect[0]

    @property
    def height(self):
        return self.rect[3] - self.rect[1]


@dataclass(frozen=True)
class RasterTransform:
    """Map continuous local edges using measured viewport and raster dimensions."""

    viewport_width: int
    viewport_height: int
    raster_width: int
    raster_height: int

    def __post_init__(self):
        sizes = self.viewport_width, self.viewport_height, self.raster_width, self.raster_height
        if any(type(size) is not int or size <= 0 for size in sizes):
            raise SelectionError("invalid viewport or raster dimensions")

    @property
    def scale_x(self):
        return self.raster_width / self.viewport_width

    @property
    def scale_y(self):
        return self.raster_height / self.viewport_height

    def selection_to_pixels(self, start, end):
        x0, y0 = start
        x1, y1 = end
        left, right = _span(x0, x1, self.viewport_width, self.raster_width)
        top, bottom = _span(y0, y1, self.viewport_height, self.raster_height)
        return left, top, right, bottom

    def pixels_to_viewport(self, box):
        left, top, right, bottom = box
        if (any(type(value) is not int for value in box)
                or not 0 <= left < right <= self.raster_width
                or not 0 <= top < bottom <= self.raster_height):
            raise SelectionError("invalid raster bounds")
        return (
            left * self.viewport_width / self.raster_width,
            top * self.viewport_height / self.raster_height,
            right * self.viewport_width / self.raster_width,
            bottom * self.viewport_height / self.raster_height,
        )


def _span(first, last, viewport, raster):
    if any(not 0 <= edge <= viewport for edge in (first, last)):
        raise SelectionError("selection is outside the overlay viewport")
    if first == last:
        raise SelectionError("selection has zero area")
    low, high = sorted((first, last))
    lower, _ = _edge_bounds(low, viewport, raster)
    _, upper = _edge_bounds(high, viewport, raster)
    return lower, upper


def _edge_bounds(edge, viewport, raster):
    nearest = round(edge * raster / viewport)
    # Preserve the inverse of a displayed pixel edge without tolerances that
    # would snap nearby, distinct mouse positions to the same edge.
    if edge == nearest * viewport / raster:
        return nearest, nearest
    numerator, denominator = edge.as_integer_ratio()
    floor, remainder = divmod(numerator * raster, denominator * viewport)
    return floor, floor + bool(remainder)


# Reserve the native BGRX DIB, detached RGB, and two 32-bpp Qt surfaces.
CAPTURE_MEMORY_LIMIT = 256 * 1024 * 1024
_CAPTURE_BYTES_PER_PIXEL = 4 + 3 + 4 + 4
_PM_V2 = -4
_DPI_CONTEXT_NAMES = {
    -1: "unaware", -2: "system", -3: "per-monitor-v1",
    _PM_V2: "per-monitor-v2", -5: "unaware-gdi-scaled",
}
_MONITORINFOF_PRIMARY = 1
_ENUM_CURRENT_SETTINGS = 0xFFFFFFFF
_DM_POSITION = 0x20
_DM_DISPLAYORIENTATION = 0x80
_SRCCOPY = 0x00CC0020
_CAPTUREBLT = 0x40000000
_NOMIRRORBITMAP = 0x80000000
_HGDI_ERROR = ctypes.c_void_p(-1).value
_DWMWA_CLOAK = 13
_DWMWA_CLOAKED = 14
_DWM_CLOAKED_APP = 1


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", win.DWORD), ("rcMonitor", win.RECT), ("rcWork", win.RECT),
        ("dwFlags", win.DWORD), ("szDevice", win.WCHAR * 32),
    ]


class _DisplayMode(ctypes.Structure):
    # The unions use their display members, which preserve the DEVMODEW layout.
    _fields_ = [
        ("dmDeviceName", win.WCHAR * 32), ("dmSpecVersion", win.WORD),
        ("dmDriverVersion", win.WORD), ("dmSize", win.WORD), ("dmDriverExtra", win.WORD),
        ("dmFields", win.DWORD), ("dmPosition", win.POINT),
        ("dmDisplayOrientation", win.DWORD), ("dmDisplayFixedOutput", win.DWORD),
        ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short),
        ("dmYResolution", ctypes.c_short), ("dmTTOption", ctypes.c_short),
        ("dmCollate", ctypes.c_short), ("dmFormName", win.WCHAR * 32), ("dmLogPixels", win.WORD),
        ("dmBitsPerPel", win.DWORD), ("dmPelsWidth", win.DWORD), ("dmPelsHeight", win.DWORD),
        ("dmDisplayFlags", win.DWORD), ("dmDisplayFrequency", win.DWORD),
        ("dmICMMethod", win.DWORD), ("dmICMIntent", win.DWORD), ("dmMediaType", win.DWORD),
        ("dmDitherType", win.DWORD), ("dmReserved1", win.DWORD), ("dmReserved2", win.DWORD),
        ("dmPanningWidth", win.DWORD), ("dmPanningHeight", win.DWORD),
    ]


class _BitmapHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", win.DWORD), ("biWidth", win.LONG), ("biHeight", win.LONG),
        ("biPlanes", win.WORD), ("biBitCount", win.WORD), ("biCompression", win.DWORD),
        ("biSizeImage", win.DWORD), ("biXPelsPerMeter", win.LONG), ("biYPelsPerMeter", win.LONG),
        ("biClrUsed", win.DWORD), ("biClrImportant", win.DWORD),
    ]


class _BitmapInfo(ctypes.Structure):
    _fields_ = [("bmiHeader", _BitmapHeader), ("bmiColors", win.DWORD * 1)]


class _Bitmap(ctypes.Structure):
    _fields_ = [
        ("bmType", win.LONG), ("bmWidth", win.LONG), ("bmHeight", win.LONG),
        ("bmWidthBytes", win.LONG), ("bmPlanes", win.WORD), ("bmBitsPixel", win.WORD),
        ("bmBits", ctypes.c_void_p),
    ]


class _DibSection(ctypes.Structure):
    _fields_ = [
        ("dsBm", _Bitmap), ("dsBmih", _BitmapHeader), ("dsBitfields", win.DWORD * 3),
        ("dshSection", win.HANDLE), ("dsOffset", win.DWORD),
    ]


class _Win32:
    def __init__(self):
        user = ctypes.WinDLL("user32.dll", use_last_error=True)
        gdi = ctypes.WinDLL("gdi32.dll", use_last_error=True)
        dwm = ctypes.WinDLL("dwmapi.dll", use_last_error=True)
        shell = ctypes.WinDLL("shcore.dll", use_last_error=True)
        ptr = ctypes.POINTER
        handle, integer, void_p = win.HANDLE, ctypes.c_int, ctypes.c_void_p
        self.MonitorEnumProc = ctypes.WINFUNCTYPE(win.BOOL, handle, win.HDC, ptr(win.RECT), win.LPARAM)

        def bind(dll, name, result, *arguments):
            function = getattr(dll, name)
            function.restype = result
            function.argtypes = arguments
            setattr(self, name, function)

        bind(user, "GetThreadDpiAwarenessContext", handle)
        bind(user, "AreDpiAwarenessContextsEqual", win.BOOL, handle, handle)
        bind(user, "SetThreadDpiAwarenessContext", handle, handle)
        bind(user, "EnumDisplayMonitors", win.BOOL, win.HDC, ptr(win.RECT), self.MonitorEnumProc, win.LPARAM)
        bind(user, "GetMonitorInfoW", win.BOOL, handle, ptr(_MonitorInfo))
        bind(user, "EnumDisplaySettingsW", win.BOOL, win.LPCWSTR, win.DWORD, ptr(_DisplayMode))
        bind(user, "IsWindow", win.BOOL, win.HWND)
        bind(user, "IsWindowVisible", win.BOOL, win.HWND)
        bind(user, "GetWindowThreadProcessId", win.DWORD, win.HWND, ptr(win.DWORD))
        bind(user, "GetClientRect", win.BOOL, win.HWND, ptr(win.RECT))
        bind(user, "ClientToScreen", win.BOOL, win.HWND, ptr(win.POINT))
        bind(user, "GetDpiForWindow", win.UINT, win.HWND)
        bind(user, "GetWindowDpiAwarenessContext", handle, win.HWND)
        bind(user, "GetDC", win.HDC, win.HWND)
        bind(user, "ReleaseDC", integer, win.HWND, win.HDC)
        bind(gdi, "CreateCompatibleDC", win.HDC, win.HDC)
        bind(gdi, "DeleteDC", win.BOOL, win.HDC)
        bind(gdi, "CreateDIBSection", handle, win.HDC, ptr(_BitmapInfo), win.UINT,
             ptr(void_p), handle, win.DWORD)
        bind(gdi, "GetObjectW", integer, handle, integer, void_p)
        bind(gdi, "SelectObject", handle, win.HDC, handle)
        bind(gdi, "DeleteObject", win.BOOL, handle)
        bind(gdi, "BitBlt", win.BOOL, win.HDC, integer, integer, integer, integer,
             win.HDC, integer, integer, win.DWORD)
        bind(gdi, "GdiFlush", win.BOOL)
        bind(dwm, "DwmFlush", win.LONG)
        bind(dwm, "DwmGetWindowAttribute", win.LONG, win.HWND, win.DWORD, void_p, win.DWORD)
        bind(dwm, "DwmSetWindowAttribute", win.LONG, win.HWND, win.DWORD, void_p, win.DWORD)
        bind(shell, "GetScaleFactorForMonitor", win.LONG, handle, ptr(integer))


@lru_cache(maxsize=1)
def _native():
    if sys.platform != "win32":
        raise CaptureError("screen capture requires Windows")
    try:
        return _Win32()
    except (OSError, AttributeError) as exc:
        raise CaptureError("Windows capture APIs are unavailable") from exc


@contextmanager
def _native_dpi(api):
    original = api.GetThreadDpiAwarenessContext()
    awareness = next((name for context, name in _DPI_CONTEXT_NAMES.items()
                      if api.AreDpiAwarenessContextsEqual(original, context)), None)
    if awareness is None:
        raise CaptureError("unknown thread DPI awareness")
    if awareness == "per-monitor-v2":
        yield awareness
        return
    previous = api.SetThreadDpiAwarenessContext(_PM_V2)
    if not previous:
        raise CaptureError("cannot enter native per-monitor-v2 DPI context")
    try:
        if not api.AreDpiAwarenessContextsEqual(api.GetThreadDpiAwarenessContext(), _PM_V2):
            raise CaptureError("native per-monitor-v2 DPI context was not applied")
        yield awareness
    finally:
        if (not api.SetThreadDpiAwarenessContext(previous)
                or not api.AreDpiAwarenessContextsEqual(api.GetThreadDpiAwarenessContext(), original)):
            raise CaptureError("cannot restore thread DPI context")


def _monitor_snapshot(api, awareness):
    handles = []

    @api.MonitorEnumProc
    def collect(handle, _dc, _rect, _data):
        handles.append(handle)
        return True

    if not api.EnumDisplayMonitors(None, None, collect, 0):
        raise CaptureError("cannot enumerate display monitors")
    primary = []
    for handle in handles:
        info = _MonitorInfo(cbSize=ctypes.sizeof(_MonitorInfo))
        if not handle or not api.GetMonitorInfoW(handle, ctypes.byref(info)):
            raise CaptureError("cannot read monitor information")
        if info.dwFlags & _MONITORINFOF_PRIMARY:
            primary.append((handle, info))
    if len(primary) != 1 or not primary[0][1].szDevice:
        raise CaptureError("cannot identify the primary monitor")
    monitor, info = primary[0]
    mode = _DisplayMode(dmSize=ctypes.sizeof(_DisplayMode))
    if not api.EnumDisplaySettingsW(info.szDevice, _ENUM_CURRENT_SETTINGS, ctypes.byref(mode)):
        raise CaptureError("cannot read the primary monitor display mode")
    scale = ctypes.c_int()
    result = api.GetScaleFactorForMonitor(monitor, ctypes.byref(scale))
    if result != 0 or scale.value <= 0:
        raise CaptureError("cannot read the primary monitor scale")
    rect = info.rcMonitor
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if (not 0 < width == mode.dmPelsWidth or not 0 < height == mode.dmPelsHeight
            or (mode.dmFields & _DM_POSITION
                and (mode.dmPosition.x, mode.dmPosition.y) != (rect.left, rect.top))):
        raise CaptureError("primary monitor dimensions or origin do not match the native display mode")
    if (not mode.dmFields & _DM_DISPLAYORIENTATION or mode.dmDisplayOrientation > 3
            or not mode.dmBitsPerPel):
        raise CaptureError("primary monitor display mode is incomplete")
    return MonitorSnapshot(
        monitor, info.szDevice, (rect.left, rect.top, rect.right, rect.bottom),
        mode.dmDisplayOrientation, mode.dmBitsPerPel, mode.dmDisplayFrequency,
        mode.dmDisplayFlags, scale.value, awareness,
    )


def primary_monitor():
    api = _native()
    with _native_dpi(api) as awareness:
        return _monitor_snapshot(api, awareness)


def _require_hidden(api, hwnd):
    if not hwnd or not api.IsWindow(hwnd) or api.IsWindowVisible(hwnd):
        raise CaptureError("capture requires a valid, hidden application window")


def require_hidden_window(hwnd):
    _require_hidden(_native(), hwnd)


def _require_native_window(api, hwnd):
    context = api.GetWindowDpiAwarenessContext(hwnd)
    if not context or not api.AreDpiAwarenessContextsEqual(context, _PM_V2):
        raise CaptureError("overlay window does not use per-monitor-v2 DPI awareness")


def window_dpi(hwnd):
    api = _native()
    _require_native_window(api, hwnd)
    dpi = api.GetDpiForWindow(hwnd)
    if not dpi:
        raise CaptureError("cannot read the overlay window DPI")
    return dpi


def native_window_rect(hwnd):
    """Return client coverage as half-open physical virtual-screen coordinates."""
    api = _native()
    _require_native_window(api, hwnd)
    with _native_dpi(api):
        rect = win.RECT()
        if not api.GetClientRect(hwnd, ctypes.byref(rect)):
            raise CaptureError("cannot read the overlay viewport")
        first, last = win.POINT(rect.left, rect.top), win.POINT(rect.right, rect.bottom)
        if (not api.ClientToScreen(hwnd, ctypes.byref(first))
                or not api.ClientToScreen(hwnd, ctypes.byref(last))
                or first.x >= last.x or first.y >= last.y):
            raise CaptureError("cannot verify native overlay coverage")
        return first.x, first.y, last.x, last.y


def _check_dwm(result, operation):
    if result != 0:
        raise CaptureError(f"{operation} failed (HRESULT 0x{result & 0xFFFFFFFF:08x})")


def _require_owned_window(api, hwnd):
    process = win.DWORD()
    thread = api.GetWindowThreadProcessId(hwnd, ctypes.byref(process))
    if not thread or process.value != os.getpid() or thread != threading.get_native_id():
        raise CaptureError("window cloak must be managed on the application's owning thread")


def _has_app_cloak(api, hwnd):
    state = win.DWORD()
    result = api.DwmGetWindowAttribute(hwnd, _DWMWA_CLOAKED, ctypes.byref(state), ctypes.sizeof(state))
    _check_dwm(result, "DwmGetWindowAttribute(DWMWA_CLOAKED)")
    return bool(state.value & _DWM_CLOAKED_APP)


def _set_app_cloak(api, hwnd, enabled):
    value = win.BOOL(enabled)
    result = api.DwmSetWindowAttribute(hwnd, _DWMWA_CLOAK, ctypes.byref(value), ctypes.sizeof(value))
    _check_dwm(result, "DwmSetWindowAttribute(DWMWA_CLOAK)")


@contextmanager
def cloak_window_for_capture(hwnd):
    """On the owner thread, cloak a window before hiding it and restore on exit."""
    api = _native()
    _require_owned_window(api, hwnd)
    if _has_app_cloak(api, hwnd):
        yield
        return
    _set_app_cloak(api, hwnd, True)
    try:
        if not _has_app_cloak(api, hwnd):
            raise CaptureError("DWM application cloak was not applied")
        yield
    finally:
        if api.IsWindow(hwnd):
            _require_owned_window(api, hwnd)
            _set_app_cloak(api, hwnd, False)
            if _has_app_cloak(api, hwnd):
                raise CaptureError("DWM application cloak was not restored")


def _flush_compositor(api):
    _check_dwm(api.DwmFlush(), "DwmFlush")


def flush_compositor():
    _flush_compositor(_native())


def _require_current(api, awareness, expected):
    if _monitor_snapshot(api, awareness) != expected:
        raise CaptureError("primary monitor configuration changed during capture")


def capture_primary(expected, *, hidden_hwnd=None):
    """Return a detached, read-only uint8 RGB raster from one native transfer."""
    width, height = expected.width, expected.height
    if not width > 0 or not height > 0 or width * height * _CAPTURE_BYTES_PER_PIXEL > CAPTURE_MEMORY_LIMIT:
        raise CaptureError("primary monitor raster exceeds the capture memory budget")
    api = _native()
    try:
        with _native_dpi(api) as awareness:
            _require_current(api, awareness, expected)
            if hidden_hwnd is not None:
                _require_hidden(api, hidden_hwnd)
            return _read_screen(api, awareness, expected, hidden_hwnd)
    except MemoryError as exc:
        raise CaptureError("cannot allocate the primary monitor raster") from exc


def _read_screen(api, awareness, expected, hidden_hwnd):
    left, top, _, _ = expected.rect
    width, height = expected.width, expected.height
    stride = width * 4
    size = stride * height
    display = api.GetDC(None)
    if not display:
        raise CaptureError("GetDC failed")
    memory = bitmap = previous = None
    rgb = None
    cleanup_failures = []
    try:
        memory = api.CreateCompatibleDC(display)
        if not memory:
            raise CaptureError("CreateCompatibleDC failed")
        # A positive BI_RGB height requests bottom-up BGRX. GetObjectW reports
        # positive heights even for a top-down request on this Windows build.
        info = _BitmapInfo(_BitmapHeader(ctypes.sizeof(_BitmapHeader), width, height, 1, 32, 0, size))
        bits = ctypes.c_void_p()
        bitmap = api.CreateDIBSection(display, ctypes.byref(info), 0, ctypes.byref(bits), None, 0)
        if not bitmap or not bits.value:
            raise CaptureError("CreateDIBSection failed")
        _verify_dib(api, bitmap, bits.value, width, height, stride, size)
        selected = api.SelectObject(memory, bitmap)
        if not selected or selected == _HGDI_ERROR:
            raise CaptureError("SelectObject failed")
        previous = selected
        _flush_compositor(api)
        _require_current(api, awareness, expected)
        if hidden_hwnd is not None:
            _require_hidden(api, hidden_hwnd)
        if not api.BitBlt(memory, 0, 0, width, height, display, left, top,
                          _SRCCOPY | _CAPTUREBLT | _NOMIRRORBITMAP):
            raise CaptureError("BitBlt failed")
        if not api.GdiFlush():
            raise CaptureError("GdiFlush failed")
        _require_current(api, awareness, expected)
        if hidden_hwnd is not None:
            _require_hidden(api, hidden_hwnd)
        rgb = _copy_rgb(bits.value, width, height, size)
    finally:
        if previous:
            restored = api.SelectObject(memory, previous)
            if restored != bitmap:
                cleanup_failures.append("SelectObject")
        if memory and not api.DeleteDC(memory):
            cleanup_failures.append("DeleteDC")
        if bitmap and not api.DeleteObject(bitmap):
            cleanup_failures.append("DeleteObject")
        if api.ReleaseDC(None, display) != 1:
            cleanup_failures.append("ReleaseDC")
        if cleanup_failures:
            message = "GDI cleanup failed for " + ", ".join(cleanup_failures)
            if rgb is not None:
                raise CaptureError(message)
            sys.exception().add_note(message)
    return rgb


def _verify_dib(api, bitmap, address, width, height, stride, size):
    section = _DibSection()
    if api.GetObjectW(bitmap, ctypes.sizeof(section), ctypes.byref(section)) != ctypes.sizeof(section):
        raise CaptureError("GetObjectW did not return DIB metadata")
    raster, header = section.dsBm, section.dsBmih
    if ((raster.bmType, raster.bmWidth, raster.bmHeight, raster.bmWidthBytes,
         raster.bmPlanes, raster.bmBitsPixel, raster.bmBits) != (0, width, height, stride, 1, 32, address)
            or (header.biSize, header.biWidth, header.biHeight, header.biPlanes,
                header.biBitCount, header.biCompression) != (ctypes.sizeof(_BitmapHeader), width, height, 1, 32, 0)
            or header.biSizeImage not in (0, size) or section.dshSection or section.dsOffset):
        raise CaptureError("unexpected DIB dimensions, stride, orientation or color format")


def _copy_rgb(address, width, height, size):
    raw = (ctypes.c_ubyte * size).from_address(address)
    bgrx = np.ctypeslib.as_array(raw).reshape(height, width, 4)
    rgb = bgrx[::-1, :, 2::-1].copy(order="C")
    rgb.setflags(write=False)
    return rgb
