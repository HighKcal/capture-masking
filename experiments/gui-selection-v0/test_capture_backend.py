import ctypes
from dataclasses import replace
import math
import os
import sys
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

import capture_backend as backend


class RasterTransformTests(unittest.TestCase):
    def test_fractional_scales_with_nonzero_origins_and_reversed_drag(self):
        for raster_size, native_origin, expected, expected_native in (
            ((40, 24), (-280, 40), (1, 2, 9, 13), (-279, 42, -271, 53)),
            ((50, 30), (-350, 50), (1, 3, 12, 17), (-349, 53, -338, 67)),
            ((60, 36), (-420, 60), (2, 3, 14, 20), (-418, 63, -406, 80)),
        ):
            with self.subTest(raster_size=raster_size):
                transform = backend.RasterTransform(40, 24, *raster_size)
                ox, oy = -71, 13
                start_global = ox + 1.5, oy + 2.5
                end_global = ox + 9, oy + 13
                start = start_global[0] - ox, start_global[1] - oy
                end = end_global[0] - ox, end_global[1] - oy
                box = transform.selection_to_pixels(end, start)
                self.assertEqual(box, expected)
                nx, ny = native_origin
                left, top, right, bottom = box
                self.assertEqual((nx + left, ny + top, nx + right, ny + bottom), expected_native)
                left, top, right, bottom = transform.pixels_to_viewport(expected)
                self.assertEqual(transform.selection_to_pixels((left, top), (right, bottom)), expected)

    def test_full_extent_one_pixel_and_non_uniform_ratios(self):
        transform = backend.RasterTransform(31, 25, 39, 31)
        self.assertEqual((transform.scale_x, transform.scale_y), (39 / 31, 31 / 25))
        self.assertEqual(transform.selection_to_pixels((31, 25), (0, 0)), (0, 0, 39, 31))
        for y in range(31):
            for x in range(39):
                box = x, y, x + 1, y + 1
                l, t, r, b = transform.pixels_to_viewport(box)
                self.assertEqual(transform.selection_to_pixels((l, t), (r, b)), box)

    def test_positive_tiny_extent_and_strict_edge_neighbors(self):
        transform = backend.RasterTransform(20, 20, 25, 25)
        minimum = math.nextafter(0.0, 1.0)
        self.assertEqual(transform.selection_to_pixels((0, 0), (minimum, minimum)), (0, 0, 1, 1))
        edge = 12 * 20 / 25
        before = math.nextafter(edge, 0.0)
        after = math.nextafter(edge, 20.0)
        self.assertEqual(transform.selection_to_pixels((before, before), (edge, edge)), (11, 11, 12, 12))
        self.assertEqual(transform.selection_to_pixels((edge, edge), (after, after)), (12, 12, 13, 13))

    def test_invalid_and_zero_continuous_area_are_rejected(self):
        transform = backend.RasterTransform(40, 24, 50, 30)
        for start, end in (
            ((1.2, 2), (1.2, 3)), ((1, 2.3), (4, 2.3)),
            ((0, 0), (0, 0)), ((-0.000001, 2), (1, 3)),
            ((3, 4), (40.000001, 10)), ((3, 4), (39, 24.000001)),
            ((math.nan, 4), (39, 24)), ((3, 4), (39, math.inf)),
        ):
            with self.subTest(start=start, end=end), self.assertRaises(backend.SelectionError):
                transform.selection_to_pixels(start, end)
        with self.assertRaises(backend.SelectionError):
            backend.RasterTransform(40, 0, 50, 30)
        with self.assertRaises(backend.SelectionError):
            transform.pixels_to_viewport((-1, 0, 3, 1))


class NativeDouble:
    """Minimal GDI/User32 double with real ctypes buffer layout and recorded calls."""

    snapshot = backend.MonitorSnapshot(11, "synthetic", (-1, -1, 2, 1), 0, 32, 60, 0, 125,
                                       "per-monitor-v2")

    def __init__(self):
        self.bytes = (ctypes.c_ubyte * 24)(
            90, 80, 70, 6, 120, 110, 100, 5, 150, 140, 130, 4,
            9, 8, 7, 3, 12, 11, 10, 2, 15, 14, 13, 1,
        )
        self.header = None
        self.awareness = backend._PM_V2
        self.api = Mock()
        self.api.GetThreadDpiAwarenessContext.side_effect = lambda: self.awareness
        self.api.AreDpiAwarenessContextsEqual.side_effect = lambda a, b: a == b
        self.api.SetThreadDpiAwarenessContext.side_effect = self.set_awareness
        self.api.MonitorEnumProc.side_effect = lambda callback: callback
        self.api.EnumDisplayMonitors.side_effect = self.enumerate_monitors
        self.api.GetMonitorInfoW.side_effect = self.monitor_info
        self.api.EnumDisplaySettingsW.side_effect = self.display_mode
        self.api.GetScaleFactorForMonitor.side_effect = self.monitor_scale
        self.api.IsWindow.return_value = 1
        self.api.IsWindowVisible.return_value = 0
        self.api.GetDC.return_value = 21
        self.api.CreateCompatibleDC.return_value = 22
        self.api.CreateDIBSection.side_effect = self.create_dib
        self.api.GetObjectW.side_effect = self.dib_metadata
        self.api.SelectObject.side_effect = lambda dc, handle: 24 if handle == 23 else 23
        self.api.DwmFlush.return_value = 0
        self.api.BitBlt.return_value = 1
        self.api.GdiFlush.return_value = 1
        self.api.DeleteDC.return_value = 1
        self.api.DeleteObject.return_value = 1
        self.api.ReleaseDC.return_value = 1

    def set_awareness(self, value):
        previous, self.awareness = self.awareness, value
        return previous

    @staticmethod
    def enumerate_monitors(dc, rect, callback, data):
        return callback(10, dc, rect, data) and callback(11, dc, rect, data)

    def monitor_info(self, handle, pointer):
        info = ctypes.cast(pointer, ctypes.POINTER(backend._MonitorInfo)).contents
        if handle == 10:
            info.rcMonitor = backend.win.RECT(-5, -1, -1, 1)
            info.szDevice = "synthetic-secondary"
            return 1
        info.rcMonitor = backend.win.RECT(*self.snapshot.rect)
        info.dwFlags = backend._MONITORINFOF_PRIMARY
        info.szDevice = self.snapshot.device
        return 1

    def display_mode(self, device, mode, pointer):
        current = ctypes.cast(pointer, ctypes.POINTER(backend._DisplayMode)).contents
        current.dmFields = backend._DM_POSITION | backend._DM_DISPLAYORIENTATION
        current.dmPosition = backend.win.POINT(*self.snapshot.rect[:2])
        current.dmPelsWidth = self.snapshot.width
        current.dmPelsHeight = self.snapshot.height
        current.dmBitsPerPel = self.snapshot.bits_per_pixel
        current.dmDisplayFrequency = self.snapshot.refresh_hz
        current.dmDisplayOrientation = self.snapshot.orientation
        return 1

    def monitor_scale(self, handle, pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = self.snapshot.scale_percent
        return 0

    def create_dib(self, dc, info_pointer, usage, bits_pointer, section, offset):
        info = ctypes.cast(info_pointer, ctypes.POINTER(backend._BitmapInfo)).contents
        self.header = backend._BitmapHeader.from_buffer_copy(info.bmiHeader)
        ctypes.cast(bits_pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value = ctypes.addressof(self.bytes)
        return 23

    def dib_metadata(self, handle, size, pointer):
        section = ctypes.cast(pointer, ctypes.POINTER(backend._DibSection)).contents
        section.dsBm = backend._Bitmap(0, 3, 2, 12, 1, 32, ctypes.addressof(self.bytes))
        section.dsBmih = self.header
        return size


class CaptureTests(unittest.TestCase):
    def test_single_transfer_native_extent_color_order_and_detached_buffer(self):
        native = NativeDouble()
        with patch.object(backend, "_native", return_value=native.api):
            image = backend.capture_primary(native.snapshot, hidden_hwnd=41)
        native.api.BitBlt.assert_called_once_with(22, 0, 0, 3, 2, 21, -1, -1, 0xC0CC0020)
        native.api.DwmFlush.assert_called_once_with()
        native.api.GdiFlush.assert_called_once_with()
        names = [call[0] for call in native.api.mock_calls]
        self.assertLess(names.index("DwmFlush"), names.index("BitBlt"))
        self.assertLess(names.index("BitBlt"), names.index("GdiFlush"))
        self.assertEqual(names.count("EnumDisplaySettingsW"), 3)
        self.assertEqual(names.count("IsWindowVisible"), 3)
        self.assertEqual(names[-4:], ["SelectObject", "DeleteDC", "DeleteObject", "ReleaseDC"])
        native.api.SelectObject.assert_called_with(22, 24)
        native.bytes[:] = [0] * 24
        np.testing.assert_array_equal(image, [[[7, 8, 9], [10, 11, 12], [13, 14, 15]],
                                              [[70, 80, 90], [100, 110, 120], [130, 140, 150]]])
        self.assertTrue(image.flags.owndata and image.flags.c_contiguous)
        self.assertFalse(image.flags.writeable)
        self.assertEqual(image.dtype, np.uint8)

    def test_black_capture_is_valid_when_win32_operations_succeed(self):
        native = NativeDouble()
        native.bytes[:] = [0] * 24
        with patch.object(backend, "_native", return_value=native.api):
            result = backend.capture_primary(native.snapshot)
        self.assertEqual(result.shape, (2, 3, 3))
        self.assertFalse(result.any())

    def test_resource_failures_release_acquired_handles(self):
        for failed, return_value, expected_cleanup in (
            ("GetDC", 0, ()),
            ("CreateCompatibleDC", 0, ("ReleaseDC",)),
            ("CreateDIBSection", 0, ("DeleteDC", "ReleaseDC")),
            ("GetObjectW", 0, ("DeleteDC", "DeleteObject", "ReleaseDC")),
            ("SelectObject", backend._HGDI_ERROR, ("DeleteDC", "DeleteObject", "ReleaseDC")),
            ("DwmFlush", -1, ("DeleteDC", "DeleteObject", "ReleaseDC")),
            ("BitBlt", 0, ("DeleteDC", "DeleteObject", "ReleaseDC")),
            ("GdiFlush", 0, ("DeleteDC", "DeleteObject", "ReleaseDC")),
        ):
            native = NativeDouble()
            operation = getattr(native.api, failed)
            operation.side_effect = None
            operation.return_value = return_value
            with self.subTest(failed=failed), patch.object(backend, "_native", return_value=native.api):
                with self.assertRaisesRegex(backend.CaptureError, failed):
                    backend.capture_primary(native.snapshot)
                cleanup = tuple(call[0] for call in native.api.mock_calls
                                if call[0] in {"DeleteDC", "DeleteObject", "ReleaseDC"})
                self.assertEqual(cleanup, expected_cleanup)

    def test_wrong_stride_and_cleanup_failure_are_errors(self):
        native = NativeDouble()

        def bad_metadata(handle, size, pointer):
            native.dib_metadata(handle, size, pointer)
            section = ctypes.cast(pointer, ctypes.POINTER(backend._DibSection)).contents
            section.dsBm.bmWidthBytes += 1
            return size

        native.api.GetObjectW.side_effect = bad_metadata
        native.api.DeleteObject.return_value = 0
        with patch.object(backend, "_native", return_value=native.api):
            with self.assertRaisesRegex(backend.CaptureError, "stride") as error:
                backend.capture_primary(native.snapshot)
        self.assertEqual(error.exception.__notes__, ["GDI cleanup failed for DeleteObject"])
        native.api.BitBlt.assert_not_called()
        native.api.ReleaseDC.assert_called_once_with(None, 21)

    def test_cleanup_failure_after_copy_still_rejects_capture(self):
        native = NativeDouble()
        native.api.ReleaseDC.return_value = 0
        with patch.object(backend, "_native", return_value=native.api):
            with self.assertRaisesRegex(backend.CaptureError, "cleanup failed for ReleaseDC"):
                backend.capture_primary(native.snapshot)

    def test_configuration_change_before_and_after_transfer(self):
        for event in ("DwmFlush", "BitBlt"):
            native = NativeDouble()
            expected = native.snapshot

            def change_scale(*args):
                native.snapshot = replace(expected, scale_percent=150)
                return 0 if event == "DwmFlush" else 1

            getattr(native.api, event).side_effect = change_scale
            with self.subTest(event=event), patch.object(backend, "_native", return_value=native.api):
                with self.assertRaisesRegex(backend.CaptureError, "configuration changed"):
                    backend.capture_primary(expected)
                self.assertEqual(native.api.BitBlt.call_count, int(event == "BitBlt"))
                native.api.ReleaseDC.assert_called_once_with(None, 21)

    def test_window_that_becomes_visible_aborts_before_transfer(self):
        for checks, transfer_count in (((0, 1), 0), ((0, 0, 1), 1)):
            native = NativeDouble()
            native.api.IsWindowVisible.side_effect = checks
            with self.subTest(checks=checks), patch.object(backend, "_native", return_value=native.api):
                with self.assertRaisesRegex(backend.CaptureError, "hidden"):
                    backend.capture_primary(native.snapshot, hidden_hwnd=41)
            self.assertEqual(native.api.BitBlt.call_count, transfer_count)
            native.api.ReleaseDC.assert_called_once_with(None, 21)

    def test_capture_budget_precedes_any_win32_calls(self):
        oversized = replace(NativeDouble.snapshot, rect=(0, 0, 7680, 4320))
        with patch.object(backend, "_native") as native:
            with self.assertRaisesRegex(backend.CaptureError, "memory budget"):
                backend.capture_primary(oversized)
            native.assert_not_called()

    def test_temporary_thread_dpi_context_is_restored_on_failure(self):
        native = NativeDouble()
        native.awareness = -1
        native.api.GetMonitorInfoW.side_effect = None
        native.api.GetMonitorInfoW.return_value = 0
        with patch.object(backend, "_native", return_value=native.api):
            with self.assertRaisesRegex(backend.CaptureError, "read monitor information"):
                backend.primary_monitor()
        self.assertEqual(native.awareness, -1)
        self.assertEqual([call.args for call in native.api.SetThreadDpiAwarenessContext.call_args_list],
                         [(-4,), (-1,)])

    def test_capture_can_run_from_thread_with_different_initial_dpi_context(self):
        native = NativeDouble()
        native.awareness = -1
        with patch.object(backend, "_native", return_value=native.api):
            self.assertEqual(backend.primary_monitor().thread_awareness, "unaware")
            self.assertEqual(backend.capture_primary(native.snapshot).shape, (2, 3, 3))
        self.assertEqual(native.awareness, -1)


class CloakTests(unittest.TestCase):
    def setUp(self):
        self.flags = 0
        self.api = Mock()
        self.api.IsWindow.return_value = 1
        self.api.GetWindowThreadProcessId.side_effect = self.window_owner
        self.api.DwmGetWindowAttribute.side_effect = self.get_state
        self.api.DwmSetWindowAttribute.side_effect = self.set_state
        native = patch.object(backend, "_native", return_value=self.api)
        native.start()
        self.addCleanup(native.stop)

    def window_owner(self, hwnd, pointer):
        ctypes.cast(pointer, ctypes.POINTER(backend.win.DWORD)).contents.value = os.getpid()
        return threading.get_native_id()

    def get_state(self, hwnd, attribute, pointer, size):
        self.assertEqual((hwnd, attribute, size), (41, 14, ctypes.sizeof(backend.win.DWORD)))
        ctypes.cast(pointer, ctypes.POINTER(backend.win.DWORD)).contents.value = self.flags
        return 0

    def set_state(self, hwnd, attribute, pointer, size):
        self.assertEqual((hwnd, attribute, size), (41, 13, ctypes.sizeof(backend.win.BOOL)))
        enabled = ctypes.cast(pointer, ctypes.POINTER(backend.win.BOOL)).contents.value
        self.flags = self.flags | 1 if enabled else self.flags & ~1
        return 0

    def test_restores_only_app_cloak_even_when_body_raises(self):
        self.flags = 2
        with self.assertRaisesRegex(ValueError, "capture cancelled"):
            with backend.cloak_window_for_capture(41):
                self.assertEqual(self.flags, 3)
                raise ValueError("capture cancelled")
        self.assertEqual(self.flags, 2)

    def test_preserves_existing_app_cloak(self):
        self.flags = 5
        with backend.cloak_window_for_capture(41):
            self.assertEqual(self.flags, 5)
        self.assertEqual(self.flags, 5)
        self.api.DwmSetWindowAttribute.assert_not_called()

    def test_rejects_window_outside_owning_process_or_thread(self):
        self.api.GetWindowThreadProcessId.side_effect = None
        self.api.GetWindowThreadProcessId.return_value = threading.get_native_id() + 1
        with self.assertRaisesRegex(backend.CaptureError, "owning thread"):
            with backend.cloak_window_for_capture(41):
                self.fail("unowned window was accepted")
        self.api.DwmGetWindowAttribute.assert_not_called()
        self.api.DwmSetWindowAttribute.assert_not_called()

    def test_restores_after_enable_reported_success_but_verification_failed(self):
        self.api.DwmSetWindowAttribute.side_effect = None
        self.api.DwmSetWindowAttribute.return_value = 0
        with self.assertRaisesRegex(backend.CaptureError, "not applied"):
            with backend.cloak_window_for_capture(41):
                self.fail("uncloaked window was accepted")
        self.assertEqual(self.api.DwmSetWindowAttribute.call_count, 2)

    def test_restore_failure_surfaces_hresult(self):
        with self.assertRaisesRegex(backend.CaptureError, "0x80070005"):
            with backend.cloak_window_for_capture(41):
                self.api.DwmSetWindowAttribute.side_effect = None
                self.api.DwmSetWindowAttribute.return_value = -2147024891


@unittest.skipUnless(sys.platform == "win32", "Win32 memory DIB test")
class NativeMemoryDibTests(unittest.TestCase):
    def test_real_win32_bottom_up_dib_with_synthetic_pixels(self):
        # Draw in a RAM-only DC. This test never acquires a desktop DC.
        api = backend._native()
        memory = api.CreateCompatibleDC(None)
        self.assertTrue(memory)
        self.addCleanup(api.DeleteDC, memory)
        width, height, size = 3, 2, 24
        info = backend._BitmapInfo(backend._BitmapHeader(40, width, height, 1, 32, 0, size))
        bits = ctypes.c_void_p()
        bitmap = api.CreateDIBSection(None, ctypes.byref(info), 0, ctypes.byref(bits), None, 0)
        self.assertTrue(bitmap and bits.value)
        self.addCleanup(api.DeleteObject, bitmap)
        backend._verify_dib(api, bitmap, bits.value, width, height, width * 4, size)
        original = api.SelectObject(memory, bitmap)
        self.assertNotIn(original, (None, 0, backend._HGDI_ERROR))
        self.addCleanup(api.SelectObject, memory, original)

        set_pixel = ctypes.WinDLL("gdi32.dll").SetPixelV
        set_pixel.argtypes = backend.win.HDC, ctypes.c_int, ctypes.c_int, backend.win.DWORD
        set_pixel.restype = backend.win.BOOL
        expected = np.array([[[7, 8, 9], [10, 11, 12], [13, 14, 15]],
                             [[70, 80, 90], [100, 110, 120], [130, 140, 150]]], dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                red, green, blue = map(int, expected[y, x])
                self.assertTrue(set_pixel(memory, x, y, red | green << 8 | blue << 16))
        self.assertTrue(api.GdiFlush())
        result = backend._copy_rgb(bits.value, width, height, size)
        np.testing.assert_array_equal(result, expected)


@unittest.skipUnless(sys.platform == "win32", "Win32 hidden window test")
class NativeCloakTests(unittest.TestCase):
    def test_real_win32_cloak_and_restore_on_hidden_window(self):
        win = backend.win
        user = ctypes.WinDLL("user32.dll", use_last_error=True)
        create = user.CreateWindowExW
        create.argtypes = (win.DWORD, win.LPCWSTR, win.LPCWSTR, win.DWORD, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int, ctypes.c_int, win.HWND, win.HMENU,
                           win.HINSTANCE, ctypes.c_void_p)
        create.restype = win.HWND
        destroy = user.DestroyWindow
        destroy.argtypes = (win.HWND,)
        destroy.restype = win.BOOL
        # WS_POPUP without WS_VISIBLE keeps the test window off the desktop.
        hwnd = create(0, "STATIC", "", 0x80000000, 0, 0, 8, 8, None, None, None, None)
        self.assertTrue(hwnd)
        self.addCleanup(destroy, hwnd)
        api = backend._native()
        backend.require_hidden_window(hwnd)
        self.assertFalse(backend._has_app_cloak(api, hwnd))
        with backend.cloak_window_for_capture(hwnd):
            self.assertTrue(backend._has_app_cloak(api, hwnd))
            with backend.cloak_window_for_capture(hwnd):
                self.assertTrue(backend._has_app_cloak(api, hwnd))
            self.assertTrue(backend._has_app_cloak(api, hwnd))
        self.assertFalse(backend._has_app_cloak(api, hwnd))


if __name__ == "__main__":
    unittest.main()
