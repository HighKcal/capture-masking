"""Offscreen checks of overlay pixel selection and frame lifetime."""

from dataclasses import replace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication

from capture_backend import MonitorSnapshot
from capture_ui import ScreenBinding, SelectionOverlay


def frame_rgb(width, height):
    y, x = np.indices((height, width), dtype=np.uint8)
    return np.stack((7 + x, 91 + y, 201 + x + y), axis=-1)


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def overlay(self, frame, viewport, *, current=None):
        height, width, _ = frame.shape
        origin = (-250, 120)
        monitor = MonitorSnapshot(11, 'synthetic', (*origin, origin[0] + width, origin[1] + height),
                                  0, 32, 60, 0, round(100 * width / viewport[0]), 'per-monitor-v2')
        binding = ScreenBinding(1, monitor, 'synthetic', (*origin, *viewport),
                                width / viewport[0], 96, 96, 'landscape')
        if current is not None:
            current.append(binding)
        overlay = SelectionOverlay(frame, binding, 'a' * 32,
                                   validator=lambda expected: current is None or current[-1] == expected)
        self.addCleanup(overlay.close)
        overlay.resize(*viewport)
        overlay.first_paint = True
        self.enterContext(patch.object(overlay, 'isActiveWindow', return_value=True))
        self.enterContext(patch.object(overlay, 'winId', return_value=123))
        self.enterContext(patch('capture_ui.backend.native_window_rect', return_value=monitor.rect))
        self.enterContext(patch('capture_ui.backend.window_dpi', return_value=round(96 * binding.qt_dpr)))
        confirmed, cancelled = [], []
        overlay.confirmed.connect(confirmed.append)
        overlay.cancelled.connect(cancelled.append)
        self.assertTrue(overlay.ready())
        return overlay, confirmed, cancelled

    def begin_drag(self, overlay, start, finish):
        left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
        for kind, point, button, buttons in (
            (QEvent.Type.MouseButtonPress, start, left, left),
            (QEvent.Type.MouseMove, finish, none, left),
        ):
            pos = QPointF(*point)
            event = QMouseEvent(kind, pos, pos, button, buttons, Qt.KeyboardModifier.NoModifier)
            self.qt.sendEvent(overlay, event)

    def release(self, overlay, finish):
        pos = QPointF(*finish)
        self.qt.sendEvent(overlay, QMouseEvent(QEvent.Type.MouseButtonRelease, pos, pos,
                          Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))

    def key(self, overlay, key, modifiers=Qt.KeyboardModifier.NoModifier):
        event = QKeyEvent(QEvent.Type.KeyPress, key, modifiers)
        self.qt.sendEvent(overlay, event)

    def test_confirm_exact_padded_rgb_with_reverse_and_edge_drags(self):
        cases = [
            ((10, 8), (10, 8), (10, 8), (7, 5), (7, 5, 10, 8)),
            ((10, 10), (8, 8), (8, 8), (5.6, 5.6), (7, 7, 10, 10)),
            ((12, 12), (8, 8), (0, 0), (2, 2), (0, 0, 3, 3)),
            ((10, 10), (8, 8), (0.8, 0.8), (1.6, 1.6), (1, 1, 2, 2)),
        ]
        for raster, viewport, start, finish, box in cases:
            with self.subTest(raster=raster, viewport=viewport, box=box):
                frame = frame_rgb(*raster)
                x0, y0, x1, y1 = box
                expected = frame[y0:y1, x0:x1].tobytes()
                source_ref = weakref.ref(frame)
                overlay, confirmed, cancelled = self.overlay(frame, viewport)
                full_frame = overlay.image
                frame[:] = 0
                del frame
                self.assertIsNone(source_ref())
                self.assertGreater(full_frame.copy(x0, y0, x1-x0, y1-y0).bytesPerLine(), (x1-x0) * 3)

                self.begin_drag(overlay, start, finish)
                self.assertEqual(overlay.pixel_box, box)
                self.assertEqual(confirmed, [])
                self.release(overlay, finish)
                self.assertEqual(len(confirmed), 1)
                self.release(overlay, finish)
                self.key(overlay, Qt.Key.Key_Return)
                self.assertEqual(len(confirmed), 1)
                self.assertEqual(cancelled, [])
                self.assertTrue(overlay.image.isNull())
                full_frame.fill(0)
                (selected,) = confirmed
                self.assertEqual((selected.width, selected.height), (x1-x0, y1-y0))
                self.assertEqual(selected.metadata['roi'], box)
                self.assertEqual(selected.metadata['raster_size'], raster)
                self.assertEqual(selected.metadata['viewport_size'], viewport)
                self.assertEqual(selected.screen_generation, overlay.binding.generation)
                self.assertIs(type(selected.rgb_bytes), bytes)
                self.assertEqual(selected.rgb_bytes, expected)

    def test_ctrl_a_selects_last_row_and_column_for_explicit_confirmation(self):
        frame = frame_rgb(10, 10)
        overlay, confirmed, cancelled = self.overlay(frame, (8, 8))
        self.key(overlay, Qt.Key.Key_A)
        self.assertIsNone(overlay.pixel_box)

        self.key(overlay, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(overlay.pixel_box, (0, 0, 10, 10))
        self.assertEqual(confirmed, [])
        self.key(overlay, Qt.Key.Key_Return)
        (selected,) = confirmed
        self.assertEqual((selected.width, selected.height), (10, 10))
        self.assertEqual(selected.metadata['roi'], (0, 0, 10, 10))
        self.assertEqual(selected.rgb_bytes, frame.tobytes())
        self.assertEqual(cancelled, [])
        self.assertTrue(overlay.image.isNull())

    def test_ctrl_a_over_limit_clears_previous_selection_and_blocks_enter(self):
        overlay, confirmed, cancelled = self.overlay(frame_rgb(10, 10), (8, 8))
        self.key(overlay, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        self.assertIsNotNone(overlay.pixel_box)
        with patch('capture_ui.MAX_PIXELS', 99):
            self.key(overlay, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
            self.assertIsNone(overlay.pixel_box)
            self.assertIn('한도', overlay.selection_error)
            self.assertIn(overlay.selection_error, overlay.accessibleDescription())
            self.key(overlay, Qt.Key.Key_Return)
        self.assertEqual(confirmed, [])
        self.assertEqual(cancelled, [])
        self.assertTrue(overlay.armed)
        self.assertFalse(overlay.image.isNull())

    def test_zero_area_and_outside_release_clear_previous_selection(self):
        overlay, confirmed, cancelled = self.overlay(frame_rgb(10, 10), (8, 8))
        for invalid in ((1, 1), (8.01, 3), (-0.01, 3)):
            with self.subTest(release=invalid):
                self.key(overlay, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
                self.assertIsNotNone(overlay.pixel_box)
                self.assertEqual(overlay.selection_error, '')
                self.begin_drag(overlay, (1, 1), (3, 3))
                self.release(overlay, invalid)
                self.assertIsNone(overlay.pixel_box)
                self.assertIn('드래그', overlay.selection_error)
                self.assertIn(overlay.selection_error, overlay.accessibleDescription())
                self.key(overlay, Qt.Key.Key_Return)
                self.assertEqual(confirmed, [])
                self.assertEqual(cancelled, [])
                self.assertFalse(overlay.image.isNull())
        self.begin_drag(overlay, (1, 1), (3, 3))
        self.release(overlay, (4, 4))
        self.assertEqual(confirmed[0].metadata['roi'], (1, 1, 5, 5))

    def test_escape_and_focus_loss_discard_frame_and_block_enter(self):
        for reason in ('cancelled', 'focus_lost'):
            with self.subTest(reason=reason):
                overlay, confirmed, cancelled = self.overlay(frame_rgb(10, 10), (8, 8))
                self.begin_drag(overlay, (0, 0), (2, 2))
                if reason == 'cancelled':
                    self.key(overlay, Qt.Key.Key_Escape)
                else:
                    self.qt.sendEvent(overlay, QEvent(QEvent.Type.WindowDeactivate))
                self.release(overlay, (2, 2))
                self.key(overlay, Qt.Key.Key_Return)
                self.assertEqual(confirmed, [])
                self.assertEqual(cancelled, [reason])
                self.assertTrue(overlay.image.isNull())
                self.assertIsNone(overlay.pixel_box)
                self.assertFalse(overlay.armed)

    def test_changed_back_screen_geometry_still_rejects_confirmation(self):
        current = []
        overlay, confirmed, cancelled = self.overlay(frame_rgb(10, 10), (8, 8), current=current)
        self.begin_drag(overlay, (0, 0), (2, 2))
        original = current[-1]
        moved = replace(original.monitor, rect=(-249, 120, -239, 130))
        current.append(replace(original, epoch=2, monitor=moved))
        current.append(replace(original, epoch=3))
        self.assertEqual(current[-1].monitor, original.monitor)
        self.assertEqual(current[-1].logical_rect, original.logical_rect)
        self.assertNotEqual(current[-1].generation, original.generation)
        self.release(overlay, (2, 2))
        self.assertEqual(confirmed, [])
        self.assertEqual(cancelled, ['screen_changed'])
        self.assertTrue(overlay.image.isNull())


if __name__ == '__main__':
    unittest.main()
