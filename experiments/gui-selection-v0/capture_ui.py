"""Hide -> one RAM capture -> frozen, pixel-bound ROI selection -> detached input."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import ctypes
from ctypes import wintypes
from pathlib import Path
import time
import uuid

import numpy as np
from shiboken6 import getCppPointer
from PySide6.QtCore import (QAbstractNativeEventFilter, QEvent, QObject, QPointF,
                           QRectF, Qt, QTimer, Signal)
from PySide6.QtGui import QColor, QFont, QGuiApplication, QIcon, QImage, QPainter, QPen
from PySide6.QtWidgets import QApplication, QToolTip, QWidget

import capture_backend as backend
from capture_input import CapturedInput
from fresh_protocol import digest, MAX_PIXELS, MAX_SIDE


@dataclass(frozen=True)
class ScreenBinding:
    epoch: int
    monitor: backend.MonitorSnapshot
    qt_name: str
    logical_rect: tuple
    qt_dpr: float
    qt_logical_dpi: float
    qt_physical_dpi: float
    qt_orientation: str
    qt_identity: int = 0

    def metadata(self):
        return dict(screen_epoch=self.epoch, monitor=asdict(self.monitor),
                    logical_rect=self.logical_rect, qt_name=self.qt_name, qt_dpr=self.qt_dpr,
                    qt_logical_dpi=self.qt_logical_dpi, qt_physical_dpi=self.qt_physical_dpi,
                    qt_orientation=self.qt_orientation)

    @property
    def generation(self):
        return digest(self.metadata())


class _NativeChanges(QAbstractNativeEventFilter):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def nativeEventFilter(self, event_type, message):
        try:
            native = wintypes.MSG.from_address(int(message))
            # Display, device and DPI messages invalidate even changed-back state.
            if native.message in (0x007E, 0x0219, 0x02E0):
                self.tracker.invalidate()
        except (ValueError, TypeError, OSError):
            self.tracker.invalidate()
        return False, 0


class ScreenTracker(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.epoch = 0
        self.app = QApplication.instance()
        self.native_filter = _NativeChanges(self)
        self.app.installNativeEventFilter(self.native_filter)
        self.app.primaryScreenChanged.connect(self.invalidate)
        self.app.screenAdded.connect(self.added)
        self.app.screenRemoved.connect(self.invalidate)
        for screen in self.app.screens():
            self.watch(screen)

    def watch(self, screen):
        for name in ('geometryChanged', 'availableGeometryChanged', 'virtualGeometryChanged',
                     'logicalDotsPerInchChanged', 'physicalDotsPerInchChanged', 'orientationChanged'):
            getattr(screen, name).connect(self.invalidate)

    def added(self, screen):
        self.watch(screen)
        self.invalidate()

    def invalidate(self, *_):
        self.epoch += 1
        self.changed.emit()

    def binding(self):
        screen = self.app.primaryScreen()
        if screen is None:
            raise backend.CaptureError('primary Qt screen is unavailable')
        monitor = backend.primary_monitor()
        g = screen.geometry()
        return ScreenBinding(self.epoch, monitor, screen.name(),
            (g.x(), g.y(), g.width(), g.height()), screen.devicePixelRatio(),
            screen.logicalDotsPerInch(), screen.physicalDotsPerInch(), str(screen.orientation()),
            qt_identity=getCppPointer(screen)[0])

    def current(self, expected):
        return self.binding() == expected


class SelectionOverlay(QWidget):
    confirmed = Signal(object)
    cancelled = Signal(str)
    painted = Signal()

    def __init__(self, frame, binding, attempt_id, *, validator):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        if (frame.dtype != np.uint8 or frame.shape != (binding.monitor.height, binding.monitor.width, 3)
                or not frame.flags.c_contiguous):
            raise backend.CaptureError('capture RGB is invalid')
        h, w, _ = frame.shape
        # This is the only full-frame Qt buffer. It is detached before the
        # acquisition future and numpy buffer are released.
        self.image = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_RGB888).copy()
        if self.image.isNull() or (self.image.width(), self.image.height()) != (w, h):
            raise backend.CaptureError('capture display allocation failed')
        self.binding, self.attempt_id, self.validator = binding, attempt_id, validator
        self.start_point = None
        self.pixel_box = None
        self.dragging = False
        self.armed = False
        self.ending = False
        self.first_paint = False
        self.viewport_at_arm = None
        self.selection_started = None
        self.selection_error = ''
        self.setAccessibleDescription('드래그를 놓으면 캡처 · 전체 화면은 Ctrl+A 후 Enter · Esc 취소')
        self.setWindowTitle('가림캡처 · 주 모니터 영역 선택')
        self.focus_icon = QIcon(str(Path(__file__).with_name('flow_icons') / 'overlay-focus.svg'))
        self.hud_font = QFont('Noto Sans KR')
        self.hud_font.setPixelSize(13)
        self.hud_bold = QFont(self.hud_font)
        self.hud_bold.setPixelSize(16)
        self.hud_bold.setBold(True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)

    def transform(self):
        return backend.RasterTransform(self.width(), self.height(), self.image.width(), self.image.height())

    def ready(self):
        if self.ending or self.armed or not self.first_paint or not self.isActiveWindow():
            return False
        if (not self.validator(self.binding)
                or backend.native_window_rect(int(self.winId())) != self.binding.monitor.rect):
            raise backend.CaptureError('overlay coverage changed')
        self.viewport_at_arm = (self.width(), self.height())
        self.armed = True
        self.selection_started = time.perf_counter()
        self.update()
        return True

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(QRectF(0, 0, self.width(), self.height()), self.image,
                          QRectF(0, 0, self.image.width(), self.image.height()))
        w, h = self.width(), self.height()
        shade = QColor(23, 28, 44, 112)
        selection = QRectF()
        if self.pixel_box:
            left, top, right, bottom = self.transform().pixels_to_viewport(self.pixel_box)
            selection = QRectF(left, top, right - left, bottom - top)
            for outside in (QRectF(0, 0, w, top), QRectF(0, bottom, w, h-bottom),
                            QRectF(0, top, left, bottom-top), QRectF(right, top, w-right, bottom-top)):
                painter.fillRect(outside, shade)
            self.paint_selection(painter, selection)
        else:
            painter.fillRect(self.rect(), shade)
        self.paint_instructions(painter, selection)
        painter.end()
        if not self.first_paint:
            self.first_paint = True
            self.painted.emit()

    def paint_selection(self, painter, selection):
        # Paint onto the overlay only. accept_selection copies the untouched input QImage.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor('#4640b8'), 2))
        painter.drawRect(selection)
        painter.setPen(QPen(QColor('#4640b8'), 1))
        painter.setBrush(QColor('white'))
        for corner in (selection.topLeft(), selection.topRight(), selection.bottomLeft(), selection.bottomRight()):
            painter.drawRect(QRectF(corner.x()-3, corner.y()-3, 6, 6))
        x0, y0, x1, y1 = self.pixel_box
        size = f'{x1-x0} × {y1-y0} px'
        painter.setFont(self.hud_bold)
        width = painter.fontMetrics().horizontalAdvance(size) + 24
        x = max(0, min(selection.left(), self.width()-width))
        y = selection.top()-38 if selection.top() >= 38 else min(self.height()-32, selection.top()+7)
        badge = QRectF(x,y,width,32)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#302b86'))
        painter.drawRoundedRect(badge,5,5)
        painter.setPen(QColor('white'))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, size)

    def paint_instructions(self, painter, selection):
        width = min(500, max(1, self.width()-16))
        panel = QRectF((self.width()-width)/2, min(36, self.height()/12), width, 62)
        if not selection.isEmpty() and self.dragging and panel.intersects(selection):
            panel.moveBottom(self.height()-12)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#171c2c'))
        painter.drawRoundedRect(panel,10,10)
        self.focus_icon.paint(painter, int(panel.x()+18), int(panel.center().y()-12), 24, 24)
        text = panel.adjusted(56, 4, -78, -4)
        painter.setPen(QColor('white'))
        painter.setFont(self.hud_bold)
        title = '놓으면 캡처' if self.armed else '정지 화면 준비 중'
        if self.pixel_box and not self.dragging:
            title = 'Enter로 전체 화면 캡처'
        elif self.selection_error:
            title = '영역을 다시 선택하세요'
        painter.drawText(text.adjusted(0,0,0,-24), Qt.AlignmentFlag.AlignVCenter, title)
        painter.setFont(self.hud_font)
        subtitle = self.selection_error or '주 모니터 · 선택 범위를 드래그하세요.  전체 화면 Ctrl+A'
        painter.drawText(text.adjusted(0,23,0,0), Qt.AlignmentFlag.AlignVCenter,
                         painter.fontMetrics().elidedText(subtitle, Qt.TextElideMode.ElideRight, int(text.width())))
        painter.drawText(panel.adjusted(0,0,-18,0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, 'Esc  취소')

    def mousePressEvent(self, event):
        if not self.armed or self.ending or event.button() != Qt.MouseButton.LeftButton:
            return
        self.start_point = (event.position().x(), event.position().y())
        self.pixel_box = None
        self.dragging = True
        self.set_selection_error('')
        self.update()

    def mouseMoveEvent(self, event):
        if not self.dragging or self.ending:
            return
        self.choose((event.position().x(), event.position().y()))

    def choose(self, end):
        error = ''
        try:
            box = self.transform().selection_to_pixels(self.start_point, end)
            x0, y0, x1, y1 = box
            if x1-x0 > MAX_SIDE or y1-y0 > MAX_SIDE or (x1-x0)*(y1-y0) > MAX_PIXELS:
                self.pixel_box = None
                error = '입력 한도를 넘었어요. 더 작은 영역을 선택하세요.'
            else:
                self.pixel_box = box
        except backend.SelectionError:
            self.pixel_box = None
            error = '주 모니터 안에서 너비와 높이가 있는 영역을 드래그하세요.'
        self.set_selection_error(error)
        self.update()

    def set_selection_error(self, message):
        if self.selection_error != message:
            self.selection_error = message
            self.setAccessibleDescription(
                message + ' · Esc 취소' if message else
                '드래그를 놓으면 캡처 · 전체 화면은 Ctrl+A 후 Enter · Esc 취소')

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragging:
            self.choose((event.position().x(), event.position().y()))
            self.dragging = False
            self.accept_selection()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.abort('cancelled')
        elif (event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier
              and self.armed and not self.dragging and not self.ending):
            # The last physical pixel is not an outer edge pointer position.
            # Explicit selection makes all [0,width) x [0,height) reachable.
            self.start_point = (0.0, 0.0)
            self.choose((float(self.width()), float(self.height())))
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.pixel_box and self.armed and not self.dragging:
            self.accept_selection()
        else:
            super().keyPressEvent(event)

    def accept_selection(self):
        if self.ending or not self.armed or self.dragging or self.pixel_box is None:
            return
        try:
            if (not self.isActiveWindow() or not self.validator(self.binding)
                    or self.viewport_at_arm != (self.width(), self.height())
                    or backend.native_window_rect(int(self.winId())) != self.binding.monitor.rect):
                self.abort('screen_changed')
                return
            x0, y0, x1, y1 = self.pixel_box
            crop = self.image.copy(x0, y0, x1-x0, y1-y0)
            if crop.isNull() or (crop.width(), crop.height()) != (x1-x0, y1-y0):
                raise backend.CaptureError('crop allocation failed')
            rows = np.frombuffer(crop.constBits(), dtype=np.uint8).reshape(crop.height(), crop.bytesPerLine())
            selected = rows[:, :crop.width()*3].reshape(crop.height(), crop.width(), 3)
            metadata = dict(self.binding.metadata(), raster_size=[self.image.width(), self.image.height()],
                            viewport_size=[self.width(), self.height()], roi=list(self.pixel_box),
                            overlay_dpi=backend.window_dpi(int(self.winId())),
                            selection_seconds=time.perf_counter() - self.selection_started)
            payload = CapturedInput.from_rgb(selected, self.attempt_id, self.binding.generation, metadata)
            self.ending = True
            self.release_frame()
            self.confirmed.emit(payload)
        except Exception:
            self.abort('capture_failed')

    def release_frame(self):
        self.image = QImage()
        self.start_point = self.pixel_box = None
        self.dragging = self.armed = False

    def abort(self, reason):
        if not self.ending:
            self.ending = True
            self.release_frame()
            self.cancelled.emit(reason)

    def event(self, event):
        if hasattr(self, 'armed') and self.armed and not self.ending:
            if event.type() == QEvent.Type.WindowDeactivate:
                self.abort('focus_lost')
            elif event.type() in (QEvent.Type.ScreenChangeInternal, QEvent.Type.DevicePixelRatioChange):
                self.abort('screen_changed')
        return super().event(event)

    def closeEvent(self, event):
        if not self.ending:
            self.abort('cancelled')
        event.accept()


class _Bridge(QObject):
    captured = Signal(str, object, object)


class CaptureFlow(QObject):
    finished = Signal(str, object)

    def __init__(self, window, *, acquire=backend.capture_primary, tracker=None):
        super().__init__(window)
        self.window = window
        self.tracker = tracker or ScreenTracker(self)
        self.acquire = acquire
        self.tracker.changed.connect(self.screen_changed)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='gui-capture')
        self.bridge = _Bridge()
        self.bridge.captured.connect(self.acquired)
        self.future = None
        self.overlay = None
        self.attempt_id = None
        self.binding = None
        self.closed = False
        self.previous_quit = None
        self.hidden = []
        self.hidden_handles = []
        self.pending_finish = None
        self.visibility_scope = None
        self.timings = {}
        self.watchdog = QTimer(self)
        self.watchdog.setInterval(100)
        self.watchdog.timeout.connect(self.check_current)

    @property
    def active(self):
        return self.attempt_id is not None

    @property
    def quiet(self):
        # A completed Future may still have its RGB queued to the GUI thread.
        return self.future is None

    def start(self):
        if self.closed or self.active or not self.quiet:
            return False
        self.attempt_id = uuid.uuid4().hex
        attempt = self.attempt_id
        self.timings = dict(started=time.perf_counter())
        try:
            self.binding = self.tracker.binding()
            self.previous_quit = QApplication.quitOnLastWindowClosed()
            QApplication.setQuitOnLastWindowClosed(False)
            QToolTip.hideText()
            visible = [w for w in QApplication.topLevelWidgets() if w.isVisible()]
            self.hidden_handles = [int(w.winId()) for w in visible]
            self.root_handle = int(self.window.winId())
            self.visibility_scope = ExitStack()
            for handle in self.hidden_handles:
                self.visibility_scope.enter_context(backend.cloak_window_for_capture(handle))
            # ToolTip.hideText can defer hiding; explicitly hide every own
            # visible top-level, including unparented transient windows.
            self.hidden = [w for w in visible if w.windowType() not in
                           (Qt.WindowType.ToolTip, Qt.WindowType.Popup)]
            for widget in visible:
                widget.hide()
            self.watchdog.start()
            # Return to the Qt event loop so hide/native messages are dispatched.
            # The worker then verifies native visibility and DwmFlush before reading.
            QTimer.singleShot(0, lambda: self.acquire_after_hide(attempt))
            return True
        except Exception:
            self.finish('capture_failed')
            return False

    def acquire_after_hide(self, attempt):
        if (attempt != self.attempt_id or not self.active or self.closed
                or self.pending_finish is not None or self.future is not None):
            return
        try:
            if not self.tracker.current(self.binding):
                self.finish('screen_changed')
                return
            handles, root_handle = tuple(self.hidden_handles), self.root_handle
            for handle in handles:
                backend.require_hidden_window(handle)
            backend.flush_compositor()
            attempt, binding = self.attempt_id, self.binding
            self.timings['hide_prepare_seconds'] = time.perf_counter() - self.timings['started']
            started = time.perf_counter()
            def task():
                for handle in handles:
                    backend.require_hidden_window(handle)
                frame = self.acquire(binding.monitor, hidden_hwnd=root_handle)
                for handle in handles:
                    backend.require_hidden_window(handle)
                return frame, time.perf_counter() - started
            self.future = self.pool.submit(task)
            def done(future):
                try:
                    value = (True, future.result())
                except Exception:
                    value = (False, 'capture_failed')
                self.bridge.captured.emit(attempt, value, future)
            self.future.add_done_callback(done)
        except Exception:
            self.finish('capture_failed')

    def acquired(self, attempt, value, future):
        if self.future is not future:
            return
        self.future = None
        if self.pending_finish is not None:
            reason, payload = self.pending_finish
            self.pending_finish = None
            self.finish(reason, payload)
            if self.closed:
                self.pool.shutdown(wait=False, cancel_futures=True)
            return
        if self.closed or attempt != self.attempt_id:
            return
        success, result = value
        if not success:
            self.finish(result)
            return
        try:
            if not self.tracker.current(self.binding):
                self.finish('screen_changed')
                return
            frame, capture_seconds = result
            self.timings['capture_seconds'] = capture_seconds
            self.overlay = SelectionOverlay(frame, self.binding, attempt, validator=self.tracker.current)
            self.overlay.confirmed.connect(self.confirmed)
            self.overlay.cancelled.connect(self.finish)
            self.overlay.painted.connect(self.arm_later)
            self.overlay.winId()
            self.overlay.windowHandle().setScreen(QGuiApplication.primaryScreen())
            self.overlay.setGeometry(QGuiApplication.primaryScreen().geometry())
            self.arm_deadline = time.perf_counter() + 2
            self.overlay.showFullScreen()
            self.overlay.raise_()
            self.overlay.activateWindow()
            self.arm_later()
        except Exception:
            self.finish('capture_failed')

    def arm_later(self):
        QTimer.singleShot(0, self.arm)

    def arm(self):
        if self.overlay is None or not self.active:
            return
        try:
            if self.overlay.armed:
                return
            if self.overlay.ready():
                return
            if time.perf_counter() >= self.arm_deadline:
                self.finish('focus_lost')
            else:
                QTimer.singleShot(25, self.arm)
        except Exception:
            self.finish('screen_changed')

    def confirmed(self, payload):
        try:
            if (not self.active or payload.attempt_id != self.attempt_id
                    or payload.screen_generation != self.binding.generation):
                self.finish('capture_failed')
                return
            if not self.tracker.current(self.binding):
                self.finish('screen_changed')
                return
            from capture_input import thaw_metadata
            metadata = thaw_metadata(payload.metadata)
            metadata.update({k: v for k, v in self.timings.items() if k != 'started'})
            selected = CapturedInput(payload.width, payload.height, payload.rgb_bytes,
                                     payload.attempt_id, payload.screen_generation, metadata)
            self.finish('confirmed', selected)
        except Exception:
            self.finish('capture_failed')

    def screen_changed(self):
        if self.active:
            self.finish('screen_changed')

    def check_current(self):
        if self.active:
            try:
                if not self.tracker.current(self.binding):
                    self.finish('screen_changed')
            except Exception:
                self.finish('capture_failed')

    def finish(self, reason, payload=None):
        if not self.active:
            return
        self.watchdog.stop()
        if self.future is not None:
            # Do not expose our windows while BitBlt might still be pending.
            # Invalidate the attempt logically; drain its queued RGB before
            # restoring windows or allowing the next acquisition.
            self.pending_finish = (reason, None)
            return
        self.attempt_id = None
        if self.overlay is not None:
            overlay, self.overlay = self.overlay, None
            overlay.ending = True
            overlay.release_frame()
            overlay.close()
            overlay.deleteLater()
        if self.visibility_scope is not None:
            scope, self.visibility_scope = self.visibility_scope, None
            try:
                scope.close()
            except Exception:
                reason, payload = 'capture_failed', None
        if not self.closed:
            for widget in self.hidden:
                try:
                    widget.show()
                except RuntimeError:
                    pass  # A transient dialog may already have been destroyed.
            if reason != 'focus_lost':
                self.window.raise_()
                self.window.activateWindow()
        self.hidden = []
        self.hidden_handles = []
        self.binding = None
        if self.previous_quit is not None:
            QApplication.setQuitOnLastWindowClosed(self.previous_quit)
            self.previous_quit = None
        self.finished.emit(reason, payload)

    def shutdown(self):
        self.closed = True
        self.finish('closed')
        if self.quiet:
            self.pool.shutdown(wait=False, cancel_futures=True)
