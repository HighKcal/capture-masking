"""Short, text-only status messages for background captures."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor, QGuiApplication, QPalette
from PySide6.QtWidgets import QLabel


class CaptureNotice(QLabel):
    def __init__(self, parent):
        flags = (
            Qt.WindowType.ToolTip
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
            | Qt.WindowType.WindowTransparentForInput
        )
        super().__init__(parent, flags)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWindowTitle("가림캡처 알림")
        self.setAccessibleName("가림캡처 알림")
        self.setBackgroundRole(QPalette.ColorRole.ToolTipBase)
        self.setForegroundRole(QPalette.ColorRole.ToolTipText)
        self.setAutoFillBackground(True)
        self.setMargin(10)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.dismiss)

    def show_message(self, text):
        screen = QGuiApplication.screenAt(QCursor.pos()) or self.screen()
        area = screen.availableGeometry()
        inset = 16
        self.setText(text)
        self.setWordWrap(False)
        natural = self.sizeHint()
        width = min(natural.width(), 460, max(1, area.width() - 2 * inset))
        self.setWordWrap(width < natural.width())
        height = self.heightForWidth(width) if self.wordWrap() else natural.height()
        self.resize(width, height)
        self.move(area.right() + 1 - inset - width, area.bottom() + 1 - inset - height)
        self.show()
        self.timer.start(4000)

    def dismiss(self):
        self.timer.stop()
        self.hide()
