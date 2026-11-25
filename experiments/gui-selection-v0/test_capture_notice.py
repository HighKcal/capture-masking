import unittest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget

from capture_notice import CaptureNotice


class CaptureNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.parent = QWidget()
        self.notice = CaptureNotice(self.parent)

    def tearDown(self):
        self.notice.dismiss()
        self.parent.close()

    def test_popup_uses_plain_text_and_refuses_focus_and_pointer_input(self):
        notice = self.notice
        self.assertEqual(notice.textFormat(), Qt.TextFormat.PlainText)
        self.assertEqual(notice.focusPolicy(), Qt.FocusPolicy.NoFocus)
        self.assertTrue(notice.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating))
        self.assertTrue(notice.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus)
        self.assertTrue(notice.windowFlags() & Qt.WindowType.WindowTransparentForInput)

    def test_replacement_uses_one_popup_and_dismiss_stops_timeout(self):
        notice = self.notice
        notice.show_message('가린 이미지 복사됨')
        notice.show_message('후보 0개 · 이번 캡처 원본 복사됨')
        self.assertTrue(notice.isVisible())
        self.assertEqual(notice.text(), '후보 0개 · 이번 캡처 원본 복사됨')
        self.assertTrue(notice.timer.isActive())
        self.assertTrue(notice.timer.isSingleShot())
        self.assertTrue(notice.screen().availableGeometry().contains(notice.geometry()))
        notice.timer.timeout.emit()
        self.assertFalse(notice.isVisible())
        self.assertFalse(notice.timer.isActive())
