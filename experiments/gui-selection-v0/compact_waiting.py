"""Input and masking controls for the compact Windows capture window."""

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)


ICONS = Path(__file__).with_name('compact_icons')


def app_icon():
    tile = QPixmap(64, 64)
    tile.fill(Qt.GlobalColor.transparent)
    painter = QPainter(tile)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor('#4640b8'))
    painter.drawRoundedRect(0, 0, 64, 64, 15, 15)
    QIcon(str(ICONS / 'crop.svg')).paint(painter, 10, 10, 44, 44)
    painter.end()
    return QIcon(tile)


class CompactWaiting(QWidget):
    capture_requested = Signal()
    file_requested = Signal()
    enabled_changed = Signal(bool)
    mode_changed = Signal(str)
    expanded_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('compactWaiting')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setWindowIcon(app_icon())
        self.setStyleSheet("""
            QWidget#compactWaiting { background: white; color: #171c2c; }
            QLabel, QPushButton { font-family: 'Noto Sans KR'; color: #171c2c; }
            QPushButton { background: white; border: 1px solid #d9dfea;
                border-radius: 7px; font-size: 15px; font-weight: bold; }
            QPushButton:hover { border-color: #aca9e0; }
            QPushButton#capture { color: white; background: #4640b8; border-color: #4640b8; }
            QPushButton#capture:hover { background: #39339b; }
            QPushButton#settings { background: #eeedfc; border-color: #eeedfc; }
            QPushButton:focus { border: 2px solid #726bce; }
            QLabel#modeCaption { color: #302b86; font-size: 14px; }
            QLabel#shortcutHint { color: #5b667e; font-size: 15px; }
            QFrame#settingsPanel { border-top: 1px solid #d9dfea; }
            QPushButton#toggle { border: 0; background: white; }
            QPushButton#toggle:focus { border: 1px solid #726bce; }
            QLabel#toggleText { color: #302b86; font-size: 13px; }
            QLabel#track { background: #838b9c; border-radius: 10px; }
            QLabel#track[on="true"] { background: #4640b8; }
            QPushButton[modeOption="true"] { font-size: 14px; color: #49536b; font-weight: normal; }
            QPushButton[modeOption="true"]:checked { background: #eeedfc; color: #302b86;
                border-color: #aca9e0; font-weight: bold; }
            QPushButton[modeOption="true"]:disabled { color: #687185; background: #f7f8fb; }
            QLabel#policyHint { color: #49536b; font-size: 12px; }
            QLabel#maskingTitle { font-size: 15px; font-weight: bold; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 0)
        layout.setSpacing(0)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.capture_button = QPushButton('영역 캡처')
        self.capture_button.setObjectName('capture')
        self.capture_button.setFixedSize(135, 34)
        self.capture_button.clicked.connect(self.capture_requested)
        self.file_button = QPushButton('파일 업로드')
        self.file_button.setFixedSize(105, 34)
        self.file_button.setToolTip('이 PC의 PNG 파일 열기')
        self.file_button.clicked.connect(self.file_requested)
        row.addWidget(self.capture_button)
        row.addWidget(self.file_button)
        row.addStretch()
        self.settings_button = QPushButton()
        self.settings_button.setObjectName('settings')
        self.settings_button.setFixedSize(140, 34)
        self.settings_button.setCheckable(True)
        settings_row = QHBoxLayout(self.settings_button)
        settings_row.setContentsMargins(10, 0, 12, 0)
        settings_row.setSpacing(8)
        shield = self.icon_label('shield', 17)
        self.mode_caption = QLabel()
        self.mode_caption.setObjectName('modeCaption')
        self.arrow = QLabel()
        for part in (shield, self.mode_caption, self.arrow):
            part.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        settings_row.addWidget(shield)
        settings_row.addWidget(self.mode_caption, 1)
        settings_row.addWidget(self.arrow)
        row.addWidget(self.settings_button)
        layout.addLayout(row)

        self.shortcut_hint = QLabel('앱에서 <b>Alt+C</b>로 영역 캡처')
        self.shortcut_hint.setObjectName('shortcutHint')
        self.shortcut_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.shortcut_hint.setAccessibleName('앱에서 Alt+C로 영역 캡처')
        layout.addWidget(self.shortcut_hint, 1)

        self.settings_panel = QFrame()
        self.settings_panel.setObjectName('settingsPanel')
        self.settings_panel.setFixedHeight(96)
        panel = QVBoxLayout(self.settings_panel)
        panel.setContentsMargins(0, 12, 0, 10)
        panel.setSpacing(5)
        modes = QHBoxLayout()
        modes.setSpacing(6)
        title = QLabel('마스킹')
        title.setObjectName('maskingTitle')
        modes.addWidget(title)
        self.masking_toggle = QPushButton()
        self.masking_toggle.setObjectName('toggle')
        self.masking_toggle.setCheckable(True)
        self.masking_toggle.setFixedSize(81, 29)
        self.toggle_text = QLabel(self.masking_toggle)
        self.toggle_text.setObjectName('toggleText')
        self.toggle_text.setGeometry(2, 0, 36, 29)
        self.track = QLabel(self.masking_toggle)
        self.track.setObjectName('track')
        self.track.setGeometry(39, 4, 38, 21)
        self.knob = self.icon_label('knob', 15, self.track)
        for part in (self.toggle_text, self.track, self.knob):
            part.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.masking_toggle.clicked.connect(self.enabled_changed)
        modes.addWidget(self.masking_toggle)
        modes.addStretch()
        self.group = QButtonGroup(self)
        self.auto_button = self.mode_button('완전 자동', 'auto')
        self.individual_button = self.mode_button('개별 승인', 'individual')
        modes.addWidget(self.auto_button)
        modes.addWidget(self.individual_button)
        panel.addLayout(modes)
        self.policy_hint = QLabel()
        self.policy_hint.setObjectName('policyHint')
        self.policy_hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.policy_hint.setWordWrap(True)
        panel.addWidget(self.policy_hint)
        layout.addWidget(self.settings_panel)
        self.settings_panel.hide()
        self.settings_button.toggled.connect(self.show_settings_panel)
        self.show_settings_panel(False)
        self.set_settings(True, 'auto')

    @staticmethod
    def icon_label(name, size, parent=None):
        label = QLabel(parent)
        label.setPixmap(QIcon(str(ICONS / (name + '.svg'))).pixmap(QSize(size, size)))
        label.setFixedSize(size, size)
        return label

    def mode_button(self, label, mode):
        button = QPushButton(label)
        button.setProperty('modeOption', True)
        button.setFixedSize(120, 34)
        button.setCheckable(True)
        self.group.addButton(button)
        button.clicked.connect(lambda: self.mode_changed.emit(mode))
        return button

    def set_expanded(self, expanded):
        self.settings_button.setChecked(expanded)

    def show_settings_panel(self, expanded):
        self.settings_panel.setVisible(expanded)
        arrow = 'collapse.svg' if expanded else 'expand.svg'
        self.arrow.setPixmap(QIcon(str(ICONS / arrow)).pixmap(QSize(16, 16)))
        self.settings_button.setToolTip('가림 설정 접기' if expanded else '가림 설정 펼치기')
        self.expanded_changed.emit(expanded)

    def set_settings(self, enabled, mode):
        text = '완전 자동' if mode == 'auto' else '개별 승인'
        self.mode_caption.setText(text if enabled else '꺼짐')
        self.settings_button.setAccessibleName('가림 설정 · ' + self.mode_caption.text())
        self.masking_toggle.setChecked(enabled)
        self.toggle_text.setText('켜짐' if enabled else '꺼짐')
        self.masking_toggle.setAccessibleName('마스킹 ' + self.toggle_text.text())
        self.knob.move(19 if enabled else 4, 3)
        if self.track.property('on') != enabled:
            self.track.setProperty('on', enabled)
            self.track.style().unpolish(self.track)
            self.track.style().polish(self.track)
        self.auto_button.setChecked(mode == 'auto')
        self.individual_button.setChecked(mode == 'individual')
        self.auto_button.setEnabled(enabled)
        self.individual_button.setEnabled(enabled)
        if not enabled:
            note = '가림 없이 원본을 확인한 뒤 직접 복사'
        elif mode == 'individual':
            note = '가릴 항목을 선택한 뒤 미리보기에서 직접 복사'
        else:
            note = '놓으면 캡처 · 분석 후 자동 복사 (후보 0개면 원본) · 파일은 직접 복사'
        self.policy_hint.setText(note)
