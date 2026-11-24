"""Presentation of the current capture session. This view never publishes images."""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPen, QShortcut, QTextOption
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame, QHeaderView, QHBoxLayout, QLabel,
    QMenu, QProgressBar, QPushButton, QSizePolicy, QSplitter, QStackedWidget,
    QStyle, QStyledItemDelegate, QTextEdit, QToolButton, QVBoxLayout, QWidget, QWidgetAction,
)


ICONS = Path(__file__).with_name('flow_icons')


@dataclass(frozen=True)
class FlowState:
    page: str
    title: str
    metadata: str = ''
    heading: str = ''
    description: str = ''
    detail_title: str = ''
    detail_text: str = ''
    footer_title: str = ''
    footer_text: str = ''
    tone: str = 'neutral'
    individual: bool = False
    source: str = ''
    mode: str = ''
    next_step: str = ''


class WindowPages(QStackedWidget):
    """Only the visible page contributes minimum and preferred dimensions."""

    def sizeHint(self):
        return self.currentWidget().sizeHint() if self.currentWidget() else QSize(448, 182)

    def minimumSizeHint(self):
        return self.currentWidget().minimumSizeHint() if self.currentWidget() else QSize()


def label(role, text=''):
    item = QLabel(text)
    item.setObjectName(role)
    item.setTextFormat(Qt.TextFormat.PlainText)
    item.setWordWrap(True)
    item.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    return item


class InputLabel(QLabel):
    """Keep a long filename inside the viewport; retain the full input identity."""

    def __init__(self, role):
        super().__init__()
        self.setObjectName(role)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.full_text = ''

    def setText(self, text):
        self.full_text = text
        self.setToolTip(text)
        self.setAccessibleName(text)
        self.fit_text()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_text()

    def fit_text(self):
        value = self.fontMetrics().elidedText(
            self.full_text, Qt.TextElideMode.ElideMiddle, self.contentsRect().width())
        if value != self.text():
            super().setText(value)


def line():
    rule = QFrame()
    rule.setObjectName('rule')
    rule.setFixedHeight(1)
    return rule


class SummaryText(QTextEdit):
    """A short caption that can also hold a scrollable list of unresolved spans."""

    def __init__(self, role):
        super().__init__()
        self.setObjectName(role)
        self.setReadOnly(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse |
                                     Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.document().setDocumentMargin(0)
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._value = None

    def setText(self, value):
        if value != self._value:
            self._value = value
            self.setPlainText(value)
            self.update_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self.update_height()

    def update_height(self):
        height = int(self.document().size().height()) + 3
        self.setFixedHeight(min(108, max(self.fontMetrics().height()+2, height)))


class CandidateDelegate(QStyledItemDelegate):
    """A checkable row with a localized type and a separately elided value."""

    types = {'NAME': '이름', 'PHONE': '전화번호', 'EMAIL': '이메일', 'ACCOUNT_ID': '계정'}

    def __init__(self, tree):
        super().__init__(tree)
        self.tick = QIcon(str(ICONS / 'tick.svg'))

    def sizeHint(self, option, index):
        return QSize(120, 65)

    def paint(self, painter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outer = option.rect.adjusted(2, 2, -3, -4)
        checked = index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked.value
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        focused = selected and bool(option.state & QStyle.StateFlag.State_HasFocus)
        painter.setPen(QPen(QColor('#4640b8' if focused else '#aca9e0' if checked else '#d9dfea'), 2 if focused else 1))
        painter.setBrush(QColor('#eeedfc' if checked else '#ffffff'))
        painter.drawRoundedRect(outer, 6, 6)
        check = QRect(outer.x() + 12, outer.center().y() - 10, 20, 20)
        painter.setPen(QColor('#4640b8' if checked else '#8793a9'))
        painter.setBrush(QColor('#4640b8' if checked else '#ffffff'))
        painter.drawRoundedRect(check, 4, 4)
        if checked:
            self.tick.paint(painter, check.adjusted(3, 3, -3, -3))
        text_x = check.right() + 13
        width = max(0, outer.right() - text_x - 8)
        font = QFont(option.font)
        font.setPixelSize(15)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor('#171c2c'))
        kind = index.data() or ''
        painter.drawText(QRect(text_x, outer.y()+4, width, 24), Qt.AlignmentFlag.AlignVCenter, self.types.get(kind, kind))
        font.setBold(False)
        font.setPixelSize(13)
        painter.setFont(font)
        painter.setPen(QColor('#49536b'))
        raw = str(index.siblingAtColumn(1).data() or '').replace('\n', ' ')
        raw = painter.fontMetrics().elidedText(raw, Qt.TextElideMode.ElideRight, width)
        painter.drawText(QRect(text_x, outer.y()+27, width, 24), Qt.AlignmentFlag.AlignVCenter, raw)
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if (event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton
                and option.rect.contains(event.position().toPoint()) and index.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            was_checked = index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked.value
            value = Qt.CheckState.Unchecked if was_checked else Qt.CheckState.Checked
            # Match the integer role written by QTreeWidgetItem.setCheckState and Qt's Space key handler.
            return model.setData(index, value.value, Qt.ItemDataRole.CheckStateRole)
        return super().editorEvent(event, model, option, index)


class FlowView:
    def __init__(self, window):
        self.window = w = window
        self.state = None
        self.icon_cache = {}
        self.root = w.root
        for button in (w.prepare_button, w.copy_button, w.cancel_button, w.retry_button):
            button.setProperty('primary', True)
        self.root.setObjectName('flowRoot')
        self.root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.root.setStyleSheet(Path(__file__).with_name('flow_view.qss').read_text(encoding='utf8'))
        w.statusBar().setSizeGripEnabled(False)
        w.statusBar().setStyleSheet('QStatusBar { background: white; color: #49536b; border: none; }')
        outer = QVBoxLayout(self.root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.content = QWidget()
        body = QVBoxLayout(self.content)
        body.setContentsMargins(36, 24, 36, 22)
        body.setSpacing(14)
        header = QHBoxLayout()
        self.title = label('pageTitle')
        header.addWidget(self.title, 1)
        self.tools = QHBoxLayout()
        self.tools.setSpacing(6)
        header.addLayout(self.tools)
        self.settings_button = self.make_settings()
        # The two original controls now live in a popup and a separate dialog.
        # Keep their mnemonics active while focus is inside the main review page.
        for key, control in (('Alt+S', w.masking_on), ('Alt+F', w.fallback)):
            shortcut = QShortcut(QKeySequence(key), self.root)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(control.click)
        header.addWidget(self.settings_button)
        body.addLayout(header)
        self.metadata = InputLabel('muted')
        body.addWidget(self.metadata)
        self.stack = WindowPages()
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body.addWidget(self.stack, 1)
        outer.addWidget(self.content, 1)
        self.activity = self.make_activity()
        self.review = self.make_review()
        self.stack.addWidget(self.activity)
        self.stack.addWidget(self.review)
        self.footer = QFrame()
        self.footer.setObjectName('footer')
        self.footer_layout = foot = QHBoxLayout(self.footer)
        foot.setContentsMargins(36, 14, 36, 14)
        self.footer_icon = QLabel()
        foot.addWidget(self.footer_icon)
        words = QVBoxLayout()
        words.setSpacing(2)
        self.footer_title = label('footTitle')
        self.footer_text = label('small')
        words.addWidget(self.footer_title)
        words.addWidget(self.footer_text)
        foot.addLayout(words, 1)
        foot.addWidget(w.input_button)
        foot.addWidget(w.prepare_button)
        foot.addWidget(w.copy_button)
        outer.addWidget(self.footer)

    def make_settings(self):
        w = self.window
        button = QToolButton()
        button.setText('가림 설정')
        button.setAccessibleName('가림 설정 및 처리 상세')
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(button)
        container = QWidget()
        controls = QVBoxLayout(container)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.addWidget(w.masking_on)
        controls.addWidget(w.mode_combo)
        action = QWidgetAction(menu)
        action.setDefaultWidget(container)
        menu.addAction(action)
        menu.addSeparator()
        menu.addAction('처리 상세 및 위치 설정', self.show_diagnostics)
        button.setMenu(menu)
        self.diagnostics = QDialog(w)
        self.diagnostics.setWindowTitle('처리 상세 및 위치 설정')
        self.diagnostics.resize(640, 470)
        detail = QVBoxLayout(self.diagnostics)
        for widget in (w.input_label, w.status, w.details, w.fallback, w.open_button):
            detail.addWidget(widget)
        if hasattr(w, 'legend'):
            detail.addWidget(w.legend)
        w.details.setMaximumHeight(16777215)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self.diagnostics)
        close.rejected.connect(self.diagnostics.close)
        detail.addWidget(close)
        return button

    def show_diagnostics(self):
        self.diagnostics.show()
        self.diagnostics.raise_()
        self.diagnostics.activateWindow()

    def make_activity(self):
        w = self.window
        page = QWidget()
        arrangement = QVBoxLayout(page)
        arrangement.setContentsMargins(0, 8, 0, 0)
        arrangement.setSpacing(24)
        card = QFrame()
        card.setObjectName('card')
        area = QVBoxLayout(card)
        area.setContentsMargins(30, 28, 30, 28)
        area.setSpacing(20)
        row = QHBoxLayout()
        row.setSpacing(18)
        self.activity_icon = label('iconTile')
        self.activity_icon.setFixedSize(54, 54)
        self.activity_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.activity_icon)
        texts = QVBoxLayout()
        self.activity_heading = label('cardTitle')
        self.activity_description = label('muted')
        texts.addWidget(self.activity_heading)
        texts.addWidget(self.activity_description)
        row.addLayout(texts, 1)
        area.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        area.addWidget(self.progress)
        self.info = QWidget()
        facts = QVBoxLayout(self.info)
        facts.setContentsMargins(0, 6, 0, 0)
        facts.setSpacing(10)
        self.facts = []
        for title in ('입력', '승인 방식', '다음 단계'):
            fact = QHBoxLayout()
            key = label('small', title)
            key.setFixedWidth(88)
            key.setMinimumHeight(34)
            value = InputLabel('fact') if title == '입력' else label('fact')
            fact.addWidget(key)
            fact.addWidget(value, 1)
            facts.addLayout(fact)
            self.facts.append(value)
        area.addWidget(self.info)
        self.recovery_rule = line()
        area.addWidget(self.recovery_rule)
        self.recovery_title = label('footTitle')
        self.recovery_text = label('muted')
        area.addWidget(self.recovery_title)
        area.addWidget(self.recovery_text)
        arrangement.addWidget(card)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(10)
        self.actions.addWidget(w.cancel_button)
        self.actions.addWidget(w.retry_button)
        self.actions.addStretch()
        arrangement.addLayout(self.actions)
        arrangement.addStretch()
        return page

    def make_review(self):
        w = self.window
        frame = QSplitter(Qt.Orientation.Horizontal)
        frame.setChildrenCollapsible(False)
        frame.setHandleWidth(14)
        canvas = QWidget()
        preview = QVBoxLayout(canvas)
        preview.setSpacing(6)
        preview.setContentsMargins(0, 0, 0, 0)
        zoom = QHBoxLayout()
        self.view_toggle = QToolButton()
        self.view_toggle.setText('원본 위치')
        self.view_toggle.setAccessibleName('원본의 후보 위치와 결과 미리보기 전환')
        self.view_toggle.setCheckable(True)
        self.view_toggle.clicked.connect(lambda checked: w.tabs.setCurrentIndex(0 if checked else 1))
        zoom.addWidget(self.view_toggle)
        zoom.addStretch()
        self.percentage = label('scale')
        self.percentage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.percentage.setMinimumWidth(50)
        zoom.addWidget(self.percentage)
        for button in w.zoom_buttons:
            button.setProperty('small', True)
            zoom.addWidget(button)
        preview.addLayout(zoom)
        w.tabs.tabBar().hide()
        w.tabs.currentChanged.connect(self.sync_view_tools)
        preview.addWidget(w.tabs, 1)
        for view in (w.source_view, w.result_view):
            view.setBackgroundBrush(QColor('#e9ecf4'))
            view.setFrameShape(QFrame.Shape.NoFrame)
            view.scale_changed.connect(self.sync_view_tools)
        self.sync_view_tools()
        frame.addWidget(canvas)
        self.sidebar = QFrame()
        self.sidebar.setObjectName('card')
        self.sidebar.setMinimumWidth(205)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(16, 16, 16, 16)
        side.setSpacing(8)
        self.summary_icon = label('iconTile')
        self.summary_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.summary_icon.setFixedSize(44, 44)
        side.addWidget(self.summary_icon)
        self.side_heading = label('cardTitle')
        self.side_text = SummaryText('sideDescription')
        self.side_text.setAccessibleName('후보 및 가림 상태')
        side.addWidget(self.side_heading)
        side.addWidget(self.side_text)
        w.tree.setRootIsDecorated(False)
        w.tree.setHeaderHidden(True)
        w.tree.setColumnHidden(1, True)
        w.tree.setColumnHidden(2, True)
        w.tree.header().setStretchLastSection(False)
        w.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        w.tree.setFrameShape(QFrame.Shape.NoFrame)
        w.tree.setIndentation(0)
        w.tree.setUniformRowHeights(True)
        w.tree.setItemDelegateForColumn(0, CandidateDelegate(w.tree))
        side.addWidget(w.tree, 1)
        self.selection_actions = QWidget()
        choices = QHBoxLayout(self.selection_actions)
        choices.setContentsMargins(0, 2, 0, 2)
        for button in (w.all_button, w.none_button):
            button.setProperty('linkButton', True)
            choices.addWidget(button)
        choices.addStretch()
        side.addWidget(self.selection_actions)
        self.summary_space = QWidget()
        side.addWidget(self.summary_space, 1)
        side.addWidget(line())
        self.side_detail_title = label('footTitle')
        self.side_detail_text = SummaryText('sideDetail')
        self.side_detail_text.setAccessibleName('결과 및 위치 검토 안내')
        side.addWidget(self.side_detail_title)
        side.addWidget(self.side_detail_text)
        frame.addWidget(self.sidebar)
        frame.setStretchFactor(0, 2)
        frame.setStretchFactor(1, 1)
        frame.setSizes([620, 318])
        return frame

    def sync_view_tools(self, *_):
        w = self.window
        self.view_toggle.setChecked(w.tabs.currentIndex() == 0)
        self.percentage.setText(f'{round(w.active_view().transform().m11()*100)}%')

    def icon(self, widget, name, size):
        key = name, size
        if key not in self.icon_cache:
            self.icon_cache[key] = QIcon(str(ICONS / (name + '.svg'))).pixmap(size, size)
        widget.setPixmap(self.icon_cache[key])

    def update_state(self, state):
        old = self.state
        self.state = state
        w = self.window
        result = state.page == 'result'
        working = state.page == 'progress'
        if old is None or (old.page == 'result') != result:
            self.stack.setCurrentWidget(self.review if result else self.activity)
            self.stack.updateGeometry()
            for button in (w.capture_button, w.file_button):
                if result:
                    self.tools.addWidget(button)
                else:
                    self.actions.insertWidget(self.actions.count()-1, button)
                button.show()
            if result:
                self.footer_layout.insertWidget(self.footer_layout.indexOf(w.prepare_button), w.input_button)
            else:
                self.tools.addWidget(w.input_button)
            w.input_button.show()
            self.footer.setVisible(result)
        w.cancel_button.setVisible(working and w.cancel_button.isEnabled())
        w.retry_button.setVisible(state.page == 'recovery' and w.retry_button.isEnabled())
        self.title.setText(state.title)
        self.metadata.setText('아직 복사 전이에요. 기존 클립보드를 유지해요.'
                              if working and self.window.session.result is None and not self.window.closing
                              else state.metadata)
        def set_text(widget, value):
            widget.setText(value)
            widget.setVisible(bool(value))
        if result:
            self.summary_icon.setVisible(not state.individual)
            w.tree.setVisible(state.individual)
            self.selection_actions.setVisible(state.individual)
            self.summary_space.setVisible(not state.individual)
            self.side_heading.setText(state.heading)
            set_text(self.side_text, state.description)
            set_text(self.side_detail_title, state.detail_title)
            set_text(self.side_detail_text, state.detail_text)
            self.footer_title.setText(state.footer_title)
            set_text(self.footer_text, state.footer_text)
            # A render can change the action caption before Qt lays out the
            # previous (shorter) "preparing" button. Reserve its full label
            # and the primary button's padding immediately.
            for button in (w.prepare_button, w.copy_button):
                if not button.isHidden():
                    button.ensurePolished()
                    width = button.fontMetrics().horizontalAdvance(button.text().replace('&', ''))
                    button.setMinimumWidth(max(button.sizeHint().width(), width + 40))
            self.footer_layout.activate()
        else:
            self.activity_heading.setText(state.heading)
            self.activity_description.setText(state.description)
            self.progress.setVisible(working)
            self.info.setVisible(working)
            for widget, value in zip(self.facts, (state.source, state.mode, state.next_step)):
                widget.setText(value)
            self.recovery_rule.setVisible(not working)
            set_text(self.recovery_title, '' if working else state.detail_title)
            set_text(self.recovery_text, '' if working else state.detail_text)
        tone = state.tone
        if old is None or (old.tone,old.page) != (tone,state.page):
            icon, small, bg, ink = {
                'success': ('success','success-small','#e5f3ed','#09634a'),
                'warning': ('warning','warning-small','#fff0d7','#914600'),
                'error': ('error','warning-small','#fae9ec','#ad243e'),
                'working': ('scan','clock-small','#eeedfc','#4640b8'),
                'neutral': ('stopped' if state.page == 'recovery' else 'layers','layers-small','#eeedfc','#302b86'),
            }.get(tone, ('layers','layers-small','#eeedfc','#49536b'))
            for widget, size in ((self.activity_icon,27),(self.summary_icon,22)):
                self.icon(widget, icon, size)
                widget.setStyleSheet('background: '+bg+'; border-radius: 10px;')
            self.icon(self.footer_icon, small, 19)
            self.footer_title.setStyleSheet('color: '+ink+';')
            self.side_detail_title.setStyleSheet('color: '+ink+';')
