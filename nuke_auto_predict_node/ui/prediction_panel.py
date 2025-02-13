import os
import glob
import nuke
import nukescripts

from PySide2 import QtWidgets, QtCore

from ..request_handler import get_request_handler

import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BUILT_IN_TABS = [
    "Properties.1",
    "DAG.1",
    "DopeSheet.1",
    "Viewer.1",
    "Toolbar.1",
]

_g_prediction_panel = None


def get_panel_instance():
    global _g_prediction_panel
    if not _g_prediction_panel:
        _g_prediction_panel = PredictionPanel()
    return _g_prediction_panel


def register_prediction_panel():
    _register_recommendation_panel()


def _register_recommendation_panel():
    pane_menu = nuke.menu("Pane")
    pane_menu.addCommand("Node Recommendation✨", show_prediction_panel)
    nukescripts.registerPanel("mushyfruit.node_recommendation", show_prediction_panel)


def show_prediction_panel():
    global _g_prediction_panel

    if not _g_prediction_panel:
        _g_prediction_panel = PredictionPanel()

    existing_pane = None
    for tab_name in BUILT_IN_TABS:
        existing_pane = nuke.getPaneFor(tab_name)
        if existing_pane:
            break

    return _g_prediction_panel.addToPane(pane=existing_pane)


class PredictionPanel(nukescripts.panels.PythonPanel):
    def __init__(self):
        nukescripts.panels.PythonPanel.__init__(
            self, "NodePrediction", "mushyfruit.node_recommendation"
        )

        self._init_layout()

    def _init_layout(self):
        self.prediction_widget = nuke.PyCustom_Knob(
            "prediction_widget",
            "",
            ("nuke_auto_predict_node.ui.prediction_panel.PredictionWidget()"),
        )
        self.prediction_widget.setFlag(nuke.STARTLINE)

        knob_list = [self.prediction_widget]

        for knob in knob_list:
            self.addKnob(knob)

    def show_prediction(self, predictions):
        prediction_widget = self.prediction_widget.getObject()
        if prediction_widget:
            prediction_widget.update_predictions(predictions)


class PredictionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setup_ui()
        self._stored_nuke_files = []

    def setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        self.tab_widget = QtWidgets.QTabWidget(self)

        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.tab_widget)

        self._setup_prediction_page()
        self._setup_training_page()

        self.tab_widget.addTab(self.prediction_page, "Prediction")
        self.tab_widget.addTab(self.training_page, "Training")

    def _on_folder_selected(self, folder_path):
        self._clear_status_label()
        if not os.path.exists(folder_path):
            self._update_status_label("Invalid path selected!")
            return

        glob_pattern = os.path.join(folder_path, "*.nk")
        nuke_files = glob.glob(glob_pattern)
        if not nuke_files:
            self._update_status_label("No valid .nk files located!")
            return

        self._update_list_widget(nuke_files)

    def _update_list_widget(self, nuke_files):
        self.file_list_widget.clear()
        for nuke_file in nuke_files:
            filename, _ = os.path.splitext(os.path.basename(nuke_file))
            self.file_list_widget.addItem(filename)
        self._stored_nuke_files = nuke_files

    def _update_status_label(self, status):
        self.status_label.setText(status)
        self.status_label.setHidden(False)

    def _clear_status_label(self):
        self.status_label.setText("")
        self.status_label.setHidden(True)

    def _setup_training_page(self):
        self.training_page = QtWidgets.QWidget()
        training_layout = QtWidgets.QVBoxLayout(self.training_page)

        self.file_input = FileInputWidget()
        self.file_input.folder_selected.connect(self._on_folder_selected)
        training_layout.addWidget(self.file_input)

        self.status_label = QtWidgets.QLabel("")
        training_layout.addWidget(self.status_label)
        self.status_label.setHidden(True)

        self.file_list_widget = QtWidgets.QListWidget(self)
        self.file_list_widget.setMinimumHeight(200)
        self.file_list_widget.setMaximumHeight(450)
        training_layout.addWidget(self.file_list_widget)

        self.training_progress_bar = QtWidgets.QProgressBar(self)
        training_layout.addWidget(self.training_progress_bar)

        stat_layout = QtWidgets.QHBoxLayout()

        self.epoch_label = QtWidgets.QLabel("Epoch:")
        stat_layout.addWidget(self.epoch_label)

        self.epoch_value = QtWidgets.QLineEdit(self)
        stat_layout.addWidget(self.epoch_value)

        self.loss_label = QtWidgets.QLabel("Loss:")
        stat_layout.addWidget(self.loss_label)

        self.loss_value = QtWidgets.QLineEdit(self)
        stat_layout.addWidget(self.loss_value)

        self.training_accuracy_label = QtWidgets.QLabel("Training")
        stat_layout.addWidget(self.training_accuracy_label)

        self.training_accuracy_value = QtWidgets.QLineEdit(self)
        stat_layout.addWidget(self.training_accuracy_value)

        self.validation_accuracy_label = QtWidgets.QLabel("Validation:")
        stat_layout.addWidget(self.validation_accuracy_label)

        self.validation_accuracy_value = QtWidgets.QLineEdit(self)
        stat_layout.addWidget(self.validation_accuracy_value)

        training_layout.addLayout(stat_layout)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch(1)

        self.training_btn = QtWidgets.QPushButton("Train")
        button_layout.addWidget(self.training_btn)
        self.training_btn.clicked.connect(self._on_training_btn_clicked)
        training_layout.addLayout(button_layout)

    def _on_training_btn_clicked(self):
        handler = get_request_handler()
        if not self._stored_nuke_files:
            self._update_status_label("Unable to start training. Please select a valid directory.")
            return

        handler.kickoff_training(self._stored_nuke_files)


    def _setup_prediction_page(self):
        self.prediction_page = QtWidgets.QWidget()
        prediction_layout = QtWidgets.QVBoxLayout(self.prediction_page)

        header_layout = QtWidgets.QHBoxLayout()
        self.selected_node_label = QtWidgets.QLabel("Selected Node: None")
        header_layout.addWidget(self.selected_node_label)

        self.prediction_tree = QtWidgets.QTreeWidget()
        self.prediction_tree.setHeaderLabels(["Node Type", "Confidence"])

        self.prediction_tree.setRootIsDecorated(False)
        self.prediction_tree.setUniformRowHeights(True)
        self.prediction_tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        self.prediction_tree.setMinimumHeight(200)

        self.prediction_tree.header().setStretchLastSection(True)
        self.prediction_tree.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )

        button_layout = QtWidgets.QHBoxLayout()

        self.create_button = QtWidgets.QPushButton("Create Node")
        self.create_button.setEnabled(False)

        self.refresh_button = QtWidgets.QPushButton("Refresh")
        button_layout.addWidget(self.refresh_button)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)

        prediction_layout.addLayout(header_layout)
        prediction_layout.addWidget(self.prediction_tree)
        prediction_layout.addLayout(button_layout)
        prediction_layout.addWidget(self.status_label)

    def _update_prediction_tree(self):
        """Update the prediction list widget with current predictions."""
        self.prediction_tree.clear()
        for node_type, confidence_score in self.predictions:
            item = QtWidgets.QTreeWidgetItem([node_type, f"{confidence_score:.2f}"])
            item.setTextAlignment(1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            self.prediction_tree.addTopLevelItem(item)

    def update_predictions(self, node_predictions):
        """Get predictions from the GAT model for the selected node."""
        if not node_predictions:
            return

        self.predictions = node_predictions["prediction"][0]
        self._update_prediction_tree()
        self.status_label.setText("Predictions updated successfully")

    def makeUI(self):
        return self

    def updateValue(self):
        pass


class FileInputWidget(QtWidgets.QWidget):
    folder_selected = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create line edit
        self.line_edit = QtWidgets.QLineEdit()

        # Create browse button
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_folders)

        # Add widgets to layout
        layout.addWidget(self.line_edit)
        layout.addWidget(self.browse_button)

    def browse_folders(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder", "", QtWidgets.QFileDialog.ShowDirsOnly
        )
        if folder_path:
            self.line_edit.setText(folder_path)
            self.folder_selected.emit(folder_path)

    def get_file_path(self):
        return self.line_edit.text()