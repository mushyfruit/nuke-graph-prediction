import os
import glob
import json
import nuke
from typing import List, Tuple

from ...api import RequestHandler, PredictionManager
from ...core.model.constants import TrainingPhase, DirectoryConfig
from ...core.model.utilities import check_for_model_on_disk
from ...server.launcher import get_inference_launcher
from ...logging_config import get_logger


try:
    from PySide6 import QtWidgets, QtCore, QtGui, QtNetwork
except ImportError:
    from PySide2 import QtWidgets, QtCore, QtGui, QtNetwork

log = get_logger(__name__)


class PredictionWidget(QtWidgets.QWidget):
    def __init__(
        self,
        request_handler: RequestHandler,
        prediction_manager: PredictionManager,
        parent=None,
    ):
        super().__init__(parent=parent)
        self._request_handler = request_handler
        self._prediction_manager = prediction_manager

        self._model_exists = False
        self._stored_nuke_files = []
        self._current_node_name = None
        self._pending_requests = set()
        self.network_manager = QtNetwork.QNetworkAccessManager(self)
        self.network_manager.finished.connect(self._on_request_finished)

        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self._check_training_status)
        self.status_timer.setInterval(1000)

        self.health_timer = QtCore.QTimer(self)
        self.health_timer.timeout.connect(self.update_server_status)

        self.setup_ui()
        self._check_for_model_on_disk()

        self.health_timer.start(2500)

    def _check_for_model_on_disk(self):
        self._model_exists = check_for_model_on_disk()
        self.fine_tune_checkbox.setEnabled(self._model_exists)
        self.fine_tune_label.setEnabled(self._model_exists)
        return self._model_exists

    def setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        self.tab_widget = QtWidgets.QTabWidget(self)

        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.tab_widget)

        self._setup_prediction_page()
        self._setup_training_page()
        self._setup_model_page()

        self.tab_widget.addTab(self.prediction_page, "Inference")
        self.tab_widget.addTab(self.training_page, "Training")
        self.tab_widget.addTab(self.model_page, "Model")

    def _on_folder_selected(self, folder_path):
        self._update_training_status_label("")
        self._update_list_widget([])
        if not os.path.exists(folder_path):
            self._update_training_status_label("Invalid path selected!")
            return

        glob_pattern = os.path.join(folder_path, "*.nk")
        nuke_files = glob.glob(glob_pattern)
        if not nuke_files:
            self._update_training_status_label("No valid .nk files located!")
            return

        self._update_list_widget(nuke_files)

    def _update_list_widget(self, nuke_files):
        self.file_list_widget.clear()
        for nuke_file in nuke_files:
            filename, _ = os.path.splitext(os.path.basename(nuke_file))
            self.file_list_widget.addItem(filename)
        self._stored_nuke_files = nuke_files

    def _update_training_status_label(self, status):
        self.training_status_label.setText(status)
        self.training_status_label.setHidden(not status)

    def _setup_training_page(self):
        self.training_page = QtWidgets.QWidget()
        training_layout = QtWidgets.QVBoxLayout(self.training_page)

        status_layout = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Server Status:")
        self.status_label.setAlignment(QtCore.Qt.AlignVCenter)
        self.status_indicator = QtWidgets.QLabel()
        self.status_indicator.setAlignment(QtCore.Qt.AlignVCenter)
        self.status_indicator.setText("●")
        self.status_indicator.setStyleSheet("color: green;")

        self.host_lbl = QtWidgets.QLabel("Host:")
        self.host = QtWidgets.QLineEdit("127.0.0.1")
        self.host.setMaximumWidth(75)

        self.port_lbl = QtWidgets.QLabel("Port:")
        self.port = QtWidgets.QLineEdit(self._request_handler.port)
        self.port.setMaximumWidth(40)

        ui_directory = os.path.dirname(os.path.dirname(__file__))
        icon_path = os.path.join(ui_directory, "icons", "refresh.png")
        self.restart_btn = QtWidgets.QPushButton()
        self.restart_btn.setIcon(QtGui.QIcon(icon_path))
        self.restart_btn.setFixedSize(24, 24)
        self.restart_btn.clicked.connect(self._restart_inference_server)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.status_indicator)
        status_layout.addStretch()
        status_layout.addWidget(self.host_lbl)
        status_layout.addWidget(self.host)
        status_layout.addSpacing(3)
        status_layout.addWidget(self.port_lbl)
        status_layout.addWidget(self.port)
        status_layout.addSpacing(5)
        status_layout.addWidget(self.restart_btn)

        training_layout.addLayout(status_layout)

        self.file_input = FileInputWidget("Scripts Folder:")
        self.file_input.folder_selected.connect(self._on_folder_selected)
        training_layout.addWidget(self.file_input)

        self.training_status_label = QtWidgets.QLabel("")
        training_layout.addWidget(self.training_status_label)
        self.training_status_label.setHidden(True)

        self.file_list_widget = QtWidgets.QListWidget(self)
        self.file_list_widget.setMinimumHeight(200)
        training_layout.addWidget(self.file_list_widget)

        self.training_progress_bar = QtWidgets.QProgressBar(self)
        training_layout.addWidget(self.training_progress_bar)

        stat_layout = QtWidgets.QHBoxLayout()

        self.epoch_label = QtWidgets.QLabel("Epoch:")
        stat_layout.addWidget(self.epoch_label)

        self.epoch_value = QtWidgets.QLineEdit(self)
        self.epoch_value.setReadOnly(True)
        stat_layout.addWidget(self.epoch_value)

        self.loss_label = QtWidgets.QLabel("Loss:")
        stat_layout.addWidget(self.loss_label)

        self.loss_value = QtWidgets.QLineEdit(self)
        self.loss_value.setReadOnly(True)
        stat_layout.addWidget(self.loss_value)

        self.training_accuracy_label = QtWidgets.QLabel("Training")
        stat_layout.addWidget(self.training_accuracy_label)

        self.training_accuracy_value = QtWidgets.QLineEdit(self)
        self.training_accuracy_value.setReadOnly(True)
        stat_layout.addWidget(self.training_accuracy_value)

        self.validation_accuracy_label = QtWidgets.QLabel("Validation:")
        stat_layout.addWidget(self.validation_accuracy_label)

        self.validation_accuracy_value = QtWidgets.QLineEdit(self)
        self.validation_accuracy_value.setReadOnly(True)
        stat_layout.addWidget(self.validation_accuracy_value)
        training_layout.addLayout(stat_layout)

        bottom_layout = QtWidgets.QHBoxLayout()

        self.memory_allocation_label = QtWidgets.QLabel("Memory Allocation:")
        self.memory_allocation_value = QtWidgets.QDoubleSpinBox(self)
        self.memory_allocation_value.setDecimals(1)
        self.memory_allocation_value.setSingleStep(0.1)
        self.memory_allocation_value.setRange(0.1, 0.9)
        self.memory_allocation_value.setValue(0.5)

        self.fine_tune_label = QtWidgets.QLabel("Enable Fine Tuning:")
        self.fine_tune_checkbox = QtWidgets.QCheckBox(self)

        bottom_layout.addWidget(self.memory_allocation_label)
        bottom_layout.addWidget(self.memory_allocation_value)
        bottom_layout.addWidget(self.fine_tune_label)
        bottom_layout.addWidget(self.fine_tune_checkbox)

        bottom_layout.addStretch(1)
        self.training_btn = QtWidgets.QPushButton("Train")
        bottom_layout.addWidget(self.training_btn)
        self.training_btn.clicked.connect(self._on_training_btn_clicked)
        training_layout.addLayout(bottom_layout)

    def _setup_model_page(self):
        self.model_page = QtWidgets.QWidget()
        model_layout = QtWidgets.QVBoxLayout(self.model_page)

        self.loss_image_widget = QtWidgets.QLabel()
        self.loss_image_widget.setScaledContents(True)
        self.loss_image_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )

        self.topk_image_widget = QtWidgets.QLabel()
        self.topk_image_widget.setScaledContents(True)
        self.topk_image_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )

        self.graph_vis_timer = QtCore.QTimer(self)
        self.graph_vis_timer.setInterval(3000)
        self.graph_vis_timer.timeout.connect(self.update_image)
        self.update_image()

        model_layout.addWidget(QtWidgets.QLabel("Training/Validation Loss:"))
        model_layout.addWidget(self.loss_image_widget)

        model_layout.addWidget(QtWidgets.QLabel("Top-K Accuracy:"))
        model_layout.addWidget(self.topk_image_widget)
        model_layout.addStretch()

    def update_image(self):
        for widget, image_path in (
            (self.loss_image_widget, DirectoryConfig.LOSS_PLOT),
            (self.topk_image_widget, DirectoryConfig.TOPK_PLOT),
        ):
            if os.path.exists(image_path):
                pixmap = QtGui.QPixmap(image_path)
                if not pixmap.isNull():
                    widget.setPixmap(pixmap)
                else:
                    widget.setText("Failed to load image")

    def update_server_status(self):
        self._request("health", timeout=self._request_handler.timeout)

    def _request(self, endpoint, data=None, timeout=None):
        if endpoint in self._pending_requests:
            return
        request = QtNetwork.QNetworkRequest(
            QtCore.QUrl(f"{self._request_handler.base_url}{endpoint}")
        )
        if timeout:
            request.setTransferTimeout(int(1000 * timeout))
        if data is None:
            reply = self.network_manager.get(request)
        else:
            request.setHeader(
                QtNetwork.QNetworkRequest.ContentTypeHeader, "application/json"
            )
            reply = self.network_manager.post(request, json.dumps(data).encode("utf-8"))
        reply.setProperty("endpoint", endpoint)
        self._pending_requests.add(endpoint)

    def _on_request_finished(self, reply):
        endpoint = reply.property("endpoint")
        self._pending_requests.remove(endpoint)
        response = {}
        try:
            if reply.error() != QtNetwork.QNetworkReply.NoError:
                raise RuntimeError(reply.errorString())
            response = json.loads(bytes(reply.readAll()))
            if endpoint == "train":
                if response["status"] != TrainingPhase.SERIALIZING.value:
                    raise RuntimeError(
                        response.get("error") or response.get("label") or str(response)
                    )
                self._update_training_status_label(response.get("label", ""))
                self.status_timer.start()
                self.graph_vis_timer.start()
            elif endpoint == "training_status":
                self._update_training_status(response)
        except Exception as e:
            if endpoint != "health":
                log.error(f"Training request failed: {e}")
                self._update_training_status_label(str(e))
                self._on_model_finish(successful=False)
        if endpoint == "health":
            healthy = response.get("status") == "ok"
            self.status_indicator.setText("●" if healthy else "○")
            self.status_indicator.setStyleSheet(
                "color: green;" if healthy else "color: red;"
            )
        reply.deleteLater()

    def _restart_inference_server(self):
        host = self.host.text()
        port = self.port.text()

        inference_launcher = get_inference_launcher()
        inference_launcher.restart_service(host=host, port=port)

        self._request_handler.set_host(host)
        self._request_handler.set_port(port)

    def _on_training_btn_clicked(self):
        if not self._stored_nuke_files:
            self._update_training_status_label(
                "Unable to start training. Please select a valid directory."
            )
            return

        self.training_btn.setEnabled(False)
        self.file_input.setEnabled(False)
        self._request(
            "train",
            {
                "file_paths": self._stored_nuke_files,
                "memory_allocation": self.memory_allocation_value.value(),
                "enable_fine_tuning": self.fine_tune_checkbox.isChecked(),
            },
        )

    def _on_model_finish(self, successful=True):
        self.status_timer.stop()
        self.graph_vis_timer.stop()
        self.training_btn.setEnabled(True)
        self.file_input.setEnabled(True)

        if successful and self._check_for_model_on_disk():
            self.fine_tune_checkbox.setChecked(True)
            self._prediction_manager.refresh_manager()
            self.update_prediction_page_status("")

    def _check_training_status(self):
        if self.status_timer.isActive():
            self._request("training_status")

    def _update_training_status(self, response):
        status = response["status"]
        if status in (TrainingPhase.COMPLETE.value, TrainingPhase.ERROR.value):
            self._on_model_finish(successful=status == TrainingPhase.COMPLETE.value)

        label = response.get("error") or response.get("label")
        if label:
            self._update_training_status_label(label)
        if response.get("progress") is not None:
            self.training_progress_bar.setValue(int(response["progress"] * 100))

        for key, widget, format_string in (
            ("current_epoch", self.epoch_value, "{}"),
            ("training_loss", self.loss_value, "{:.2f}"),
            ("training_accuracy", self.training_accuracy_value, "{:.2f}%"),
            ("validation_accuracy", self.validation_accuracy_value, "{:.2f}%"),
        ):
            if response.get(key) is not None:
                widget.setText(format_string.format(response[key]))

    def _setup_prediction_page(self):
        self.prediction_page = QtWidgets.QWidget()
        prediction_layout = QtWidgets.QVBoxLayout(self.prediction_page)

        header_layout = QtWidgets.QHBoxLayout()
        self.selected_node_label = QtWidgets.QLabel("Selected Node: None")
        header_layout.addWidget(self.selected_node_label)
        header_layout.addStretch(1)

        self.callback_label = QtWidgets.QLabel("Predict on Selection:")
        self.callback_checkbox = QtWidgets.QCheckBox()
        self.callback_checkbox.toggled.connect(self._on_callback_toggle)

        header_layout.addWidget(self.callback_label)
        header_layout.addWidget(self.callback_checkbox)

        self.prediction_tree = QtWidgets.QTreeWidget()
        self.prediction_tree.itemDoubleClicked.connect(
            self._on_prediction_double_clicked
        )
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

        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._on_refresh_button_clicked)
        button_layout.addWidget(self.refresh_button)

        bottom_status_layout = QtWidgets.QHBoxLayout()
        bottom_status_layout.addStretch(1)
        self.bottom_status_label = QtWidgets.QLabel("")
        bottom_status_layout.addWidget(self.bottom_status_label)
        bottom_status_layout.addStretch(1)

        prediction_layout.addLayout(header_layout)
        prediction_layout.addWidget(self.prediction_tree)
        prediction_layout.addLayout(button_layout)
        prediction_layout.addLayout(bottom_status_layout)

    def _on_callback_toggle(self, enabled):
        self._prediction_manager.toggle_callback(enabled)

    def _on_prediction_double_clicked(self, item, column):
        node_type = item.data(0, QtCore.Qt.UserRole)
        if not node_type:
            log.error("Unable to create prediction node!")
            return

        try:
            nuke.createNode(node_type, inpanel=False)
        except Exception as e:
            log.error(f"Failed to create prediction node: {e}")

    def _on_refresh_button_clicked(self):
        if not self._model_exists:
            self._check_for_model_on_disk()
            self._prediction_manager.refresh_manager()

        self._prediction_manager.perform_recommendation(self._current_node_name)

    def update_prediction_page_status(self, label):
        self.bottom_status_label.setText(label)

    def update_selected_node(self, selected_node):
        self._current_node_name = selected_node
        self.selected_node_label.setText("Selected Node: {}".format(selected_node))

    def update_prediction(self, node_predictions: List[Tuple[str, float]]):
        self.prediction_tree.clear()
        for node_type, confidence_score in node_predictions:
            item = QtWidgets.QTreeWidgetItem([node_type, f"{confidence_score:.2f}"])
            item.setData(0, QtCore.Qt.UserRole, node_type)
            item.setTextAlignment(1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            self.prediction_tree.addTopLevelItem(item)

    def makeUI(self):
        return self

    def updateValue(self):
        pass


class FileInputWidget(QtWidgets.QWidget):
    folder_selected = QtCore.Signal(str)

    def __init__(self, label=None, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = None
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.editingFinished.connect(
            lambda: self.folder_selected.emit(self.line_edit.text())
        )
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_folders)

        if label:
            self.label = QtWidgets.QLabel(label)
            layout.addWidget(self.label)

        layout.addWidget(self.line_edit)
        layout.addWidget(self.browse_button)

    def browse_folders(self):
        file_dialog = QtWidgets.QFileDialog(self)
        file_dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        file_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        file_dialog.setWindowTitle("Select Folder")

        file_dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, False)
        file_dialog.setViewMode(QtWidgets.QFileDialog.Detail)

        if file_dialog.exec_() == QtWidgets.QFileDialog.Accepted:
            selected_dirs = file_dialog.selectedFiles()
            if selected_dirs:
                folder_path = selected_dirs[0]
                self.line_edit.setText(folder_path)
                self.folder_selected.emit(folder_path)
