import os
import json
from typing import Dict, Optional

from .request_handler import RequestHandler

from ..core.model.constants import DirectoryConfig
from ..core.dcc_backend import DCCBackend
from ..logging_config import get_logger
from ..core.model.utilities import check_for_model_on_disk

log = get_logger(__name__)

_prediction_manager = None


def get_prediction_manager():
    global _prediction_manager
    if not _prediction_manager:
        _prediction_manager = PredictionManager()
    return _prediction_manager


class PredictionManager:
    def __init__(self, backend: DCCBackend = None):
        self._backend = backend or self._get_dcc_backend()
        self._request_handler = RequestHandler()
        self._min_node_requirement = 3
        self._model_exists = check_for_model_on_disk()
        self._load_model_vocab()

    def toggle_callback(self, enable_callback: bool):
        if enable_callback:
            self._backend.enable_callback(self.predict_selected)
        else:
            self._backend.disable_callback(self.predict_selected)

    def predict_selected(self):
        if not self._backend.should_perform_callback():
            return

        self.perform_recommendation()

    def perform_recommendation(self, node_name: Optional[str] = None) -> None:
        try:
            target_node = self._backend.get_target_node(node_name=node_name)
            if not target_node:
                return

            if not self._backend.validate_node_for_inference(target_node):
                return

            error_msg = None
            if not self._model_exists:
                error_msg = "Please train a local model first!"
            elif not self._vocabulary:
                error_msg = (
                    "No model vocabulary found! Please train a local model first."
                )

            if error_msg:
                self._get_panel_instance().update_training_page_label(error_msg)
                return

            serialized_graph_data = self._backend.serialize_upstream_nodes(target_node)

            root_nodes = serialized_graph_data["root"]["nodes"]
            if len(root_nodes) <= self._min_node_requirement:
                msg = f"Not enough upstream nodes to predict for {target_node.name()}!"
                self._get_panel_instance().update_training_page_label(msg)
                log.info(msg)
                return

            self._process_prediction(serialized_graph_data)

        except Exception as e:
            log.error(f"Error during recommendation: {str(e)}")
            self._get_panel_instance().update_training_page_label(str(e))

    def refresh_manager(self):
        model_exists = check_for_model_on_disk()
        if model_exists and not self._model_exists:
            self._get_panel_instance().update_training_page_label(
                "Found valid PyTorch model."
            )

        self._model_exists = model_exists
        self._load_model_vocab()

    def _get_dcc_backend(self):
        from ..core.nuke import nuke_backend

        return nuke_backend.NukeBackend()

    def _get_panel_instance(self):
        from ..ui.nuke import prediction_panel

        return prediction_panel.get_panel_instance()

    def _process_prediction(self, graph_data: Dict) -> None:
        prediction = self._request_handler.post("predict", graph_data, custom_timeout=30)
        if not prediction:
            return

        self._get_panel_instance().update_prediction_state(prediction)

    def _load_model_vocab(self):
        self._vocabulary = {}
        if not self._model_exists:
            return

        if not os.path.exists(DirectoryConfig.VOCAB_PATH):
            return

        try:
            with open(DirectoryConfig.VOCAB_PATH, "r") as f:
                self._vocabulary = json.load(f).get("node_type_to_idx", {})
        except (json.JSONDecodeError, IOError) as e:
            log.error(f"Failed to load vocabulary: {str(e)}")
