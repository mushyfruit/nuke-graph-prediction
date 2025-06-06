import os
import json
from typing import Dict, Optional, Any

import nuke

from .request_handler import RequestHandler

from .. import model_cnst
from ..core.dcc_backend import DCCBackend
from ..logging_config import get_logger
from ..core.utilities import detect_current_dcc
from ..core.model.utilities import check_for_model_on_disk

log = get_logger(__name__)

_prediction_manager = None


def get_prediction_manager():
    global _prediction_manager
    if not _prediction_manager:
        _prediction_manager = PredictionManager()
    return _prediction_manager


class PredictionManager:
    """Manages graph traversal and serialization for the recommendation system.

    Serialized graph data is posted via the :class:`RequestHandler` to the
    inference service running in a separate process.
    """

    def __init__(self, backend: DCCBackend = None):
        self._vocabulary: Dict[str, int] = {}

        self._panel = self._get_panel_instance()

        # Handle DCC specific API calls.
        self._backend = backend or self._get_dcc_backend()

        # Handles request to inference server.
        self._request_handler = RequestHandler()

        # Minimum number of upstream nodes required to make valid inference request.
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
        """Retrieves the next-node predictions for the given or selected node.

        :param node_name: Optional node name to perform predictions for.
        """
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
                self._panel.update_training_page_label(error_msg)
                return

            # Convert upstream nodes to dictionary representations.
            serialized_graph_data = self._serialize_upstream(start_node=target_node)
            if not serialized_graph_data:
                return

            root_nodes = serialized_graph_data["root"]["nodes"]
            if len(root_nodes) <= self._min_node_requirement:
                msg = f"Not enough upstream nodes to predict for {target_node.name()}!"
                self._panel.update_training_page_label(msg)
                log.info(msg)
                return

            self._process_prediction(serialized_graph_data)

        except Exception as e:
            log.error(f"Error during recommendation: {str(e)}")
            self._panel.update_training_page_label(str(e))

    def refresh_manager(self):
        # Check if the model now exists on disk.
        model_exists = check_for_model_on_disk()
        if not model_exists:
            return

        if not self._model_exists:
            self._panel.update_training_page_label("Found valid PyTorch model.")

        self._model_exists = True
        self._load_model_vocab()

    def _get_dcc_backend(self):
        if detect_current_dcc() == "nuke":
            from ..core.nuke import nuke_backend

            return nuke_backend.NukeBackend()
        else:
            raise NotImplementedError("TBD Houdini")

    def _get_panel_instance(self):
        """Returns a Python panel instance for the target DCC."""
        if detect_current_dcc() == "nuke":
            from ..ui.nuke import prediction_panel

            return prediction_panel.get_panel_instance()
        else:
            raise NotImplementedError("TBD Houdini")

    def _process_prediction(self, graph_data: Dict) -> None:
        prediction = self._request_handler.post("predict", graph_data)
        if not prediction:
            return

        if not self._panel:
            log.info(f"Prediction: {prediction}")
            return

        self._panel.update_prediction_state(prediction)

    def _serialize_upstream(self, start_node: Any) -> Optional[Dict]:
        serialized_nodes = self._backend.serialize_upstream_nodes(start_node=start_node)
        return {
            "root": {"name": "root", "parent": None, "nodes": serialized_nodes},
            "start_node": self._backend.get_node_name(start_node),
        }

    def _load_model_vocab(self):
        if not self._model_exists:
            return

        if not os.path.exists(model_cnst.DirectoryConfig.VOCAB_PATH):
            return

        try:
            with open(model_cnst.DirectoryConfig.VOCAB_PATH, "r") as f:
                self._vocabulary = json.load(f).get("node_type_to_idx", {})
        except (json.JSONDecodeError, IOError) as e:
            log.error(f"Failed to load vocabulary: {str(e)}")
