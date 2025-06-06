import os
import sys
import json
from typing import Dict, Optional

import nuke

from .request_handler import RequestHandler
from .utilities import get_all_parameters
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
    """Manages graph traversal and serialization for the recommendation system.

    Serialized graph data is posted via the :class:`RequestHandler` to the
    inference service running in a separate process.
    """

    def __init__(self):
        self._vocabulary: Dict[str, int] = {}
        self._request_handler = RequestHandler()
        self._min_node_requirement = 3
        self._panel = self.get_panel_instance()

        self._callback_installed = False

        self._model_exists = check_for_model_on_disk()
        self.load_model_vocab()

    def get_panel_instance(self):
        """Returns a Python panel instance for the target DCC."""
        exe = os.path.basename(sys.executable).lower()
        if "nuke" in exe:
            from ..ui.nuke import prediction_panel

            return prediction_panel.get_panel_instance()
        else:
            raise NotImplementedError("TBD Houdini")

    def enable_callback(self):
        if not self._callback_installed:
            nuke.addKnobChanged(self.predict_selected)
            self._callback_installed = True

    def disable_callback(self):
        if self._callback_installed:
            nuke.removeKnobChanged(self.predict_selected)
            self._callback_installed = False

    def predict_selected(self):
        knob = nuke.thisKnob()
        if knob is None or knob.name() != "selected":
            return

        # Return early when deselecting nodes.
        if not knob.value():
            return

        self.perform_recommendation()

    def load_model_vocab(self):
        from .. import model_cnst

        if not self._model_exists:
            return

        if not os.path.exists(model_cnst.DirectoryConfig.VOCAB_PATH):
            return

        try:
            with open(model_cnst.DirectoryConfig.VOCAB_PATH, "r") as f:
                self._vocabulary = json.load(f).get("node_type_to_idx", {})
        except (json.JSONDecodeError, IOError) as e:
            log.error(f"Failed to load vocabulary: {str(e)}")

    def perform_recommendation(self, node_name: Optional[str] = None):
        """Retrieves the next-node predictions for the given or selected node.

        :param node_name: Optional node name to perform predictions for.
        """
        try:
            target_node = self._get_target_node(node_name)
            if not target_node:
                return

            # Ensure the node has inputs/outputs.
            if not self._validate_node_for_prediction(target_node):
                return

            error_msg = None
            if not self._model_exists:
                error_msg = "Please train a local model first!"

            if not error_msg and not self._vocabulary:
                error_msg = (
                    "No model vocabulary found! Please train a local model first."
                )

            if error_msg:
                self._panel.update_training_page_label(error_msg)
                return None

            serialized_graph_data = self._serialize_upstream(node_name=node_name)
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
        self.load_model_vocab()

    def _process_prediction(self, graph_data: Dict) -> None:
        prediction = self._request_handler.post("predict", graph_data)
        if not prediction:
            return

        if not self._panel:
            log.info(f"Prediction: {prediction}")
            return

        selected_node = nuke.selectedNode()
        self._panel.update_prediction_state(selected_node.fullName(), prediction)

    def _serialize_upstream(self, node_name: Optional[str] = None) -> Optional[Dict]:
        selected = self._get_target_node(node_name=node_name)
        if not selected:
            return None

        serialized_nodes = self.traverse_upstream(selected)
        return {
            "script_name": os.path.basename(nuke.scriptName()),
            "root": {"name": "root", "parent": None, "nodes": serialized_nodes},
            "start_node": selected.name(),
        }

    def traverse_upstream(
        self, node, target_nodes=None, visited=None, length=0, max_length=15
    ):
        """Traverse upstream nodes to construct a serialized graph for node prediction."""

        target_nodes = target_nodes or {}
        visited = visited or set()

        if length >= max_length:
            return target_nodes

        node_full_name = node.fullName()
        if node_full_name in visited:
            return target_nodes

        visited.add(node_full_name)
        target_nodes[node_full_name] = self._serialize_node(node)

        dependency_flag = nuke.INPUTS | nuke.HIDDEN_INPUTS
        for ancestor in node.dependencies(dependency_flag):
            if self._is_group_or_gizmo(node):
                self.traverse_upstream(
                    ancestor,
                    target_nodes=target_nodes,
                    visited=visited,
                    length=length + 1,
                )

            self.traverse_upstream(
                ancestor, target_nodes=target_nodes, visited=visited, length=length + 1
            )

        return target_nodes

    @staticmethod
    def _validate_node_for_prediction(node: nuke.Node) -> bool:
        if not node or node.maxOutputs() < 1 or node.inputs() < 1:
            return False
        return True

    @staticmethod
    def _is_group_or_gizmo(node: nuke.Node) -> bool:
        return node.Class() in {"Group", "Gizmo"} or "gizmo_file" in node.knobs()

    @staticmethod
    def _get_target_node(node_name: Optional[str]) -> Optional[nuke.Node]:
        if node_name:
            return nuke.toNode(node_name)

        selected = nuke.selectedNode()
        if not selected:
            nuke.message("Please select a node!")
            return None

        return selected

    @staticmethod
    def _serialize_node(node: nuke.Node, include_parameters: bool = False) -> Dict:
        dependency_flag = nuke.INPUTS | nuke.HIDDEN_INPUTS
        ancestors = node.dependencies(dependency_flag)

        json_node = {
            "name": node.name(),
            "node_type": node.Class(),
            "inputs": node.inputs(),
            "input_connections": [a.name() for a in ancestors],
        }

        if include_parameters:
            json_node["parameters"] = get_all_parameters(node)

        return json_node

    @staticmethod
    def _is_node_disabled(node):
        disable_knob = node.knob("disable")
        if not disable_knob:
            return False

        return disable_knob.value()
