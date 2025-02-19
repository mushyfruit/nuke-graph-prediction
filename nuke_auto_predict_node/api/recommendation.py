import os
import json
from functools import lru_cache
from typing import Dict, Optional

import nuke

from .request_handler import get_request_handler
from .utilities import get_all_parameters
from ..logging_config import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_prediction_manager():
    return PredictionManager()


def perform_recommendation(node_name=None):
    manager = get_prediction_manager()
    manager.perform_recommendation(node_name=node_name)


class PredictionManager:
    """Manages graph traversal and serialization for the recommendation system.

    Serialized graph data is posted via the :class:`RequestHandler` to the
    inference service running in a separate process.
    """

    def __init__(self):
        self._vocabulary: Dict[str, int] = {}
        self._request_handler = get_request_handler()
        self._min_node_requirement = 3

        self.load_model_vocab()

    def load_model_vocab(self):
        from .. import model_cnst

        if not os.path.exists(model_cnst.DirectoryConfig.VOCAB_PATH):
            return

        try:
            with open(model_cnst.DirectoryConfig.VOCAB_PATH, "r") as f:
                self._vocabulary = json.load(f).get("node_type_to_idx", {})
        except (json.JSONDecodeError, IOError) as e:
            log.error(f"Failed to load vocabulary: {str(e)}")

    def perform_recommendation(self, node_name=None):
        """Retrieves the next-node predictions for the given or selected node.

        :param node_name: Optional node name to perform predictions for.
        """
        try:
            serialized_graph_data = self._serialize_upstream(node_name=node_name)
            if not serialized_graph_data:
                return

            root_nodes = serialized_graph_data["root"]["nodes"]
            if len(root_nodes) <= self._min_node_requirement:
                log.info("Not enough nodes for meaningful prediction!")
                return

            log.info("Making prediction POST request...")
            self._process_prediction(serialized_graph_data)

        except Exception as e:
            log.error(f"Error during recommendation: {str(e)}")

    def _process_prediction(self, graph_data: Dict) -> None:
        prediction = self._request_handler.post("predict", graph_data)
        if not prediction:
            return

        from ..ui import prediction_panel

        panel = prediction_panel.get_panel_instance()
        if not panel:
            log.info(f"Prediction: {prediction}")
            return

        selected_node = nuke.selectedNode()
        panel.update_prediction_state(selected_node.fullName(), prediction)

    def _serialize_upstream(self, node_name: Optional[str] = None) -> Optional[Dict]:
        selected = self._get_target_node(node_name=node_name)
        if not selected:
            return None

        if not self._vocabulary:
            log.error("No model vocabulary found! Please train a local model first.")
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
    def _is_group_or_gizmo(node: nuke.Node):
        return node.Class() in {"Group", "Gizmo"} or "gizmo_file" in node.knobs()

    @staticmethod
    def _get_target_node(node_name: Optional[str]) -> Optional[nuke.Node]:
        if node_name:
            return nuke.toNode(node_name)

        selected = nuke.selectedNode()
        if not selected:
            nuke.message("Please select a node!")
            return

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
