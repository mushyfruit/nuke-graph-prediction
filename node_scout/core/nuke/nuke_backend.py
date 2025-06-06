import os
from typing import Optional, Dict, Any, Callable

import nuke

from ..dcc_backend import DCCBackend


class NukeBackend(DCCBackend):
    def get_node_name(self, node: Any) -> str:
        return node.fullName()

    def get_target_node(self, node_name: Optional[str] = None) -> Optional[nuke.Node]:
        if node_name:
            return nuke.toNode(node_name)

        selected = nuke.selectedNode()
        if not selected:
            nuke.message("Please select a node!")
            return None

        return selected

    def serialize_upstream_nodes(self, start_node: Any) -> Dict:
        serialized_nodes = self._traverse_and_serialize_upstream_nodes(start_node)
        return {
            "script_name": os.path.basename(nuke.scriptName()),
            "root": {"name": "root", "parent": None, "nodes": serialized_nodes},
            "start_node": start_node.name(),
        }

    def serialize_node(self, node: Any, **kwargs) -> Dict:
        dependency_flag = nuke.INPUTS | nuke.HIDDEN_INPUTS
        ancestors = node.dependencies(dependency_flag)

        json_node = {
            "name": node.name(),
            "node_type": node.Class(),
            "inputs": node.inputs(),
            "input_connections": [a.name() for a in ancestors],
        }

        if kwargs.get("include_parameters"):
            json_node["parameters"] = get_all_parameters(node)

        return json_node

    def should_perform_callback(self) -> bool:
        knob = nuke.thisKnob()
        if knob is None or knob.name() != "selected":
            return False

        # Return early when deselecting nodes.
        if not knob.value():
            return False

        return True

    def enable_callback(self, callback_fn: Callable[[], None]) -> None:
        nuke.addKnobChanged(callback_fn)

    def disable_callback(self, callback_fn: Callable[[], None]) -> None:
        nuke.removeKnobChanged(callback_fn)

    def _traverse_and_serialize_upstream_nodes(
        self, node, target_nodes=None, visited=None, length=0, max_length=15
    ):
        target_nodes = target_nodes or {}
        visited = visited or set()

        if length >= max_length:
            return target_nodes

        node_full_name = node.fullName()
        if node_full_name in visited:
            return target_nodes

        visited.add(node_full_name)
        target_nodes[node_full_name] = self.serialize_node(node)

        dependency_flag = nuke.INPUTS | nuke.HIDDEN_INPUTS
        for ancestor in node.dependencies(dependency_flag):
            if self._is_group_or_gizmo(node):
                self._traverse_and_serialize_upstream_nodes(
                    ancestor,
                    target_nodes=target_nodes,
                    visited=visited,
                    length=length + 1,
                )

            self._traverse_and_serialize_upstream_nodes(
                ancestor, target_nodes=target_nodes, visited=visited, length=length + 1
            )

        return target_nodes

    def validate_node_for_inference(self, node: nuke.Node) -> bool:
        if not node or node.maxOutputs() < 1 or node.inputs() < 1:
            return False
        return True

    def show_message(self, text: str) -> None:
        pass

    @staticmethod
    def _is_group_or_gizmo(node: nuke.Node) -> bool:
        return node.Class() in {"Group", "Gizmo"} or "gizmo_file" in node.knobs()
