from typing import Optional, Dict, Any, Callable

import nuke

from ..dcc_backend import DCCBackend


class NukeBackend(DCCBackend):
    def get_target_node(self, node_name: Optional[str] = None) -> Optional[nuke.Node]:
        if node_name:
            return nuke.toNode(node_name)

        try:
            return nuke.selectedNode()
        except ValueError:
            nuke.message("Please select a node!")
            return None

    def serialize_upstream_nodes(self, start_node: Any) -> Dict:
        serialized_nodes = {}
        pending = [start_node]
        while pending:
            node = pending.pop()
            name = node.fullName()
            if name in serialized_nodes:
                continue
            serialized_nodes[name] = self.serialize_node(node)
            pending.extend(node.dependencies(nuke.INPUTS | nuke.HIDDEN_INPUTS))

        return {
            "root": {"name": "root", "parent": None, "nodes": serialized_nodes},
            "start_node": start_node.fullName(),
        }

    def serialize_node(self, node: Any) -> Dict:
        dependency_flag = nuke.INPUTS | nuke.HIDDEN_INPUTS
        ancestors = node.dependencies(dependency_flag)

        return {
            "name": node.fullName(),
            "node_type": node.Class(),
            "inputs": node.inputs(),
            "input_connections": [a.fullName() for a in ancestors],
            "xpos": node.xpos(),
            "ypos": node.ypos(),
        }

    def should_perform_callback(self) -> bool:
        knob = nuke.thisKnob()
        if knob is None or knob.name() != "selected":
            return False

        return bool(knob.value())

    def enable_callback(self, callback_fn: Callable[[], None]) -> None:
        nuke.addKnobChanged(callback_fn)

    def disable_callback(self, callback_fn: Callable[[], None]) -> None:
        nuke.removeKnobChanged(callback_fn)

    def validate_node_for_inference(self, node: nuke.Node) -> bool:
        if not node or node.maxOutputs() < 1 or node.inputs() < 1:
            return False
        return True
