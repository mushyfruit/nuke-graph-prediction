from typing import Optional, Dict, Any, Callable

import nuke

from .parser import NukeScript, NukeGroup, NukeNode, NukeScriptParser, GROUP_INPUT_NODE, IGNORE_NODES
from .serialization import NukeGraphSerializer


class NukeBackend:
    def get_target_node(self, node: Optional[nuke.Node] = None) -> Optional[nuke.Node]:
        if node is not None:
            return node

        try:
            return nuke.selectedNode()
        except ValueError:
            nuke.message("Please select a node!")
            return None

    def serialize_upstream_nodes(self, start_node: Any) -> Dict:
        script = NukeScript()
        nodes = {}
        groups = {nuke.root(): script.root_group}
        pending = [start_node]
        while pending:
            node = pending.pop()
            if node is None or node in nodes or node.Class() in IGNORE_NODES | {"Viewer"}:
                continue
            parent = node.parent()
            if parent not in groups:
                if parent.Class() not in NukeScriptParser.GROUP_TYPES:
                    break
                pending.extend((node, parent))
                continue
            group = groups[parent]
            data = NukeNode(
                node.fullName() + (f"#{node.cloneInstanceIndex()}" if node.clones() else ""),
                node.Class(), node.inputs(),
                {"xpos": node.xpos(), "ypos": node.ypos()},
                script.node_name_counter,
            )
            nodes[node] = group.nodes[data.name] = data
            if data.node_type in GROUP_INPUT_NODE:
                data.parameters["number"] = node["number"].value()
                group.input_stack.append(data)
            if data.node_type in NukeScriptParser.GROUP_TYPES:
                groups[node] = script.groups[data.name] = NukeGroup(data.name, group)
                pending.extend(n for n in node.nodes() if n.Class() == "Output")
            pending.extend(node.input(i) for i in range(data.inputs))
        for node, data in nodes.items():
            data.input_connections = [
                nodes[upstream].name if upstream in nodes else None
                for upstream in (node.input(i) for i in range(data.inputs))
            ]

        target = nodes[start_node].name if start_node in nodes else None
        graph = NukeGraphSerializer().serialize_graph("", script, start_node=target)
        serialized_nodes = graph["root"]["nodes"]
        upstream = {}
        pending = [graph.setdefault("start_node", None)]
        while pending:
            name = pending.pop()
            if name not in serialized_nodes or name in upstream:
                continue
            upstream[name] = serialized_nodes[name]
            pending.extend(serialized_nodes[name]["input_connections"])
        graph["root"]["nodes"] = upstream
        return graph

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
