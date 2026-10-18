import os
import json


class NukeGraphSerializer:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir

        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)

    def _serialize_group(self, group):
        return {
            "name": group.name,
            "parent": group.parent.name if group.parent else None,
            "nodes": {
                node_id: {
                    "name": node.name,
                    "node_id": node_id,
                    "node_type": node.node_type,
                    "inputs": node.inputs,
                    "input_connections": list(node.input_connections),
                    "xpos": int(float(node.parameters.get("xpos", 0))),
                    "ypos": int(float(node.parameters.get("ypos", 0))),
                }
                for node_id, node in group.nodes.items()
            },
        }

    def serialize_graph(self, script_name, nuke_script, start_node=None):
        graph_data = {
            "script_name": script_name,
            nuke_script.root_group.name: self._serialize_group(nuke_script.root_group),
        }
        if start_node is not None:
            graph_data["start_node"] = start_node

        for group in nuke_script.groups.values():
            if group.parent is None:
                continue

            self._flatten_child_group(graph_data, group)

        replacements = self._remove_passthrough_nodes(graph_data["root"]["nodes"])
        if start_node is not None:
            graph_data["start_node"] = replacements.get(
                graph_data["start_node"], graph_data["start_node"]
            )

        if self.output_dir is not None:
            output_path = os.path.join(self.output_dir, "{0}.json".format(script_name))
            with open(output_path, "w") as f:
                json.dump(graph_data, f, indent=2)
        return graph_data

    def _flatten_child_group(self, graph_data, group):
        root_nodes = graph_data["root"]["nodes"]
        connections = root_nodes[group.name]["input_connections"]
        input_nodes = {}
        for node in group.input_stack:
            index = int(node.parameters.get("number", 0))
            input_nodes[node.name] = connections[index] if index < len(connections) else None
        if graph_data.get("start_node") in input_nodes:
            graph_data["start_node"] = input_nodes[graph_data["start_node"]]

        child_nodes = self._serialize_group(group)["nodes"]
        for node in child_nodes.values():
            node["input_connections"] = [
                input_nodes.get(conn, conn) for conn in node["input_connections"]
            ]
        output_node = next(
            (node for node in child_nodes.values() if node["node_type"] == "Output"), None
        )
        if output_node is None:
            if any(group.name in n["input_connections"] for n in root_nodes.values()):
                raise RuntimeError(
                    f"Group '{group.name}' is referenced as an input but has no output node: {group}"
                )
        else:
            output_input = next(iter(output_node["input_connections"]), None)
            if graph_data.get("start_node") in {group.name, output_node["name"]}:
                graph_data["start_node"] = output_input
            for node in root_nodes.values():
                node["input_connections"] = [
                    output_input if conn == group.name else conn
                    for conn in node["input_connections"]
                ]

        new_nodes = {}
        for name, node in root_nodes.items():
            if name == group.name:
                new_nodes.update(
                    (key, child) for key, child in child_nodes.items()
                    if key not in input_nodes and child["node_type"] != "Output"
                )
            else:
                new_nodes[name] = node
        graph_data["root"]["nodes"] = new_nodes

    def _remove_passthrough_nodes(self, nodes_dict):
        replacements = {
            name: next(iter(node["input_connections"]), None)
            for name, node in nodes_dict.items()
            if node["node_type"] in {"Dot", "NoOp", "Input", "LiveInput", "Output"}
        }
        for name, conn in replacements.items():
            path = [name]
            while conn in replacements:
                path.append(conn)
                conn = replacements[conn]
            for key in path:
                replacements[key] = conn
        for name in replacements:
            del nodes_dict[name]
        for node in nodes_dict.values():
            node["input_connections"] = [
                replacements.get(conn, conn) for conn in node["input_connections"]
            ]
        return replacements
