import os
import json
import logging
import traceback

from tqdm import tqdm

from .parser import NukeNode

log = logging.getLogger(__name__)


class NukeGraphSerializer:
    def __init__(self, output_dir, include_params=False, remove_passthrough_nodes=True):
        self.output_dir = output_dir
        self.include_params = include_params
        self.remove_passthrough_nodes = remove_passthrough_nodes

        os.makedirs(self.output_dir, exist_ok=True)

    def _serialize_node(self, node: NukeNode, node_id: str):
        try:
            node_dict = {
                "name": node.name,
                "node_id": node_id,
                "node_type": node.node_type,
                "inputs": node.inputs,
                "input_connections": node.input_connections or [],
                "xpos": int(float(node.parameters.get("xpos", 0))),
                "ypos": int(float(node.parameters.get("ypos", 0))),
            }
        except Exception as e:
            log.info(node.parameters)
            log.info(node)
            log.error(e)
            raise

        if self.include_params:
            node_dict["parameters"] = node.parameters

        return node_dict

    def _serialize_group(self, group):
        # Ensure we maintain the relative order!
        child_nodes = {}
        for node_id, node in group.nodes.items():
            child_nodes[node_id] = self._serialize_node(node, node_id)

        return {
            "name": group.name,
            "parent": group.parent.name if group.parent else None,
            "nodes": child_nodes,
        }

    def serialize_script_graph(self, script_content_dict):
        for script_name, nuke_script in tqdm(
            script_content_dict.items(), desc="Serializing scripts"
        ):
            full = script_name
            script_base_path = os.path.basename(script_name)
            script_name, _ = os.path.splitext(script_base_path)

            if len(nuke_script) <= 1:
                log.info("Filtering stub script: {0}".format(script_name))
                continue

            try:
                self.serialize_graph(script_name, nuke_script)
            except Exception:
                log.info(f"Processing failed for {full}: {traceback.format_exc()}")
                return

    def serialize_graph(self, script_name, nuke_script):
        graph_data = {
            "script_name": script_name,
            nuke_script.root_group.name: self._serialize_group(nuke_script.root_group),
        }

        for node_name, group in nuke_script.groups.items():
            if group.parent is None:
                continue

            self._flatten_child_group(graph_data, group)

        if self.remove_passthrough_nodes:
            self._remove_passthrough_nodes(graph_data["root"]["nodes"])

        output_path = os.path.join(self.output_dir, "{0}.json".format(script_name))
        with open(output_path, "w") as f:
            json.dump(graph_data, f, indent=2)

    def _flatten_child_group(self, graph_data, group):
        """Flatten group structures, integrating nodes into root's node structure.

        1. Find the group's input and output nodes.
        2. Redirect internal connections to external nodes.
        3. Replace group references with direct connections.
        4. Reorganize nodes to ensure correct node structure.
        """
        root_nodes = graph_data["root"]["nodes"]

        # 'root' group is handled separately.
        # Each group thus has a parent, but can be nested.
        output_node = next(
            (node for node in group.nodes.values() if node.node_type == "Output"), None
        )

        # Store "LiveGroupInput/Input" internal nodes for the current group.
        input_nodes = [node.name for node in group.input_stack]

        matching_groups = [node for node in root_nodes if node == group.name]
        if not matching_groups:
            # The parent node was not connected to the node graph and wasn't added.
            return

        # Store input connections feeding into the current group.
        group_node_name = matching_groups[0]
        group_input_connections = list(
            reversed(root_nodes[group_node_name]["input_connections"])
        )

        nodes_to_remove = set()
        if group_input_connections:
            for node in group.nodes.values():
                # Check for nodes connected to "Input" nodes.
                for i, conn in enumerate(node.input_connections):
                    if conn in input_nodes:
                        # Find the associated parent input.
                        target_index = min(
                            len(group_input_connections) - 1, input_nodes.index(conn)
                        )
                        external_node = group_input_connections[target_index]
                        node.input_connections[i] = external_node
        else:
            if not any(
                group.name == node_name
                for node in root_nodes.values()
                for node_name in node.get("input_connections", [])
            ):
                nodes_to_remove.update(list(group.nodes))

        # It's fine for a group to have no output if it's a terminal node.
        if not output_node:
            referenced_in_inputs = any(
                group.name == node_name
                for node in root_nodes.values()
                for node_name in node.get("input_connections", [])
            )
            if referenced_in_inputs:
                raise RuntimeError(
                    f"Group '{group.name}' is referenced as an input but has no output node: {group}"
                )
        else:
            # Replace group references with direct output connection
            output_input = output_node.input_connections[0]
            for node in root_nodes.values():
                inputs = node["input_connections"]
                for i, node_name in enumerate(inputs):
                    if node_name == group.name:
                        inputs[i] = output_input

        nodes_to_remove.update(
            [
                node_name
                for node_name in group.nodes
                if output_node
                and node_name == output_node.name
                or node_name in input_nodes
            ]
        )

        for node_name in nodes_to_remove:
            group.nodes.pop(node_name)

        serialized_value = self._serialize_group(group)
        group_nodes = serialized_value["nodes"]

        group_key = group.name
        original_keys = list(root_nodes.keys())
        group_index = original_keys.index(group_key)
        new_nodes = {}

        # Add nodes before the group
        for i in range(group_index):
            key = original_keys[i]
            new_nodes[key] = root_nodes[key]

        # Add the group's nodes
        for node_name, node_data in group_nodes.items():
            new_nodes[node_name] = node_data

        # Add nodes after the group
        for i in range(group_index + 1, len(original_keys)):
            key = original_keys[i]
            new_nodes[key] = root_nodes[key]

        # Update the parent's nodes with our new ordered dictionary
        graph_data["root"]["nodes"] = new_nodes

    def _remove_passthrough_nodes(self, nodes_dict):
        pass_through_nodes = {"Dot", "NoOp"}

        reverse_connections = {}
        for node in nodes_dict.values():
            for input_connection in node.get("input_connections", []):
                if input_connection not in reverse_connections:
                    reverse_connections[input_connection] = []
                reverse_connections[input_connection].append(node["name"])

        nodes_to_remove = []
        for node_name, node in nodes_dict.items():
            if node["node_type"] in pass_through_nodes:
                input_connection = (
                    node["input_connections"][0] if node["input_connections"] else None
                )

                # Update dependent nodes
                dependent_nodes = reverse_connections.get(node["name"], [])
                for dependent_name in dependent_nodes:
                    for dep_node in nodes_dict.values():
                        if dep_node["name"] == dependent_name:
                            for j, conn in enumerate(dep_node["input_connections"]):
                                if conn == node["name"]:
                                    dep_node["input_connections"][j] = input_connection

                nodes_to_remove.append(node_name)

        # Remove nodes in reverse order to maintain correct indices
        for node_name in nodes_to_remove:
            nodes_dict.pop(node_name)
