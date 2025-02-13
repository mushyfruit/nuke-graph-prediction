import os
import sys
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from nuke_auto_predict_node.model.utilities import download_remote_files
from nuke_auto_predict_node.model.constants import NukeScript, INVALID_NODE_CLASSES
import torch
from torch_geometric.data import Data, Dataset


class NukeGraphConverter:
    def __init__(self, node_type_vocab):
        self.node_type_to_idx = node_type_vocab
        self.feature_config = {"numerical": [1, 2]}

    def _normalize_seq_features(self, raw_features, numerical_indices):
        """Normalize numerical features using the same method as training."""
        features = np.array(raw_features)
        normalized = features.copy()

        if len(features) > 1:
            for idx in numerical_indices:
                values = features[:, idx]
                mean = np.mean(values)
                std = np.std(values)

                if std > 1e-6:
                    normalized[:, idx] = (values - mean) / std
                else:
                    normalized[:, idx] = values - mean

        return normalized.tolist()

    def convert_json_to_pyg(self, json_data, min_context=3, max_context=15):
        """
        Convert JSON data to PyTorch Geometric Data format.

        Args:
            json_data (dict): JSON data in the same format as training data
            min_context (int): Minimum number of nodes to process
            max_context (int): Maximum number of nodes to consider

        Returns:
            torch_geometric.data.Data: Graph data ready for model inference
        """
        nodes = json_data.get("root", {}).get("nodes", {})
        ordered_nodes = list(nodes.items())

        # Take the last max_context nodes
        start_idx = max(0, len(ordered_nodes) - max_context)
        curr_nodes = ordered_nodes[start_idx:]

        if len(curr_nodes) < min_context:
            raise ValueError(f"Not enough nodes (minimum {min_context} required)")

        # Extract and normalize features
        raw_features = []
        for node_name, node_data in curr_nodes:
            raw_features.append(get_node_features(node_data, self.node_type_to_idx))

        normalized_features = self._normalize_seq_features(
            raw_features, self.feature_config["numerical"]
        )

        # Build edge connections
        edge_index = [[], []]
        edge_attr = []
        node_name_to_idx = {
            node_name: idx for idx, (node_name, _) in enumerate(curr_nodes)
        }

        for node_name, node_data in curr_nodes:
            curr_idx = node_name_to_idx[node_name]

            for input_idx, input_node in enumerate(
                node_data.get("input_connections", [])
            ):
                if input_node in node_name_to_idx:
                    source_idx = node_name_to_idx[input_node]
                    edge_index[0].append(source_idx)
                    edge_index[1].append(curr_idx)
                    edge_attr.append(input_idx)

        # Create tensors
        node_features = torch.tensor(normalized_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long).unsqueeze(1)

        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(curr_nodes),
        )


class NukeGraphDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, should_download=False):
        super().__init__(root=root_dir, transform=transform)
        self.root_dir = root_dir
        self.node_type_to_idx = {}
        self.file_paths = []

        self.examples = []
        self.feature_config = {"numerical": [1, 2, 3]}

        self.all_features = defaultdict(list)

        self.feature_stats = {
            "min_vals": None,
            "max_vals": None,
            "means": None,
            "stds": None,
        }

        self.converter = NukeGraphConverter(self.feature_config)
        self.validity_mask = None

        if should_download:
            print("Currently ignoring the request to download remote files!")
            # utilities.download_remote_files()

        for file in os.listdir(root_dir):
            if file.endswith(".json"):
                self.file_paths.append(os.path.join(root_dir, file))

        # Build node type vocabulary prior to vocab.
        self._perform_dataset_preprocessing()
        self._populate_validity_mask()
        self._process_all_graphs()

    def _populate_validity_mask(self):
        self.validity_mask = torch.ones(len(self.node_type_to_idx), dtype=torch.bool)
        for node_type, idx in self.node_type_to_idx.items():
            if node_type not in INVALID_NODE_CLASSES:
                self.validity_mask[idx] = False

    def _perform_dataset_preprocessing(self):
        for file_path in tqdm(self.file_paths, desc="Preprocessing all graphs..."):
            with open(file_path, "r") as f:
                data = json.load(f)

            self._build_node_type_vocab(data)

    def _get_all_node_features(self, data):
        root_group = data[NukeScript.ROOT]
        nodes = root_group[NukeScript.NODES]

        for node_data in nodes.values():
            node_features = get_node_features(node_data, self.node_type_to_idx)
            self.all_features[data["script_name"]].append(node_features)

    def _build_node_type_vocab(self, data):
        for group_data in data.values():
            if isinstance(group_data, dict) and "nodes" in group_data:
                for node in group_data["nodes"].values():
                    node_type = node["node_type"]
                    if node_type not in self.node_type_to_idx:
                        self.node_type_to_idx[node_type] = len(self.node_type_to_idx)

    def _process_all_graphs(self):
        """Process all graphs and store their examples."""
        for file_path in tqdm(self.file_paths, desc="Processing all graphs"):
            with open(file_path, "r") as f:
                data = json.load(f)
            graph_examples = self._process_graph(data)

            self.examples.extend(graph_examples)

        print(f"Processed {len(self.examples)} total examples")

    def len(self):
        return len(self.file_paths)

    def get(self, idx: int) -> Data:
        file_path = self.file_paths[idx]
        with open(file_path, "r") as f:
            data = json.load(f)

        graph_data = self._process_graph(data)

        if self.transform is not None:
            graph_data = self.transform(graph_data)

        return graph_data

    def _normalize_seq_features(self, raw_features, numerical_indices):
        features = np.array(raw_features)
        normalized = features.copy()

        if len(features) > 1:
            for idx in numerical_indices:
                values = features[:, idx]
                mean = np.mean(values)
                std = np.std(values)

                # Avoid division by zero
                if std > 1e-6:
                    normalized[:, idx] = (values - mean) / std
                else:
                    normalized[:, idx] = values - mean

        return normalized.tolist()

    def _get_upstream_nodes(self, all_nodes_dict, start_node_dict, max_context):
        """Generate a correct list of upstream nodes based on input connections."""
        queue = deque([start_node_dict])
        upstream_nodes = []
        visited = {start_node_dict["name"]}

        while queue and len(upstream_nodes) < max_context:
            node_dict = queue.popleft()

            if node_dict != start_node_dict:
                upstream_nodes.append(node_dict)

            for upstream_name in node_dict.get("input_connections", []):
                if upstream_name is None:
                    continue

                if upstream_name in visited:
                    continue

                if upstream_name not in all_nodes_dict:
                    continue

                upstream_dict = all_nodes_dict[upstream_name]
                visited.add(upstream_name)
                queue.append(upstream_dict)

        return upstream_nodes

    def _process_graph(self, data, min_context=3, max_context=25, stride=3):
        root_group = data[NukeScript.ROOT]
        nodes = root_group[NukeScript.NODES]
        ordered_nodes = list(nodes.items())
        examples = []

        for i in reversed(range(min_context, len(ordered_nodes), stride)):
            # This is the prediction node.
            start_node = ordered_nodes[i][1]
            upstreams = self._get_upstream_nodes(nodes, start_node, max_context)
            if len(upstreams) < min_context:
                continue

            raw_features = []
            for node_data in upstreams:
                raw_features.append(get_node_features(node_data, self.node_type_to_idx))

            normalized_features = self._normalize_seq_features(
                raw_features,
                self.feature_config["numerical"],
            )

            edge_index = [[], []]
            edge_attr = []

            # Map node IDs to indices
            node_name_to_idx = {
                node_dict["name"]: idx for idx, node_dict in enumerate(upstreams)
            }

            for node_data in upstreams:
                curr_idx = node_name_to_idx[node_data["name"]]

                for input_idx, input_node in enumerate(
                    node_data.get("input_connections", [])
                ):
                    if input_node in node_name_to_idx:
                        source_idx = node_name_to_idx[input_node]
                        edge_index[0].append(source_idx)
                        edge_index[1].append(curr_idx)
                        edge_attr.append(input_idx)

            # Create tensors
            node_features = torch.tensor(normalized_features, dtype=torch.float32)
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            edge_attr = torch.tensor(edge_attr, dtype=torch.long).unsqueeze(1)

            target = torch.tensor(
                self.node_type_to_idx[ordered_nodes[i][1]["node_type"]],
                dtype=torch.long,
            )

            examples.append(
                Data(
                    x=node_features,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=target,
                    num_nodes=len(upstreams),
                )
            )

        return examples


def get_node_features(node_data, node_type_to_idx):
    is_merge = 1 if node_data["node_type"] in {"Merge2", "Merge"} else 0
    return [
        # Categorical node infomration.
        node_type_to_idx[node_data["node_type"]],  # Node Type
        # Numerical node information.
        # These are normalized in graph preprocessing.
        node_data["inputs"],
        len(node_data.get("input_connections", [])),
        # Binary node information.
        is_merge,
    ]
