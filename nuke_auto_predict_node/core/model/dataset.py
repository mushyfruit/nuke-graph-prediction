import os
import json
import pickle
import logging
import numpy as np
from tqdm import tqdm
from collections import deque

from .constants import NukeScript, DirectoryConfig, VOCAB

import torch
from torch_geometric.data import Data, Dataset

from typing import Dict, Optional

log = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self, vocab_file: Optional[str] = None):
        self._type_to_idx: Dict[str, int] = {}
        self._idx_to_type: Dict[int, str] = {}

        if vocab_file:
            self.load(vocab_file)

    def get_idx(self, node_type: str) -> int:
        return self.add(node_type)

    def get_type(self, idx: int) -> str:
        return self._idx_to_type[idx]

    def add(self, node_type: str) -> int:
        """Adds a new node type to the vocabulary if it doesn't exist."""
        if node_type not in self._type_to_idx:
            idx = len(self._type_to_idx)
            self._type_to_idx[node_type] = idx
            self._idx_to_type[idx] = node_type
            return idx
        return self._type_to_idx[node_type]

    def load(self, file_path) -> None:
        """Loads the vocabulary from a JSON file."""
        if not os.path.exists(file_path):
            log.warning(
                f"Vocabulary file {file_path} not found. Starting with empty vocabulary."
            )
            return
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            self._type_to_idx = data["node_type_to_idx"]
            self._idx_to_type = {idx: type_ for type_, idx in self._type_to_idx.items()}
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid vocabulary file format: {e}")

    def save(self, file_path: str) -> None:
        """Saves the vocabulary to a JSON file."""
        data = {"node_type_to_idx": self._type_to_idx, "num_node_types": len(self)}

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def __contains__(self, node_type: str) -> bool:
        return node_type in self._type_to_idx

    def __len__(self) -> int:
        return len(self._type_to_idx)


class NukeGraphBuilder:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def create_graph_data(
        self,
        serialized_graph: Dict[str, dict],
        start_node_data: Dict[str, any],
        update_vocab: bool = False,
        min_upstream_nodes: int = 5,
        include_start_node=False,
    ) -> Optional[Data]:
        """Creates a PyG graph from serialized json Nuke graph data.

        :param serialized_graph: Dictionary containing the entire graph structure.
        :param start_node_data: Dictionary containing the start node's data.
        :param update_vocab: Whether to update the vocabulary.
        :param include_start_node: Whether to include the start node in the
            upstream traversal. Used for model inference only.
        :param min_upstream_nodes: Minimum number of nodes required for a
            valid graph example.
        :returns: A PyTorch Geometric Data object containing the processed graph
        :rtype: Data or None.
        """
        nodes = serialized_graph.get("root", {}).get("nodes", {})
        if not nodes:
            raise RuntimeError("No nodes found in serialized graph!")

        # Construct the upstream graph.
        upstreams = get_upstream_nodes(
            nodes, start_node_data, include_start=include_start_node
        )

        # Ensure a minimum number of upstream nodes for the graph data.
        if len(upstreams) < min_upstream_nodes:
            return None

        raw_features = []
        for node_data in upstreams:
            if update_vocab and self.vocabulary:
                self.vocabulary.add(node_data["node_type"])

            raw_features.append(get_node_features(node_data, self.vocabulary))

        normalized_features = normalize_features(raw_features)

        # Build edge connections
        edge_index = [[], []]
        edge_attr = []
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

        # Create node, edge connectivity, and edge feature matrices.
        node_features = torch.tensor(normalized_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long).unsqueeze(1)

        # Return the initialized data object (excludes a ground-truth label)
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(upstreams),
        )


class NukeGraphDataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        transform=None,
        force_rebuild=False,
    ):
        super().__init__(root=root_dir, transform=transform)
        self.root_dir = root_dir
        self.file_paths = []

        self.processed_files = []
        self.examples = []

        self.processed_graphs_file = os.path.join(
            DirectoryConfig.DATA_CACHE_PATH, "process_graphs.pt"
        )
        self.metadata_file = os.path.join(
            DirectoryConfig.DATA_CACHE_PATH, "graph_metadata.json"
        )

        # Load the Nuke node type vocabulary.
        self.vocabulary_path = os.path.join(DirectoryConfig.DATA_CACHE_PATH, VOCAB)
        self.vocab = Vocabulary(self.vocabulary_path)

        self.graph_builder = NukeGraphBuilder(self.vocab)

        if not force_rebuild:
            self._load_cache()

    def _load_cache(self):
        # Check if the files have already been processed.
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.processed_files = json.load(f).get("processed_files", [])

        # Load processed graph data.
        if os.path.exists(self.processed_graphs_file):
            saved_data = torch.load(self.processed_graphs_file, weights_only=False)
            self.examples = saved_data["examples"]

    def process_all_graphs_in_dir(self, target_dir):
        for file in os.listdir(target_dir):
            if file.endswith(".json"):
                self.file_paths.append(os.path.join(target_dir, file))

        for file_path in tqdm(self.file_paths, desc="Processing all graphs"):
            # We've already processed this file.
            if file_path in self.processed_files:
                continue

            with open(file_path, "r") as f:
                data = json.load(f)

            graph_examples = self.generate_graph_training_examples(data)
            self.examples.extend(graph_examples)
            self.processed_files.append(file_path)

        self.save_graph_state()

        log.info(f"Processed {len(self.examples)} total examples")

    def save_graph_state(self):
        # Ensure the metadata parent dir exists.
        os.makedirs(DirectoryConfig.DATA_CACHE_PATH, exist_ok=True)

        torch.save(
            {
                "examples": self.examples,
            },
            self.processed_graphs_file,
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )
        self._save_metadata()
        self.vocab.save(self.vocabulary_path)

    def _save_metadata(self):
        with open(self.metadata_file, "w") as f:
            json.dump(
                {
                    "processed_files": self.processed_files,
                },
                f,
                indent=2,
            )

    def generate_graph_training_examples(
        self, data, min_context=3, max_context=25, stride=3
    ):
        root_group = data[NukeScript.ROOT]
        nodes = root_group[NukeScript.NODES]

        graph_node_data = list(nodes.values())
        examples = []

        for i in reversed(range(min_context, len(graph_node_data), stride)):
            # This is the prediction node.
            target_node_data = graph_node_data[i]
            graph_data = self.graph_builder.create_graph_data(
                data, target_node_data, update_vocab=True
            )
            if not graph_data:
                continue

            # Ensure we include graph-level ground-truth label.
            target = torch.tensor(
                self.vocab.get_idx(target_node_data["node_type"]),
                dtype=torch.long,
            )
            graph_data.y = target

            graph_data.validate(raise_on_error=True)

            examples.append(graph_data)

        return examples

    def len(self):
        return len(self.file_paths)

    def get(self, idx: int) -> Data:
        file_path = self.file_paths[idx]
        with open(file_path, "r") as f:
            data = json.load(f)

        graph_data = self.generate_graph_training_examples(data)

        if self.transform is not None:
            graph_data = self.transform(graph_data)

        return graph_data


def get_upstream_nodes(
    all_nodes_dict, start_node_dict, max_context=1000, include_start=False
):
    queue = deque([start_node_dict])
    upstream_nodes = [start_node_dict] if include_start else []
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


def normalize_features(raw_features, numerical_indices=None):
    if numerical_indices is None:
        numerical_indices = [1, 2]

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


def get_node_features(node_data, vocab: Vocabulary):
    node_type = node_data["node_type"]
    is_merge = 1 if node_type in {"Merge2", "Merge"} else 0
    return [
        # Categorical node information.
        vocab.get_idx(node_type),
        # Numerical node information.
        # These are normalized in graph preprocessing.
        node_data["inputs"],
        len(node_data.get("input_connections", [])),
        # Binary node information.
        is_merge,
    ]
