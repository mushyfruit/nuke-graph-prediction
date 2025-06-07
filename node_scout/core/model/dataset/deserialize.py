import logging
import numpy as np
from collections import deque

from .vocabulary import Vocabulary

import torch
from torch_geometric.data import Data

from typing import Dict, Optional, List, Any

log = logging.getLogger(__name__)


def create_graph_data(
    serialized_graph: Dict[str, dict],
    start_node_data: Dict[str, any],
    vocabulary: Vocabulary,
    update_vocab: bool = False,
    min_upstream_nodes: int = 5,
    max_upstream_nodes: int = 50,
    filter_graphs: bool = True,
    include_start_node: bool = False,
) -> Optional[Data]:
    """Creates a PyG graph from serialized JSON Nuke graph data.

    :param serialized_graph: Dictionary containing the entire graph structure.
    :param start_node_data: Dictionary containing the start node's data.
    :param vocabulary: Vocabulary instance.
    :param update_vocab: Whether to update the vocabulary.
    :param min_upstream_nodes: Minimum number of nodes required for a
        valid graph example.
    :param max_upstream_nodes: Maximum number of upstream nodes stored
        for a given graph example.
    :param filter_graphs: Whether to filter out invalid graphs.
    :param include_start_node: Whether to include the start node in the
        upstream traversal. Used for model inference only.
    :returns: A PyTorch Geometric Data object containing the processed graph
    :rtype: Data or None.
    """
    nodes = serialized_graph.get("root", {}).get("nodes", {})
    if not nodes:
        raise RuntimeError("No nodes found in serialized graph!")

    # Construct the upstream graph.
    upstreams = get_upstream_nodes(
        nodes,
        start_node_data,
        max_context=max_upstream_nodes,
        include_start=include_start_node,
    )

    # Ensure a minimum number of upstream nodes for the graph data.
    if filter_graphs and len(upstreams) < min_upstream_nodes:
        return None

    raw_features = []
    for node_data in upstreams:
        if update_vocab:
            vocabulary.add(node_data["node_type"])

        raw_features.append(get_node_features(node_data, vocabulary))

    normalized_features = normalize_features(raw_features)

    # Build edge connections
    edge_index = [[], []]
    edge_attr = []
    node_name_to_idx = {
        node_dict["name"]: idx for idx, node_dict in enumerate(upstreams)
    }

    for node_data in upstreams:
        curr_idx = node_name_to_idx[node_data["name"]]
        for input_idx, input_node in enumerate(node_data.get("input_connections", [])):
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


def get_upstream_nodes(
    all_nodes_dict: Dict[str, Any],
    start_node_dict: Dict[str, Any],
    max_context: Optional[int] = 1000,
    include_start: bool = False,
) -> List[Dict[str, Any]]:
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


def normalize_features(
    raw_features: List[List[Any]],
    numerical_indices: Optional[List[int]] = None,
) -> List[List[Any]]:
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


def get_node_features(node_data: Dict[str, Any], vocab: Vocabulary) -> List[Any]:
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
