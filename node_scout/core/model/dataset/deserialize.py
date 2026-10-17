from collections import deque

from .vocabulary import Vocabulary

import torch
from torch_geometric.data import Data

from typing import Dict, Optional, List, Any


def create_graph_data(
    serialized_graph: Dict[str, dict],
    start_node_data: Dict[str, any],
    vocabulary: Vocabulary,
    update_vocab: bool = False,
    min_upstream_nodes: int = 0,
    max_upstream_nodes: int = 50,
    include_start_node: bool = False,
) -> Optional[Data]:
    """Build context; exclude the training target, include the inference selection."""
    nodes = serialized_graph.get("root", {}).get("nodes", {})
    if not nodes:
        raise RuntimeError("No nodes found in serialized graph!")

    upstreams = get_upstream_nodes(
        nodes,
        start_node_data,
        max_context=max_upstream_nodes,
        include_start=include_start_node,
    )

    if len(upstreams) < min_upstream_nodes:
        return None

    raw_features = []
    for node_data in upstreams:
        if update_vocab:
            vocabulary.add(node_data["node_type"])

        raw_features.append(get_node_features(node_data, vocabulary))

    edge_index = [[], []]
    node_name_to_idx = {
        node_dict["name"]: idx for idx, node_dict in enumerate(upstreams)
    }

    for node_data in upstreams:
        curr_idx = node_name_to_idx[node_data["name"]]
        for input_node in node_data.get("input_connections", []):
            if input_node in node_name_to_idx:
                source_idx = node_name_to_idx[input_node]
                edge_index[0].append(source_idx)
                edge_index[1].append(curr_idx)

    return Data(
        x=normalize_features(raw_features),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        num_nodes=len(upstreams),
    )


def get_upstream_nodes(
    all_nodes_dict: Dict[str, Any],
    start_node_dict: Dict[str, Any],
    max_context: int,
    include_start: bool,
) -> List[Dict[str, Any]]:
    queue = deque([start_node_dict])
    upstream_nodes = [start_node_dict] if include_start else []
    visited = {start_node_dict["name"]}

    while queue and len(upstream_nodes) < max_context:
        node_dict = queue.popleft()

        if node_dict != start_node_dict:
            upstream_nodes.append(node_dict)

        for upstream_name in node_dict.get("input_connections", []):
            if upstream_name in visited or upstream_name not in all_nodes_dict:
                continue

            upstream_dict = all_nodes_dict[upstream_name]
            visited.add(upstream_name)
            queue.append(upstream_dict)

    return upstream_nodes


def normalize_features(raw_features: List[List[Any]]) -> torch.Tensor:
    numerical_indices = [1, 2, 4, 5]

    features = torch.tensor(raw_features, dtype=torch.float32)
    if len(features) > 1:
        values = features[:, numerical_indices]
        mean = values.mean(dim=0)
        std = values.std(dim=0, correction=0)
        std = torch.where(std > 1e-6, std, 1)
        features[:, numerical_indices] = (values - mean) / std

    return features


def get_node_features(node_data: Dict[str, Any], vocab: Vocabulary) -> List[Any]:
    node_type = node_data["node_type"]
    is_merge = 1 if node_type in {"Merge2", "Merge"} else 0

    xpos = node_data.get("xpos", 0)
    ypos = node_data.get("ypos", 0)

    return [
        vocab.get_idx(node_type),
        node_data["inputs"],
        len(node_data.get("input_connections", [])),
        is_merge,
        xpos,
        ypos,
    ]
