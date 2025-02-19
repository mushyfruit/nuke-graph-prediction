import os
import json

import torch
from torch_geometric.data import Data

from constants import NukeScript


def test_node_suggestions(model, dataset, device="cuda", top_k=5):
    """
    Interactive testing function for node suggestions.

    Args:
        model: Trained GNN model
        dataset: NukeGraphDataset instance
        device: Device to run inference on
        top_k: Number of suggestions to show
    """
    # First, create a reverse mapping from indices to node types
    idx_to_node_type = {
        idx: node_type for node_type, idx in dataset.node_type_to_idx.items()
    }

    model.eval()  # Set model to evaluation mode

    while True:
        # Let user pick a graph to test
        print("\nAvailable graphs:")
        for i in range(min(5, len(dataset.file_paths))):
            print(f"{i}: {os.path.basename(dataset.file_paths[i])}")
        print("...")

        try:
            graph_idx = int(input("\nEnter graph number to test (or -1 to quit): "))
            if graph_idx == -1:
                break

            # Load the selected graph
            with open(dataset.file_paths[graph_idx], "r") as f:
                data = json.load(f)

            root_group = data[NukeScript.ROOT]
            nodes = root_group[NukeScript.NODES]
            ordered_nodes = list(nodes.items())

            # Show current graph structure
            print("\nCurrent graph structure:")
            for i, (node_id, node_data) in enumerate(ordered_nodes):
                print(f"{i}: {node_data['node_type']} (id: {node_id})")

            # Let user select where to test suggestions
            seq_length = int(input("\nEnter how many nodes to use as context: "))
            if seq_length >= len(ordered_nodes):
                print("Selected length exceeds graph size!")
                continue

            # Create test example using the same processing as training
            curr_nodes = ordered_nodes[:seq_length]
            node_types = []
            edge_index = [[], []]

            node_id_to_idx = {
                node_id: idx for idx, (node_id, _) in enumerate(curr_nodes)
            }

            for node_id, node_data in curr_nodes:
                node_types.append(dataset.node_type_to_idx[node_data["node_type"]])
                curr_idx = node_id_to_idx[node_id]
                for input_node in node_data.get("input_connections", []):
                    if input_node in node_id_to_idx:
                        source_idx = node_id_to_idx[input_node]
                        edge_index[0].append(source_idx)
                        edge_index[1].append(curr_idx)

            # Convert to tensors
            node_features = torch.tensor(node_types, dtype=torch.long).unsqueeze(1)
            edge_index = torch.tensor(edge_index, dtype=torch.long)

            # Create Data object
            test_data = Data(
                x=node_features, edge_index=edge_index, num_nodes=len(curr_nodes)
            )

            # Get model predictions
            test_data = test_data.to(device)
            with torch.no_grad():
                predictions = model(test_data)
                probabilities = torch.nn.functional.softmax(predictions, dim=1)

                # Get top k suggestions
                top_probs, top_indices = torch.topk(probabilities, k=top_k)

                print(f"\nTop {top_k} suggested next nodes:")
                print("\nActual next node:", ordered_nodes[seq_length][1]["node_type"])
                print("\nSuggestions:")
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    node_type = idx_to_node_type[idx.item()]
                    print(f"{node_type}: {prob.item() * 100:.2f}%")

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            continue

        again = input("\nTest another graph? (y/n): ")
        if again.lower() != "y":
            break
