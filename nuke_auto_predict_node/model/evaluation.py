import os
import random
import torch

from constants import MODEL_NAME
from utilities import load_model_checkpoint


def test_nuke_predictor(top_k: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from dataset import NukeGraphDataset  # Your dataset class

    graph_dir = os.path.join(os.path.dirname(__file__), "data/nuke_graphs")
    dataset = NukeGraphDataset(graph_dir, should_download=False)
    test_graph = dataset[1][11]

    print(f"Retrieving: {dataset.file_paths[50]}")

    model_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    model, _, vocab = load_model_checkpoint(model_dir, MODEL_NAME, device)
    print(print_graph_details(test_graph, vocab))

    model.eval()
    idx_to_node_type = {
        idx: node_type for node_type, idx in vocab["node_type_to_idx"].items()
    }
    test_graph = test_graph.to(device)

    with torch.no_grad():
        predictions = model(test_graph)
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        print("\nPredictions:")
        print("-" * 50)

        for node_idx in range(len(predictions)):
            print(f"\nNode {node_idx}:")
            print("Top {} predictions:".format(top_k))

            for k in range(top_k):
                node_type = idx_to_node_type[top_indices[node_idx][k].item()]
                prob = top_probs[node_idx][k].item() * 100
                print(f"  {k + 1}. {node_type:<30} ({prob:.2f}%)")

            if hasattr(test_graph, "y"):
                actual_type = idx_to_node_type[test_graph.y.item()]
                print(f"\n  Actual: {actual_type}")

            print("-" * 50)


def print_graph_details(graph, vocab=None):
    """Print detailed information about the graph structure"""
    print("\nGraph Details:")
    print("-" * 50)
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.edge_index.size(1)}")
    print(f"Node feature dimensions: {graph.x.size()}")

    print("\nEdge Connections:")
    for i in range(graph.edge_index.size(1)):
        src = graph.edge_index[0, i].item()
        dst = graph.edge_index[1, i].item()
        if hasattr(graph, "edge_attr"):
            attr = graph.edge_attr[i].item()
            print(f"Node {src} → Node {dst} (attr: {attr})")
        else:
            print(f"Node {src} → Node {dst}")

    if vocab and hasattr(graph, "y"):
        idx_to_node_type = {
            idx: node_type for node_type, idx in vocab["node_type_to_idx"].items()
        }
        print(f"\nTarget Node Type: {idx_to_node_type[graph.y.item()]}")


if __name__ == "__main__":
    test_nuke_predictor()
