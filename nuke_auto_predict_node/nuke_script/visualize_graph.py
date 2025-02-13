import os
import networkx as nx
import matplotlib.pyplot as plt


def visualize_nuke_script(script):
    # Create directed graph
    G = nx.DiGraph()

    # Add nodes and edges
    for node_id, node in script.nodes.items():
        G.add_node(node_id, label=node.node_type)
        for input_id in node.input_connections:
            G.add_edge(input_id, node_id)

    # Assign layers based on topological generations
    for layer, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            G.nodes[node]["layer"] = layer

    # Get positions using multipartite layout
    pos = nx.spring_layout(
        G,
        k=1.5,  # Increase space between nodes (default is 1.0)
        iterations=50,  # More iterations for better layout
        weight=None,  # Don't use edge weights
        scale=2.0,  # Scale up the layout
    )

    # Create plot
    plt.figure(figsize=(12, 8))
    plt.xticks([])
    plt.yticks([])

    # Draw the graph
    nx.draw_networkx(
        G,
        pos=pos,
        node_color="lightblue",
        node_size=25,
        arrows=True,
        arrowsize=2,
        width=0.5,
        labels=nx.get_node_attributes(G, "label"),
        font_size=2,
    )

    current_dir = os.path.dirname(__file__)
    output_dir = os.path.join(current_dir, "test")

    output_path = os.path.join(output_dir, f"nuke_graph_{len(script.nodes)}_nodes.png")
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()  # Close the figure to free memory

    print(f"Graph saved to: {output_path}")


def get_node_layers(G):
    """Determine layer for each node based on longest path from root"""
    layers = {}
    # Find root nodes (nodes with no incoming edges)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]

    def set_layer(node, layer):
        if node not in layers or layer > layers[node]:
            layers[node] = layer
            # Set children's layers
            for child in G.successors(node):
                set_layer(child, layer + 1)

    # Process from roots
    for root in roots:
        set_layer(root, 0)

    return layers
