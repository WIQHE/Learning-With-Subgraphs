"""
Utility functions for subgraph generation and processing.

This module contains common functions used across different subgraph extraction methods
including BFS-based and sliding window approaches.
"""

import torch
import random
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset


def create_bfs_subgraphs(G, original_features, graph_label, depth_limit=8, min_nodes=10, min_edges=8):
    """
    Create subgraphs using BFS (Breadth-First Search) traversal.
    
    Args:
        G: NetworkX graph
        original_features: Node features from the original graph
        graph_label: Label of the original graph
        depth_limit: Maximum depth for BFS traversal
        min_nodes: Minimum number of nodes a subgraph must have
        min_edges: Minimum number of edges a subgraph must have
    
    Returns:
        List of PyTorch Geometric Data objects representing subgraphs
    """
    visited = set()
    subgraphs = []

    def bfs(G, start_node, depth_limit):
        """BFS implementation for subgraph extraction."""
        bfs_nodes = set()
        bfs_edges = set()
        queue = [(start_node, 0)]  # (node, depth)
        visited.add(start_node)

        while queue:
            node, depth = queue.pop(0)
            if depth < depth_limit:
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        bfs_edges.add((node, neighbor))
                        bfs_nodes.add(neighbor)
            bfs_nodes.add(node)

        return bfs_nodes, bfs_edges

    # Perform BFS iteratively from random unvisited nodes
    while len(visited) < len(G.nodes):
        unvisited_nodes = list(set(G.nodes) - visited)
        if not unvisited_nodes:
            break

        start_node = random.choice(unvisited_nodes)
        bfs_nodes, bfs_edges = bfs(G, start_node, depth_limit)

        # Check minimum requirements
        if len(bfs_nodes) < min_nodes or len(bfs_edges) < min_edges:
            continue

        # Map original indices to subgraph indices
        node_indices = {node: i for i, node in enumerate(bfs_nodes)}

        # Create edge index in subgraph format
        subgraph_edges = []
        for u, v in bfs_edges:
            if u in bfs_nodes and v in bfs_nodes:
                subgraph_edges.append((node_indices[u], node_indices[v]))

        if not subgraph_edges:
            continue

        edge_index = torch.tensor(subgraph_edges, dtype=torch.long).t().contiguous()
        features = original_features[list(bfs_nodes)]

        # Create Data object for the subgraph
        data = Data(
            x=features,
            edge_index=edge_index,
            y=torch.tensor([graph_label.item()], dtype=torch.long),
            original_node_indices=torch.tensor(list(bfs_nodes), dtype=torch.long),
        )
        
        if data.edge_index.size(1) > 0:  # Check if edges exist
            subgraphs.append(data)

    return subgraphs


def create_sliding_window_subgraphs(graph, window_size, step_size):
    """
    Create subgraphs using sliding window approach.
    
    Args:
        graph: PyTorch Geometric Data object
        window_size: Size of the sliding window (number of nodes)
        step_size: Step size for the sliding window
    
    Returns:
        List of PyTorch Geometric Data objects representing subgraphs
    """
    subgraphs = []
    num_nodes = graph.num_nodes

    for start in range(0, num_nodes - window_size + 1, step_size):
        end = start + window_size

        # Create subgraph node features
        subgraph_x = graph.x[start:end]

        # Create masks for edges within the subgraph
        subgraph_edge_index = graph.edge_index
        mask = (subgraph_edge_index[0] >= start) & (subgraph_edge_index[0] < end) & \
               (subgraph_edge_index[1] >= start) & (subgraph_edge_index[1] < end)

        # Filter edges using the mask
        subgraph_edge_index = subgraph_edge_index[:, mask]

        # Adjust indices to reflect position in subgraph
        subgraph_edge_index[0] -= start
        subgraph_edge_index[1] -= start

        # Ensure edge_index is not empty
        if subgraph_edge_index.size(1) == 0:
            # Create self-loops if no edges exist
            subgraph_edge_index = torch.stack([
                torch.arange(end - start), 
                torch.arange(end - start)
            ], dim=0)

        # Create the subgraph
        subgraph = Data(
            x=subgraph_x, 
            edge_index=subgraph_edge_index, 
            y=graph.y,
            original_node_indices=list(range(start, end))
        )

        subgraphs.append(subgraph)

    return subgraphs


def move_to_device(batch, device):
    """Move batch data to specified device (CPU/GPU)."""
    return batch.to(device)


def evaluate_with_attention(model, dataset, subgraph_func, k=3, device='cpu', **subgraph_kwargs):
    """
    Evaluate model using top-k subgraph selection based on attention weights.
    
    Args:
        model: Trained GAT model
        dataset: Test dataset
        subgraph_func: Function to create subgraphs (BFS or sliding window)
        k: Number of top subgraphs to select
        device: Device to run evaluation on
        **subgraph_kwargs: Additional arguments for subgraph function
    
    Returns:
        accuracy: Classification accuracy
        predictions: List of predicted labels
        true_labels: List of true labels
    """
    model.eval()
    all_predictions = []
    all_true_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for graph in dataset:
            # Generate subgraphs based on the specified method
            if 'G' in subgraph_kwargs:  # BFS method
                G = nx.from_edgelist(graph.edge_index.t().tolist())
                subgraphs = subgraph_func(G, graph.x, graph.y, **subgraph_kwargs)
            else:  # Sliding window method
                subgraphs = subgraph_func(graph, **subgraph_kwargs)
            
            if not subgraphs:
                continue
                
            subgraph_outputs = []
            subgraph_attention_scores = []

            for subgraph in subgraphs:
                subgraph = subgraph.to(device)
                
                # Skip subgraphs with empty edge_index
                if subgraph.edge_index.size(1) == 0:
                    continue
                
                try:
                    output, attn_weights = model(subgraph)
                except IndexError:
                    continue
                
                # Extract attention weights
                if isinstance(attn_weights, (tuple, list)):
                    attention_tensor = attn_weights[-1]
                else:
                    attention_tensor = attn_weights
                
                attention_score = attention_tensor.mean().item()
                subgraph_outputs.append(output.unsqueeze(0))
                subgraph_attention_scores.append(attention_score)

            if not subgraph_outputs:
                continue

            subgraph_outputs = torch.cat(subgraph_outputs, dim=0)
            subgraph_attention_scores = torch.tensor(subgraph_attention_scores)

            # Select top-k subgraphs
            current_k = min(k, len(subgraph_outputs))
            if current_k == 0:
                continue

            top_k_values, top_k_indices = subgraph_attention_scores.topk(
                current_k, dim=0, largest=True, sorted=True
            )
            top_k_subgraphs = subgraph_outputs[top_k_indices]

            # Aggregate predictions
            final_prediction = top_k_subgraphs.mean(dim=0)
            final_prediction = torch.softmax(final_prediction, dim=1)
            final_prediction_class = final_prediction.argmax(dim=1).item()
            true_label = graph.y.item()

            all_predictions.append(final_prediction_class)
            all_true_labels.append(true_label)

            if final_prediction_class == true_label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, all_predictions, all_true_labels