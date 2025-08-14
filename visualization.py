"""
Visualization utilities for graphs and subgraphs.

This module provides functions for visualizing graphs, subgraphs,
and attention-based subgraph selection results.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def visualize_graph(nx_graph, title, pos, color='lightblue', with_labels=False, 
                   node_size=300, edge_color='gray'):
    """
    Visualize a complete graph.
    
    Args:
        nx_graph: NetworkX graph object
        title: Title for the plot
        pos: Node positions
        color: Node color
        with_labels: Whether to show node labels
        node_size: Size of nodes
        edge_color: Color of edges
    """
    plt.figure(figsize=(8, 6))
    nx.draw(nx_graph, pos, node_color=color, with_labels=with_labels, 
            node_size=node_size, edge_color=edge_color)
    plt.title(title)
    plt.show()


def visualize_subgraph_highlight(nx_graph, subgraph, pos, color, title):
    """
    Visualize a subgraph highlighted within the original graph.
    
    Args:
        nx_graph: Original NetworkX graph
        subgraph: Subgraph to highlight
        pos: Node positions
        color: Color for highlighted subgraph
        title: Title for the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Draw full graph in light gray
    nx.draw(nx_graph, pos, node_color='lightgray', with_labels=False, 
            node_size=300, edge_color='lightgray')
    
    # Highlight subgraph nodes and edges
    subgraph_nodes = subgraph.original_node_indices
    subgraph_edges = subgraph.edge_index.cpu().numpy().T
    
    # Map subgraph edges back to original indices
    original_edges = [(subgraph.original_node_indices[src], 
                      subgraph.original_node_indices[dst]) 
                     for src, dst in subgraph_edges]
    
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=subgraph_nodes, 
                          node_color=color, node_size=300)
    nx.draw_networkx_edges(nx_graph, pos, edgelist=original_edges, 
                          edge_color=color, width=2)
    
    plt.title(title)
    plt.show()


def plot_confusion_matrix(true_labels, predictions, class_names, title="Confusion Matrix"):
    """
    Plot confusion matrix for classification results.
    
    Args:
        true_labels: True class labels
        predictions: Predicted class labels
        class_names: Names of classes
        title: Title for the plot
    """
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_training_curves(train_acc_list, test_acc_list, loss_list, num_epochs):
    """
    Plot training and testing accuracy curves along with loss curve.
    
    Args:
        train_acc_list: List of training accuracies
        test_acc_list: List of testing accuracies  
        loss_list: List of training losses
        num_epochs: Number of training epochs
    """
    # Plot accuracy curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_acc_list, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), test_acc_list, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")
    
    # Plot loss curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), loss_list, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss over Epochs")
    
    plt.tight_layout()
    plt.show()


def evaluate_and_visualize_top_k(model, dataset, subgraph_func, device, k=3, 
                                random_seed=42, **subgraph_kwargs):
    """
    Evaluate model and visualize top-k subgraphs based on attention weights.
    
    Args:
        model: Trained model
        dataset: Test dataset  
        subgraph_func: Function to create subgraphs
        device: Device for computation
        k: Number of top subgraphs to visualize
        random_seed: Seed for consistent layout
        **subgraph_kwargs: Arguments for subgraph function
    """
    model.eval()
    
    import torch
    with torch.no_grad():
        for graph in dataset:
            # Generate subgraphs
            if 'G' in subgraph_kwargs:  # BFS method
                import networkx as nx
                G = nx.from_edgelist(graph.edge_index.t().tolist())
                subgraphs = subgraph_func(G, graph.x, graph.y, **subgraph_kwargs)
            else:  # Sliding window method
                subgraphs = subgraph_func(graph, **subgraph_kwargs)
            
            print(f'Number of subgraphs: {len(subgraphs)}')
            
            subgraph_outputs = []
            subgraph_attention_scores = []

            # Convert to NetworkX for visualization
            nx_graph = to_networkx(graph, to_undirected=True)
            pos = nx.spring_layout(nx_graph, seed=random_seed)

            for subgraph in subgraphs:
                subgraph = subgraph.to(device)
                output, attn_weights = model(subgraph)

                # Extract attention scores
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

            # Visualize original graph
            visualize_graph(nx_graph, title="Original Graph", pos=pos)

            # Visualize top-k subgraphs
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            for i, idx in enumerate(top_k_indices):
                top_subgraph = subgraphs[idx]
                color = colors[i % len(colors)]
                title = f'Top-{i+1} Subgraph (Attention Score: {subgraph_attention_scores[idx]:.4f})'
                visualize_subgraph_highlight(nx_graph, top_subgraph, pos, color, title)

            break  # Only visualize first graph