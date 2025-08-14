"""Visualization utilities for graphs and subgraphs."""

import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx


def visualize_graph(nx_graph, title, pos=None, color='lightblue', 
                   with_labels=False, node_size=300, edge_color='gray'):
    """Visualize a NetworkX graph.
    
    Args:
        nx_graph: NetworkX graph to visualize
        title (str): Title for the plot
        pos: Node positions (if None, uses spring layout)
        color: Node color
        with_labels (bool): Whether to show node labels
        node_size (int): Size of nodes
        edge_color: Color of edges
    """
    plt.figure(figsize=(8, 6))
    
    if pos is None:
        pos = nx.spring_layout(nx_graph, seed=42)
    
    nx.draw(nx_graph, pos, node_color=color, with_labels=with_labels, 
            node_size=node_size, edge_color=edge_color)
    plt.title(title)
    plt.show()


def visualize_subgraph_highlight(nx_graph, subgraph, pos, color, title):
    """Visualize a subgraph highlighted within the original graph.
    
    Args:
        nx_graph: Original NetworkX graph
        subgraph: Subgraph data object with original_node_indices
        pos: Node positions for consistent layout
        color: Color to highlight the subgraph
        title (str): Title for the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Draw the full graph with light gray nodes and edges
    nx.draw(nx_graph, pos, node_color='lightgray', with_labels=False, 
            node_size=300, edge_color='lightgray')
    
    # Highlight the subgraph nodes and edges
    subgraph_nodes = subgraph.original_node_indices
    subgraph_edges = subgraph.edge_index.cpu().numpy().T
    
    # Map subgraph edge indices back to original graph node indices
    original_edges = [
        (subgraph.original_node_indices[src], subgraph.original_node_indices[dst]) 
        for src, dst in subgraph_edges
        if src < len(subgraph.original_node_indices) and dst < len(subgraph.original_node_indices)
    ]
    
    # Draw highlighted nodes and edges
    nx.draw_networkx_nodes(nx_graph, pos, nodelist=subgraph_nodes, 
                          node_color=color, node_size=300)
    nx.draw_networkx_edges(nx_graph, pos, edgelist=original_edges, 
                          edge_color=color, width=2)
    
    plt.title(title)
    plt.show()


def evaluate_and_visualize_top_k_separately(model, dataset, subgraph_method, device, 
                                           k=3, random_seed=42, **subgraph_kwargs):
    """Evaluate model and visualize top-k subgraphs for the first graph.
    
    This function demonstrates how the attention mechanism selects important
    subgraphs by visualizing the top-k subgraphs based on attention scores.
    
    Args:
        model: Trained GAT model
        dataset: Test dataset
        subgraph_method: Function to create subgraphs
        device: Device to run on
        k (int): Number of top subgraphs to visualize
        random_seed (int): Seed for consistent layout
        **subgraph_kwargs: Arguments for subgraph creation
    """
    model.eval()
    
    with torch.no_grad():
        for graph in dataset:
            # Create subgraphs
            if 'bfs' in subgraph_method.__name__.lower():
                import networkx as nx
                G = nx.from_edgelist(graph.edge_index.t().tolist())
                subgraphs = subgraph_method(G, graph.x, graph.y, **subgraph_kwargs)
                nx_graph = G
            else:
                subgraphs = subgraph_method(graph, **subgraph_kwargs)
                nx_graph = to_networkx(graph, to_undirected=True)
            
            print(f'Number of subgraphs: {len(subgraphs)}')
            
            subgraph_outputs = []
            subgraph_attention_scores = []

            # Generate consistent layout
            pos = nx.spring_layout(nx_graph, seed=random_seed)

            for subgraph in subgraphs:
                subgraph = subgraph.to(device)
                
                if subgraph.edge_index.size(1) == 0:
                    continue
                
                try:
                    output, attn_weights = model(subgraph)
                except (IndexError, RuntimeError) as e:
                    print(f"Error processing subgraph: {e}")
                    continue

                # Extract attention weights and compute score
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

            # Visualize top-k subgraphs separately
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            for i, idx in enumerate(top_k_indices):
                if i >= len(colors):
                    break
                    
                top_subgraph = subgraphs[idx]
                attention_score = subgraph_attention_scores[idx].item()
                
                visualize_subgraph_highlight(
                    nx_graph, top_subgraph, pos, 
                    color=colors[i], 
                    title=f'Top-{i+1} Subgraph (Attention Score: {attention_score:.4f})'
                )

            # Only visualize the first graph
            break


def plot_dataset_statistics(dataset_info):
    """Plot basic statistics about the dataset.
    
    Args:
        dataset_info (dict): Dictionary with dataset statistics
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Basic stats
    stats = ['num_graphs', 'num_classes', 'num_features']
    values = [dataset_info[stat] for stat in stats]
    
    axes[0, 0].bar(stats, values, color=['skyblue', 'lightgreen', 'salmon'])
    axes[0, 0].set_title('Dataset Overview')
    axes[0, 0].set_ylabel('Count')
    
    # Average nodes and edges
    axes[0, 1].bar(['Avg Nodes', 'Avg Edges'], 
                   [dataset_info['avg_nodes'], dataset_info['avg_edges']], 
                   color=['orange', 'purple'])
    axes[0, 1].set_title('Average Graph Size')
    axes[0, 1].set_ylabel('Count')
    
    # Class distribution (if available)
    if 'class_distribution' in dataset_info:
        classes = list(dataset_info['class_distribution'].keys())
        counts = list(dataset_info['class_distribution'].values())
        axes[1, 0].bar(classes, counts, color='lightcoral')
        axes[1, 0].set_title('Class Distribution')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Count')
    
    # Remove empty subplot
    axes[1, 1].remove()
    
    plt.tight_layout()
    plt.show()