"""BFS-based subgraph extraction for weakly supervised learning."""

import torch
import random
import networkx as nx
from torch_geometric.data import Data, InMemoryDataset


class BFSSubgraphDataset(InMemoryDataset):
    """Dataset that creates subgraphs using BFS traversal.
    
    This dataset takes a collection of graphs and creates subgraphs using
    Breadth-First Search (BFS) traversal from random starting nodes.
    
    Args:
        dataset: Original dataset of graphs
        min_nodes (int): Minimum number of nodes a subgraph must have
        min_edges (int): Minimum number of edges a subgraph must have
        depth_limit (int): Maximum depth for BFS traversal
        dataset_path (str): Path to store dataset files
    """
    
    def __init__(self, dataset, min_nodes=10, min_edges=8, depth_limit=8, dataset_path='dataset'):
        super(BFSSubgraphDataset, self).__init__(root=dataset_path)
        self.data_list = []
        self.labels = []
        self.min_nodes = min_nodes
        self.min_edges = min_edges
        self.depth_limit = depth_limit

        for graph in dataset:
            # Convert to NetworkX for BFS traversal
            G = nx.from_edgelist(graph.edge_index.t().tolist())
            subgraphs = create_bfs_subgraphs(
                G, graph.x, graph.y, 
                min_nodes=min_nodes, 
                min_edges=min_edges,
                depth_limit=depth_limit
            )
            self.data_list.extend(subgraphs)
            self.labels.extend([graph.y] * len(subgraphs))
            
        self.data, self.slices = self.collate(self.data_list)

    def get_labels(self):
        """Return labels for all subgraphs."""
        return torch.tensor(self.labels)


def create_bfs_subgraphs(G, original_features, graph_label, min_nodes=10, min_edges=8, depth_limit=8):
    """Create subgraphs using BFS traversal from random starting nodes.
    
    This function performs BFS traversal from random unvisited nodes until
    all nodes in the graph are covered. Each BFS traversal creates a subgraph
    that preserves the original node features and graph label.
    
    Args:
        G (nx.Graph): NetworkX graph for BFS traversal
        original_features (torch.Tensor): Node features from original graph
        graph_label (torch.Tensor): Label for the original graph
        min_nodes (int): Minimum nodes required for a valid subgraph
        min_edges (int): Minimum edges required for a valid subgraph
        depth_limit (int): Maximum depth for BFS traversal
    
    Returns:
        list: List of PyTorch Geometric Data objects representing subgraphs
    """
    visited = set()  # Set to store visited nodes
    subgraphs = []  # List to store subgraphs from each BFS traversal

    def bfs(G, start_node, depth_limit):
        """Perform BFS traversal from a starting node.
        
        Args:
            G (nx.Graph): NetworkX graph
            start_node: Node to start BFS from
            depth_limit (int): Maximum depth to traverse
            
        Returns:
            tuple: (bfs_nodes, bfs_edges) - nodes and edges visited during BFS
        """
        bfs_nodes = set()  # To store nodes visited in this BFS
        bfs_edges = set()  # To store edges traversed in this BFS
        queue = [(start_node, 0)]  # (node, depth)
        visited.add(start_node)  # Mark the start node as visited

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

    # Perform BFS iteratively from random unvisited nodes until all nodes are visited
    while len(visited) < len(G.nodes):
        unvisited_nodes = list(set(G.nodes) - visited)
        if not unvisited_nodes:
            break

        start_node = random.choice(unvisited_nodes)

        # Run BFS from the chosen node
        bfs_nodes, bfs_edges = bfs(G, start_node, depth_limit)

        # Check if subgraph meets minimum requirements
        if len(bfs_nodes) < min_nodes or len(bfs_edges) < min_edges:
            continue  # Skip this subgraph if it doesn't meet the criteria

        # Map original indices to subgraph indices
        node_indices = {node: i for i, node in enumerate(bfs_nodes)}

        # Create edge index in subgraph format
        subgraph_edges = []
        for u, v in bfs_edges:
            if u in bfs_nodes and v in bfs_nodes:
                subgraph_edges.append((node_indices[u], node_indices[v]))

        # Handle empty edge cases
        if not subgraph_edges:
            continue

        edge_index = torch.tensor(subgraph_edges, dtype=torch.long).t().contiguous()
        features = original_features[list(bfs_nodes)]  # Extract node features for subgraph

        # Create Data object for the subgraph
        data = Data(
            x=features,
            edge_index=edge_index,
            y=torch.tensor([graph_label.item()], dtype=torch.long),
            original_node_indices=torch.tensor(list(bfs_nodes), dtype=torch.long),
        )
        
        # Final check for valid edge_index
        if data.edge_index.size(1) > 0:  # Ensure there are edges
            subgraphs.append(data)

    return subgraphs