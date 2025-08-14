"""Sliding window subgraph extraction for weakly supervised learning."""

import torch
from torch_geometric.data import Data, InMemoryDataset


class SlidingWindowSubgraphDataset(InMemoryDataset):
    """Dataset that creates subgraphs using sliding window approach.
    
    This dataset takes a collection of graphs and creates subgraphs by
    sliding a fixed-size window over the nodes in sequential order.
    
    Args:
        dataset: Original dataset of graphs
        window_size (int): Size of the sliding window (number of nodes)
        step_size (int): Step size for the sliding window
        dataset_path (str): Path to store dataset files
    """
    
    def __init__(self, dataset, window_size=62, step_size=5, dataset_path='dataset'):
        super(SlidingWindowSubgraphDataset, self).__init__(root=dataset_path)
        self.data_list = []
        self.labels = []
        self.window_size = window_size
        self.step_size = step_size

        for graph in dataset:
            subgraphs = create_sliding_window_subgraphs(
                graph, 
                window_size=window_size, 
                step_size=step_size
            )
            self.data_list.extend(subgraphs)
            self.labels.extend([graph.y] * len(subgraphs))

        # Convert to tensor format required by PyTorch Geometric
        self.data, self.slices = self.collate(self.data_list)

    def get_labels(self):
        """Return labels for all subgraphs."""
        return torch.tensor(self.labels)


def create_sliding_window_subgraphs(graph, window_size=62, step_size=5):
    """Create subgraphs using sliding window approach.
    
    This function creates subgraphs by sliding a fixed-size window over
    the nodes in sequential order. Each window creates a subgraph that
    includes the nodes within the window and all edges between them.
    
    Args:
        graph: PyTorch Geometric data object
        window_size (int): Number of nodes in each subgraph window
        step_size (int): Step size for moving the window
    
    Returns:
        list: List of PyTorch Geometric Data objects representing subgraphs
    """
    subgraphs = []
    num_nodes = graph.num_nodes

    for start in range(0, num_nodes - window_size + 1, step_size):
        end = start + window_size

        # Extract node features for the current window
        subgraph_x = graph.x[start:end]  # Node features

        # Create masks for edges within the subgraph
        subgraph_edge_index = graph.edge_index
        mask = (subgraph_edge_index[0] >= start) & (subgraph_edge_index[0] < end) & \
               (subgraph_edge_index[1] >= start) & (subgraph_edge_index[1] < end)

        # Filter edges using the mask
        subgraph_edge_index = subgraph_edge_index[:, mask]

        # Adjust the indices of the filtered edges to reflect their position in the subgraph
        subgraph_edge_index[0] -= start
        subgraph_edge_index[1] -= start

        # Handle case where no edges exist within the window
        if subgraph_edge_index.size(1) == 0:
            # Create self-loops for each node to avoid empty edge_index
            num_window_nodes = end - start
            subgraph_edge_index = torch.stack([
                torch.arange(num_window_nodes), 
                torch.arange(num_window_nodes)
            ], dim=0)

        # Create the subgraph data object
        subgraph = Data(
            x=subgraph_x, 
            edge_index=subgraph_edge_index, 
            y=graph.y  # Keep the original graph label
        )
        
        # Add the original node indices as an attribute for visualization/analysis
        subgraph.original_node_indices = list(range(start, end))

        subgraphs.append(subgraph)

    return subgraphs