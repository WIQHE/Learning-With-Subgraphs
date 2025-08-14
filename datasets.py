"""
Dataset handling for subgraph-based learning.

This module provides dataset classes for creating subgraph datasets
from original graph datasets using different extraction methods.
"""

import torch
from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split


class SubgraphDataset(InMemoryDataset):
    """
    Dataset class for subgraph-based learning.
    
    This class creates a dataset of subgraphs from an original graph dataset
    using a specified subgraph extraction method.
    """
    
    def __init__(self, dataset, subgraph_function, root='dataset', **subgraph_kwargs):
        """
        Initialize the subgraph dataset.
        
        Args:
            dataset: Original graph dataset
            subgraph_function: Function to extract subgraphs
            root: Root directory for dataset
            **subgraph_kwargs: Additional arguments for subgraph function
        """
        super(SubgraphDataset, self).__init__(root=root)
        self.data_list = []
        self.labels = []

        for graph in dataset:
            # Extract subgraphs using the specified method
            if 'G' in subgraph_kwargs or 'original_features' in subgraph_kwargs:
                # BFS-based extraction
                import networkx as nx
                G = nx.from_edgelist(graph.edge_index.t().tolist())
                subgraphs = subgraph_function(G, graph.x, graph.y, **subgraph_kwargs)
            else:
                # Sliding window extraction
                subgraphs = subgraph_function(graph, **subgraph_kwargs)
            
            self.data_list.extend(subgraphs)
            self.labels.extend([graph.y] * len(subgraphs))

        # Convert to tensor format
        if self.data_list:
            self.data, self.slices = self.collate(self.data_list)
        else:
            self.data = None
            self.slices = None

    def get_labels(self):
        """Get labels for all subgraphs."""
        return torch.tensor(self.labels)


def create_dataset_splits(dataset, test_size=0.2, random_state=42):
    """
    Create train/test splits from a dataset.
    
    Args:
        dataset: Original dataset to split
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        train_dataset: Training portion of dataset
        test_dataset: Testing portion of dataset
    """
    train_idx, test_idx = train_test_split(
        range(len(dataset)), 
        test_size=test_size, 
        random_state=random_state
    )
    
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    
    return train_dataset, test_dataset