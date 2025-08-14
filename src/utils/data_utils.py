"""Data loading and preprocessing utilities."""

import torch_geometric
from sklearn.model_selection import train_test_split


def load_dataset(name="MSRC_21", dataset_path='dataset'):
    """Load a graph dataset.
    
    Args:
        name (str): Name of the dataset to load
        dataset_path (str): Path to store/load dataset files
    
    Returns:
        dataset: PyTorch Geometric dataset
    """
    dataset = torch_geometric.datasets.TUDataset(root=dataset_path, name=name)
    return dataset


def split_dataset(dataset, test_size=0.2, random_state=42):
    """Split dataset into train and test sets.
    
    Args:
        dataset: PyTorch Geometric dataset
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    train_graphs, test_graphs = train_test_split(
        dataset, 
        test_size=test_size, 
        random_state=random_state
    )
    return train_graphs, test_graphs


def get_dataset_info(dataset):
    """Get basic information about a dataset.
    
    Args:
        dataset: PyTorch Geometric dataset
    
    Returns:
        dict: Dictionary containing dataset statistics
    """
    num_graphs = len(dataset)
    num_classes = dataset.num_classes
    num_features = dataset.num_node_features
    
    # Get labels for all graphs
    labels = [data.y.item() for data in dataset]
    
    # Calculate average number of nodes and edges
    total_nodes = sum(data.num_nodes for data in dataset)
    total_edges = sum(data.num_edges for data in dataset)
    avg_nodes = total_nodes / num_graphs
    avg_edges = total_edges / num_graphs
    
    return {
        'num_graphs': num_graphs,
        'num_classes': num_classes,
        'num_features': num_features,
        'unique_labels': set(labels),
        'avg_nodes': avg_nodes,
        'avg_edges': avg_edges
    }