#!/usr/bin/env python3
"""
Example: BFS-based Subgraph Learning

This script demonstrates weakly supervised learning on graphs using
BFS-based subgraph extraction with Graph Attention Networks (GAT).
"""

import sys
import os
import torch
from torch_geometric.loader import DataLoader

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gat import GAT
from subgraph_methods.bfs_subgraphs import BFSSubgraphDataset, create_bfs_subgraphs
from utils.data_utils import load_dataset, split_dataset, get_dataset_info
from utils.training import train_model_epochs
from utils.evaluation import evaluate_model_with_attention, plot_confusion_matrix, plot_training_curves, print_evaluation_report
from utils.visualization import evaluate_and_visualize_top_k_separately

import warnings
warnings.filterwarnings('ignore')


def main():
    """Main function to run BFS-based subgraph learning."""
    
    print("=== BFS-based Subgraph Learning ===\n")
    
    # Configuration
    DATASET_NAME = "MSRC_21"
    DATASET_PATH = 'dataset'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}\n")
    
    # Load and split dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_PATH)
    dataset_info = get_dataset_info(dataset)
    
    print("Dataset Information:")
    for key, value in dataset_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Split dataset
    train_graphs, test_graphs = split_dataset(dataset, test_size=0.2, random_state=21)
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Test graphs: {len(test_graphs)}\n")
    
    # Create BFS subgraph datasets
    print("Creating BFS subgraphs...")
    train_subgraph_dataset = BFSSubgraphDataset(
        train_graphs, 
        min_nodes=10, 
        min_edges=8, 
        depth_limit=8,
        dataset_path=DATASET_PATH
    )
    test_subgraph_dataset = BFSSubgraphDataset(
        test_graphs,
        min_nodes=10,
        min_edges=8, 
        depth_limit=8,
        dataset_path=DATASET_PATH
    )
    
    print(f"Train subgraphs: {len(train_subgraph_dataset)}")
    print(f"Test subgraphs: {len(test_subgraph_dataset)}\n")
    
    # Create data loaders
    train_loader = DataLoader(train_subgraph_dataset.data_list, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subgraph_dataset.data_list, batch_size=32, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Initialize model
    print("Initializing GAT model...")
    model = GAT(
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        hidden_channels=64,
        heads=8,
        dropout=0.6
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"Model: {model}\n")
    
    # Train model
    print("Training model...")
    history = train_model_epochs(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=DEVICE,
        num_epochs=20,  # Reduced for demonstration
        criterion=criterion,
        verbose=True
    )
    
    print("\nTraining completed!\n")
    
    # Plot training curves
    plot_training_curves(history, "BFS Subgraph Learning")
    
    # Evaluate with attention-based selection
    print("Evaluating with attention-based subgraph selection...")
    accuracy, predictions, true_labels = evaluate_model_with_attention(
        model=model,
        dataset=test_graphs,
        subgraph_method=create_bfs_subgraphs,
        device=DEVICE,
        k=4,  # Select top-4 subgraphs
        min_nodes=10,
        min_edges=8,
        depth_limit=8
    )
    
    # Print evaluation results
    class_names = [f'Class {i}' for i in range(dataset.num_classes)]
    print_evaluation_report(true_labels, predictions, class_names, accuracy)
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, class_names, 
                         "BFS Subgraph Learning - Confusion Matrix")
    
    # Visualize top subgraphs for first test graph
    print("\nVisualizing top subgraphs...")
    evaluate_and_visualize_top_k_separately(
        model=model,
        dataset=test_graphs,
        subgraph_method=create_bfs_subgraphs,
        device=DEVICE,
        k=3,
        min_nodes=10,
        min_edges=8,
        depth_limit=8
    )
    
    print("BFS-based subgraph learning completed!")


if __name__ == "__main__":
    main()