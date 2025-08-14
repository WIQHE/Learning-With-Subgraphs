#!/usr/bin/env python3
"""
Example: Comparison of BFS vs Sliding Window Subgraph Learning

This script compares the performance of BFS-based and sliding window
subgraph extraction methods for weakly supervised graph learning.
"""

import sys
import os
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import time

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.gat import GAT
from subgraph_methods.bfs_subgraphs import BFSSubgraphDataset, create_bfs_subgraphs
from subgraph_methods.sliding_window import SlidingWindowSubgraphDataset, create_sliding_window_subgraphs
from utils.data_utils import load_dataset, split_dataset, get_dataset_info
from utils.training import train_model_epochs
from utils.evaluation import evaluate_model_with_attention, print_evaluation_report

import warnings
warnings.filterwarnings('ignore')


def train_and_evaluate_method(method_name, subgraph_dataset_class, subgraph_method, 
                             train_graphs, test_graphs, dataset, device, **method_kwargs):
    """Train and evaluate a specific subgraph method.
    
    Args:
        method_name (str): Name of the method
        subgraph_dataset_class: Class for creating subgraph dataset
        subgraph_method: Function for creating subgraphs
        train_graphs: Training graphs
        test_graphs: Testing graphs
        dataset: Original dataset
        device: Computing device
        **method_kwargs: Method-specific parameters
    
    Returns:
        dict: Results including accuracy, training time, etc.
    """
    print(f"\n=== {method_name} ===")
    
    start_time = time.time()
    
    # Create subgraph datasets
    print("Creating subgraphs...")
    train_subgraph_dataset = subgraph_dataset_class(train_graphs, **method_kwargs)
    test_subgraph_dataset = subgraph_dataset_class(test_graphs, **method_kwargs)
    
    print(f"Train subgraphs: {len(train_subgraph_dataset)}")
    print(f"Test subgraphs: {len(test_subgraph_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_subgraph_dataset.data_list, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_subgraph_dataset.data_list, batch_size=32, shuffle=False)
    
    # Initialize model
    model = GAT(
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
        hidden_channels=64,
        heads=8,
        dropout=0.6
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Train model
    print("Training...")
    history = train_model_epochs(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=30,
        criterion=criterion,
        verbose=False
    )
    
    # Evaluate with attention
    print("Evaluating...")
    accuracy, predictions, true_labels = evaluate_model_with_attention(
        model=model,
        dataset=test_graphs,
        subgraph_method=subgraph_method,
        device=device,
        k=3,
        **method_kwargs
    )
    
    training_time = time.time() - start_time
    
    # Print results
    class_names = [f'Class {i}' for i in range(dataset.num_classes)]
    print(f"Final Accuracy: {accuracy:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    return {
        'method_name': method_name,
        'accuracy': accuracy,
        'training_time': training_time,
        'history': history,
        'predictions': predictions,
        'true_labels': true_labels,
        'num_train_subgraphs': len(train_subgraph_dataset),
        'num_test_subgraphs': len(test_subgraph_dataset)
    }


def plot_comparison_results(results):
    """Plot comparison results between methods.
    
    Args:
        results (list): List of result dictionaries
    """
    methods = [r['method_name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    times = [r['training_time'] for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy comparison
    bars1 = axes[0].bar(methods, accuracies, color=['skyblue', 'lightcoral'])
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    bars2 = axes[1].bar(methods, times, color=['lightgreen', 'orange'])
    axes[1].set_title('Training Time Comparison')
    axes[1].set_ylabel('Time (seconds)')
    
    # Add value labels on bars
    for bar, time_val in zip(bars2, times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time_val:.1f}s', ha='center', va='bottom')
    
    # Training curves comparison
    for result in results:
        history = result['history']
        epochs = range(1, len(history['test_accuracies']) + 1)
        axes[2].plot(epochs, history['test_accuracies'], 
                    label=f"{result['method_name']} Test Acc", marker='o', markersize=3)
    
    axes[2].set_title('Learning Curves Comparison')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Test Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_detailed_comparison(results):
    """Print detailed comparison table.
    
    Args:
        results (list): List of result dictionaries
    """
    print("\n" + "="*80)
    print("DETAILED COMPARISON RESULTS")
    print("="*80)
    
    print(f"{'Method':<25} {'Accuracy':<12} {'Time (s)':<12} {'Train Sub.':<12} {'Test Sub.':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['method_name']:<25} "
              f"{result['accuracy']:<12.4f} "
              f"{result['training_time']:<12.1f} "
              f"{result['num_train_subgraphs']:<12} "
              f"{result['num_test_subgraphs']:<12}")
    
    print("-" * 80)
    
    # Find best method
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    fastest_method = min(results, key=lambda x: x['training_time'])
    
    print(f"\nBest Accuracy: {best_accuracy['method_name']} ({best_accuracy['accuracy']:.4f})")
    print(f"Fastest Training: {fastest_method['method_name']} ({fastest_method['training_time']:.1f}s)")


def main():
    """Main comparison function."""
    
    print("=== BFS vs Sliding Window Subgraph Learning Comparison ===\n")
    
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
    
    train_graphs, test_graphs = split_dataset(dataset, test_size=0.2, random_state=42)
    print(f"\nTrain graphs: {len(train_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    
    # Define methods to compare
    methods = [
        {
            'name': 'BFS Subgraphs',
            'dataset_class': BFSSubgraphDataset,
            'method_func': create_bfs_subgraphs,
            'kwargs': {
                'min_nodes': 10,
                'min_edges': 8,
                'depth_limit': 8,
                'dataset_path': DATASET_PATH
            }
        },
        {
            'name': 'Sliding Window',
            'dataset_class': SlidingWindowSubgraphDataset,
            'method_func': create_sliding_window_subgraphs,
            'kwargs': {
                'window_size': 62,
                'step_size': 5,
                'dataset_path': DATASET_PATH
            }
        }
    ]
    
    # Run comparison
    results = []
    for method in methods:
        result = train_and_evaluate_method(
            method_name=method['name'],
            subgraph_dataset_class=method['dataset_class'],
            subgraph_method=method['method_func'],
            train_graphs=train_graphs,
            test_graphs=test_graphs,
            dataset=dataset,
            device=DEVICE,
            **method['kwargs']
        )
        results.append(result)
    
    # Display comparison results
    print_detailed_comparison(results)
    plot_comparison_results(results)
    
    print("\nComparison completed!")


if __name__ == "__main__":
    main()