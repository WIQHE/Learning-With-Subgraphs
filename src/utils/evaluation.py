"""Evaluation utilities for subgraph-based learning."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def evaluate_model_with_attention(model, dataset, subgraph_method, device, k=3, **subgraph_kwargs):
    """Evaluate model using top-k subgraphs based on attention weights.
    
    This function creates subgraphs for each graph in the dataset, computes
    attention scores, selects the top-k subgraphs, and aggregates their
    predictions for final classification.
    
    Args:
        model: Trained GAT model
        dataset: Test dataset of graphs
        subgraph_method: Function to create subgraphs (BFS or sliding window)
        device: Device to run on (cuda/cpu)
        k (int): Number of top subgraphs to select
        **subgraph_kwargs: Additional arguments for subgraph creation
    
    Returns:
        tuple: (accuracy, predictions, true_labels)
    """
    model.eval()
    all_predictions = []
    all_true_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for graph in dataset:
            # Create subgraphs based on the method
            if 'bfs' in subgraph_method.__name__.lower():
                import networkx as nx
                G = nx.from_edgelist(graph.edge_index.t().tolist())
                subgraphs = subgraph_method(G, graph.x, graph.y, **subgraph_kwargs)
            else:
                subgraphs = subgraph_method(graph, **subgraph_kwargs)
            
            subgraph_outputs = []
            subgraph_attention_scores = []

            for subgraph in subgraphs:
                subgraph = subgraph.to(device)
                
                # Skip subgraphs with empty edge_index
                if subgraph.edge_index.size(1) == 0:
                    continue
                
                try:
                    # Forward pass through model
                    output, attn_weights = model(subgraph)
                except (IndexError, RuntimeError) as e:
                    print(f"Error processing subgraph: {e}")
                    continue
                
                # Extract attention weights
                if isinstance(attn_weights, (tuple, list)):
                    attention_tensor = attn_weights[-1]
                else:
                    attention_tensor = attn_weights
                
                # Compute single attention score for the subgraph
                attention_score = attention_tensor.mean().item()
                subgraph_outputs.append(output.unsqueeze(0))
                subgraph_attention_scores.append(attention_score)

            # Skip if no valid subgraph outputs
            if not subgraph_outputs:
                continue

            subgraph_outputs = torch.cat(subgraph_outputs, dim=0)
            subgraph_attention_scores = torch.tensor(subgraph_attention_scores)

            # Select top-k subgraphs based on attention scores
            current_k = min(k, len(subgraph_outputs))
            if current_k == 0:
                continue

            top_k_values, top_k_indices = subgraph_attention_scores.topk(
                current_k, dim=0, largest=True, sorted=True
            )
            top_k_subgraphs = subgraph_outputs[top_k_indices]

            # Aggregate top-k subgraph outputs (mean aggregation)
            final_prediction = top_k_subgraphs.mean(dim=0)

            # Apply softmax to get probabilities
            final_prediction = torch.softmax(final_prediction, dim=1)

            # Get predicted class
            final_prediction_class = final_prediction.argmax(dim=1).item()
            true_label = graph.y.item()

            all_predictions.append(final_prediction_class)
            all_true_labels.append(true_label)

            if final_prediction_class == true_label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, all_predictions, all_true_labels


def plot_confusion_matrix(true_labels, predictions, class_names, title="Confusion Matrix"):
    """Plot confusion matrix.
    
    Args:
        true_labels: True class labels
        predictions: Predicted class labels
        class_names: List of class names
        title (str): Title for the plot
    """
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_training_curves(history, title="Training Progress"):
    """Plot training and validation curves.
    
    Args:
        history: Dictionary with training history
        title (str): Title for the plots
    """
    epochs = range(1, len(history['losses']) + 1)
    
    # Plot accuracy curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_accuracies'], label="Train Accuracy", marker='o', markersize=2)
    plt.plot(epochs, history['test_accuracies'], label="Test Accuracy", marker='s', markersize=2)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(f"{title} - Accuracy")
    plt.grid(True, alpha=0.3)

    # Plot loss curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['losses'], label="Training Loss", color='red', marker='^', markersize=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"{title} - Loss")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_evaluation_report(true_labels, predictions, class_names, accuracy):
    """Print detailed evaluation report.
    
    Args:
        true_labels: True class labels
        predictions: Predicted class labels  
        class_names: List of class names
        accuracy: Overall accuracy
    """
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))
    
    # Additional statistics
    print(f"\nTotal Samples: {len(true_labels)}")
    print(f"Correct Predictions: {sum(1 for t, p in zip(true_labels, predictions) if t == p)}")
    print(f"Incorrect Predictions: {sum(1 for t, p in zip(true_labels, predictions) if t != p)}")