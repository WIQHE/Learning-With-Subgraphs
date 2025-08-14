"""
Graph neural network models for subgraph-based learning.

This module contains the GAT (Graph Attention Network) model definition
used for weakly supervised learning on graphs.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(torch.nn.Module):
    """
    Graph Attention Network (GAT) for graph classification.
    
    This model uses two GAT layers followed by global mean pooling
    and a linear classifier. It returns both predictions and attention weights.
    """
    
    def __init__(self, num_features, num_classes, hidden_channels=64, heads=8, dropout=0.6):
        """
        Initialize the GAT model.
        
        Args:
            num_features: Number of node features
            num_classes: Number of output classes
            hidden_channels: Hidden layer size
            heads: Number of attention heads for first layer
            dropout: Dropout rate
        """
        super(GAT, self).__init__()
        torch.manual_seed(42)
        
        self.conv1 = GATConv(
            num_features, 
            hidden_channels, 
            heads=heads, 
            dropout=dropout
        )
        
        # Second layer combines heads by averaging (concat=False)
        self.conv2 = GATConv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=1, 
            concat=False, 
            dropout=dropout
        )
        
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object containing graph
        
        Returns:
            log_softmax: Log softmax predictions
            attn_weights: Attention weights from second layer
        """
        x, edge_index = data.x, data.edge_index

        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GAT layer with attention weights
        x, attn_weights = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Global mean pooling
        x = global_mean_pool(x, data.batch)

        # Classifier
        x = self.lin(x)

        return F.log_softmax(x, dim=1), attn_weights


def train_model(model, train_loader, optimizer, device, criterion=None):
    """
    Train the model for one epoch.
    
    Args:
        model: GAT model to train
        train_loader: Training data loader
        optimizer: Optimizer for training
        device: Device to run training on
        criterion: Loss function (if None, uses NLL loss)
    
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    if criterion is None:
        criterion = F.nll_loss
        
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out, _ = model(batch)
        
        if hasattr(criterion, '__call__') and criterion.__name__ == 'nll_loss':
            loss = criterion(out, batch.y)
        else:
            loss = criterion(out, batch.y)
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        if hasattr(criterion, '__call__') and criterion.__name__ == 'nll_loss':
            pred = out.argmax(dim=1)
        else:
            _, pred = torch.max(out, dim=1)
            
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def test_model(model, test_loader, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: GAT model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
    
    Returns:
        accuracy: Test accuracy
        predictions: List of predictions
        labels: List of true labels
    """
    model.eval()
    correct = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out, _ = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            predictions.extend(pred.tolist())
            labels.extend(data.y.tolist())
    
    accuracy = correct / len(test_loader.dataset)
    return accuracy, predictions, labels