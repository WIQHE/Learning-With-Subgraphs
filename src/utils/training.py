"""Training and testing utilities for graph neural networks."""

import torch
import torch.nn.functional as F


def train_model(model, train_loader, optimizer, device, criterion=None):
    """Train model for one epoch.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        optimizer: PyTorch optimizer
        device: Device to run on (cuda/cpu)
        criterion: Loss function (if None, uses NLL loss)
    
    Returns:
        tuple: (average_loss, accuracy)
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
        
        # Forward pass
        if hasattr(model, 'forward') and 'attn' in str(model.forward):
            # Model returns attention weights
            out, _ = model(batch)
        else:
            out = model(batch)
        
        # Compute loss
        if hasattr(criterion, '__call__') and criterion != F.nll_loss:
            loss = criterion(out, batch.y)
        else:
            loss = F.nll_loss(out, batch.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        if out.dim() > 1:  # Multi-class classification
            _, predicted = torch.max(out, dim=1)
        else:  # Binary classification
            predicted = (out > 0.5).long()
            
        correct += (predicted == batch.y).sum().item()
        total += batch.y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def test_model(model, test_loader, device):
    """Test model and return accuracy and predictions.
    
    Args:
        model: PyTorch model to test
        test_loader: DataLoader for test data
        device: Device to run on (cuda/cpu)
    
    Returns:
        tuple: (accuracy, predictions, true_labels)
    """
    model.eval()
    correct = 0
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            if hasattr(model, 'forward') and 'attn' in str(model.forward):
                # Model returns attention weights
                out, _ = model(batch)
            else:
                out = model(batch)
            
            # Get predictions
            if out.dim() > 1:  # Multi-class classification
                pred = out.argmax(dim=1)
            else:  # Binary classification
                pred = (out > 0.5).long()
                
            correct += (pred == batch.y).sum().item()
            predictions.extend(pred.tolist())
            labels.extend(batch.y.tolist())
    
    accuracy = correct / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0
    return accuracy, predictions, labels


def train_model_epochs(model, train_loader, test_loader, optimizer, device, 
                      num_epochs=200, criterion=None, verbose=True):
    """Train model for multiple epochs and track performance.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: PyTorch optimizer
        device: Device to run on (cuda/cpu)
        num_epochs (int): Number of epochs to train
        criterion: Loss function (if None, uses NLL loss)
        verbose (bool): Whether to print progress
    
    Returns:
        dict: Training history with losses and accuracies
    """
    train_acc_list = []
    test_acc_list = []
    loss_list = []
    
    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        loss, train_acc = train_model(model, train_loader, optimizer, device, criterion)
        
        # Test the model
        test_acc, predictions, true_labels = test_model(model, test_loader, device)
        
        # Store metrics
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        loss_list.append(loss)
        
        # Print progress
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    return {
        'train_accuracies': train_acc_list,
        'test_accuracies': test_acc_list,
        'losses': loss_list,
        'final_predictions': predictions,
        'final_true_labels': true_labels
    }