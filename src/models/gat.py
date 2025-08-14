"""Graph Attention Network (GAT) implementation for graph classification."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GAT(torch.nn.Module):
    """Graph Attention Network for graph classification.
    
    This model uses multi-head attention to learn node representations
    and performs graph-level classification using global mean pooling.
    
    Args:
        num_features (int): Number of input node features
        num_classes (int): Number of output classes
        hidden_channels (int): Size of hidden layers (default: 64)
        heads (int): Number of attention heads in first layer (default: 8)
        dropout (float): Dropout rate (default: 0.6)
    """
    
    def __init__(self, num_features, num_classes, hidden_channels=64, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        torch.manual_seed(42)
        
        # First GAT layer with multiple heads
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=dropout)
        
        # Second GAT layer that combines heads (concat=False means averaging)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, 
                            concat=False, dropout=dropout)
        
        # Final linear layer for classification
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):
        """Forward pass through the network.
        
        Args:
            data: PyTorch Geometric data object containing:
                - x: Node features [num_nodes, num_features]
                - edge_index: Graph connectivity [2, num_edges]
                - batch: Batch vector [num_nodes] (for batched graphs)
        
        Returns:
            tuple: (log_softmax_output, attention_weights)
                - log_softmax_output: Log probabilities for each class
                - attention_weights: Attention weights from second layer
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

        # Global mean pooling to get graph-level representation
        x = global_mean_pool(x, data.batch)

        # Final classification layer
        x = self.lin(x)

        return F.log_softmax(x, dim=1), attn_weights