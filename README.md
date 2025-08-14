# Learning-With-Subgraphs

A comprehensive framework for **weakly supervised learning on graphs** using subgraph-based approaches. This project explores two different subgraph extraction methodologies to improve graph classification performance by identifying and leveraging meaningful subgraph structures.

## Overview

Traditional graph neural networks process entire graphs for classification tasks. This project investigates how breaking graphs into smaller, meaningful subgraphs can enhance learning performance in weakly supervised settings. We implement and compare two distinct approaches:

1. **BFS-based Subgraph Extraction**: Uses breadth-first search to create connected subgraphs
2. **Sliding Window Subgraph Extraction**: Creates fixed-size subgraphs using a sliding window approach

Both methods are combined with Graph Attention Networks (GAT) to leverage attention mechanisms for identifying the most important subgraphs for classification.

## Key Features

- **Dual Methodology**: Implementation of both BFS and sliding window subgraph extraction
- **Attention-Based Selection**: Uses GAT attention weights to identify top-k most relevant subgraphs
- **Modular Design**: Clean, reusable code structure with separate modules for different components
- **Comprehensive Evaluation**: Includes visualization tools and detailed performance metrics
- **Weakly Supervised Learning**: Designed for scenarios with limited supervision

## Installation

1. Clone the repository:
```bash
git clone https://github.com/WIQHE/Learning-With-Subgraphs.git
cd Learning-With-Subgraphs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the **MSRC_21** dataset from the TU Dataset collection, which contains:
- Graph-level classification tasks
- Multiple node features per graph
- 20 different object classes
- Varying graph sizes and structures

## Methodology

### 1. BFS-Based Subgraph Extraction

The BFS approach creates subgraphs by:
- Starting from random unvisited nodes
- Performing breadth-first traversal up to a specified depth
- Ensuring minimum node and edge requirements
- Preserving graph connectivity within subgraphs

**Key Parameters:**
- `depth_limit`: Maximum BFS traversal depth (default: 8)
- `min_nodes`: Minimum nodes per subgraph (default: 10)
- `min_edges`: Minimum edges per subgraph (default: 8)

### 2. Sliding Window Subgraph Extraction

The sliding window approach:
- Creates fixed-size subgraphs by moving a window across node sequences
- Maintains consistent subgraph sizes
- Captures local neighborhood structures
- Handles disconnected components with self-loops

**Key Parameters:**
- `window_size`: Number of nodes in each subgraph (default: 62)
- `step_size`: Step size for window movement (default: 5)

### 3. Graph Attention Network (GAT) Architecture

Our GAT model includes:
- Two GAT convolutional layers with multi-head attention
- ELU activation functions
- Dropout for regularization
- Global mean pooling for graph-level representation
- Linear classifier for final predictions

### 4. Attention-Based Subgraph Selection

For evaluation, we:
1. Generate subgraphs using either method
2. Compute attention scores for each subgraph
3. Select top-k subgraphs based on attention weights
4. Aggregate predictions using mean pooling
5. Make final classification decisions

## Usage

### Basic Example

```python
from utils import create_bfs_subgraphs, create_sliding_window_subgraphs
from models import GAT, train_model, test_model
from datasets import SubgraphDataset, create_dataset_splits
import torch_geometric.datasets as datasets

# Load dataset
dataset = datasets.TUDataset(root='dataset', name='MSRC_21')

# Create train/test splits
train_data, test_data = create_dataset_splits(dataset)

# BFS-based approach
train_subgraphs = SubgraphDataset(train_data, create_bfs_subgraphs)

# Initialize and train model
model = GAT(dataset.num_node_features, dataset.num_classes)
# ... training code
```

### Running Experiments

1. **BFS-based experiment**: Open and run `bfs_based.ipynb`
2. **Sliding window experiment**: Open and run `sliding_window.ipynb`

Both notebooks include:
- Data loading and preprocessing
- Model training with progress tracking
- Evaluation with attention-based subgraph selection
- Visualization of results and top-k subgraphs

## Results and Evaluation

The framework provides comprehensive evaluation including:

- **Classification Accuracy**: Standard accuracy metrics on test data
- **Confusion Matrix**: Detailed per-class performance analysis
- **Classification Report**: Precision, recall, and F1-scores
- **Attention Visualization**: Visual representation of top-k selected subgraphs
- **Training Curves**: Loss and accuracy progression over epochs

## File Structure

```
Learning-With-Subgraphs/
├── README.md                 # This file
├── requirements.txt          # Project dependencies
├── utils.py                 # Subgraph extraction utilities
├── models.py                # GAT model definitions
├── datasets.py              # Dataset handling classes
├── visualization.py         # Plotting and visualization tools
├── bfs_based.ipynb          # BFS methodology experiment
└── sliding_window.ipynb     # Sliding window methodology experiment
```

## Key Advantages

1. **Improved Interpretability**: Attention-based subgraph selection shows which parts of graphs are most important
2. **Scalability**: Subgraph-based approach can handle large graphs more efficiently
3. **Flexibility**: Two different extraction methods suit different graph types
4. **Weakly Supervised**: Requires only graph-level labels, not node-level annotations

## Future Work

- Integration of more sophisticated subgraph extraction methods
- Extension to other graph neural network architectures
- Application to larger-scale graph datasets
- Investigation of hierarchical subgraph structures

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is available under the MIT License. See LICENSE file for details.
