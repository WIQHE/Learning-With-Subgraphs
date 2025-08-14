# Learning-With-Subgraphs

A comprehensive framework for weakly supervised learning on graphs using subgraph extraction and Graph Attention Networks (GAT). This project explores how extracting meaningful subgraphs can improve graph classification performance by focusing on the most informative parts of the graph structure.

## Overview

Graph neural networks have shown remarkable success in various graph-based tasks. However, many real-world scenarios involve weakly supervised learning where we only have graph-level labels without node-level supervision. This project addresses this challenge by:

1. **Extracting meaningful subgraphs** from original graphs using two different approaches
2. **Learning to identify important subgraphs** using attention mechanisms
3. **Aggregating subgraph predictions** for improved graph classification

## Key Features

- **Two Subgraph Extraction Methods**:
  - **BFS-based**: Uses Breadth-First Search traversal to create subgraphs that preserve local connectivity patterns
  - **Sliding Window**: Creates subgraphs by sliding a fixed-size window over nodes in sequential order

- **Attention-based Selection**: Uses Graph Attention Networks to identify the most important subgraphs for classification

- **Modular Design**: Clean, reusable code structure with separate modules for models, subgraph methods, and utilities

- **Comprehensive Evaluation**: Includes visualization tools, performance metrics, and comparison utilities

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.12.0+
- PyTorch Geometric 2.0.0+
- Other dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:
```bash
git clone https://github.com/WIQHE/Learning-With-Subgraphs.git
cd Learning-With-Subgraphs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch Geometric (if not automatically installed):
```bash
pip install torch-geometric
```

## Project Structure

```
Learning-With-Subgraphs/
├── src/                          # Source code
│   ├── models/                   # Neural network models
│   │   ├── __init__.py
│   │   └── gat.py               # Graph Attention Network implementation
│   ├── subgraph_methods/         # Subgraph extraction methods
│   │   ├── __init__.py
│   │   ├── bfs_subgraphs.py     # BFS-based subgraph extraction
│   │   └── sliding_window.py    # Sliding window subgraph extraction
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── data_utils.py        # Data loading and preprocessing
│       ├── training.py          # Training utilities
│       ├── evaluation.py        # Evaluation metrics and functions
│       └── visualization.py     # Visualization tools
├── examples/                     # Example usage scripts
│   ├── bfs_example.py           # BFS subgraph learning example
│   ├── sliding_window_example.py # Sliding window example
│   └── comparison_example.py    # Compare both methods
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── bfs_based.ipynb            # Original BFS notebook (legacy)
└── sliding_window.ipynb       # Original sliding window notebook (legacy)
```

## Methodology

### Graph Attention Network (GAT)

The core model uses a Graph Attention Network with:
- Multi-head attention mechanism for learning node representations
- Global mean pooling for graph-level representation
- Dropout for regularization
- Log-softmax output for classification

### Subgraph Extraction Methods

#### 1. BFS-based Subgraphs
- Performs Breadth-First Search from random starting nodes
- Creates subgraphs that preserve local connectivity patterns
- Ensures minimum node and edge requirements for meaningful subgraphs
- Covers the entire graph through iterative BFS traversals

#### 2. Sliding Window Subgraphs
- Slides a fixed-size window over nodes in sequential order
- Creates overlapping subgraphs with consistent size
- Maintains structural relationships within the window
- Provides systematic coverage of the graph

### Attention-based Subgraph Selection
- Computes attention scores for each subgraph during inference
- Selects top-k subgraphs based on attention weights
- Aggregates predictions from selected subgraphs for final classification

## Usage

### Quick Start

Run the BFS-based example:
```bash
python examples/bfs_example.py
```

Run the sliding window example:
```bash
python examples/sliding_window_example.py
```

Compare both methods:
```bash
python examples/comparison_example.py
```

### Using the Framework

#### Basic Usage

```python
import torch
from src.models.gat import GAT
from src.subgraph_methods.bfs_subgraphs import BFSSubgraphDataset
from src.utils.data_utils import load_dataset, split_dataset

# Load dataset
dataset = load_dataset("MSRC_21")
train_graphs, test_graphs = split_dataset(dataset)

# Create BFS subgraphs
train_subgraphs = BFSSubgraphDataset(train_graphs)

# Initialize model
model = GAT(
    num_features=dataset.num_node_features,
    num_classes=dataset.num_classes
)

# Train and evaluate (see examples for complete training loops)
```

#### Custom Subgraph Parameters

```python
# BFS with custom parameters
bfs_dataset = BFSSubgraphDataset(
    graphs,
    min_nodes=15,      # Minimum nodes per subgraph
    min_edges=10,      # Minimum edges per subgraph
    depth_limit=6      # Maximum BFS depth
)

# Sliding window with custom parameters
sw_dataset = SlidingWindowSubgraphDataset(
    graphs,
    window_size=50,    # Nodes per window
    step_size=10       # Step size for sliding
)
```

## Results and Performance

The framework has been tested on the MSRC_21 dataset with the following observations:

- **BFS subgraphs** tend to capture local connectivity patterns effectively
- **Sliding window** provides systematic coverage and consistent subgraph sizes
- **Attention mechanism** successfully identifies important subgraphs for classification
- **Top-k selection** improves performance by focusing on most relevant subgraphs

### Performance Metrics
- Classification accuracy on graph-level tasks
- Training time comparison between methods
- Subgraph attention score distributions
- Confusion matrices and detailed classification reports

## Visualization

The framework includes comprehensive visualization tools:
- Original graph visualization
- Highlighted subgraph visualization
- Top-k subgraph attention score visualization
- Training curve plots
- Confusion matrices

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is open source and available under the [MIT License](LICENSE).

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{learning-with-subgraphs,
  title={Learning-With-Subgraphs: A Framework for Weakly Supervised Graph Learning},
  author={Learning-With-Subgraphs Contributors},
  year={2024},
  url={https://github.com/WIQHE/Learning-With-Subgraphs}
}
```

## Acknowledgments

- PyTorch Geometric team for the excellent graph neural network library
- NetworkX developers for graph manipulation utilities
- The graph neural network research community for foundational work on attention mechanisms

---

For questions or issues, please open an issue on GitHub or contact the maintainers.
