#!/usr/bin/env python3
"""
Setup script for Learning-With-Subgraphs framework.
This script helps users get started with the framework.
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def test_installation():
    """Test if the installation works correctly."""
    print("Testing installation...")
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        
        # Test imports
        from models.gat import GAT
        from subgraph_methods.bfs_subgraphs import BFSSubgraphDataset
        from subgraph_methods.sliding_window import SlidingWindowSubgraphDataset
        from utils.data_utils import load_dataset
        
        print("✓ All imports successful!")
        
        # Test basic functionality
        import torch
        model = GAT(num_features=3, num_classes=2)
        print("✓ GAT model can be created!")
        
        return True
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def print_usage_examples():
    """Print usage examples."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE! 🎉")
    print("="*60)
    print("\nYou can now use the Learning-With-Subgraphs framework!")
    print("\nQuick start examples:")
    print("  1. Run BFS example:           python examples/bfs_example.py")
    print("  2. Run sliding window:       python examples/sliding_window_example.py")
    print("  3. Compare both methods:     python examples/comparison_example.py")
    print("  4. Test functionality:       python test_refactored.py")
    print("\nFor more information, see the README.md file.")
    print("="*60)


def main():
    """Main setup function."""
    print("=== Learning-With-Subgraphs Setup ===\n")
    
    # Check if we're in the right directory
    if not os.path.exists("README.md") or not os.path.exists("src"):
        print("❌ Please run this script from the project root directory")
        return 1
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during requirements installation")
        return 1
    
    # Test installation
    if not test_installation():
        print("\n❌ Setup failed during installation test")
        return 1
    
    # Print usage examples
    print_usage_examples()
    
    return 0


if __name__ == "__main__":
    exit(main())