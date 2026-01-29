#!/usr/bin/env python3
"""
Generate Layer 3 Activations for Mixed-Motif Graphs

Loads mixed-motif graphs (4000-4999) and generates layer3 activations
using the current GNN checkpoint (checkpoints/gnn_model.pt).

Saves to: outputs/activations/layer2_new/mixed/

Usage:
    python generate_mixed_motif_activations.py
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from pathlib import Path
import pickle
import networkx as nx
from tqdm import tqdm
from typing import Optional

# Import GCN model from gnn_train_copy (in parent directory)
import sys
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
from gnn_train_copy import GCNModel


def load_graph_from_pickle(graph_path: Path) -> Optional[Data]:
    """
    Load graph from pickle and convert to PyG Data object.

    Args:
        graph_path: Path to pickled NetworkX graph

    Returns:
        PyG Data object or None if loading fails
    """
    try:
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        # Extract features
        node_ids = sorted(G.nodes())
        x = torch.tensor([[G.nodes[n].get('input', 0.0),
                          G.nodes[n].get('value', 0.0)] for n in node_ids],
                        dtype=torch.float32)

        # Extract edges
        edge_index = []
        edge_attr = []
        for u, v, data in G.edges(data=True):
            edge_index.append([node_ids.index(u), node_ids.index(v)])
            edge_attr.append(data.get('weight', 1.0))

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float32)

        # Extract target values
        y = torch.tensor([G.nodes[n].get('value', 0.0) for n in node_ids],
                        dtype=torch.float32)

        # Create mask (all nodes)
        mask = torch.ones(len(node_ids), dtype=torch.bool)

        # Extract graph ID from path
        graph_id = int(graph_path.stem.replace('graph_', ''))

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                   y=y, mask=mask, graph_id=graph_id)

        return data

    except Exception as e:
        print(f"Error loading {graph_path}: {e}")
        return None


def load_gnn_model(checkpoint_path: Path, device: str = 'cuda') -> nn.Module:
    """
    Load GNN model from checkpoint.

    Args:
        checkpoint_path: Path to GNN checkpoint
        device: Device to load model on

    Returns:
        Loaded GNN model
    """
    # Load checkpoint to infer architecture
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)

    # Infer dimensions from state dict
    hidden_dim1 = state_dict['conv1.bias'].shape[0]  # Should be 88
    hidden_dim2 = state_dict['conv2.bias'].shape[0]  # Should be 64

    print(f"Detected architecture: hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}")

    # Create model
    model = GCNModel(
        input_dim=2,
        hidden_dim=hidden_dim1,
        output_dim=1,
        dropout=0.2
    )

    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded GNN model from {checkpoint_path}")

    return model


def extract_layer2_activation(model: GCNModel, data: Data, device: str = 'cuda') -> torch.Tensor:
    """
    Extract layer 3 activation for a single graph.

    Args:
        model: GNN model
        data: PyG Data object
        device: Device

    Returns:
        Layer 2 activation tensor (n_nodes, 64)
    """
    data = data.to(device)

    with torch.no_grad():
        # Forward pass with activation storage
        _ = model.forward(data, store_activations=True)

        # Get layer3 activation (64-dim)
        layer3_act = model.layer2_activations.cpu()

    return layer3_act


def main():
    """Main function to generate activations for mixed-motif graphs."""

    print("="*70)
    print("GENERATING ACTIVATIONS FOR MIXED-MOTIF GRAPHS")
    print("="*70)

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    checkpoint_path = Path("checkpoints/gnn_model.pt")
    graphs_dir = Path("virtual_graphs/data/all_graphs/raw_graphs")
    output_dir = Path("outputs/activations/layer2_new/mixed")

    # Validate paths
    if not checkpoint_path.exists():
        print(f"ERROR: GNN checkpoint not found at {checkpoint_path}")
        return

    if not graphs_dir.exists():
        print(f"ERROR: Graphs directory not found at {graphs_dir}")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GNN model
    model = load_gnn_model(checkpoint_path, device)

    # Find all mixed-motif graphs (4000-4999)
    mixed_graph_files = sorted(graphs_dir.glob("graph_4[0-9][0-9][0-9].pkl"))

    if len(mixed_graph_files) == 0:
        print(f"ERROR: No mixed-motif graphs found in {graphs_dir}")
        print("Expected files like: graph_4000.pkl, graph_4001.pkl, ..., graph_4999.pkl")
        return

    print(f"\nFound {len(mixed_graph_files)} mixed-motif graphs")
    print(f"Range: {mixed_graph_files[0].stem} to {mixed_graph_files[-1].stem}")

    # Process each graph
    success_count = 0
    failed_count = 0

    print("\nGenerating activations...")
    for graph_file in tqdm(mixed_graph_files, desc="Processing graphs"):
        # Load graph
        data = load_graph_from_pickle(graph_file)

        if data is None:
            failed_count += 1
            continue

        # Extract layer3 activation
        try:
            layer3_act = extract_layer2_activation(model, data, device)

            # Save activation
            graph_id = data.graph_id
            output_file = output_dir / f"graph_{graph_id}.pt"
            torch.save(layer3_act, output_file)

            success_count += 1

        except Exception as e:
            print(f"\nError processing {graph_file.stem}: {e}")
            failed_count += 1

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total graphs processed: {len(mixed_graph_files)}")
    print(f"Successfully saved: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"\nActivations saved to: {output_dir}/")
    print(f"Expected shape per graph: (n_nodes, 64)")

    # Verify a sample
    if success_count > 0:
        sample_file = output_dir / "graph_4500.pt"
        if sample_file.exists():
            sample_act = torch.load(sample_file, weights_only=True)
            print(f"\nSample verification (graph_4500):")
            print(f"  Shape: {sample_act.shape}")
            print(f"  Mean: {sample_act.mean():.4f}")
            print(f"  Std: {sample_act.std():.4f}")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
