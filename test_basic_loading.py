#!/usr/bin/env python3
"""
Simple test to check if graph loading and basic structures work.
"""

import json
import pickle
from pathlib import Path
import torch

print("Testing basic data loading...")

# Test 1: Load test graph IDs
print("\n1. Loading test graph IDs...")
with open('outputs/test_graph_ids.json', 'r') as f:
    test_graph_ids = json.load(f)['graph_ids']
print(f"   ✓ Found {len(test_graph_ids)} test graph IDs")

# Test 2: Load a sample graph
print("\n2. Loading first test graph...")
graph_dir = Path("virtual_graphs/data/all_graphs/raw_graphs")
graph_id = test_graph_ids[0]
graph_path = graph_dir / f"graph_{graph_id}.pkl"

if not graph_path.exists():
    print(f"   ✗ Graph file not found: {graph_path}")
else:
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    print(f"   ✓ Loaded graph {graph_id}")
    print(f"     Nodes: {len(G.nodes())}")
    print(f"     Edges: {len(G.edges())}")

    # Check edge attributes
    sample_edge = list(G.edges())[0]
    print(f"     Sample edge attributes: {G[sample_edge[0]][sample_edge[1]]}")

# Test 3: Check GNN checkpoint
print("\n3. Checking GNN checkpoint...")
gnn_path = Path('checkpoints/gnn_model.pt')
if not gnn_path.exists():
    print(f"   ✗ Not found: {gnn_path}")
else:
    checkpoint = torch.load(gnn_path, map_location='cpu', weights_only=False)
    print(f"   ✓ Loaded checkpoint")
    if isinstance(checkpoint, dict):
        print(f"     Keys: {list(checkpoint.keys())}")
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Check layer dimensions
    print(f"     Checking layer dimensions:")
    for key in list(state_dict.keys())[:10]:
        if 'weight' in key:
            print(f"       {key}: {state_dict[key].shape}")

# Test 4: Check SAE checkpoint
print("\n4. Checking SAE checkpoint...")
sae_path = Path('checkpoints/sae_latent128_k16.pt')
if not sae_path.exists():
    print(f"   ✗ Not found: {sae_path}")
else:
    sae_checkpoint = torch.load(sae_path, map_location='cpu', weights_only=False)
    print(f"   ✓ Loaded SAE checkpoint")
    if 'model_state_dict' in sae_checkpoint:
        sae_state = sae_checkpoint['model_state_dict']
        print(f"     State dict keys: {list(sae_state.keys())[:5]}")

print("\n✓ All basic checks passed!")
print("\nNext: Run the full comparison with debug output:")
print("  python compare_sae_vs_gnnexplainer.py")
