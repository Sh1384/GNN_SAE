#!/usr/bin/env python3
"""
Quick test to verify models and data load correctly before running full comparison.
"""

import sys
import torch
from pathlib import Path

print("="*70)
print("TESTING COMPARISON SETUP")
print("="*70)

# Test 1: Check CUDA availability
print("\n1. Checking CUDA...")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# Test 2: Check GNN checkpoint
print("\n2. Checking GNN checkpoint...")
gnn_path = Path('checkpoints/gnn_model.pt')
if not gnn_path.exists():
    print(f"   ✗ Not found: {gnn_path}")
    sys.exit(1)
else:
    print(f"   ✓ Found: {gnn_path}")
    ckpt = torch.load(gnn_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        print(f"   Format: Dictionary with keys: {list(ckpt.keys())}")
    else:
        print(f"   Format: State dict directly (type: {type(ckpt).__name__})")

# Test 3: Check SAE checkpoint
print("\n3. Checking SAE checkpoint...")
sae_path = Path('checkpoints/sae_latent128_k16.pt')
if not sae_path.exists():
    print(f"   ✗ Not found: {sae_path}")
    print("   Available SAE checkpoints:")
    for p in Path('checkpoints').glob('sae_*.pt'):
        print(f"     - {p.name}")
    sys.exit(1)
else:
    print(f"   ✓ Found: {sae_path}")

# Test 4: Check test graph IDs
print("\n4. Checking test graph IDs...")
test_ids_path = Path('outputs/test_graph_ids.json')
if not test_ids_path.exists():
    print(f"   ✗ Not found: {test_ids_path}")
    sys.exit(1)
else:
    import json
    with open(test_ids_path) as f:
        test_ids = json.load(f)['graph_ids']
    print(f"   ✓ Found {len(test_ids)} test graph IDs")

# Test 5: Check activations
print("\n5. Checking layer 3 activations...")
act_dir = Path('outputs/activations/layer3_new/test')
if not act_dir.exists():
    print(f"   ✗ Not found: {act_dir}")
    sys.exit(1)
else:
    act_files = list(act_dir.glob('graph_*.pt'))
    print(f"   ✓ Found {len(act_files)} activation files")

# Test 6: Check graph files
print("\n6. Checking graph files...")
graph_dir = Path('virtual_graphs/data/all_graphs/raw_graphs')
if not graph_dir.exists():
    print(f"   ✗ Not found: {graph_dir}")
    sys.exit(1)
else:
    graph_files = list(graph_dir.glob('graph_*.pkl'))
    print(f"   ✓ Found {len(graph_files)} graph files")

# Test 7: Check metadata with degree features
print("\n7. Checking metadata with degree features...")
import pandas as pd
meta_path = Path('virtual_graphs/data/all_graphs/graph_motif_metadata/graph_0_metadata.csv')
if not meta_path.exists():
    print(f"   ✗ Not found: {meta_path}")
    sys.exit(1)
else:
    df = pd.read_csv(meta_path, index_col=0)
    print(f"   ✓ Columns: {list(df.columns)}")
    if 'degree' in df.columns:
        print(f"   ✓ Degree features present")
    else:
        print(f"   ⚠ Degree features missing - run compute_node_features.py")

# Test 8: Load models to verify
print("\n8. Loading models...")
try:
    from sparse_autoencoder import SparseAutoencoder
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv

    class GCNModel(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=88, output_dim=1, dropout=0.3):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim, normalize=False)
            self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
            self.conv3 = GCNConv(hidden_dim, 64, normalize=False)
            self.conv4 = GCNConv(64, output_dim, normalize=False)
            self.dropout = dropout

    # Load GNN
    gnn = GCNModel()
    gnn_ckpt = torch.load(gnn_path, map_location='cpu', weights_only=False)
    if isinstance(gnn_ckpt, dict) and 'model_state_dict' in gnn_ckpt:
        gnn.load_state_dict(gnn_ckpt['model_state_dict'])
    else:
        gnn.load_state_dict(gnn_ckpt)
    print("   ✓ GNN loaded successfully")

    # Load SAE
    sae = SparseAutoencoder(input_dim=64, latent_dim=128, k=16)
    sae_ckpt = torch.load(sae_path, map_location='cpu', weights_only=False)
    sae.load_state_dict(sae_ckpt['model_state_dict'])
    print("   ✓ SAE loaded successfully")

except Exception as e:
    print(f"   ✗ Error loading models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ ALL CHECKS PASSED")
print("="*70)
print("\nYou're ready to run the comparison:")
print("  python compare_sae_vs_gnnexplainer.py")
print("\nOr run feature identification first:")
print("  python identify_top_sae_features.py")
