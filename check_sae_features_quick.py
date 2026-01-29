#!/usr/bin/env python3
"""
Quick check: Which SAE features are actually active?
"""

import torch
import json
from pathlib import Path
from sparse_autoencoder import SparseAutoencoder

print("="*70)
print("QUICK SAE FEATURE CHECK")
print("="*70)

# Load SAE
sae = SparseAutoencoder(input_dim=64, latent_dim=128, k=16)
checkpoint = torch.load('checkpoints/sae_latent128_k16.pt', weights_only=False)
sae.load_state_dict(checkpoint['model_state_dict'])
sae.eval()

print("\n✓ Loaded SAE (latent_dim=128, k=16)")

# Load a few test activations
activation_dir = Path("outputs/activations/layer3_new/test")
with open('outputs/test_graph_ids.json', 'r') as f:
    test_ids = json.load(f)['graph_ids']

print(f"\nChecking first 20 test graphs...")

all_activations_count = torch.zeros(128)
all_max_activations = torch.zeros(128)

for graph_id in test_ids[:20]:
    act_file = activation_dir / f"graph_{graph_id}.pt"
    if not act_file.exists():
        continue

    # Load activations
    h = torch.load(act_file, weights_only=True)

    # Encode
    with torch.no_grad():
        z_dense = sae.encoder(h)
        z_dense = torch.relu(z_dense)  # Pre-TopK

        # Count active (>0.01) per feature
        active_per_feature = (z_dense > 0.01).sum(dim=0)
        all_activations_count += active_per_feature.cpu()

        # Track max activation per feature
        max_per_feature = z_dense.max(dim=0)[0]
        all_max_activations = torch.maximum(all_max_activations, max_per_feature.cpu())

# Show results
print("\n" + "="*70)
print("FEATURE ACTIVATION SUMMARY (first 20 graphs)")
print("="*70)

active_features = (all_max_activations > 0.01).sum().item()
print(f"\nActive features (max > 0.01): {active_features}/128")

# Show top 20 most active features
top_indices = torch.argsort(all_max_activations, descending=True)[:20]

print("\nTop 20 most active features:")
print(f"{'Feature':<10} {'Max Activation':>15} {'Total Active Nodes':>20}")
print("-" * 48)
for idx in top_indices:
    print(f"z{idx.item():<9} {all_max_activations[idx].item():>15.4f} {all_activations_count[idx].item():>20.0f}")

# Check the current feature indices
current_features = [111, 119]  # z112 and z120 in 0-indexed
print("\n" + "="*70)
print("CURRENT FEATURE INDICES (from comparison script)")
print("="*70)
for idx in current_features:
    print(f"\nz{idx} (z{idx+1} in 1-indexed):")
    print(f"  Max activation: {all_max_activations[idx].item():.4f}")
    print(f"  Total active nodes: {all_activations_count[idx].item():.0f}")
    if all_max_activations[idx].item() < 0.01:
        print(f"  ⚠ WARNING: This feature has very low activation!")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("\nThe current features (z112, z120) may not be optimal.")
print("Run this to find the best features for each motif:")
print("  python identify_top_sae_features.py")
print("\nThen update the 'targets' dictionary in compare_sae_vs_gnnexplainer.py")
print("with the feature indices from the output.")
