#!/usr/bin/env python3
"""
Quick test to verify GNNExplainer comparison dimensions are compatible.
Tests that layer 2 (80-dim) activations work with new SAE models.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add sae directory to path
sys.path.insert(0, str(Path(__file__).parent / 'sae'))
from sparse_autoencoder import TopKSAE, GatedSAE, JumpReLUSAE, SwitchSAE

# Import GCN model from comparison script
sys.path.insert(0, str(Path(__file__).parent / 'sae'))
from compare_sae_vs_gnnexplainer import GCNModel

def test_dimension_compatibility():
    """Test that GNN layer 2 activations (80-dim) work with SAE models."""
    print("="*70)
    print("TESTING DIMENSION COMPATIBILITY")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}\n")

    # Create test data
    num_nodes = 10
    num_edges = 20
    x = torch.randn(num_nodes, 2)  # 2D node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.rand(num_edges, 1)

    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(device)

    # Load GNN model
    print("1. Testing GNN activation extraction...")
    gnn = GCNModel(input_dim=2, hidden_dim=80, output_dim=1, dropout=0.2)

    # Try to load checkpoint, but don't fail if it doesn't exist
    ckpt_path = Path('checkpoints/gnn_model.pt')
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        gnn.load_state_dict(checkpoint)
        print("   ✓ Loaded GNN checkpoint")
    else:
        print("   ⚠ GNN checkpoint not found, using random weights")

    gnn.to(device)
    gnn.eval()

    # Extract activations
    with torch.no_grad():
        activations = gnn.get_intermediate_activations(data)

    print(f"   ✓ Layer 2 activations shape: {activations.shape}")

    if activations.shape[1] != 80:
        print(f"   ✗ ERROR: Expected 80-dim, got {activations.shape[1]}-dim")
        return False

    print("   ✓ Correct dimension (80)")

    # Test each SAE variant
    variants = {
        'topk': (TopKSAE, {'input_dim': 80, 'latent_dim': 256, 'k': 16}),
        'gated': (GatedSAE, {'input_dim': 80, 'latent_dim': 256, 'sparsity_coef': 0.001}),
        'jumprelu': (JumpReLUSAE, {'input_dim': 80, 'latent_dim': 256, 'threshold_init': 0.01, 'bandwidth': 0.01}),
        'switch': (SwitchSAE, {'input_dim': 80, 'num_experts': 4, 'latent_per_expert': 64, 'k_per_expert': 8}),
    }

    print("\n2. Testing SAE variant compatibility...")

    for variant_name, (SAEClass, config) in variants.items():
        print(f"\n   Testing {variant_name.upper()}...")

        # Create SAE model
        sae = SAEClass(**config).to(device)
        sae.eval()

        # Test forward pass
        with torch.no_grad():
            try:
                if hasattr(sae, 'encoder'):
                    z = sae.encoder(activations)
                    z = torch.relu(z)
                else:
                    z = sae.encode(activations)

                print(f"      ✓ SAE forward pass successful")
                print(f"      ✓ Latent shape: {z.shape}")

                # Test gradient computation
                activations_grad = activations.clone().requires_grad_(True)
                if hasattr(sae, 'encoder'):
                    z_grad = sae.encoder(activations_grad)
                    z_grad = torch.relu(z_grad)
                else:
                    z_grad = sae.encode(activations_grad)

                # Compute gradient for first feature
                target = z_grad[:, 0].sum()
                grad = torch.autograd.grad(target, activations_grad, create_graph=False)[0]

                print(f"      ✓ Gradient computation successful")
                print(f"      ✓ Gradient shape: {grad.shape}")

            except Exception as e:
                print(f"      ✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                return False

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\nDimensions are compatible. You can now run:")
    print("  python3 sae/compare_sae_vs_gnnexplainer.py --all")

    return True


if __name__ == "__main__":
    success = test_dimension_compatibility()
    sys.exit(0 if success else 1)
