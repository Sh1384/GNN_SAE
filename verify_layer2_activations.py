#!/usr/bin/env python3
"""
Verify Layer 2 Activations

This script checks if layer 2 activations exist and have the correct dimensions (80-dim).
"""

import torch
from pathlib import Path

def check_activations(split_name: str):
    """Check activations for a given split."""
    activation_dir = Path(f"outputs/activations/layer2_new/{split_name}")

    if not activation_dir.exists():
        print(f"❌ {split_name.upper()}: Directory not found: {activation_dir}")
        return False

    activation_files = list(activation_dir.glob("graph_*.pt"))

    if len(activation_files) == 0:
        print(f"❌ {split_name.upper()}: No activation files found")
        return False

    print(f"✓ {split_name.upper()}: Found {len(activation_files)} activation files")

    # Check dimensions of first file
    try:
        first_file = activation_files[0]
        activations = torch.load(first_file, weights_only=True)

        num_nodes, dim = activations.shape

        if dim == 80:
            print(f"  ✓ Dimensions correct: {activations.shape} (expected: [num_nodes, 80])")
        else:
            print(f"  ❌ Dimensions incorrect: {activations.shape} (expected: [num_nodes, 80])")
            print(f"     Found {dim}-dim instead of 80-dim")
            return False

        # Check a few more files to be sure
        for test_file in activation_files[1:min(5, len(activation_files))]:
            act = torch.load(test_file, weights_only=True)
            if act.shape[1] != 80:
                print(f"  ❌ Inconsistent dimensions in {test_file.name}: {act.shape}")
                return False

        print(f"  ✓ All sampled files have 80-dim activations")
        return True

    except Exception as e:
        print(f"  ❌ Error loading activation file: {e}")
        return False


def main():
    print("="*70)
    print("VERIFYING LAYER 2 ACTIVATIONS")
    print("="*70)
    print()

    splits = ['train', 'val', 'test']
    results = {}

    for split in splits:
        results[split] = check_activations(split)
        print()

    print("="*70)
    print("SUMMARY")
    print("="*70)

    all_good = all(results.values())

    if all_good:
        print("✓ All checks passed!")
        print()
        print("Layer 2 activations are ready to use.")
        print("You can now run: ./switch_to_layer2.sh")
    else:
        print("❌ Some checks failed!")
        print()
        print("Layer 2 activations are missing or have wrong dimensions.")
        print()
        print("To generate layer 2 activations, run:")
        print("  python3 gnn_train_copy.py")
        print()
        print("This will:")
        print("  1. Load the trained GNN model from checkpoints/gnn_model.pt")
        print("  2. Extract layer 1, 2, and 3 activations for all graphs")
        print("  3. Save them to outputs/activations/layer{1,2,3}_new/")

    print()
    return 0 if all_good else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
