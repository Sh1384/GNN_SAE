#!/usr/bin/env python3
"""
Check SAE Checkpoint Compatibility

Verifies that existing SAE checkpoints are compatible with the current INPUT_DIM.
"""

import torch
from pathlib import Path
import sys

# Import current INPUT_DIM from pipeline
sys.path.insert(0, '.')
from run_sae_pipeline_multi_gpu import INPUT_DIM

def check_checkpoint(checkpoint_path: Path, expected_input_dim: int):
    """Check if checkpoint has correct input dimension."""
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' not in checkpoint:
            return None, "No model_state_dict found"

        state_dict = checkpoint['model_state_dict']

        # Check encoder weight shape to determine input_dim
        # All SAE variants have encoder.weight as first layer
        if 'encoder.weight' in state_dict:
            encoder_weight = state_dict['encoder.weight']
            input_dim = encoder_weight.shape[1]  # [latent_dim, input_dim]

            if input_dim == expected_input_dim:
                return True, f"{input_dim}-dim (correct)"
            else:
                return False, f"{input_dim}-dim (expected {expected_input_dim}-dim)"
        else:
            # Check for switch SAE expert encoders
            expert_keys = [k for k in state_dict.keys() if k.startswith('experts.0.encoder.weight')]
            if expert_keys:
                encoder_weight = state_dict[expert_keys[0]]
                input_dim = encoder_weight.shape[1]

                if input_dim == expected_input_dim:
                    return True, f"{input_dim}-dim (correct)"
                else:
                    return False, f"{input_dim}-dim (expected {expected_input_dim}-dim)"

            return None, "Could not determine input_dim"

    except Exception as e:
        return None, f"Error: {str(e)}"


def main():
    print("="*70)
    print("CHECKING SAE CHECKPOINT COMPATIBILITY")
    print("="*70)
    print(f"\nCurrent INPUT_DIM in pipeline: {INPUT_DIM}")
    print()

    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        print("No checkpoints directory found.")
        return 0

    sae_checkpoints = list(ckpt_dir.glob("sae_*.pt"))

    if len(sae_checkpoints) == 0:
        print("No SAE checkpoints found.")
        print("✓ Checkpoints directory is clean - ready for training!")
        return 0

    print(f"Found {len(sae_checkpoints)} SAE checkpoint files\n")

    compatible = []
    incompatible = []
    unknown = []

    for ckpt_path in sorted(sae_checkpoints):
        is_compatible, msg = check_checkpoint(ckpt_path, INPUT_DIM)

        if is_compatible is None:
            status = "?"
            symbol = "⚠"
            unknown.append((ckpt_path.name, msg))
        elif is_compatible:
            status = "✓"
            symbol = "✓"
            compatible.append(ckpt_path.name)
        else:
            status = "✗"
            symbol = "✗"
            incompatible.append((ckpt_path.name, msg))

        print(f"{symbol} {ckpt_path.name}: {msg}")

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Compatible:   {len(compatible)} checkpoints")
    print(f"Incompatible: {len(incompatible)} checkpoints")
    print(f"Unknown:      {len(unknown)} checkpoints")
    print()

    if len(incompatible) > 0:
        print("❌ INCOMPATIBLE CHECKPOINTS FOUND!")
        print()
        print("These checkpoints were trained with a different INPUT_DIM")
        print("and will cause errors if used with the current pipeline.")
        print()
        print("Solution: Backup and remove old checkpoints")
        print("  ./backup_old_checkpoints.sh")
        print()
        print("Then retrain with correct INPUT_DIM:")
        print("  python3 run_sae_pipeline_multi_gpu.py --phase 1")
        return 1

    elif len(compatible) == len(sae_checkpoints):
        print("✓ All checkpoints are compatible!")
        print()
        print("You can proceed with Phases 2-5:")
        print("  python3 run_sae_pipeline_multi_gpu.py --phase 2")
        return 0

    else:
        print("⚠ Some checkpoints could not be verified")
        print()
        print("Recommend backing up and retraining to be safe:")
        print("  ./backup_old_checkpoints.sh")
        print("  python3 run_sae_pipeline_multi_gpu.py --phase 1")
        return 1


if __name__ == "__main__":
    sys.exit(main())
