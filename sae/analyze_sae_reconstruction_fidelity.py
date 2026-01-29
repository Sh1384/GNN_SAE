#!/usr/bin/env python3
"""
SAE Reconstruction Fidelity Analysis using PCA Component Histograms

Analyzes how faithfully each SAE variant reconstructs the native GNN activation space
by comparing PCA component distributions.

Key Insight: A generator (or SAE) may have low point-wise error but still miss
important aspects of the data distribution. PCA histograms instantly reveal:
  - Distribution shape preservation (multimodality, skewness, tails)
  - Variance in different directions
  - Systematic biases or mode collapse

Usage:
    python analyze_sae_reconstruction_fidelity.py \\
        --variant topk \\
        --latent_dim 512 \\
        --k 8 \\
        --use-mixed-motifs  # Optional: use mixed-motif graphs instead of test set
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import argparse
import json


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


def load_sae_model(variant: str, config: Dict, device: str = 'cuda') -> nn.Module:
    """
    Load a trained SAE model checkpoint.

    Args:
        variant: SAE variant name ('topk', 'gated', 'jumprelu', 'switch')
        config: Configuration dict with hyperparameters
        device: Device to load on ('cuda' or 'cpu')

    Returns:
        Loaded SAE model
    """
    # Import SAE variants
    from sparse_autoencoder import TopKSAE, GatedSAE, JumpReLUSAE, SwitchSAE

    variant_classes = {
        'topk': TopKSAE,
        'gated': GatedSAE,
        'jumprelu': JumpReLUSAE,
        'switch': SwitchSAE,
    }

    if variant not in variant_classes:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variant_classes.keys())}")

    # Build checkpoint filename based on variant and config
    if variant == 'topk':
        ckpt_name = f"sae_topk_latent{config['latent_dim']}_k{config['k']}_seed42.pt"
    elif variant == 'gated':
        ckpt_name = f"sae_gated_latent{config['latent_dim']}_lambda{config['sparsity_coef']:.0e}_seed42.pt"
    elif variant == 'jumprelu':
        ckpt_name = f"sae_jumprelu_latent{config['latent_dim']}_thresh{config['threshold_init']:.0e}_bw{config['bandwidth']:.0e}_seed42.pt"
    elif variant == 'switch':
        total_latent = config['num_experts'] * config['latent_per_expert']
        ckpt_name = f"sae_switch_experts{config['num_experts']}_latent{total_latent}_k{config['k_per_expert']}_seed42.pt"

    ckpt_path = Path("checkpoints") / ckpt_name

    # ========================================================================
    # VALIDATION 1: Check checkpoint file exists
    # ========================================================================
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"\n{'='*80}\n"
            f"❌ CHECKPOINT NOT FOUND: {ckpt_path}\n"
            f"{'='*80}\n"
            f"\nVariant:     {variant.upper()}\n"
            f"Config:      {config}\n"
            f"Expected:    {ckpt_path.absolute()}\n"
            f"\n{'─'*80}\n"
            f"This likely means:\n"
            f"  1. Phase 1 training hasn't completed for the {variant.upper()} variant\n"
            f"  2. Phase 2 identified a config that wasn't actually trained\n"
            f"  3. Checkpoint files were moved or deleted\n"
            f"{'─'*80}\n"
            f"\nTo fix:\n"
            f"  • Verify Phase 1 training completed successfully\n"
            f"  • Check that checkpoints/ directory contains the file:\n"
            f"    ls checkpoints/sae_{variant}_*seed42.pt\n"
            f"  • Re-run Phase 1 training if needed\n"
            f"{'='*80}\n"
        )

    # Create model instance
    if variant == 'topk':
        model = TopKSAE(input_dim=80, latent_dim=config['latent_dim'], k=config['k'])
    elif variant == 'gated':
        model = GatedSAE(input_dim=80, latent_dim=config['latent_dim'], sparsity_coef=config['sparsity_coef'])
    elif variant == 'jumprelu':
        model = JumpReLUSAE(input_dim=80, latent_dim=config['latent_dim'],
                           threshold_init=config['threshold_init'],
                           bandwidth=config['bandwidth'])
    elif variant == 'switch':
        model = SwitchSAE(input_dim=80,
                         num_experts=config['num_experts'],
                         latent_per_expert=config['latent_per_expert'],
                         k_per_expert=config['k_per_expert'])

    # ========================================================================
    # VALIDATION 2: Try to load checkpoint file
    # ========================================================================
    try:
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location=device)
    except Exception as e:
        raise RuntimeError(
            f"\n{'='*80}\n"
            f"❌ FAILED TO LOAD CHECKPOINT: {ckpt_path}\n"
            f"{'='*80}\n"
            f"\nError: {str(e)}\n"
            f"\n{'─'*80}\n"
            f"The checkpoint file exists but cannot be loaded. This could mean:\n"
            f"  • Checkpoint file is corrupted or incomplete\n"
            f"  • Saved with incompatible PyTorch version\n"
            f"  • Training was interrupted before completion\n"
            f"  • File was partially written\n"
            f"{'─'*80}\n"
            f"\nTo fix:\n"
            f"  • Verify file integrity: file {ckpt_path}\n"
            f"  • Check file size: ls -lh {ckpt_path}\n"
            f"  • Re-run Phase 1 training for this variant\n"
            f"{'='*80}\n"
        ) from e

    # ========================================================================
    # VALIDATION 3: Verify checkpoint structure
    # ========================================================================
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError(
            f"\n{'='*80}\n"
            f"❌ INVALID CHECKPOINT FORMAT: {ckpt_path}\n"
            f"{'='*80}\n"
            f"\nExpected: dict with 'model_state_dict' key\n"
            f"Got:      {type(checkpoint)}\n"
            f"Keys:     {checkpoint.keys() if isinstance(checkpoint, dict) else 'N/A'}\n"
            f"\n{'─'*80}\n"
            f"The checkpoint file was loaded but has unexpected structure.\n"
            f"This suggests the checkpoint was corrupted or saved incorrectly.\n"
            f"{'─'*80}\n"
            f"\nTo fix:\n"
            f"  • Re-run Phase 1 training for the {variant.upper()} variant\n"
            f"{'='*80}\n"
        )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    return model


def load_activations(graph_ids: List[int], use_mixed_motifs: bool = False) -> torch.Tensor:
    """
    Load native GNN layer2 activations.

    Args:
        graph_ids: List of graph IDs to load
        use_mixed_motifs: If True, load from mixed-motif directory (4000-4999)
                         Otherwise load from test directory (0-3999)

    Returns:
        Tensor of shape (num_samples, 64)
    """
    activations = []

    if use_mixed_motifs:
        activation_dir = Path("outputs/activations/layer2_new/mixed")
    else:
        activation_dir = Path("outputs/activations/layer2_new/test")

    if not activation_dir.exists():
        raise FileNotFoundError(f"Activation directory not found: {activation_dir}")

    for graph_id in tqdm(graph_ids, desc="Loading activations"):
        activation_file = activation_dir / f"graph_{graph_id}.pt"

        if not activation_file.exists():
            print(f"Warning: Activation file not found: {activation_file}")
            continue

        h = torch.load(activation_file, weights_only=True)  # Shape: (num_nodes, 64)
        activations.append(h)

    # Concatenate all activations
    if len(activations) == 0:
        raise RuntimeError("No activations were loaded!")

    activations = torch.cat(activations, dim=0)  # (total_nodes, 64)

    return activations


def fit_pca(activations: torch.Tensor, n_components: int = 5) -> Tuple[PCA, torch.Tensor]:
    """
    Fit PCA on native activations.

    Args:
        activations: Native activations (num_samples, 64)
        n_components: Number of principal components to compute

    Returns:
        (fitted PCA object, PCA-projected activations)
    """
    activations_np = activations.cpu().numpy()

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(activations_np)

    print(f"\nPCA Analysis:")
    print(f"  Input dimension: {activations.shape[1]}")
    print(f"  Number of samples: {activations.shape[0]}")
    print(f"  Explained variance ratio (top {n_components}):")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {var:.4f} ({100*var:.2f}%)")
    print(f"  Cumulative variance: {100*pca.explained_variance_ratio_.sum():.2f}%")

    return pca, torch.tensor(projected, dtype=torch.float32)


def compute_reconstruction(model: nn.Module, activations: torch.Tensor,
                          batch_size: int = 256, device: str = 'cuda') -> torch.Tensor:
    """
    Compute SAE reconstructions for all activations.

    Args:
        model: SAE model
        activations: Native activations (num_samples, 64)
        batch_size: Batch size for inference
        device: Device to compute on

    Returns:
        Reconstructed activations (num_samples, 64)
    """
    reconstructions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(activations), batch_size), desc="Computing reconstructions"):
            batch = activations[i:i+batch_size].to(device)
            recon, _ = model(batch)
            reconstructions.append(recon.cpu())

    reconstructions = torch.cat(reconstructions, dim=0)

    return reconstructions


def plot_pca_component_histograms(native: torch.Tensor, reconstructed: torch.Tensor,
                                  pca: PCA, variant: str, config: Dict,
                                  output_dir: Optional[Path] = None, n_components: int = 5):
    """
    Plot PCA component histograms comparing native vs reconstructed activations.

    Misalignment between histograms indicates reconstruction issues.

    Args:
        native: Native activations (num_samples, 64) or projected (num_samples, n_components)
        reconstructed: Reconstructed activations, same shape
        pca: Fitted PCA object
        variant: SAE variant name
        config: Configuration dict
        output_dir: Directory to save plots
        n_components: Number of components to visualize
    """
    # Project reconstructed onto the same PCA basis
    reconstructed_np = reconstructed.cpu().numpy()
    native_projected = native.cpu().numpy()
    reconstructed_projected = pca.transform(reconstructed_np)

    # Create figure with subplots for each component
    fig, axes = plt.subplots(n_components, 1, figsize=(14, 4*n_components))
    if n_components == 1:
        axes = [axes]

    colors = {'native': '#1f77b4', 'reconstructed': '#ff7f0e'}

    for i in range(n_components):
        ax = axes[i]

        native_vals = native_projected[:, i]
        reconstructed_vals = reconstructed_projected[:, i]

        # Plot histograms with transparency to see overlap
        ax.hist(native_vals, bins=50, alpha=0.6, label='Native',
                color=colors['native'], edgecolor='black', density=True)
        ax.hist(reconstructed_vals, bins=50, alpha=0.6, label='Reconstructed',
                color=colors['reconstructed'], edgecolor='black', density=True)

        # Add statistics
        native_mean = native_vals.mean()
        native_std = native_vals.std()
        recon_mean = reconstructed_vals.mean()
        recon_std = reconstructed_vals.std()

        ax.axvline(native_mean, color=colors['native'], linestyle='--', linewidth=2,
                  label=f'Native μ={native_mean:.3f}')
        ax.axvline(recon_mean, color=colors['reconstructed'], linestyle='--', linewidth=2,
                  label=f'Recon μ={recon_mean:.3f}')

        variance_explained = pca.explained_variance_ratio_[i] * 100
        ax.set_title(f'PCA Component {i+1} (explains {variance_explained:.2f}% variance)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Component Value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Overall title with variant and config info
    config_str = ', '.join([f'{k}={v}' for k, v in config.items() if k != 'device'])
    fig.suptitle(f'{variant.upper()} SAE Reconstruction Fidelity\nConfig: {config_str}',
                fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"pca_histograms_{variant}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()


def compute_reconstruction_metrics(native: torch.Tensor, reconstructed: torch.Tensor,
                                   pca: PCA, variant: str) -> Dict[str, float]:
    """
    Compute various reconstruction quality metrics.

    Args:
        native: Native activations
        reconstructed: Reconstructed activations
        pca: Fitted PCA object (for component-wise analysis)
        variant: SAE variant name

    Returns:
        Dictionary of metrics
    """
    native_np = native.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()

    # Point-wise MSE
    mse = mean_squared_error(native_np, reconstructed_np)

    # Component-wise MSE (in PCA space)
    native_projected = native_np  # Already in PCA space if passed in
    reconstructed_projected = pca.transform(reconstructed_np)

    component_mses = []
    for i in range(len(pca.components_)):
        comp_mse = mean_squared_error(native_projected[:, i], reconstructed_projected[:, i])
        component_mses.append(comp_mse)

    # Variance preservation
    native_var = native_np.var(axis=0).mean()
    reconstructed_var = reconstructed_np.var(axis=0).mean()
    variance_ratio = reconstructed_var / (native_var + 1e-8)

    # Mean and std differences
    native_mean = native_np.mean()
    reconstructed_mean = reconstructed_np.mean()
    native_std = native_np.std()
    reconstructed_std = reconstructed_np.std()

    metrics = {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'variance_ratio': float(variance_ratio),
        'mean_diff': float(abs(native_mean - reconstructed_mean)),
        'std_diff': float(abs(native_std - reconstructed_std)),
        'mean_component_mse': float(np.mean(component_mses)),
        'max_component_mse': float(np.max(component_mses)),
        'min_component_mse': float(np.min(component_mses)),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Analyze SAE reconstruction fidelity using PCA component histograms'
    )
    parser.add_argument('--variant', type=str, default='topk',
                       choices=['topk', 'gated', 'jumprelu', 'switch'],
                       help='SAE variant to analyze')
    parser.add_argument('--latent-dim', type=int, default=512,
                       help='Latent dimension')
    parser.add_argument('--k', type=int, default=8,
                       help='TopK sparsity parameter (for TopK SAE)')
    parser.add_argument('--sparsity-coef', type=float, default=5e-4,
                       help='Sparsity coefficient (for Gated SAE)')
    parser.add_argument('--threshold-init', type=float, default=0.1,
                       help='Threshold initialization (for JumpReLU SAE)')
    parser.add_argument('--bandwidth', type=float, default=0.01,
                       help='Bandwidth for STE (for JumpReLU SAE)')
    parser.add_argument('--num-experts', type=int, default=4,
                       help='Number of experts (for Switch SAE)')
    parser.add_argument('--latent-per-expert', type=int, default=128,
                       help='Latent dim per expert (for Switch SAE)')
    parser.add_argument('--k-per-expert', type=int, default=8,
                       help='K per expert (for Switch SAE)')
    parser.add_argument('--use-mixed-motifs', action='store_true',
                       help='Use mixed-motif graphs (4000-4999) instead of test set (0-3999)')
    parser.add_argument('--num-graphs', type=int, default=100,
                       help='Number of graphs to analyze (default: 100, max: 1000)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    parser.add_argument('--output-dir', type=str, default='outputs/reconstruction_fidelity',
                       help='Directory to save analysis results')

    args = parser.parse_args()

    # Build config based on variant
    if args.variant == 'topk':
        config = {
            'latent_dim': args.latent_dim,
            'k': args.k,
        }
    elif args.variant == 'gated':
        config = {
            'latent_dim': args.latent_dim,
            'sparsity_coef': args.sparsity_coef,
        }
    elif args.variant == 'jumprelu':
        config = {
            'latent_dim': args.latent_dim,
            'threshold_init': args.threshold_init,
            'bandwidth': args.bandwidth,
        }
    elif args.variant == 'switch':
        config = {
            'num_experts': args.num_experts,
            'latent_per_expert': args.latent_per_expert,
            'k_per_expert': args.k_per_expert,
        }

    device = args.device
    output_dir = Path(args.output_dir)

    print("="*70)
    print("SAE RECONSTRUCTION FIDELITY ANALYSIS")
    print("="*70)
    print(f"Variant: {args.variant}")
    print(f"Config: {config}")
    print(f"Device: {device}")
    print(f"Using mixed-motifs: {args.use_mixed_motifs}")
    print(f"Output directory: {output_dir}")
    print()

    # Load SAE model
    print("Loading SAE model...")
    model = load_sae_model(args.variant, config, device)
    print(f"✓ Loaded {args.variant.upper()} SAE")
    print()

    # Load activations
    print("Loading native GNN activations...")
    if args.use_mixed_motifs:
        graph_ids = list(range(4000, min(4000 + args.num_graphs, 5000)))
    else:
        graph_ids = list(range(0, args.num_graphs))

    native_activations = load_activations(graph_ids, use_mixed_motifs=args.use_mixed_motifs)
    print(f"✓ Loaded {native_activations.shape[0]} activation samples")
    print()

    # Fit PCA on native activations
    print("Fitting PCA on native activations...")
    pca, native_pca = fit_pca(native_activations, n_components=5)
    print()

    # Compute reconstructions
    print("Computing SAE reconstructions...")
    reconstructed_activations = compute_reconstruction(model, native_activations,
                                                       batch_size=256, device=device)
    print(f"✓ Computed reconstructions")
    print()

    # Compute metrics
    print("Computing reconstruction metrics...")
    metrics = compute_reconstruction_metrics(native_pca,
                                            reconstructed_activations, pca, args.variant)
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  Variance Ratio (Recon/Native): {metrics['variance_ratio']:.4f}")
    print(f"  Mean Diff: {metrics['mean_diff']:.6f}")
    print(f"  Std Diff: {metrics['std_diff']:.6f}")
    print()

    # Plot PCA histograms
    print("Generating PCA component histograms...")
    plot_pca_component_histograms(native_pca, reconstructed_activations, pca,
                                 args.variant, config, output_dir, n_components=5)
    print()

    # Save metrics to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / f"metrics_{args.variant}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to {metrics_file}")

    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
