#!/usr/bin/env python3
"""
Visualize SAE Training Progress

Creates plots showing training progress for all SAE configurations.
Useful for comparing convergence and identifying best configs visually.

Usage:
    python visualize_training_progress.py
    python visualize_training_progress.py --output progress_plots/
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)


def load_metrics(metrics_dir: Path = Path("outputs")) -> Dict:
    """Load all SAE training metrics."""
    metrics_files = list(metrics_dir.glob("sae_metrics_latent*.json"))

    all_metrics = {}

    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Extract config from filename
            parts = metrics_file.stem.split('_')
            latent_dim = int(parts[2].replace('latent', ''))
            k = int(parts[3].replace('k', ''))

            all_metrics[(latent_dim, k)] = metrics
        except (json.JSONDecodeError, IndexError, ValueError) as e:
            print(f"Warning: Could not load {metrics_file}: {e}")

    return all_metrics


def plot_training_curves(all_metrics: Dict, output_dir: Path):
    """Plot training and validation loss curves."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by latent_dim
    latent_dims = sorted(set(k[0] for k in all_metrics.keys()))

    for latent_dim in latent_dims:
        configs = {k: v for k, v in all_metrics.items() if k[0] == latent_dim}

        if not configs:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Training Progress: Latent Dim = {latent_dim}', fontsize=16)

        # Plot 1: Training Loss
        ax = axes[0, 0]
        for (ld, k), metrics in sorted(configs.items()):
            history = metrics.get('train_history', {})
            train_loss = history.get('train_loss', [])
            if train_loss:
                ax.plot(train_loss, label=f'k={k}', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title('Training Loss', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 2: Validation Loss
        ax = axes[0, 1]
        for (ld, k), metrics in sorted(configs.items()):
            history = metrics.get('train_history', {})
            val_loss = history.get('val_loss', [])
            if val_loss:
                ax.plot(val_loss, label=f'k={k}', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('Validation Loss', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 3: L0 Sparsity
        ax = axes[1, 0]
        for (ld, k), metrics in sorted(configs.items()):
            history = metrics.get('train_history', {})
            l0_sparsity = history.get('train_l0', [])
            target_sparsity = k / latent_dim

            if l0_sparsity:
                ax.plot(l0_sparsity, label=f'k={k} (target={target_sparsity:.3f})', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('L0 Sparsity (Fraction Active)', fontsize=12)
        ax.set_title('Sparsity During Training', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Reconstruction Loss
        ax = axes[1, 1]
        for (ld, k), metrics in sorted(configs.items()):
            history = metrics.get('train_history', {})
            train_recon = history.get('train_recon', [])
            val_recon = history.get('val_recon', [])

            if train_recon and val_recon:
                ax.plot(train_recon, '--', alpha=0.5, linewidth=1)
                ax.plot(val_recon, '-', label=f'k={k}', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Reconstruction Loss', fontsize=12)
        ax.set_title('Reconstruction Loss (Val)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        output_file = output_dir / f'training_curves_latent{latent_dim}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()


def plot_final_metrics_comparison(all_metrics: Dict, output_dir: Path):
    """Plot comparison of final test metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract final metrics
    configs = []
    test_losses = []
    test_recons = []
    test_l0s = []
    sparsities = []

    for (latent_dim, k), metrics in sorted(all_metrics.items()):
        configs.append(f"{latent_dim}d\nk={k}")
        test_losses.append(metrics.get('test_loss', np.nan))
        test_recons.append(metrics.get('test_reconstruction', np.nan))
        test_l0s.append(metrics.get('test_l0_sparsity', np.nan))
        sparsities.append(100 * k / latent_dim)

    # Create 3-panel plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Final Test Metrics Comparison', fontsize=16)

    # Plot 1: Test Reconstruction Loss
    ax = axes[0]
    bars = ax.bar(range(len(configs)), test_recons, color='steelblue', alpha=0.7)
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Reconstruction Loss', fontsize=12)
    ax.set_title('Test Reconstruction Loss', fontsize=14)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight best
    best_idx = np.nanargmin(test_recons)
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(1.0)

    # Plot 2: Achieved vs Target Sparsity
    ax = axes[1]
    x = np.arange(len(configs))
    width = 0.35

    ax.bar(x - width/2, sparsities, width, label='Target', color='orange', alpha=0.7)
    ax.bar(x + width/2, [100*l0 for l0 in test_l0s], width, label='Achieved',
           color='steelblue', alpha=0.7)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Sparsity (%)', fontsize=12)
    ax.set_title('Target vs Achieved Sparsity', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Capacity vs Reconstruction
    ax = axes[2]
    latent_dims_list = [k[0] for k in sorted(all_metrics.keys())]
    colors = plt.cm.viridis(np.linspace(0, 1, len(set(latent_dims_list))))

    for i, latent_dim in enumerate(sorted(set(latent_dims_list))):
        configs_ld = [(ld, k) for (ld, k) in all_metrics.keys() if ld == latent_dim]
        ks = [k for (ld, k) in sorted(configs_ld)]
        recons = [all_metrics[(latent_dim, k)].get('test_reconstruction', np.nan)
                  for k in ks]

        ax.plot(ks, recons, 'o-', label=f'latent={latent_dim}',
                color=colors[i], linewidth=2, markersize=8)

    ax.set_xlabel('k (Sparsity Level)', fontsize=12)
    ax.set_ylabel('Reconstruction Loss', fontsize=12)
    ax.set_title('Sparsity vs Reconstruction Trade-off', fontsize=14)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'final_metrics_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def print_summary_table(all_metrics: Dict):
    """Print summary table of all configurations."""
    print("\n" + "="*100)
    print("SAE TRAINING SUMMARY")
    print("="*100)
    print(f"{'Config':<20} {'Test Loss':<15} {'Test Recon':<15} {'L0 Sparsity':<15} {'Status':<10}")
    print("-"*100)

    for (latent_dim, k), metrics in sorted(all_metrics.items()):
        config_str = f"latent={latent_dim}, k={k}"
        test_loss = metrics.get('test_loss', float('nan'))
        test_recon = metrics.get('test_reconstruction', float('nan'))
        test_l0 = metrics.get('test_l0_sparsity', float('nan'))

        # Color coding
        if test_recon < 1e-7:
            status = "Excellent"
        elif test_recon < 5e-7:
            status = "Good"
        else:
            status = "Fair"

        print(f"{config_str:<20} {test_loss:<15.6e} {test_recon:<15.6e} "
              f"{test_l0:<15.3f} {status:<10}")

    print("="*100)

    # Find best config
    best_config = min(all_metrics.items(),
                     key=lambda x: x[1].get('test_reconstruction', float('inf')))
    (best_ld, best_k), best_metrics = best_config

    print(f"\n✓ BEST CONFIGURATION (by reconstruction loss):")
    print(f"  latent_dim = {best_ld}")
    print(f"  k = {best_k}")
    print(f"  Sparsity: {100*best_k/best_ld:.2f}%")
    print(f"  Test Reconstruction: {best_metrics['test_reconstruction']:.6e}")
    print(f"  Test L0: {best_metrics['test_l0_sparsity']:.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SAE training progress"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/training_plots',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--metrics-dir',
        type=str,
        default='outputs',
        help='Directory containing SAE metrics JSON files'
    )

    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output)

    print("Loading SAE training metrics...")
    all_metrics = load_metrics(metrics_dir)

    if not all_metrics:
        print("Error: No metrics files found!")
        print(f"Looked in: {metrics_dir}")
        print("Expected files: sae_metrics_latent*.json")
        return

    print(f"✓ Loaded {len(all_metrics)} configurations")

    # Print summary table
    print_summary_table(all_metrics)

    # Create plots
    print(f"\nGenerating plots...")
    plot_training_curves(all_metrics, output_dir)
    plot_final_metrics_comparison(all_metrics, output_dir)

    print(f"\n✓ All plots saved to: {output_dir}")
    print(f"✓ View with: eog {output_dir}/*.png")


if __name__ == "__main__":
    main()
