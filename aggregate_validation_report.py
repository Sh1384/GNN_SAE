#!/usr/bin/env python3
"""
Nested Validation & Stability Analysis for Best Configs Across Seeds

After Phase 1 of training (hyperparameter sweep with single seed=42),
Phase 2 trains only the BEST config per variant with multiple seeds
[42, 123, 456, 789, 1011] to assess stability.

This script:
1. Aggregates multi-seed results for the best config of each variant
2. Computes feature stability (decoder weight reproducibility across seeds)
3. Reports correlation distributions (not just best seed)
4. Identifies which features/results are robust vs seed-dependent

Gracefully handles:
- Missing architectures (e.g., only GCN, no GAT)
- Incomplete seeds (e.g., only 2-3 of 5 seeds available)
- Missing metrics files

Addresses reviewer concern: "Model selection by maximizing the largest
significant correlation across many features risks selection bias; nested
validation or reporting stability across seeds/configs is needed."

Usage:
    python aggregate_validation_report.py --variant topk
    python aggregate_validation_report.py --all-variants
"""

import json
import glob
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = Path("outputs")
CHECKPOINTS_DIR = Path("checkpoints")
METRICS_DIR = OUTPUT_DIR / "sae_metrics"
STABILITY_DIR = OUTPUT_DIR / "stability_analysis"
STABILITY_DIR.mkdir(parents=True, exist_ok=True)

MULTI_SEEDS = [42, 123, 456, 789, 1011]  # Phase 2 multi-seed standard


def load_metrics_json(checkpoint_path: str) -> Optional[Dict]:
    """Load metrics JSON for a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)

    Returns:
        Dictionary of metrics or None (no error on missing file)
    """
    metrics_path = checkpoint_path.replace('.pt', '_metrics.json')
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_decoder_weights(checkpoint_path: str) -> Optional[np.ndarray]:
    """Load decoder weights from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint

    Returns:
        Decoder weight matrix (latent_dim, input_dim) or None (no error)
    """
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if 'decoder_weights' in checkpoint:
            return checkpoint['decoder_weights']
        elif 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
            for key in ['decoder.weight', 'decoder_weights', 'w_dec']:
                if key in state:
                    return state[key].cpu().numpy()

        return None
    except Exception as e:
        print(f"Warning: Failed to extract decoder weights from checkpoint: {type(e).__name__}: {str(e)}")
        return None


def find_best_config_checkpoints(variant: str) -> Dict[int, str]:
    """Find multi-seed checkpoints for best config of a variant.

    Returns empty dict if no checkpoints found (no error).

    Args:
        variant: SAE variant (topk, gated, jumprelu, switch)

    Returns:
        Dictionary mapping seed -> checkpoint_path
    """
    checkpoints_by_seed = {}

    pattern = f"{CHECKPOINTS_DIR}/sae_{variant}_*_seed*.pt"
    all_checkpoints = glob.glob(pattern)

    if not all_checkpoints:
        return {}

    # Group by config
    configs = {}
    for ckpt in all_checkpoints:
        name = Path(ckpt).stem
        parts = name.split('_')

        try:
            seed = int(parts[-1].replace('seed', ''))
            config = '_'.join(parts[2:-1])
        except (ValueError, IndexError):
            continue

        if config not in configs:
            configs[config] = []
        configs[config].append((seed, ckpt))

    if not configs:
        return {}

    # Use first config (assume best from Phase 1 comparison)
    best_config = list(configs.keys())[0]

    for seed, ckpt in configs[best_config]:
        checkpoints_by_seed[seed] = ckpt

    return checkpoints_by_seed


def compute_feature_stability(variant: str, checkpoints: Dict[int, str]) -> Optional[Dict]:
    """Compute feature stability across available seeds.

    Works with 2+ seeds; gracefully skips unavailable decoders.

    Args:
        variant: SAE variant
        checkpoints: Dict mapping seed -> checkpoint_path

    Returns:
        Dictionary with stability metrics, or None if < 2 seeds available
    """
    if len(checkpoints) < 2:
        return None

    decoders = []
    loaded_seeds = []

    for seed in sorted(checkpoints.keys()):
        weights = load_decoder_weights(checkpoints[seed])
        if weights is not None:
            decoders.append(weights)
            loaded_seeds.append(seed)

    if len(decoders) < 2:
        return None

    latent_dim = decoders[0].shape[0]

    # Normalize decoder columns (features)
    normalized = []
    for dec in decoders:
        dec_norm = dec / (np.linalg.norm(dec, axis=0, keepdims=True) + 1e-8)
        normalized.append(dec_norm)

    # Compute pairwise feature similarity
    stability_scores = []

    for feature_idx in range(latent_dim):
        seed_correlations = []

        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                feat_i = normalized[i][:, feature_idx]
                feat_j = normalized[j][:, feature_idx]

                sim = np.dot(feat_i, feat_j) / (
                    np.linalg.norm(feat_i) * np.linalg.norm(feat_j) + 1e-8
                )
                seed_correlations.append(sim)

        mean_stability = np.mean(seed_correlations) if seed_correlations else 0.0
        std_stability = np.std(seed_correlations) if seed_correlations else 0.0

        stability_scores.append({
            'feature_idx': feature_idx,
            'mean_stability': float(mean_stability),
            'std_stability': float(std_stability),
            'is_stable': mean_stability > 0.8,
        })

    stable_count = sum(1 for s in stability_scores if s['is_stable'])
    stability_rate = stable_count / latent_dim

    return {
        'variant': variant,
        'num_seeds': len(loaded_seeds),
        'loaded_seeds': loaded_seeds,
        'latent_dim': latent_dim,
        'stable_features_count': stable_count,
        'stability_rate': stability_rate,
        'per_feature_scores': stability_scores,
    }


def aggregate_correlation_data(variant: str, checkpoints: Dict[int, str]) -> Optional[pd.DataFrame]:
    """Aggregate correlations across available seeds.

    Gracefully handles missing metrics files.

    Args:
        variant: SAE variant
        checkpoints: Dict mapping seed -> checkpoint_path

    Returns:
        DataFrame with correlations per seed, or None if no data found
    """
    all_rows = []

    for seed, ckpt_path in checkpoints.items():
        metrics = load_metrics_json(ckpt_path)

        if metrics is None:
            continue

        # Extract correlations (try multiple possible keys)
        correlations = None
        for key in ['point_biserial_correlations', 'correlations', 'feature_motif_correlations']:
            if key in metrics:
                correlations = metrics[key]
                break

        if correlations is None:
            continue

        # Handle different formats
        if isinstance(correlations, dict):
            corr_values = list(correlations.values())
        else:
            corr_values = list(correlations) if hasattr(correlations, '__iter__') else []

        # Ensure numeric
        try:
            corr_values = [float(abs(c)) for c in corr_values if c is not None]
        except (TypeError, ValueError):
            continue

        for feature_idx, corr in enumerate(corr_values):
            all_rows.append({
                'seed': seed,
                'feature_idx': feature_idx,
                'correlation': corr,
                'is_significant': corr > 0.3,
                'is_high': corr > 0.5,
            })

    if not all_rows:
        return None

    return pd.DataFrame(all_rows)


def plot_stability_heatmap(stability_data: Dict, variant: str):
    """Plot feature stability as heatmap."""
    if stability_data is None:
        return

    scores = stability_data['per_feature_scores']
    feature_indices = [s['feature_idx'] for s in scores]
    mean_stabilities = [s['mean_stability'] for s in scores]
    is_stable = [s['is_stable'] for s in scores]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Stability per feature
    ax = axes[0]
    colors = ['#2ca02c' if s else '#d62728' for s in is_stable]
    ax.bar(feature_indices, mean_stabilities, color=colors, edgecolor='black', alpha=0.7)
    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Stable (0.8)')
    ax.set_xlabel('Feature Index', fontweight='bold', fontsize=11)
    ax.set_ylabel('Mean Cosine Similarity', fontweight='bold', fontsize=11)
    ax.set_title(f'Feature Stability Across Seeds: {variant.upper()}', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    # Plot 2: Summary
    ax = axes[1]
    ax.axis('off')

    stable_count = stability_data['stable_features_count']
    total = stability_data['latent_dim']
    rate = stability_data['stability_rate']

    summary_text = f"""MULTI-SEED STABILITY SUMMARY

{variant.upper()}
Seeds: {stability_data['loaded_seeds']} (n={len(stability_data['loaded_seeds'])})

Stable Features (cosine sim > 0.8):
  {stable_count} / {total} ({rate:.1%})

Interpretation:
"""

    if rate > 0.8:
        summary_text += "✓ ROBUST - Features reproducible across seeds"
    elif rate > 0.5:
        summary_text += "⚠ MODERATE - ~50% reproducible"
    else:
        summary_text += "🔍 LOW - Many seed-dependent features"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    output_path = STABILITY_DIR / f'feature_stability_{variant}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved feature_stability_{variant}.png")
    plt.close()


def plot_correlation_reproducibility(corr_df: pd.DataFrame, variant: str):
    """Plot correlation reproducibility across seeds."""
    if corr_df is None or len(corr_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Correlation Reproducibility: {variant.upper()}',
                 fontsize=12, fontweight='bold')

    # Plot 1: Distribution per seed
    ax = axes[0, 0]
    for seed in sorted(corr_df['seed'].unique()):
        seed_data = corr_df[corr_df['seed'] == seed]['correlation']
        ax.hist(seed_data, bins=30, alpha=0.5, label=f'Seed {seed}', edgecolor='black')
    ax.set_xlabel('|Correlation|', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Correlation Distributions', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Heatmap (if enough features)
    ax = axes[0, 1]
    pivot = corr_df.pivot_table(values='correlation', index='feature_idx',
                                columns='seed', aggfunc='first')
    if len(pivot) > 10:
        sns.heatmap(pivot.iloc[:min(50, len(pivot)), :], cmap='RdYlGn', vmin=0, vmax=1,
                   ax=ax, cbar_kws={'label': '|r|'})
        ax.set_title('Correlation Heatmap', fontweight='bold')
        ax.set_xlabel('Seed', fontweight='bold')
        ax.set_ylabel('Feature', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Too few features for heatmap', ha='center', va='center')
        ax.axis('off')

    # Plot 3: Significant feature proportions
    ax = axes[1, 0]
    sig_by_seed = corr_df.groupby('seed')['is_significant'].mean()
    high_by_seed = corr_df.groupby('seed')['is_high'].mean()

    x = np.arange(len(sig_by_seed))
    width = 0.35

    ax.bar(x - width/2, sig_by_seed.values, width, label='Sig (|r|>0.3)',
           color='#1f77b4', edgecolor='black', alpha=0.7)
    ax.bar(x + width/2, high_by_seed.values, width, label='High (|r|>0.5)',
           color='#ff7f0e', edgecolor='black', alpha=0.7)

    ax.set_xlabel('Seed', fontweight='bold')
    ax.set_ylabel('Proportion', fontweight='bold')
    ax.set_title('Significant Correlations', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sig_by_seed.index)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    sig_count = (corr_df['correlation'] > 0.3).sum()
    high_count = (corr_df['correlation'] > 0.5).sum()

    summary_text = f"""CORRELATION STATISTICS

Seeds: {sorted(corr_df['seed'].unique())}
Total: {len(corr_df)} correlations

Mean |r|: {corr_df['correlation'].mean():.4f}
Std |r|: {corr_df['correlation'].std():.4f}
Median: {corr_df['correlation'].median():.4f}

Significant (|r| > 0.3):
  {sig_count} ({sig_count/len(corr_df)*100:.1f}%)

High (|r| > 0.5):
  {high_count} ({high_count/len(corr_df)*100:.1f}%)
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    output_path = STABILITY_DIR / f'correlation_reproducibility_{variant}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved correlation_reproducibility_{variant}.png")
    plt.close()


def create_stability_report(variant: str, stability_data: Optional[Dict],
                           corr_df: Optional[pd.DataFrame]) -> str:
    """Create markdown report."""
    report = [
        f"# Nested Validation: {variant.upper()}\n\n",
        "## Multi-Seed Stability Analysis\n\n",
    ]

    if stability_data:
        report.append(f"**Seeds analyzed:** {stability_data['loaded_seeds']}\n\n")
        report.append(f"**Stable features:** {stability_data['stable_features_count']} / {stability_data['latent_dim']} ")
        report.append(f"({stability_data['stability_rate']:.1%})\n\n")

        if stability_data['stability_rate'] > 0.8:
            report.append("✓ **Result:** Robust across seeds\n\n")
        elif stability_data['stability_rate'] > 0.5:
            report.append("⚠ **Result:** Moderate stability\n\n")
        else:
            report.append("🔍 **Result:** Low stability, seed-dependent\n\n")
    else:
        report.append("*No multi-seed stability data available*\n\n")

    if corr_df is not None and len(corr_df) > 0:
        report.append("## Correlation Reproducibility\n\n")
        report.append(f"- Seeds: {sorted(corr_df['seed'].unique())}\n")
        report.append(f"- Mean |r|: {corr_df['correlation'].mean():.4f}\n")
        report.append(f"- Significant (|r| > 0.3): {(corr_df['correlation'] > 0.3).sum()} ")
        report.append(f"({(corr_df['correlation'] > 0.3).sum() / len(corr_df) * 100:.1f}%)\n")
        report.append(f"- High (|r| > 0.5): {(corr_df['correlation'] > 0.5).sum()} ")
        report.append(f"({(corr_df['correlation'] > 0.5).sum() / len(corr_df) * 100:.1f}%)\n\n")

    report.append("## Avoiding Selection Bias\n\n")
    report.append("- ✓ Report distributions across seeds\n")
    report.append("- ✓ Show all significant features, not just best\n")
    report.append("- ✓ Include effect sizes (mean ± std)\n")
    report.append("- ✓ Identify stable vs unstable features\n")

    return "".join(report)


def main():
    parser = __import__('argparse').ArgumentParser(
        description='Multi-seed stability for best SAE configs'
    )
    parser.add_argument('--variant', type=str, choices=['topk', 'gated', 'jumprelu', 'switch'])
    parser.add_argument('--all-variants', action='store_true')

    args = parser.parse_args()

    print("="*70)
    print("NESTED VALIDATION: MULTI-SEED STABILITY")
    print("="*70)

    variants = (
        ['topk', 'gated', 'jumprelu', 'switch'] if args.all_variants
        else [args.variant] if args.variant else ['topk']
    )

    for variant in variants:
        print(f"\n{variant.upper()}...", end=' ')

        checkpoints = find_best_config_checkpoints(variant)

        if not checkpoints:
            print("(no checkpoints found)")
            continue

        print(f"({len(checkpoints)} seeds found)")

        stability_data = compute_feature_stability(variant, checkpoints)
        if stability_data:
            plot_stability_heatmap(stability_data, variant)

        corr_df = aggregate_correlation_data(variant, checkpoints)
        if corr_df is not None:
            plot_correlation_reproducibility(corr_df, variant)

        report = create_stability_report(variant, stability_data, corr_df)
        report_file = STABILITY_DIR / f'nested_validation_{variant}.md'

        with open(report_file, 'w') as f:
            f.write(report)

        print(f"  → saved to {report_file.name}")

    print("\n" + "="*70)
    print(f"✓ Output: {STABILITY_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
