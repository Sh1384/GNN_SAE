#!/usr/bin/env python3
"""
Compare SAE Latent Ablations vs Native GNN Ablations

Compares two ablation strategies to validate causal mechanistic interpretation:

1. SAE Latent Ablation (current approach):
   - Zero out SAE latent features
   - Reconstruct activations through SAE decoder
   - Measure GNN impact

2. Native GNN Ablation (new approach):
   - Directly patch nodes in 64-dimensional activation space
   - No SAE reconstruction/reconstruction error
   - More direct intervention in GNN mechanism

Analysis metrics:
- Correlation between strategies (agreement score)
- Conditional effects (with/without motif)
- Effect sizes and statistical significance
- Interpretation of differences

Usage:
    python compare_ablation_strategies.py --variant topk --latent_dim 512 --k 8
    python compare_ablation_strategies.py --all-variants
"""

import argparse
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import warnings

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = Path("outputs")
COMPARISON_DIR = OUTPUT_DIR / "ablation_strategy_comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

# Motif group constants
MOTIFS = [
    'in_feedback_loop',
    'in_cascade',
    'in_feedforward_loop',
    'in_single_input_module'
]

MOTIF_DISPLAY_NAMES = {
    'in_feedback_loop': 'Feedback Loop',
    'in_cascade': 'Cascade',
    'in_feedforward_loop': 'Feedforward Loop',
    'in_single_input_module': 'Single Input Module'
}


def load_phase_3a_config() -> Optional[Dict]:
    """Load phase_3a_config.json with all Phase 3a parameters.

    Returns:
        Dict with keys: variant, latent_dim, variant_kwargs, etc.
        Returns None if file doesn't exist or can't be loaded.
    """
    config_path = Path('phase_3a_config.json')
    if not config_path.exists():
        return None

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load phase_3a_config.json: {e}")
        return None


def generate_sae_motif_filename(motif_name: str, variant: str, latent_dim: int,
                               variant_kwargs: Dict) -> str:
    """Generate SAE motif results filename using variant-aware naming.

    This function mirrors generate_experiment_name() from run_interpretability_experiments.py
    to ensure consistent filenames across Phase 3a and Phase 3c.

    Args:
        motif_name: Display name (e.g., 'Feedback Loop')
        variant: SAE variant (topk, gated, jumprelu, switch)
        latent_dim: SAE latent dimension
        variant_kwargs: Dict of variant-specific parameters

    Returns:
        Filename string (without directory prefix)

    Examples:
        TopK: 'feedback_loop_topk_l512_k8_results.csv'
        Gated: 'feedback_loop_gated_l512_lambda1e-03_results.csv'
        JumpReLU: 'feedback_loop_jumprelu_l512_thresh1e-02_bw1e-02_results.csv'
    """
    motif_name_clean = motif_name.lower().replace(' ', '_')

    if not variant:
        k = variant_kwargs.get('k', 8)
        return f"{motif_name_clean}_l{latent_dim}_k{k}_results.csv"

    if variant == 'topk':
        k = variant_kwargs.get('k', 8)
        return f"{motif_name_clean}_{variant}_l{latent_dim}_k{k}_results.csv"

    elif variant == 'gated':
        sparsity_coef = variant_kwargs.get('sparsity_coef', 1e-3)
        return f"{motif_name_clean}_{variant}_l{latent_dim}_lambda{sparsity_coef:.0e}_results.csv"

    elif variant == 'jumprelu':
        threshold_init = variant_kwargs.get('threshold_init', 0.01)
        bandwidth = variant_kwargs.get('bandwidth', 0.01)
        return f"{motif_name_clean}_{variant}_l{latent_dim}_thresh{threshold_init:.0e}_bw{bandwidth:.0e}_results.csv"

    elif variant == 'switch':
        num_experts = variant_kwargs.get('num_experts', 8)
        latent_per_expert = variant_kwargs.get('latent_per_expert', 64)
        k_per_expert = variant_kwargs.get('k_per_expert', 8)
        return f"{motif_name_clean}_{variant}_l{latent_dim}_experts{num_experts}_latentpeexp{latent_per_expert}_k{k_per_expert}_results.csv"

    else:
        k = variant_kwargs.get('k', 8)
        return f"{motif_name_clean}_{variant}_l{latent_dim}_k{k}_results.csv"


def load_sae_ablation_results(variant: str, feature_idx: int) -> Optional[pd.DataFrame]:
    """Load SAE latent ablation results from run_ablation.py outputs.

    Args:
        variant: SAE variant name
        feature_idx: Feature index that was ablated

    Returns:
        DataFrame with SAE ablation results, or None if file not found

    Expected format: ablations/results/ablation_{variant}_feature{feature_idx}.csv
    """
    results_file = Path(f"ablations/results/ablation_{variant}_feature{feature_idx}.csv")

    if not results_file.exists():
        print(f"Warning: SAE ablation results not found: {results_file}")
        return None

    try:
        df = pd.read_csv(results_file)
        if df.empty:
            print(f"Warning: SAE ablation results file is empty: {results_file}")
            return None
        return df
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse SAE ablation CSV {results_file}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: Could not load SAE ablation results {results_file}: {str(e)}")
        return None


def load_native_ablation_results(variant: str, feature_idx: int) -> Optional[pd.DataFrame]:
    """Load native GNN ablation results from native_gnn_ablation.py outputs.

    Args:
        variant: SAE variant name
        feature_idx: Feature index that was ablated

    Returns:
        DataFrame with native ablation results, or None if file not found

    Expected format: outputs/native_gnn_ablations/native_ablation_{variant}_feature{feature_idx}.csv
    """
    results_file = Path(f"outputs/native_gnn_ablations/native_ablation_{variant}_feature{feature_idx}.csv")

    if not results_file.exists():
        print(f"Warning: Native ablation results not found: {results_file}")
        return None

    try:
        df = pd.read_csv(results_file)
        if df.empty:
            print(f"Warning: Native ablation results file is empty: {results_file}")
            return None
        return df
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse native ablation CSV {results_file}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: Could not load native ablation results {results_file}: {str(e)}")
        return None


def merge_ablation_results(sae_results: pd.DataFrame, native_results: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Merge SAE and native ablation results by graph_id.

    Args:
        sae_results: DataFrame from SAE ablation
        native_results: DataFrame from native ablation

    Returns:
        Merged DataFrame, or None if inputs are invalid

    Raises:
        ValueError: If merge fails due to incompatible columns
    """
    if sae_results is None or native_results is None:
        print("Error: Cannot merge - one or both input DataFrames are None")
        return None

    if sae_results.empty or native_results.empty:
        print("Error: Cannot merge - one or both DataFrames are empty")
        return None

    if 'graph_id' not in sae_results.columns or 'graph_id' not in native_results.columns:
        print("Error: Both DataFrames must have 'graph_id' column for merging")
        return None

    try:
        merged = pd.merge(sae_results, native_results, on=['graph_id'], suffixes=('_sae', '_native'))
        if merged.empty:
            print("Warning: Merged result is empty (no common graph_ids)")
            return None
        return merged
    except Exception as e:
        raise ValueError(f"Failed to merge ablation results: {str(e)}")


def compute_agreement_score(merged_results: pd.DataFrame) -> Dict:
    """Compute how well SAE and native ablations agree.

    Args:
        merged_results: Merged DataFrame from merge_ablation_results()

    Returns:
        Dict with agreement metrics (Pearson/Spearman correlation, direction agreement, etc.)

    Metrics:
    - Pearson correlation of Δ Loss values
    - Spearman rank correlation
    - Percent graphs with same direction of effect
    """
    if merged_results is None or len(merged_results) == 0:
        return {'error': 'No merged results'}

    sae_deltas = merged_results.get('delta_loss_sae', merged_results.get('delta_mse_sae', None))
    native_deltas = merged_results.get('delta_loss_native', merged_results.get('delta_mse_native', None))

    if sae_deltas is None or native_deltas is None:
        return {'error': 'Could not find delta loss columns'}

    # Remove NaN values
    valid_mask = ~(pd.isna(sae_deltas) | pd.isna(native_deltas))
    sae_valid = sae_deltas[valid_mask].values
    native_valid = native_deltas[valid_mask].values

    if len(sae_valid) == 0:
        return {'error': 'No valid delta loss values'}

    try:
        # Pearson correlation
        if len(sae_valid) < 2:
            return {'error': 'Insufficient data for correlation (need at least 2 points)'}

        pearson_r, pearson_p = pearsonr(sae_valid, native_valid)

        # Spearman correlation
        spearman_r, spearman_p = spearmanr(sae_valid, native_valid)

        # Direction agreement (sign)
        direction_agreement = (np.sign(sae_valid) == np.sign(native_valid)).mean()

        return {
            'n_graphs': len(sae_valid),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'direction_agreement': direction_agreement,
            'mean_sae_delta': float(sae_valid.mean()),
            'mean_native_delta': float(native_valid.mean()),
            'std_sae_delta': float(sae_valid.std()),
            'std_native_delta': float(native_valid.std()),
        }
    except Exception as e:
        return {'error': f'Failed to compute agreement score: {str(e)}'}


def analyze_conditional_agreement(merged_results: pd.DataFrame) -> Dict:
    """Analyze agreement separately for graphs with vs without target motif.

    Args:
        merged_results: Merged DataFrame from merge_ablation_results()

    Returns:
        Dict with agreement metrics broken down by motif presence
    """
    if merged_results is None or len(merged_results) == 0:
        return {'error': 'No merged results'}

    if 'has_motif' not in merged_results.columns:
        return {'error': 'Missing has_motif column for conditional analysis'}

    results = {
        'with_motif': {},
        'without_motif': {}
    }

    for motif_presence, group_name in [(True, 'with_motif'), (False, 'without_motif')]:
        try:
            subset = merged_results[merged_results['has_motif'] == motif_presence]

            if len(subset) == 0:
                continue

            sae_deltas = subset.get('delta_loss_sae', subset.get('delta_mse_sae', None))
            native_deltas = subset.get('delta_loss_native', subset.get('delta_mse_native', None))

            if sae_deltas is not None and native_deltas is not None:
                valid_mask = ~(pd.isna(sae_deltas) | pd.isna(native_deltas))
                if valid_mask.sum() > 1:
                    try:
                        r, p = pearsonr(sae_deltas[valid_mask], native_deltas[valid_mask])
                        results[group_name] = {
                            'correlation': float(r),
                            'p_value': float(p),
                            'n_graphs': int(valid_mask.sum()),
                            'mean_delta_diff': float((sae_deltas[valid_mask] - native_deltas[valid_mask]).mean()),
                        }
                    except Exception as e:
                        print(f"Error: Failed to compute correlation for {group_name}: {str(e)}")
        except Exception as e:
            print(f"Error: Failed to analyze {group_name}: {str(e)}")

    return results


def load_sae_motif_results(motif: str, latent_dim: int, k: int, variant: str = None) -> Optional[pd.DataFrame]:
    """Load SAE ablation results for a specific motif group from Phase 3a.

    First tries to load phase_3a_config.json to get exact parameters used by Phase 3a,
    then generates the correct filename with all variant-specific parameters.

    Falls back to older naming conventions if phase_3a_config.json doesn't exist.

    Expected file formats:
    - With phase_3a_config: Uses variant and variant_kwargs from config
      - TopK: ablations/results/{motif}_{variant}_l{latent_dim}_k{k}_results.csv
      - Gated: ablations/results/{motif}_{variant}_l{latent_dim}_lambda{coef}_results.csv
      - JumpReLU: ablations/results/{motif}_{variant}_l{latent_dim}_thresh{init}_bw{bw}_results.csv
      - Switch: ablations/results/{motif}_{variant}_l{latent_dim}_experts{exp}_..._results.csv

    - Fallback (no config): {motif}_l{latent_dim}_k{k}_results.csv
    """
    motif_display = motif.replace('in_', '').replace('_', ' ').title()
    motif_name = motif_display.lower().replace(' ', '_')

    # Try to load phase_3a_config first
    phase_3a_config = load_phase_3a_config()

    if phase_3a_config:
        config_variant = phase_3a_config.get('variant')
        config_latent_dim = phase_3a_config.get('latent_dim')
        config_variant_kwargs = phase_3a_config.get('variant_kwargs', {})

        # Use config values if available, fall back to arguments
        actual_variant = variant or config_variant
        actual_latent_dim = latent_dim if latent_dim else config_latent_dim
        actual_k = k if k else config_variant_kwargs.get('k', 8)

        # Generate correct filename with all variant-specific parameters
        if actual_variant and config_variant_kwargs:
            filename = generate_sae_motif_filename(motif_name, actual_variant,
                                                   actual_latent_dim, config_variant_kwargs)
            results_file = Path('ablations') / 'results' / filename
            if results_file.exists():
                try:
                    df = pd.read_csv(results_file)
                    if not df.empty:
                        return df
                except Exception:
                    pass

    # Fallback 1: Try with variant and k (standard format for all variants)
    if variant:
        results_file = Path('ablations') / 'results' / f'{motif_name}_{variant}_l{latent_dim}_k{k}_results.csv'
        if results_file.exists():
            try:
                df = pd.read_csv(results_file)
                if not df.empty:
                    return df
            except Exception:
                pass

    # Fallback 2: Try without variant (legacy format)
    results_file = Path('ablations') / 'results' / f'{motif_name}_l{latent_dim}_k{k}_results.csv'

    if not results_file.exists():
        return None

    try:
        df = pd.read_csv(results_file)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def load_native_motif_results(variant: str, motif: str) -> Optional[pd.DataFrame]:
    """Load native GNN ablation results for a specific motif from Phase 3b.

    Expected file: outputs/native_gnn_ablations/native_ablation_{variant}_rpb_{motif}.csv
    (This is the output from native_gnn_ablation.py --use-rpb --motif {motif})
    """
    results_file = Path('outputs') / 'native_gnn_ablations' / f'native_ablation_{variant}_rpb_{motif}.csv'

    if not results_file.exists():
        return None

    try:
        df = pd.read_csv(results_file)
        if df.empty:
            return None
        return df
    except Exception as e:
        return None


def merge_motif_results(sae_df: pd.DataFrame, native_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Merge SAE and native motif ablation results by graph_id."""
    if sae_df is None or native_df is None or sae_df.empty or native_df.empty:
        return None

    if 'graph_id' not in sae_df.columns or 'graph_id' not in native_df.columns:
        return None

    try:
        # For Phase 3a (run_interpretability_experiments): uses 'Ablation Impact'
        # For Phase 3b (native_gnn_ablation): uses 'delta_loss'
        sae_impact_col = 'Ablation Impact' if 'Ablation Impact' in sae_df.columns else 'delta_loss'
        native_impact_col = 'delta_loss' if 'delta_loss' in native_df.columns else 'delta_mse'

        merged = pd.merge(sae_df[['graph_id', sae_impact_col]],
                         native_df[['graph_id', native_impact_col]],
                         on='graph_id',
                         suffixes=('_sae', '_native'))

        if merged.empty:
            return None

        # Rename columns to standard names for downstream processing
        merged = merged.rename(columns={
            sae_impact_col + '_sae': 'delta_loss_sae',
            native_impact_col + '_native': 'delta_loss_native'
        })

        return merged
    except Exception as e:
        return None


def plot_strategy_comparison(merged_results: pd.DataFrame, feature_idx: int, variant: str):
    """
    Generate comprehensive comparison plots.
    """
    if merged_results is None or len(merged_results) == 0:
        print("  No data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'SAE Latent vs Native Ablation Comparison\n{variant.upper()} - Feature {feature_idx}',
                fontsize=16, fontweight='bold')

    # Plot 1: Scatter - SAE vs Native Δ Loss
    ax = axes[0, 0]

    sae_deltas = merged_results.get('delta_loss_sae', merged_results.get('delta_mse_sae', None))
    native_deltas = merged_results.get('delta_loss_native', merged_results.get('delta_mse_native', None))

    if sae_deltas is not None and native_deltas is not None:
        valid_mask = ~(pd.isna(sae_deltas) | pd.isna(native_deltas))
        sae_valid = sae_deltas[valid_mask]
        native_valid = native_deltas[valid_mask]

        # Color by motif presence
        has_motif = merged_results.loc[valid_mask, 'has_motif'].values
        colors = ['#2ca02c' if m else '#d62728' for m in has_motif]

        ax.scatter(sae_valid, native_valid, c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

        # Add diagonal line (perfect agreement)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0, label='Perfect agreement')

        ax.set_xlabel('SAE Latent Ablation Δ Loss', fontsize=11, fontweight='bold')
        ax.set_ylabel('Native GNN Ablation Δ Loss', fontsize=11, fontweight='bold')
        ax.set_title('Method Agreement', fontsize=12, fontweight='bold')

        # Add legend
        green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
                                markersize=8, label='With motif')
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
                              markersize=8, label='Without motif')
        ax.legend(handles=[green_patch, red_patch], loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)

    # Plot 2: Distribution comparison
    ax = axes[0, 1]

    if sae_deltas is not None and native_deltas is not None:
        ax.hist(sae_valid, bins=20, alpha=0.6, label='SAE Latent', edgecolor='black')
        ax.hist(native_valid, bins=20, alpha=0.6, label='Native GNN', edgecolor='black')

        ax.set_xlabel('Δ Loss', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Δ Loss Distributions', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

    # Plot 3: Box plots by motif presence
    ax = axes[1, 0]

    if 'has_motif' in merged_results.columns:
        data_with_motif = []
        data_without_motif = []

        for strategy, deltas in [('SAE', sae_deltas), ('Native', native_deltas)]:
            if deltas is not None and 'has_motif' in merged_results.columns:
                valid_mask = ~pd.isna(deltas)
                with_m = deltas[valid_mask & (merged_results.loc[valid_mask, 'has_motif'] == True)].values
                without_m = deltas[valid_mask & (merged_results.loc[valid_mask, 'has_motif'] == False)].values

                if len(with_m) > 0:
                    data_with_motif.append(with_m)
                if len(without_m) > 0:
                    data_without_motif.append(without_m)

        bp = ax.boxplot([data_with_motif, data_without_motif] if len(data_with_motif) > 0 else [],
                        labels=['With Motif', 'Without Motif'] if len(data_with_motif) > 0 else [],
                        patch_artist=True)

        if len(bp['boxes']) > 0:
            for patch in bp['boxes']:
                patch.set_facecolor('#1f77b4')
                patch.set_alpha(0.7)

            ax.set_ylabel('Δ Loss', fontsize=11, fontweight='bold')
            ax.set_title('Selective Degradation by Motif', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

    # Plot 4: Agreement metrics
    ax = axes[1, 1]
    ax.axis('off')

    agreement = compute_agreement_score(merged_results)

    if 'error' not in agreement:
        text = f"""
        AGREEMENT METRICS

        Sample Size: {agreement['n_graphs']} graphs

        Correlation:
        • Pearson r = {agreement['pearson_r']:.3f} (p={agreement['pearson_p']:.3e})
        • Spearman ρ = {agreement['spearman_r']:.3f} (p={agreement['spearman_p']:.3e})

        Direction Agreement: {agreement['direction_agreement']:.1%}

        Mean Δ Loss:
        • SAE: {agreement['mean_sae_delta']:.6f} ± {agreement['std_sae_delta']:.6f}
        • Native: {agreement['mean_native_delta']:.6f} ± {agreement['std_native_delta']:.6f}

        Interpretation:
        """

        if agreement['pearson_r'] > 0.8:
            text += "✓ Strong agreement - methods capture same effect\n"
        elif agreement['pearson_r'] > 0.5:
            text += "⚠ Moderate agreement - some confounding possible\n"
        else:
            text += "⚠ Weak agreement - significant differences between methods\n"

        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / f'strategy_comparison_{variant}_feature{feature_idx}.png',
               dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved strategy_comparison_{variant}_feature{feature_idx}.png")
    plt.close()


def compute_comprehensive_agreement_matrix(variant: str, all_features: bool = True) -> Optional[pd.DataFrame]:
    """Compute agreement matrix across all features and graphs.

    This validates that SAE latent ablations correlate with native GNN ablations
    across the full feature space, not just individual features.

    Args:
        variant: SAE variant (topk, gated, jumprelu, switch)
        all_features: If True, compute agreement for all available features

    Returns:
        DataFrame with columns: feature_idx, num_graphs, correlation, p_value,
                              mean_sae_delta, mean_native_delta, agreement_label
    """
    print(f"\n  Computing comprehensive agreement matrix for {variant}...")

    results = []

    # Get all ablation result files for this variant
    sae_files = glob.glob(f"ablations/results/ablation_{variant}_feature*.csv")

    if not sae_files:
        print(f"  Warning: No SAE ablation files found for {variant}")
        return None

    for sae_file in sorted(sae_files):
        try:
            # Extract feature index from filename
            feature_idx = int(sae_file.split('feature')[-1].split('.')[0])

            # Load both ablation types
            sae_results = pd.read_csv(sae_file)
            native_file = f"outputs/native_gnn_ablations/native_ablation_{variant}_feature{feature_idx}.csv"

            if not Path(native_file).exists():
                continue

            native_results = pd.read_csv(native_file)

            # Merge
            merged = pd.merge(sae_results, native_results, on=['graph_id'], suffixes=('_sae', '_native'))

            if len(merged) == 0:
                continue

            # Compute correlation
            sae_deltas = merged.get('delta_loss_sae', merged.get('delta_mse_sae', None))
            native_deltas = merged.get('delta_loss_native', merged.get('delta_mse_native', None))

            if sae_deltas is None or native_deltas is None:
                continue

            valid_mask = ~(pd.isna(sae_deltas) | pd.isna(native_deltas))
            sae_valid = sae_deltas[valid_mask].values
            native_valid = native_deltas[valid_mask].values

            if len(sae_valid) < 2:
                continue

            r, p = spearmanr(sae_valid, native_valid)

            # Classify agreement level
            if r > 0.8:
                agreement_label = "Strong"
            elif r > 0.5:
                agreement_label = "Moderate"
            else:
                agreement_label = "Weak"

            results.append({
                'feature_idx': feature_idx,
                'num_graphs': len(sae_valid),
                'correlation': r,
                'p_value': p,
                'mean_sae_delta': float(sae_valid.mean()),
                'mean_native_delta': float(native_valid.mean()),
                'std_sae_delta': float(sae_valid.std()),
                'std_native_delta': float(native_valid.std()),
                'agreement_label': agreement_label,
            })

        except Exception as e:
            continue

    if len(results) == 0:
        print(f"  Warning: Could not compute agreement for any features in {variant}")
        return None

    df = pd.DataFrame(results)
    return df


def plot_comprehensive_agreement(agreement_df: pd.DataFrame, variant: str):
    """Plot comprehensive agreement analysis across all features."""
    if agreement_df is None or len(agreement_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Comprehensive Ablation Strategy Agreement: {variant.upper()}',
                 fontsize=14, fontweight='bold')

    # Plot 1: Correlation by feature
    ax = axes[0, 0]
    colors = ['#2ca02c' if l == 'Strong' else '#ff7f0e' if l == 'Moderate' else '#d62728'
              for l in agreement_df['agreement_label']]
    ax.scatter(agreement_df['feature_idx'], agreement_df['correlation'], c=colors, s=80, alpha=0.7, edgecolors='black')
    ax.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='Strong (r>0.8)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (r>0.5)')
    ax.set_xlabel('Feature Index', fontweight='bold')
    ax.set_ylabel('Spearman Correlation', fontweight='bold')
    ax.set_title('Per-Feature Agreement', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Distribution of correlations
    ax = axes[0, 1]
    ax.hist(agreement_df['correlation'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=agreement_df['correlation'].mean(), color='r', linestyle='--', linewidth=2, label=f"Mean: {agreement_df['correlation'].mean():.3f}")
    ax.set_xlabel('Spearman Correlation', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Distribution of Feature Correlations', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Agreement counts
    ax = axes[1, 0]
    agreement_counts = agreement_df['agreement_label'].value_counts()
    colors_pie = {'Strong': '#2ca02c', 'Moderate': '#ff7f0e', 'Weak': '#d62728'}
    colors = [colors_pie.get(k, 'gray') for k in agreement_counts.index]
    ax.bar(agreement_counts.index, agreement_counts.values, color=colors, edgecolor='black', alpha=0.7)
    ax.set_ylabel('Number of Features', fontweight='bold')
    ax.set_title('Agreement Classification', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(agreement_counts.values):
        ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    COMPREHENSIVE AGREEMENT SUMMARY

    Total Features: {len(agreement_df)}
    Total Graphs: {agreement_df['num_graphs'].sum()}

    Correlation Statistics:
    • Mean: {agreement_df['correlation'].mean():.3f}
    • Median: {agreement_df['correlation'].median():.3f}
    • Std: {agreement_df['correlation'].std():.3f}
    • Min: {agreement_df['correlation'].min():.3f}
    • Max: {agreement_df['correlation'].max():.3f}

    Agreement Breakdown:
    • Strong (r > 0.8): {(agreement_df['agreement_label'] == 'Strong').sum()} features
    • Moderate (r > 0.5): {(agreement_df['agreement_label'] == 'Moderate').sum()} features
    • Weak (r ≤ 0.5): {(agreement_df['agreement_label'] == 'Weak').sum()} features

    Interpretation:
    """

    mean_corr = agreement_df['correlation'].mean()
    if mean_corr > 0.8:
        summary_text += "✓ STRONG agreement - SAE faithfully captures GNN"
    elif mean_corr > 0.5:
        summary_text += "⚠ MODERATE agreement - SAE valid with caveats"
    else:
        summary_text += "🔍 WEAK agreement - Use native ablations"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / f'comprehensive_agreement_{variant}.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comprehensive_agreement_{variant}.png")
    plt.close()


def create_agreement_summary_csv(agreement_data: Dict[str, pd.DataFrame]):
    """Save agreement analysis to CSV for all variants."""
    all_rows = []

    for variant, df in agreement_data.items():
        if df is None:
            continue
        df['variant'] = variant
        all_rows.append(df)

    if all_rows:
        combined_df = pd.concat(all_rows, ignore_index=True)
        combined_df = combined_df[['variant', 'feature_idx', 'num_graphs', 'correlation', 'p_value',
                                   'mean_sae_delta', 'mean_native_delta', 'agreement_label']]

        output_file = COMPARISON_DIR / 'ablation_agreement_comprehensive.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved comprehensive agreement CSV: {output_file}")

        return combined_df

    return None


def create_summary_report(all_comparisons: Dict, agreement_data: Dict = None) -> str:
    """Create markdown summary report."""
    report = [
        "# Ablation Strategy Comparison Report\n",
        "## Executive Summary\n",
        "Comprehensive comparison of two complementary ablation approaches:\n",
        "1. **SAE Latent Ablation:** Zero out SAE features, reconstruct via decoder\n",
        "2. **Native GNN Ablation:** Directly patch nodes in activation space\n\n",
    ]

    report.append("## Key Findings\n\n")

    for variant, features in all_comparisons.items():
        if len(features) == 0:
            continue

        report.append(f"### {variant.upper()}\n\n")

        agreements = [f['agreement'] for f in features.values() if 'agreement' in f]

        if len(agreements) > 0:
            mean_corr = np.mean([a.get('pearson_r', 0) for a in agreements])
            report.append(f"- Average Pearson correlation: **{mean_corr:.3f}**\n")
            report.append(f"- Features analyzed: {len(features)}\n\n")

    # Add comprehensive agreement results if available
    if agreement_data:
        report.append("## Comprehensive Agreement Analysis\n\n")
        for variant, df in agreement_data.items():
            if df is not None and len(df) > 0:
                report.append(f"### {variant.upper()}\n\n")
                report.append(f"- Features analyzed: {len(df)}\n")
                report.append(f"- Total graphs compared: {df['num_graphs'].sum()}\n")
                report.append(f"- Mean correlation: **{df['correlation'].mean():.3f}** ")
                report.append(f"(σ={df['correlation'].std():.3f})\n")
                strong = (df['agreement_label'] == 'Strong').sum()
                mod = (df['agreement_label'] == 'Moderate').sum()
                weak = (df['agreement_label'] == 'Weak').sum()
                report.append(f"- Distribution: {strong} strong, {mod} moderate, {weak} weak\n\n")

    report.append("## Interpretation Guide\n\n")
    report.append("### High Agreement (r > 0.8)\n")
    report.append("✓ SAE accurately captures causal mechanisms\n")
    report.append("✓ Reconstruction error is minimal\n")
    report.append("✓ SAE latent ablations provide valid causal claims\n\n")

    report.append("### Moderate Agreement (0.5 < r < 0.8)\n")
    report.append("⚠ SAE captures signal but with confounding\n")
    report.append("⚠ Report both methods to show robustness\n")
    report.append("⚠ Acknowledge SAE linearity assumptions\n\n")

    report.append("### Low Agreement (r < 0.5)\n")
    report.append("🔍 SAE latent space may not align with GNN mechanisms\n")
    report.append("🔍 Use native ablations as primary evidence\n")
    report.append("🔍 Investigate nonlinear SAE-GNN interactions\n\n")

    report.append("## Recommendations\n\n")
    report.append("1. Use native ablations as validation for SAE claims\n")
    report.append("2. Report agreement correlation in analysis\n")
    report.append("3. Discuss SAE limitations (linearity, reconstruction error)\n")
    report.append("4. Consider both strategies complementary, not conflicting\n")

    return "".join(report)


def main():
    parser = argparse.ArgumentParser(description='Compare SAE Latent vs Native GNN Ablation Strategies')
    parser.add_argument('--variant', type=str, choices=['topk', 'gated', 'jumprelu', 'switch'],
                       help='Specific variant to compare')
    parser.add_argument('--feature', type=int, help='Specific feature to compare')
    parser.add_argument('--all-variants', action='store_true', help='Compare all variants')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive agreement analysis')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension')
    parser.add_argument('--k', type=int, default=8, help='SAE sparsity parameter')
    parser.add_argument('--motif-mode', action='store_true',
                       help='Compare at MOTIF-GROUP level (Phase 3c mode for grouped analysis)')

    args = parser.parse_args()

    print("="*70)
    print("ABLATION STRATEGY COMPARISON: SAE Latent vs Native GNN")
    print("="*70)

    all_comparisons = {}
    agreement_data = {}

    # MOTIF-GROUP COMPARISON MODE (Phase 3c - for grouped feature analysis)
    if args.motif_mode:
        print("\n" + "="*70)
        print("MOTIF-GROUP LEVEL COMPARISON (Phase 3c Mode)")
        print("="*70)

        if not args.variant and not args.all_variants:
            print("Error: --motif-mode requires --variant or --all-variants")
            return

        variants = ['topk', 'gated', 'jumprelu', 'switch'] if args.all_variants else [args.variant]

        motif_agreement_results = []

        for variant in variants:
            print(f"\n{'-'*70}")
            print(f"Variant: {variant.upper()}")
            print(f"{'-'*70}")

            for motif in MOTIFS:
                motif_display = MOTIF_DISPLAY_NAMES.get(motif, motif)
                print(f"\n  {motif_display}...", end=' ')

                # Load Phase 3a and 3b results for this motif
                sae_results = load_sae_motif_results(motif, args.latent_dim, args.k, variant)
                native_results = load_native_motif_results(variant, motif)

                if sae_results is None or native_results is None:
                    print("⚠ Missing results")
                    continue

                # Merge by graph_id
                merged = merge_motif_results(sae_results, native_results)
                if merged is None or merged.empty:
                    print("⚠ Merge failed")
                    continue

                # Compute agreement metrics
                agreement = compute_agreement_score(merged)

                if 'error' not in agreement:
                    print(f"✓ ρ={agreement['spearman_r']:.3f} (p={agreement['spearman_p']:.3e})")

                    # Store results
                    motif_agreement_results.append({
                        'variant': variant,
                        'motif': motif,
                        'motif_display': motif_display,
                        'n_graphs': agreement['n_graphs'],
                        'pearson_r': agreement['pearson_r'],
                        'pearson_p': agreement['pearson_p'],
                        'spearman_r': agreement['spearman_r'],
                        'spearman_p': agreement['spearman_p'],
                        'direction_agreement': agreement['direction_agreement'],
                        'mean_sae_delta': agreement['mean_sae_delta'],
                        'std_sae_delta': agreement['std_sae_delta'],
                        'mean_native_delta': agreement['mean_native_delta'],
                        'std_native_delta': agreement['std_native_delta'],
                    })

                    # Generate plot
                    plot_strategy_comparison(merged, -1, f"{variant}_{motif_display}")  # -1 indicates motif mode
                else:
                    print(f"⚠ {agreement['error']}")

        # Save motif-level summary
        print("\n" + "="*70)
        print("SAVING MOTIF-LEVEL SUMMARY")
        print("="*70)

        if motif_agreement_results:
            summary_df = pd.DataFrame(motif_agreement_results)
            summary_file = COMPARISON_DIR / 'motif_agreement_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            print(f"\n✓ Saved motif-level summary: {summary_file}")
            print("\nSummary Table:")
            print(summary_df.to_string(index=False))

        print(f"\n✓ Motif-group comparison complete!")
        print(f"   Outputs saved to: {COMPARISON_DIR}/")
        return

    # Determine which variants and features to analyze
    if args.all_variants:
        variants = ['topk', 'gated', 'jumprelu', 'switch']
    elif args.variant:
        variants = [args.variant]
    else:
        variants = ['topk']

    # Comprehensive agreement analysis (across all features and graphs)
    if args.comprehensive:
        print("\n" + "="*70)
        print("COMPREHENSIVE AGREEMENT ANALYSIS (All Features × All Graphs)")
        print("="*70)

        for variant in variants:
            agreement_df = compute_comprehensive_agreement_matrix(variant)
            if agreement_df is not None:
                agreement_data[variant] = agreement_df
                plot_comprehensive_agreement(agreement_df, variant)
                print(f"  Mean correlation for {variant}: {agreement_df['correlation'].mean():.3f}")

        # Save comprehensive agreement CSV
        if agreement_data:
            create_agreement_summary_csv(agreement_data)

    # Per-feature detailed analysis
    feature_range = [args.feature] if args.feature is not None else range(0, 10)  # First 10 features

    print("\n" + "="*70)
    print("PER-FEATURE DETAILED ANALYSIS")
    print("="*70)

    for variant in variants:
        print(f"\nAnalyzing {variant.upper()}...")
        all_comparisons[variant] = {}

        for feature_idx in feature_range:
            print(f"  Feature {feature_idx}...", end=' ')

            # Load both ablation types
            sae_results = load_sae_ablation_results(variant, feature_idx)
            native_results = load_native_ablation_results(variant, feature_idx)

            if sae_results is None or native_results is None:
                print("⚠ Missing data")
                continue

            # Merge results
            merged = merge_ablation_results(sae_results, native_results)

            if merged is None or len(merged) == 0:
                print("⚠ No matches")
                continue

            # Compute agreement
            agreement = compute_agreement_score(merged)
            conditional_agreement = analyze_conditional_agreement(merged)

            all_comparisons[variant][feature_idx] = {
                'agreement': agreement,
                'conditional': conditional_agreement,
                'merged_results': merged,
            }

            if 'error' not in agreement:
                print(f"✓ r={agreement['pearson_r']:.3f}")

                # Generate plots
                plot_strategy_comparison(merged, feature_idx, variant)

    # Create summary report
    print("\nGenerating summary report...")
    report = create_summary_report(all_comparisons, agreement_data)
    report_file = COMPARISON_DIR / "ablation_strategy_comparison.md"

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"✓ Saved report to {report_file}")

    print("\n" + "="*70)
    print("✓ ABLATION STRATEGY COMPARISON COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {COMPARISON_DIR}/")
    print("\nUsage for comprehensive analysis:")
    print("  python compare_ablation_strategies.py --all-variants --comprehensive")


if __name__ == "__main__":
    main()
