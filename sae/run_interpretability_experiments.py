#!/usr/bin/env python3
"""
Motif-Specific SAE Feature Ablation Experiments with Configurable Effect Size Threshold

Performs rigorous statistical testing of SAE interpretability by:
1. Ablating features that significantly correlate with specific motifs
2. Comparing against size-matched random feature ablations
3. Computing statistical significance with z-scores, percentiles, and p-values

Key features:
- Configurable minimum effect size threshold (--min_rpb) to filter weak correlations
- Uses FDR-corrected significance + effect size filtering
- Robust random controls (20+ trials per feature count)
- Comprehensive statistical analysis

Usage:
    # Standard run with default threshold (rpb >= 0.05)
    python run_interpretability_experiments.py --latent_dim 512 --k 4

    # Strict threshold for strong correlations only
    python run_interpretability_experiments.py --latent_dim 512 --k 4 --min_rpb 0.10

    # Custom random trials
    python run_interpretability_experiments.py --latent_dim 512 --k 4 --min_rpb 0.08 --n_random_trials 50
"""

import argparse
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import sys

# Directories
ABLATION_DIR = Path("ablations")

def setup_directories(latent_dim, k, min_rpb, use_mixed_motifs=False, variant=None):
    """Create experiment-specific directories."""
    if variant:
        exp_name = f"{variant}_latent{latent_dim}_k{k}_rpb{min_rpb:.2f}"
    else:
        exp_name = f"latent{latent_dim}_k{k}_rpb{min_rpb:.2f}"
    if use_mixed_motifs:
        exp_name += "_mixed"
    results_dir = ABLATION_DIR / f"interpretability_{exp_name}_results"
    plots_dir = ABLATION_DIR / f"interpretability_{exp_name}_plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, plots_dir

def load_correlation_data(min_rpb=0.05, variant=None):
    """
    Load correlation data and extract features that are:
    1. Significant (FDR < 0.05)
    2. Have effect size above threshold (|rpb| >= min_rpb)

    Args:
        min_rpb: Minimum absolute point-biserial correlation threshold
        variant: SAE variant (topk, gated, jumprelu, switch). If specified, loads variant-specific correlations.

    Returns:
        motif_features: Dict mapping motif names to lists of features
        non_filtered_features: List of features that don't meet criteria
        filtered_df: DataFrame of features meeting both criteria
    """
    if variant:
        corr_file = f'outputs/feature_analysis_{variant}/latent_correlations.csv'
    else:
        corr_file = 'outputs/latent_correlations.csv'

    if not Path(corr_file).exists():
        print(f"ERROR: Correlation data not found at {corr_file}")
        if variant:
            print(f"  Have you run: python analyze_feature_significance.py --variant {variant}?")
        else:
            print(f"  Have you run: python compare_sae_configs.py?")
        return None, None, None

    df = pd.read_csv(corr_file)

    # Apply dual filter: significance AND effect size
    filtered_df = df[(df['significant_fdr'] == True) & (df['rpb_abs'] >= min_rpb)]

    print(f"\nFiltering criteria:")
    print(f"  - FDR < 0.05: {(df['significant_fdr'] == True).sum()} pairs")
    print(f"  - |rpb| >= {min_rpb}: {(df['rpb_abs'] >= min_rpb).sum()} pairs")
    print(f"  - Both criteria: {len(filtered_df)} pairs")

    # Group by motif
    motif_features = {}
    motif_map = {
        'in_cascade': 'Cascade',
        'in_feedback_loop': 'Feedback Loop',
        'in_feedforward_loop': 'Feedforward Loop',
        'in_single_input_module': 'Single Input Module'
    }

    for motif_raw, motif_name in motif_map.items():
        features = filtered_df[filtered_df['motif'] == motif_raw]['feature'].tolist()
        motif_features[motif_name] = sorted(features)

    # Get all features that don't meet criteria (for random controls)
    all_features = df['feature'].unique()
    filtered_features = set(filtered_df['feature'].unique())
    non_filtered_features = sorted([f for f in all_features if f not in filtered_features])

    return motif_features, non_filtered_features, filtered_df

def generate_experiment_name(feature_set_name, variant=None, latent_dim=None, variant_kwargs=None):
    """Generate variant-aware experiment name with all variant-specific parameters.

    Args:
        feature_set_name: Name of the feature set (e.g., 'Feedback Loop')
        variant: SAE variant (topk, gated, jumprelu, switch)
        latent_dim: SAE latent dimension
        variant_kwargs: Dict of variant-specific parameters

    Returns:
        Experiment name string with all parameters embedded

    Examples:
        TopK: 'feedback_loop_topk_l512_k8'
        Gated: 'feedback_loop_gated_l512_lambda1e-03'
        JumpReLU: 'feedback_loop_jumprelu_l512_thresh1e-02_bw1e-02'
        Switch: 'feedback_loop_switch_l512_experts8_latentpeexp64_k8'
    """
    if variant_kwargs is None:
        variant_kwargs = {}

    motif_name = feature_set_name.lower().replace(' ', '_')

    if not variant:
        # Fallback for non-variant runs
        k = variant_kwargs.get('k', 8)
        return f"{motif_name}_l{latent_dim}_k{k}"

    # Build variant-specific filename with all parameters
    if variant == 'topk':
        k = variant_kwargs.get('k', 8)
        return f"{motif_name}_{variant}_l{latent_dim}_k{k}"

    elif variant == 'gated':
        sparsity_coef = variant_kwargs.get('sparsity_coef', 1e-3)
        return f"{motif_name}_{variant}_l{latent_dim}_lambda{sparsity_coef:.0e}"

    elif variant == 'jumprelu':
        threshold_init = variant_kwargs.get('threshold_init', 0.01)
        bandwidth = variant_kwargs.get('bandwidth', 0.01)
        return f"{motif_name}_{variant}_l{latent_dim}_thresh{threshold_init:.0e}_bw{bandwidth:.0e}"

    elif variant == 'switch':
        num_experts = variant_kwargs.get('num_experts', 8)
        latent_per_expert = variant_kwargs.get('latent_per_expert', 64)
        k_per_expert = variant_kwargs.get('k_per_expert', 8)
        return f"{motif_name}_{variant}_l{latent_dim}_experts{num_experts}_latentpeexp{latent_per_expert}_k{k_per_expert}"

    else:
        # Fallback
        k = variant_kwargs.get('k', 8)
        return f"{motif_name}_{variant}_l{latent_dim}_k{k}"

def save_feature_mapping(motif_features, results_dir, min_rpb):
    """Save feature mapping to JSON."""
    output_path = results_dir / 'feature_motif_mapping.json'

    mapping = {
        'filtering': {
            'min_rpb': min_rpb,
            'description': f'Features with FDR<0.05 and |rpb|>={min_rpb}'
        },
        'features_per_motif': motif_features
    }

    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved feature mapping to: {output_path}")

def run_single_ablation(latent_dim, features, experiment_name, use_mixed_motifs=False, variant=None, variant_kwargs=None):
    """Run a single ablation experiment.

    Args:
        latent_dim: SAE latent dimension
        features: List of feature indices to ablate
        experiment_name: Name for this experiment
        use_mixed_motifs: Whether to use mixed-motif graphs
        variant: SAE variant ('topk', 'gated', 'jumprelu', 'switch')
        variant_kwargs: Dict of variant-specific parameters (k, sparsity_coef, threshold_init, etc.)
    """
    if variant_kwargs is None:
        variant_kwargs = {}

    # Convert features to strings if they're not already
    feature_str = ','.join(str(f) for f in features)

    cmd = [
        sys.executable, "sae/run_ablation.py",
        "--latent_dim", str(latent_dim),
        "--feature", feature_str,
        "--experiment_name", experiment_name
    ]

    # Add variant-specific parameters (variant is auto-detected by run_ablation.py from checkpoint path)
    # Note: run_ablation.py defaults to 'topk' if --variant not provided
    if variant:
        # Add variant-specific parameters
        if variant == 'topk' and 'k' in variant_kwargs:
            cmd.extend(["--k", str(variant_kwargs['k'])])
        elif variant == 'gated' and 'sparsity_coef' in variant_kwargs:
            cmd.extend(["--sparsity_coef", str(variant_kwargs['sparsity_coef'])])
        elif variant == 'jumprelu' and 'threshold_init' in variant_kwargs:
            cmd.extend(["--threshold_init", str(variant_kwargs['threshold_init'])])
            if 'bandwidth' in variant_kwargs:
                cmd.extend(["--bandwidth", str(variant_kwargs['bandwidth'])])
        elif variant == 'switch' and all(k in variant_kwargs for k in ['num_experts', 'latent_per_expert', 'k_per_expert']):
            cmd.extend(["--num_experts", str(variant_kwargs['num_experts'])])
            cmd.extend(["--latent_per_expert", str(variant_kwargs['latent_per_expert'])])
            cmd.extend(["--k_per_expert", str(variant_kwargs['k_per_expert'])])

    # Add flag for mixed motifs
    if use_mixed_motifs:
        cmd.append("--use_mixed_motifs")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error in ablation {experiment_name}:")
        print(result.stderr)
        return None

    # Load results
    results_file = ABLATION_DIR / "results" / f"{experiment_name}_results.csv"
    if not results_file.exists():
        return None

    try:
        df = pd.read_csv(results_file)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Error reading results file {results_file}: {e}")
        return None

def run_random_control_trials(latent_dim, n_features, n_trials, non_filtered_features, use_mixed_motifs=False, variant=None, variant_kwargs=None):
    """Run multiple random ablation trials and collect statistics."""
    if variant_kwargs is None:
        variant_kwargs = {}

    print(f"\nRunning {n_trials} random control trials (ablating {n_features} features each)...")

    trial_results = []

    for trial in tqdm(range(n_trials), desc=f"Random trials ({n_features} features)"):
        # Sample random non-filtered features
        if len(non_filtered_features) < n_features:
            print(f"Warning: Only {len(non_filtered_features)} non-filtered features available, need {n_features}")
            random_features = non_filtered_features
        else:
            random_features = sorted(np.random.choice(non_filtered_features, n_features, replace=False))

        # Generate experiment name with variant info - include trial number
        base_name = f"random_trial_{n_features}feat_n{trial}"
        experiment_name = generate_experiment_name(base_name, variant, latent_dim, variant_kwargs)

        df = run_single_ablation(latent_dim, random_features, experiment_name, use_mixed_motifs, variant, variant_kwargs)

        if df is not None and len(df) > 0:
            # Check if required columns exist
            if 'Motif' not in df.columns or 'Ablation Impact' not in df.columns:
                continue
                
            # Compute mean impact per motif type for this trial
            for motif in df['Motif'].unique():
                motif_data = df[df['Motif'] == motif]
                if len(motif_data) > 0:
                    trial_results.append({
                        'trial': trial,
                        'n_features': n_features,
                        'motif': motif,
                        'mean_impact': motif_data['Ablation Impact'].mean(),
                        'median_impact': motif_data['Ablation Impact'].median(),
                        'std_impact': motif_data['Ablation Impact'].std()
                    })

    return pd.DataFrame(trial_results)

def plot_comparison_with_random_controls(motif_specific_results, random_results_dict,
                                        motif_features, plots_dir, latent_dim, k, min_rpb):
    """Generate comprehensive comparison plots."""
    
    # Validate inputs
    if motif_specific_results is None or len(motif_specific_results) == 0:
        print("No motif-specific results to plot!")
        return
    
    if 'Feature_Set' not in motif_specific_results.columns:
        print(f"ERROR: motif_specific_results missing 'Feature_Set' column")
        print(f"Available columns: {motif_specific_results.columns.tolist()}")
        return

    motif_colors = {
        'Feedforward Loop': '#377eb8',
        'Feedback Loop': '#ff7f00',
        'Single Input Module': '#4daf4a',
        'Cascade': '#e41a1c'
    }

    # Determine which motifs to plot (only those with significant features)
    motifs_to_plot = [m for m, feats in motif_features.items() if len(feats) > 0]
    n_motifs = len(motifs_to_plot)

    if n_motifs == 0:
        print("No motifs with significant features to plot!")
        return

    # Create subplots
    fig, axes = plt.subplots(1, n_motifs, figsize=(6*n_motifs, 6))
    if n_motifs == 1:
        axes = [axes]

    motif_order = ['Feedforward Loop', 'Feedback Loop', 'Single Input Module', 'Cascade']
    x_pos = np.arange(len(motif_order))
    width = 0.35

    for idx, motif_name in enumerate(motifs_to_plot):
        ax = axes[idx]
        n_features = len(motif_features[motif_name])

        # Get specific motif results
        specific = motif_specific_results[motif_specific_results['Feature_Set'] == motif_name]
        
        # Check if we have random results for this feature count
        if n_features not in random_results_dict:
            print(f"Warning: No random results for {n_features} features, skipping plot for {motif_name}")
            continue
            
        random = random_results_dict[n_features]
        
        # Validate random DataFrame
        if random is None or len(random) == 0:
            print(f"Warning: Empty random results for {n_features} features, skipping plot for {motif_name}")
            continue

        # Specific features
        specific_means = [specific[specific['Motif'] == m]['Mean_Impact'].values[0]
                         if len(specific[specific['Motif'] == m]) > 0 else 0
                         for m in motif_order]
        specific_ses = [specific[specific['Motif'] == m]['SE_Impact'].values[0]
                       if len(specific[specific['Motif'] == m]) > 0 else 0
                       for m in motif_order]

        # Random controls
        random_means = [random[random['motif'] == m]['mean_impact'].mean()
                       if len(random[random['motif'] == m]) > 0 else 0
                       for m in motif_order]
        random_ses = [random[random['motif'] == m]['mean_impact'].std() / np.sqrt(len(random[random['motif'] == m]))
                     if len(random[random['motif'] == m]) > 0 else 0
                     for m in motif_order]

        bars1 = ax.bar(x_pos - width/2, specific_means, width,
                      label=f'{motif_name} Features ({n_features})',
                      yerr=specific_ses, capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5,
                      color=[motif_colors.get(m, '#999999') for m in motif_order])
        bars2 = ax.bar(x_pos + width/2, random_means, width,
                      label=f'Random Controls ({n_features} feat)',
                      yerr=random_ses, capsize=5, alpha=0.6, edgecolor='black', linewidth=1.5,
                      color='gray')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(motif_order, rotation=45, ha='right')
        ax.set_ylabel('Mean Ablation Impact\n(MSE increase)', fontsize=11)
        ax.set_title(f'{motif_name} Features vs Random Controls\n({n_features} features ablated)',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle(f'SAE Interpretability: Motif-Specific Features vs Random Controls\n'
                 f'(latent_dim={latent_dim}, k={k}, min_rpb={min_rpb:.2f})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = plots_dir / 'interpretability_vs_random_controls.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def compute_statistical_significance(motif_specific_results, random_results):
    """Compute statistical tests comparing motif-specific to random."""
    results = []
    
    # Validate inputs
    if motif_specific_results is None or len(motif_specific_results) == 0:
        print("WARNING: motif_specific_results is empty, returning empty DataFrame")
        return pd.DataFrame()
    
    if 'Feature_Set' not in motif_specific_results.columns:
        print(f"ERROR: motif_specific_results missing 'Feature_Set' column")
        print(f"Available columns: {motif_specific_results.columns.tolist()}")
        return pd.DataFrame()
    
    if random_results is None or len(random_results) == 0:
        print("WARNING: random_results is empty, returning empty DataFrame")
        return pd.DataFrame()

    for feature_set in motif_specific_results['Feature_Set'].unique():
        specific_data = motif_specific_results[motif_specific_results['Feature_Set'] == feature_set]

        # Match random controls by number of features
        n_features = specific_data['N_Features'].values[0]
        random_data = random_results[random_results['n_features'] == n_features]

        for motif in specific_data['Motif'].unique():
            specific_impact = specific_data[specific_data['Motif'] == motif]['Mean_Impact'].values[0]
            random_impacts = random_data[random_data['motif'] == motif]['mean_impact'].values

            if len(random_impacts) > 0:
                # Percentile of specific impact in random distribution
                percentile = stats.percentileofscore(random_impacts, specific_impact)

                # Z-score
                random_mean = random_impacts.mean()
                random_std = random_impacts.std()
                z_score = (specific_impact - random_mean) / random_std if random_std > 0 else 0

                # One-sample t-test
                t_stat, p_val = stats.ttest_1samp(random_impacts, specific_impact)

                results.append({
                    'Feature_Set': feature_set,
                    'Motif': motif,
                    'N_Features': n_features,
                    'Specific_Impact': specific_impact,
                    'Random_Mean': random_mean,
                    'Random_Std': random_std,
                    'Z_Score': z_score,
                    'Percentile': percentile,
                    'P_Value': p_val,
                    'Significant': p_val < 0.05
                })

    return pd.DataFrame(results)

def print_monosemanticity_report(filtered_df, min_rpb):
    """Print detailed monosemanticity analysis."""
    print("\n" + "="*70)
    print("MONOSEMANTICITY ANALYSIS")
    print("="*70)

    # Count features significant for each number of motifs
    features_per_motif = filtered_df.groupby('feature')['motif'].count()

    print(f"\nFeature specificity (after filtering):")
    print(f"  Monosemantic (1 motif only): {(features_per_motif == 1).sum()}")
    print(f"  Polysemantic (2 motifs): {(features_per_motif == 2).sum()}")
    print(f"  Polysemantic (3 motifs): {(features_per_motif == 3).sum()}")
    print(f"  Polysemantic (4 motifs): {(features_per_motif == 4).sum()}")

    total_features = len(features_per_motif)
    monosemantic = (features_per_motif == 1).sum()
    polysemantic = (features_per_motif > 1).sum()

    if total_features > 0:
        print(f"\n  Total unique features: {total_features}")
        print(f"  Monosemantic rate: {100*monosemantic/total_features:.1f}%")
        print(f"  Polysemantic rate: {100*polysemantic/total_features:.1f}%")

    # Show most polysemantic features
    if polysemantic > 0:
        print(f"\nMost polysemantic features (top 10):")
        top_poly = features_per_motif.nlargest(10)
        for feature, count in top_poly.items():
            if count > 1:
                feature_data = filtered_df[filtered_df['feature'] == feature]
                motifs = ', '.join(feature_data['motif'].values)
                rpbs = ', '.join([f"{r:.3f}" for r in feature_data['rpb'].values])
                print(f"  {feature}: {count} motifs ({motifs}) [rpb: {rpbs}]")

def main():
    parser = argparse.ArgumentParser(
        description='Run interpretability experiments with configurable effect size threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: FDR < 0.05 and |rpb| >= 0.05
  python run_interpretability_experiments.py --latent_dim 512 --k 4

  # Strict: Only strong correlations
  python run_interpretability_experiments.py --latent_dim 512 --k 4 --min_rpb 0.10

  # Very strict: Very strong correlations only
  python run_interpretability_experiments.py --latent_dim 512 --k 4 --min_rpb 0.15

  # More random trials for robust statistics
  python run_interpretability_experiments.py --latent_dim 512 --k 4 --min_rpb 0.08 --n_random_trials 50

  # Test on mixed-motif graphs using dominant motif labels
  python run_interpretability_experiments.py --latent_dim 128 --k 8 --min_rpb 0.15 --use_mixed_motifs
        """
    )
    parser.add_argument('--variant', type=str, choices=['topk', 'gated', 'jumprelu', 'switch'],
                       help='SAE variant (auto-loads best config if specified)')
    parser.add_argument('--latent_dim', type=int, default=None,
                       help='SAE latent dimension (e.g., 128, 256, 512). If --variant specified, auto-loads.')
    parser.add_argument('--k', type=int, default=None,
                       help='SAE top-k sparsity (e.g., 4, 8, 16, 32). If --variant specified, auto-loads.')
    parser.add_argument('--min_rpb', type=float, default=0,
                       help='Minimum |rpb| threshold. Recommended: 0.05-0.15')
    parser.add_argument('--n_random_trials', type=int, default=20,
                       help='Number of random control trials per feature count (default: 20)')
    parser.add_argument('--use_mixed_motifs', action='store_true',
                       help='Run ablations on mixed-motif graphs (4000-4999) using dominant motif labels')
    args = parser.parse_args()

    # Handle variant-based config loading
    variant_kwargs = {}
    if args.variant:
        config_csv = Path('outputs/sae_config_comparison.csv')
        if not config_csv.exists():
            print(f"ERROR: {config_csv} not found!")
            print(f"  Phase 2 (compare_sae_configs.py) must complete first")
            return

        df_config = pd.read_csv(config_csv)
        variant_config = df_config[df_config['variant'] == args.variant]

        if len(variant_config) == 0:
            print(f"ERROR: No configuration found for variant {args.variant}")
            return

        best_config = variant_config.iloc[0]
        args.latent_dim = int(best_config['latent_dim'])

        # Extract variant-specific parameters
        if args.variant == 'topk':
            variant_kwargs['k'] = int(best_config.get('k', 8))
        elif args.variant == 'gated':
            variant_kwargs['sparsity_coef'] = float(best_config.get('sparsity_coef', 1e-3))
        elif args.variant == 'jumprelu':
            variant_kwargs['threshold_init'] = float(best_config.get('threshold_init', 0.01))
            variant_kwargs['bandwidth'] = float(best_config.get('bandwidth', 0.01))
        elif args.variant == 'switch':
            variant_kwargs['num_experts'] = int(best_config.get('num_experts', 8))
            variant_kwargs['latent_per_expert'] = int(best_config.get('latent_per_expert', 64))
            variant_kwargs['k_per_expert'] = int(best_config.get('k_per_expert', 8))

        # For backward compatibility, set args.k for topk
        args.k = variant_kwargs.get('k', 8)

        print(f"✓ Auto-loaded best config for {args.variant.upper()}:")
        print(f"  latent_dim={args.latent_dim}")
        for key, val in variant_kwargs.items():
            print(f"  {key}={val}")
        print(f"  composite_score={best_config.get('composite_score', 'N/A')}\n")
    else:
        # If no variant specified, require explicit latent_dim and k
        if args.latent_dim is None or args.k is None:
            print("ERROR: Either --variant OR (--latent_dim AND --k) must be specified")
            return
        # For backward compatibility, set topk as default variant with k parameter
        variant_kwargs['k'] = args.k

    # Validate inputs
    if args.min_rpb < 0 or args.min_rpb > 1:
        print("ERROR: --min_rpb must be between 0 and 1")
        return

    if args.n_random_trials < 10:
        print("WARNING: Less than 10 random trials may not provide robust statistics")

    # Setup
    results_dir, plots_dir = setup_directories(args.latent_dim, args.k, args.min_rpb, args.use_mixed_motifs, args.variant)

    print("="*70)
    print("MOTIF-SPECIFIC ABLATION WITH EFFECT SIZE FILTERING")
    print("="*70)
    print(f"SAE Configuration:")
    print(f"  latent_dim={args.latent_dim}, k={args.k}")
    print(f"Graph Type:")
    print(f"  {'Mixed-motif graphs (4000-4999) with dominant motif labels' if args.use_mixed_motifs else 'Single-motif graphs (0-3999)'}")
    print(f"Feature Selection:")
    print(f"  FDR threshold: < 0.05")
    print(f"  Min |rpb| threshold: >= {args.min_rpb}")
    print(f"Random Controls:")
    print(f"  Trials per feature count: {args.n_random_trials}")

    # Load data with filtering
    motif_features, non_filtered_features, filtered_df = load_correlation_data(args.min_rpb, args.variant)

    if motif_features is None:
        print("ERROR: Could not load correlation data")
        return

    save_feature_mapping(motif_features, results_dir, args.min_rpb)

    print(f"\nNon-filtered features available for random controls: {len(non_filtered_features)}")
    print("\nFiltered features per motif:")
    for motif_name, features in motif_features.items():
        print(f"  {motif_name}: {len(features)} features")

    # Check if we have any features to test
    total_features = sum(len(feats) for feats in motif_features.values())
    if total_features == 0:
        print("\n" + "="*70)
        print("ERROR: No features meet filtering criteria!")
        print("="*70)
        print(f"Try lowering --min_rpb (current: {args.min_rpb})")
        print("Recommended values: 0.05 (inclusive), 0.08 (moderate), 0.10 (strict)")
        return

    # Print monosemanticity report
    print_monosemanticity_report(filtered_df, args.min_rpb)

    # Run motif-specific ablations
    print("\n" + "="*70)
    print("STEP 1: Motif-Specific Ablations")
    print("="*70)

    motif_specific_results = []

    for feature_set_name, features in motif_features.items():
        if len(features) == 0:
            print(f"\nSkipping {feature_set_name} (no features meet threshold)")
            continue

        print(f"\nAblating {feature_set_name} features ({len(features)})...")
        # Generate variant-aware experiment name with all variant-specific parameters
        experiment_name = generate_experiment_name(feature_set_name, args.variant, args.latent_dim, variant_kwargs)

        df = run_single_ablation(args.latent_dim, features, experiment_name, args.use_mixed_motifs, args.variant, variant_kwargs)

        if df is not None and len(df) > 0:
            # Check if required columns exist
            if 'Motif' not in df.columns or 'Ablation Impact' not in df.columns:
                print(f"  Warning: Ablation results missing required columns. Available: {df.columns.tolist()}")
                continue
                
            for motif in df['Motif'].unique():
                motif_data = df[df['Motif'] == motif]
                if len(motif_data) > 0:
                    motif_specific_results.append({
                        'Feature_Set': feature_set_name,
                        'Motif': motif,
                        'N_Features': len(features),
                        'N_Graphs': len(motif_data),
                        'Mean_Impact': motif_data['Ablation Impact'].mean(),
                        'Std_Impact': motif_data['Ablation Impact'].std(),
                        'SE_Impact': motif_data['Ablation Impact'].std() / np.sqrt(len(motif_data))
                    })
        else:
            print(f"  Warning: Ablation experiment returned no results for {feature_set_name}")

    motif_specific_df = pd.DataFrame(motif_specific_results)
    
    # Check if we have any results
    if len(motif_specific_df) == 0:
        print("\nWARNING: No motif-specific ablation results collected!")
        print("This may be because:")
        print("  1. No features met the filtering criteria")
        print("  2. Ablation experiments failed")
        print("Skipping statistical analysis and plotting.")
        
        # Still save empty results file
        motif_specific_df.to_csv(results_dir / 'motif_specific_results.csv', index=False)
        
        # Create empty config file so Phase 3b can detect that Phase 3a ran (even if unsuccessfully)
        phase_3a_config = {
            'variant': args.variant,
            'latent_dim': args.latent_dim,
            'variant_kwargs': variant_kwargs,
            'min_rpb': args.min_rpb,
            'n_random_trials': args.n_random_trials,
            'use_mixed_motifs': args.use_mixed_motifs,
            'results_dir': str(results_dir),
            'plots_dir': str(plots_dir),
            'status': 'incomplete_no_results'
        }
        config_path = Path('phase_3a_config.json')
        target_config = Path('ablations/phase_3a_config.json')
        target_config.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(phase_3a_config, f, indent=2)
        with open(target_config, 'w') as f:
            json.dump(phase_3a_config, f, indent=2)
        print(f"\nSaved phase_3a_config.json (status: incomplete)")
        return
    
    motif_specific_df.to_csv(results_dir / 'motif_specific_results.csv', index=False)

    # Run random controls for each unique feature count
    print("\n" + "="*70)
    print("STEP 2: Random Control Trials")
    print("="*70)

    unique_feature_counts = sorted(set(len(feats) for feats in motif_features.values() if len(feats) > 0))
    random_results_dict = {}

    for n_features in unique_feature_counts:
        random_df = run_random_control_trials(args.latent_dim, n_features,
                                              args.n_random_trials, non_filtered_features,
                                              args.use_mixed_motifs, args.variant, variant_kwargs)
        if random_df is not None and len(random_df) > 0:
            random_df.to_csv(results_dir / f'random_{n_features}feat_trials.csv', index=False)
            random_results_dict[n_features] = random_df

    # Statistical analysis
    print("\n" + "="*70)
    print("STEP 3: Statistical Comparison")
    print("="*70)

    if len(random_results_dict) == 0:
        print("WARNING: No random control results available. Skipping statistical analysis.")
        sig_df = pd.DataFrame()
    else:
        combined_random = pd.concat(random_results_dict.values())
        
        # Check if motif_specific_df has required columns
        if 'Feature_Set' not in motif_specific_df.columns:
            print("ERROR: motif_specific_df missing 'Feature_Set' column")
            print(f"Available columns: {motif_specific_df.columns.tolist()}")
            sig_df = pd.DataFrame()
        else:
            sig_df = compute_statistical_significance(motif_specific_df, combined_random)
    
    if len(sig_df) > 0:
        sig_df.to_csv(results_dir / 'statistical_tests.csv', index=False)

    if len(sig_df) > 0:
        print("\nStatistical Significance Results:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(sig_df.to_string())
    else:
        print("\nNo statistical significance results to display.")

    # Generate plots
    print("\n" + "="*70)
    print("STEP 4: Generate Visualizations")
    print("="*70)

    if len(motif_specific_df) > 0 and len(random_results_dict) > 0:
        plot_comparison_with_random_controls(motif_specific_df, random_results_dict,
                                            motif_features, plots_dir,
                                            args.latent_dim, args.k, args.min_rpb)
    else:
        print("Skipping plots: insufficient data")

    # Save phase_3a_config.json with all metadata
    phase_3a_config = {
        'variant': args.variant,
        'latent_dim': args.latent_dim,
        'variant_kwargs': variant_kwargs,
        'min_rpb': args.min_rpb,
        'n_random_trials': args.n_random_trials,
        'use_mixed_motifs': args.use_mixed_motifs,
        'results_dir': str(results_dir),
        'plots_dir': str(plots_dir)
    }
    # Save to both locations for compatibility
    config_path = Path('phase_3a_config.json')
    target_config = Path('ablations/phase_3a_config.json')
    target_config.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(phase_3a_config, f, indent=2)
    with open(target_config, 'w') as f:
        json.dump(phase_3a_config, f, indent=2)
    print(f"\nSaved phase_3a_config.json to both current directory and ablations/ for Phase 3b/3c to use")

    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Configuration:")
    print(f"  SAE: latent_dim={args.latent_dim}, k={args.k}")
    print(f"  Filtering: FDR<0.05 and |rpb|>={args.min_rpb}")
    print(f"  Random trials: {args.n_random_trials} per feature count")
    print(f"\nTotal features tested: {sum(len(f) for f in motif_features.values())}")
    print(f"Monosemantic features: {(filtered_df.groupby('feature')['motif'].count() == 1).sum()}")
    print(f"\nOutput directories:")
    print(f"  Results: {results_dir}")
    print(f"  Plots: {plots_dir}")

if __name__ == "__main__":
    main()
