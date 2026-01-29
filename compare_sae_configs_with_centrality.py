#!/usr/bin/env python3
"""
SAE Hyperparameter Comparison Script with Degree + Centrality Controls

Runs motif analysis across different SAE configurations (latent_dim, k combinations)
and summarizes key metrics to identify optimal hyperparameters.

This version controls for BOTH degree AND centrality confounds when computing
partial correlations between SAE features and motif membership.

Usage:
    python compare_sae_configs_with_centrality.py

Output:
    - CSV file with comparison metrics
    - Summary table printed to console
    - Recommended configuration based on composite score

Prerequisites:
    Run compute_node_features_with_centrality.py first to add centrality features
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pointbiserialr
from statsmodels.stats.multitest import multipletests
import torch
from sparse_autoencoder import SparseAutoencoder
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_DIM = 64  # Layer 3 hidden activations (updated from 80)
N_PERMUTATIONS = 1000
SIGNIFICANCE_LEVEL = 0.05
CONTROL_FOR_TOPO = True  # Enable topological confound controls (degree + centrality)

# Hyperparameter grid to test
CONFIGS = [
    # (latent_dim, k, description)
    (128, 4, "Low capacity, low sparsity"),
    (128, 16, "Low capacity, moderate sparsity"),
    (128, 8, "Low capacity, high sparsity"),
    (256, 4, "Medium capacity, very low sparsity"),
    (256, 32, "Medium capacity, low sparsity"),
    (256, 16, "Medium capacity, moderate sparsity"),
    (256, 8, "Medium capacity, high sparsity"),
    (512, 4, "High capacity, very low sparsity"),
    (512, 32, "High capacity, low sparsity"),
    (512, 16, "High capacity, moderate sparsity"),
    (512, 8, "High capacity, high sparsity"),
]

def load_data_and_model(latent_dim, k):
    """Load SAE model and prepare data."""
    # Load model
    checkpoint_path = f"checkpoints/sae_latent{latent_dim}_k{k}.pt"
    if not Path(checkpoint_path).exists():
        return None, None, None

    model = SparseAutoencoder(input_dim=INPUT_DIM, latent_dim=latent_dim, k=k)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test graph IDs
    with open('outputs/test_graph_ids.json', 'r') as f:
        test_graph_ids = json.load(f)['graph_ids']

    # Extract latent representations
    activation_dir = Path("outputs/activations/layer3_new/test")
    metadata_dir = Path("virtual_graphs/data/all_graphs/graph_motif_metadata")

    all_latents = []
    all_motifs = []

    for graph_id in test_graph_ids:
        act_file = activation_dir / f"graph_{graph_id}.pt"
        metadata_file = metadata_dir / f"graph_{graph_id}_metadata.csv"

        if not act_file.exists() or not metadata_file.exists():
            continue

        activations = torch.load(act_file, weights_only=True)
        with torch.no_grad():
            latents = model.encode(activations)

        latents_np = latents.cpu().numpy()
        df_meta = pd.read_csv(metadata_file, index_col=0)

        if len(df_meta) != latents_np.shape[0]:
            continue

        for node_idx in range(latents_np.shape[0]):
            latent_row = [graph_id, node_idx] + latents_np[node_idx].tolist()
            all_latents.append(latent_row)

            motif_row = df_meta.iloc[node_idx].to_dict()
            motif_row['graph_id'] = graph_id
            motif_row['node_idx'] = node_idx
            all_motifs.append(motif_row)

    # Create DataFrames
    latent_cols = ['graph_id', 'node_idx'] + [f'z{i+1}' for i in range(latent_dim)]
    df_latents = pd.DataFrame(all_latents, columns=latent_cols)
    df_motifs = pd.DataFrame(all_motifs)
    df = pd.merge(df_latents, df_motifs, on=['graph_id', 'node_idx'])

    # Standardize motif column names
    rename_map = {
        'feedforward_loop': 'in_feedforward_loop',
        'feedback_loop': 'in_feedback_loop',
        'single_input_module': 'in_single_input_module',
        'cascade': 'in_cascade',
    }
    for k_old, v in rename_map.items():
        if k_old in df.columns:
            df = df.rename(columns={k_old: v})

    return model, df, latent_dim

def compute_partial_correlation(binary_var, continuous_var, control_vars, df):
    """
    Compute partial point-biserial correlation controlling for confounds.

    Uses residualization approach:
    1. Regress continuous_var on control_vars, get residuals
    2. Compute point-biserial correlation between binary_var and residuals

    Args:
        binary_var: Binary variable name (e.g., 'in_single_input_module')
        continuous_var: Continuous variable name (e.g., 'z5')
        control_vars: List of control variable names (e.g., ['degree', 'out_degree'])
        df: DataFrame containing all variables

    Returns:
        Partial correlation coefficient and p-value
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np

    # Check if control variables exist and have variance
    valid_controls = []
    for ctrl in control_vars:
        if ctrl in df.columns and df[ctrl].std() > 0:
            valid_controls.append(ctrl)

    if len(valid_controls) == 0:
        # No valid controls, return regular correlation
        return pointbiserialr(df[binary_var], df[continuous_var])

    # Create a subset DataFrame with all required variables
    required_cols = valid_controls + [continuous_var, binary_var]
    df_subset = df[required_cols].copy()

    # Drop rows with NaN values in any of the required columns
    df_clean = df_subset.dropna()

    # Check if we have enough valid data points
    if len(df_clean) < 10:
        # Not enough valid data, fall back to regular correlation
        print(f"  Warning: Only {len(df_clean)} valid rows after removing NaN. Using bivariate correlation.")
        # Try regular correlation on clean data
        df_clean_basic = df[[binary_var, continuous_var]].dropna()
        if len(df_clean_basic) >= 10:
            return pointbiserialr(df_clean_basic[binary_var], df_clean_basic[continuous_var])
        else:
            return np.nan, np.nan

    # Check if binary variable has variance (both 0 and 1 present)
    if df_clean[binary_var].nunique() < 2:
        print(f"  Warning: Binary variable {binary_var} has no variance after NaN removal.")
        return np.nan, np.nan

    # Prepare control matrix and target variable
    X_control = df_clean[valid_controls].values
    y_continuous = df_clean[continuous_var].values
    y_binary = df_clean[binary_var].values

    # Regress continuous variable on controls
    reg = LinearRegression()
    reg.fit(X_control, y_continuous)
    residuals = y_continuous - reg.predict(X_control)

    # Compute correlation between binary variable and residuals
    partial_corr, partial_pval = pointbiserialr(y_binary, residuals)

    return partial_corr, partial_pval

def compute_correlations(df, latent_dim, control_for_topo=False):
    """
    Compute point-biserial correlations (with optional topological controls).

    Args:
        df: DataFrame with motif labels, SAE features, and (optionally) topo features
        latent_dim: Number of SAE latent dimensions
        control_for_topo: If True, compute partial correlations controlling for
                          degree + centrality features

    Returns:
        DataFrame with correlation results
    """
    latent_features = [f'z{i+1}' for i in range(latent_dim)]
    motif_types = ['in_feedforward_loop', 'in_feedback_loop',
                   'in_single_input_module', 'in_cascade']

    # Topological control variables (degree + centrality)
    # Include all available topological features
    topo_vars = [
        'degree', 'in_degree', 'out_degree',
        'betweenness_centrality', 'in_closeness_centrality',
        'out_closeness_centrality'
    ]
    available_topo_vars = [v for v in topo_vars if v in df.columns and df[v].std() > 0]

    correlations = []
    for motif in motif_types:
        if motif not in df.columns:
            continue
        for z_col in latent_features:
            if df[z_col].std() == 0:  # Skip constant features
                continue

            # Compute bivariate correlation
            corr, pval = pointbiserialr(df[motif], df[z_col])

            # Compute partial correlation (if controls available)
            partial_corr, partial_pval = None, None
            if control_for_topo and len(available_topo_vars) > 0:
                partial_corr, partial_pval = compute_partial_correlation(
                    motif, z_col, available_topo_vars, df
                )

            result = {
                'feature': z_col,
                'motif': motif,
                'rpb': corr,
                'pval': pval,
                'rpb_abs': abs(corr),
            }

            if partial_corr is not None:
                result['rpb_partial'] = partial_corr
                result['pval_partial'] = partial_pval
                result['rpb_partial_abs'] = abs(partial_corr)

            correlations.append(result)

    return pd.DataFrame(correlations)

def sanity_check_topo_confounds(df):
    """
    Sanity check: Compute correlations between motif labels and topological features.

    This quantifies how much motif membership is predicted by topology alone
    (degree + centrality). High correlations indicate potential confounding.

    Args:
        df: DataFrame with motif labels and topological features

    Returns:
        DataFrame with topology-motif correlations
    """
    motif_types = ['in_feedforward_loop', 'in_feedback_loop',
                   'in_single_input_module', 'in_cascade']

    # All topological features (degree + centrality)
    topo_vars = [
        'degree', 'in_degree', 'out_degree',
        'betweenness_centrality', 'in_closeness_centrality',
        'out_closeness_centrality'
    ]

    # Check if topological variables exist
    available_topo_vars = [v for v in topo_vars if v in df.columns]

    if len(available_topo_vars) == 0:
        print("  ⚠ No topological features found in data")
        print("     Run: python virtual_graphs/compute_node_features_with_centrality.py")
        return None

    results = []
    for motif in motif_types:
        if motif not in df.columns:
            continue
        for topo_var in available_topo_vars:
            if df[topo_var].std() == 0:
                continue
            corr, pval = pointbiserialr(df[motif], df[topo_var])
            results.append({
                'motif': motif,
                'topo_feature': topo_var,
                'rpb': corr,
                'pval': pval,
                'rpb_abs': abs(corr)
            })

    df_sanity = pd.DataFrame(results)

    if len(df_sanity) > 0:
        print("\n" + "="*70)
        print("SANITY CHECK: Topology-Motif Correlations")
        print("="*70)
        print("High correlations indicate potential confounding:")
        print("(SAE features may be detecting topology rather than motif semantics)\n")

        # Show highest correlations
        df_sanity_sorted = df_sanity.sort_values('rpb_abs', ascending=False)
        print("Top 12 strongest correlations:")
        for _, row in df_sanity_sorted.head(12).iterrows():
            motif_short = row['motif'].replace('in_', '')
            print(f"  {motif_short:20s} ~ {row['topo_feature']:25s}: rpb={row['rpb']:6.3f}, p={row['pval']:.2e}")

        # Separate analysis by feature type
        print("\nBreakdown by feature type:")
        degree_features = ['degree', 'in_degree', 'out_degree']
        centrality_features = [v for v in available_topo_vars if v not in degree_features]

        degree_corrs = df_sanity[df_sanity['topo_feature'].isin(degree_features)]
        centrality_corrs = df_sanity[df_sanity['topo_feature'].isin(centrality_features)]

        if len(degree_corrs) > 0:
            max_degree_corr = degree_corrs['rpb_abs'].max()
            print(f"  Degree features: max |rpb|={max_degree_corr:.3f}")

        if len(centrality_corrs) > 0:
            max_centrality_corr = centrality_corrs['rpb_abs'].max()
            print(f"  Centrality features: max |rpb|={max_centrality_corr:.3f}")

        print("\nInterpretation:")
        max_corr = df_sanity['rpb_abs'].max()
        if max_corr > 0.5:
            print(f"  ⚠ Strong confounding detected (max |rpb|={max_corr:.3f})")
            print(f"    → Partial correlations are CRITICAL for valid interpretation")
        elif max_corr > 0.3:
            print(f"  ⚠ Moderate confounding detected (max |rpb|={max_corr:.3f})")
            print(f"    → Partial correlations recommended")
        else:
            print(f"  ✓ Weak confounding (max |rpb|={max_corr:.3f})")
            print(f"    → Topology controls may not be necessary")

        print("="*70 + "\n")

    return df_sanity

def permutation_test(df, df_corr, latent_dim, n_permutations=1000):
    """Run permutation testing and compute empirical p-values."""
    latent_features = [f'z{i+1}' for i in range(latent_dim)]
    motif_types = ['in_feedforward_loop', 'in_feedback_loop',
                   'in_single_input_module', 'in_cascade']

    # Store null distributions
    null_distributions = {motif: {f: [] for f in latent_features} for motif in motif_types}

    for perm_idx in range(n_permutations):
        for motif in motif_types:
            if motif not in df.columns:
                continue
            shuffled_labels = df[motif].sample(frac=1, random_state=42+perm_idx).reset_index(drop=True)

            for z_col in latent_features:
                if df[z_col].std() == 0:
                    continue
                corr_perm, _ = pointbiserialr(shuffled_labels, df[z_col])
                null_distributions[motif][z_col].append(corr_perm)

    # Calculate empirical p-values
    df_corr['p_empirical'] = 1.0

    for idx, row in df_corr.iterrows():
        feature = row['feature']
        motif = row['motif']
        obs_rpb_abs = abs(row['rpb'])

        null_dist = null_distributions[motif][feature]
        if len(null_dist) == 0:
            continue

        p_empirical = (np.abs(null_dist) >= obs_rpb_abs).sum() / n_permutations
        df_corr.loc[idx, 'p_empirical'] = p_empirical

    # FDR correction
    reject, pvals_fdr, _, _ = multipletests(df_corr['p_empirical'],
                                            alpha=SIGNIFICANCE_LEVEL,
                                            method='fdr_bh')
    df_corr['p_fdr'] = pvals_fdr
    df_corr['significant_fdr'] = reject

    return df_corr

def compute_precision_recall(df, feature, motif, percentile=95):
    """Compute precision and recall for a feature-motif pair."""
    threshold = np.percentile(df[feature], percentile)
    activated = df[feature] > threshold
    present = df[motif] == 1
    tp = (activated & present).sum()

    precision = tp / activated.sum() if activated.sum() > 0 else 0
    recall = tp / present.sum() if present.sum() > 0 else 0

    return precision, recall

def analyze_configuration(latent_dim, k, run_sanity_check=False):
    """Run full analysis for one configuration."""
    print(f"\nAnalyzing: latent_dim={latent_dim}, k={k} ({100*k/latent_dim:.2f}% active)")

    # Load model and data
    model, df, latent_dim = load_data_and_model(latent_dim, k)
    if model is None or df is None:
        print(f"  ⚠ Checkpoint not found, skipping...")
        return None

    # Run sanity check (only for first config)
    if run_sanity_check:
        sanity_check_topo_confounds(df)

    # Compute correlations (with degree controls if enabled)
    df_corr = compute_correlations(df, latent_dim, control_for_topo=CONTROL_FOR_TOPO)
    if len(df_corr) == 0:
        print(f"  ⚠ No valid correlations, skipping...")
        return None

    # Determine which correlation column to use for metrics
    corr_col = 'rpb_partial_abs' if 'rpb_partial_abs' in df_corr.columns else 'rpb_abs'
    if CONTROL_FOR_TOPO and corr_col == 'rpb_partial_abs':
        print(f"  Using partial correlations (controlling for degree + centrality)")
    else:
        print(f"  Using bivariate correlations")

    print(f"  Skipping significance testing...")

    # Compute precision/recall for top features
    latent_features = [f'z{i+1}' for i in range(latent_dim)]
    motif_types = ['in_feedforward_loop', 'in_feedback_loop',
                   'in_single_input_module', 'in_cascade']

    precision_recall_results = []
    for motif in motif_types:
        if motif not in df.columns:
            continue
        motif_corrs = df_corr[df_corr['motif'] == motif].nlargest(10, corr_col)
        for _, row in motif_corrs.iterrows():
            feature = row['feature']
            precision, recall = compute_precision_recall(df, feature, motif)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            result_row = {
                'feature': feature,
                'motif': motif,
                'rpb': row['rpb'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            if 'rpb_partial' in row:
                result_row['rpb_partial'] = row['rpb_partial']
            precision_recall_results.append(result_row)

    df_pr = pd.DataFrame(precision_recall_results)

    # Calculate summary metrics
    n_features_tested = df_corr['feature'].nunique()
    activation_counts = {f'z{i+1}': (df[f'z{i+1}'] > 0).sum() for i in range(latent_dim)}
    n_active_features = sum(1 for count in activation_counts.values() if count > 0)
    dead_feature_rate = 1 - (n_active_features / latent_dim)

    # Top metrics (use partial correlation if available)
    max_rpb = df_corr[corr_col].max() if len(df_corr) > 0 else 0
    max_rpb_feature = df_corr.loc[df_corr[corr_col].idxmax()] if len(df_corr) > 0 else None

    best_f1 = df_pr['f1_score'].max() if len(df_pr) > 0 else 0
    best_f1_row = df_pr.loc[df_pr['f1_score'].idxmax()] if len(df_pr) > 0 else None

    best_precision = df_pr['precision'].max() if len(df_pr) > 0 else 0
    best_recall = df_pr['recall'].max() if len(df_pr) > 0 else 0

    # Composite quality score (without significance testing)
    capacity_utilization = 1 - dead_feature_rate

    composite_score = (
        0.50 * min(max_rpb / 0.5, 1.0) +            # Effect size (50%, capped at 0.5)
        0.35 * best_f1 +                             # Predictive power (35%)
        0.15 * min(capacity_utilization, 1.0)        # Capacity utilization (15%)
    )

    # Load reconstruction metrics from SAE training
    metrics_path = f"outputs/sae_metrics_latent{latent_dim}_k{k}.json"
    test_reconstruction = None
    if Path(metrics_path).exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                test_reconstruction = metrics.get('test_reconstruction', None)
        except (json.JSONDecodeError, KeyError):
            print(f"  ⚠ Could not load reconstruction metrics from {metrics_path}")

    results = {
        'latent_dim': latent_dim,
        'k': k,
        'sparsity_pct': 100 * k / latent_dim,
        'n_features_tested': n_features_tested,
        'n_active_features': n_active_features,
        'dead_feature_rate': dead_feature_rate,
        'max_rpb_abs': max_rpb,
        'max_rpb_feature': max_rpb_feature['feature'] if max_rpb_feature is not None else 'N/A',
        'max_rpb_motif': max_rpb_feature['motif'] if max_rpb_feature is not None else 'N/A',
        'best_f1': best_f1,
        'best_f1_feature': best_f1_row['feature'] if best_f1_row is not None else 'N/A',
        'best_f1_motif': best_f1_row['motif'] if best_f1_row is not None else 'N/A',
        'best_precision': best_precision,
        'best_recall': best_recall,
        'test_reconstruction': test_reconstruction,
        'composite_score': composite_score,
    }

    print(f"  ✓ Max |rpb|: {max_rpb:.3f}")
    print(f"  ✓ Best F1: {best_f1:.3f}")
    print(f"  ✓ Composite score: {composite_score:.3f}")

    return results

def main():
    """Run two-stage analysis: fast screening then deep analysis on top configs."""
    print("="*70)
    print("SAE HYPERPARAMETER COMPARISON - TWO-STAGE ANALYSIS")
    print("="*70)
    print(f"\nTotal configurations to test: {len(CONFIGS)}")

    # STAGE 1: Fast screening (all configs, no permutation testing)
    print("\n" + "="*70)
    print("STAGE 1: Fast Screening (All Configurations)")
    print("="*70)
    print("Running correlation analysis and precision/recall metrics...")
    print("Skipping permutation tests in this stage for speed.\n")

    results = []

    for idx, (latent_dim, k, description) in enumerate(tqdm(CONFIGS, desc="Stage 1")):
        # Run sanity check only on first config
        run_sanity = (idx == 0) and CONTROL_FOR_TOPO
        config_result = analyze_configuration(latent_dim, k, run_sanity_check=run_sanity)
        if config_result is not None:
            config_result['description'] = description
            config_result['significance_tested'] = False
            config_result['n_significant_features'] = None
            config_result['max_significant_rpb'] = None
            results.append(config_result)

    if len(results) == 0:
        print("\n⚠ No configurations completed successfully!")
        return

    # Create summary DataFrame
    df_results = pd.DataFrame(results)

    # Sort by composite score
    df_results = df_results.sort_values('composite_score', ascending=False)

    # Print Stage 1 summary
    print("\n" + "="*70)
    print("STAGE 1 RESULTS: Top 5 by Composite Score")
    print("="*70)
    display_cols_stage1 = [
        'latent_dim', 'k', 'sparsity_pct', 'max_rpb_abs',
        'best_f1', 'test_reconstruction', 'composite_score'
    ]
    top5_stage1 = df_results.head(5)[display_cols_stage1].copy()
    top5_stage1['sparsity_pct'] = top5_stage1['sparsity_pct'].round(2)
    top5_stage1['max_rpb_abs'] = top5_stage1['max_rpb_abs'].round(3)
    top5_stage1['best_f1'] = top5_stage1['best_f1'].round(3)
    top5_stage1['composite_score'] = top5_stage1['composite_score'].round(3)
    print(top5_stage1.to_string(index=False))

    # STAGE 2: Deep analysis on top 3 configs
    print("\n" + "="*70)
    print("STAGE 2: Deep Analysis (Top 3 Candidates)")
    print("="*70)
    print("Running full permutation tests (1000 permutations) on top candidates...")
    print("This may take several minutes per configuration.\n")

    top_n = min(3, len(df_results))
    top_configs = df_results.head(top_n)

    for idx, row in top_configs.iterrows():
        latent_dim_val = int(row['latent_dim'])
        k_val = int(row['k'])

        print(f"\n[{idx+1}/{top_n}] Testing latent_dim={latent_dim_val}, k={k_val}...")
        print(f"      Composite score: {row['composite_score']:.3f}")

        # Re-load data and run permutation tests
        model, df, latent_dim = load_data_and_model(latent_dim_val, k_val)
        if model is None or df is None:
            print(f"  ⚠ Could not load data, skipping significance test")
            continue

        # Compute correlations (with degree controls if enabled)
        df_corr = compute_correlations(df, latent_dim, control_for_topo=CONTROL_FOR_TOPO)
        if len(df_corr) == 0:
            print(f"  ⚠ No valid correlations, skipping significance test")
            continue

        # Determine which correlation column to use
        corr_col = 'rpb_partial_abs' if 'rpb_partial_abs' in df_corr.columns else 'rpb_abs'

        # Run permutation tests
        print(f"  Running permutation tests...")
        df_corr = permutation_test(df, df_corr, latent_dim, n_permutations=N_PERMUTATIONS)

        # Extract significance metrics (use controlled correlations if available)
        n_significant = df_corr['significant_fdr'].sum()
        max_significant_rpb = df_corr[df_corr['significant_fdr']][corr_col].max() if n_significant > 0 else 0.0

        # Update results
        df_results.loc[idx, 'significance_tested'] = True
        df_results.loc[idx, 'n_significant_features'] = int(n_significant)
        df_results.loc[idx, 'max_significant_rpb'] = float(max_significant_rpb)

        print(f"  ✓ Found {n_significant} significant features (FDR < 0.05)")
        print(f"  ✓ Max significant |rpb|: {max_significant_rpb:.3f}")

    # Re-sort by composite score after adding significance data
    df_results = df_results.sort_values('composite_score', ascending=False)

    # Apply tiebreaker logic using reconstruction loss
    print("\n" + "="*70)
    print("FINAL SELECTION (with Tiebreaker Logic)")
    print("="*70)

    top_score = df_results.iloc[0]['composite_score']
    top_candidates = df_results[df_results['composite_score'] >= top_score - 0.05]

    if len(top_candidates) > 1 and top_candidates['test_reconstruction'].notna().any():
        # Multiple configs within 0.05 composite score - use reconstruction as tiebreaker
        valid_recon = top_candidates[top_candidates['test_reconstruction'].notna()]
        if len(valid_recon) > 0:
            best_idx = valid_recon['test_reconstruction'].idxmin()
            best = df_results.loc[best_idx]
            print(f"\nFound {len(top_candidates)} configs within 0.05 of top composite score.")
            print(f"Using reconstruction loss as tiebreaker (lower is better).\n")
            print("Tiebreaker candidates:")
            tb_cols = ['latent_dim', 'k', 'composite_score', 'test_reconstruction']
            print(top_candidates[tb_cols].to_string(index=False))
        else:
            best = df_results.iloc[0]
            print(f"\nTop scorer selected (no reconstruction data for tiebreaking).")
    else:
        best = df_results.iloc[0]
        print(f"\nClear winner: latent_dim={int(best['latent_dim'])}, k={int(best['k'])}")
        print(f"Composite score: {best['composite_score']:.3f} (no close competitors)")

    # Save to CSV
    output_file = 'outputs/sae_config_comparison.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Saved detailed results to {output_file}")

    # Print Stage 2 summary table
    print("\n" + "="*70)
    print("STAGE 2 RESULTS: Top 3 with Significance Testing")
    print("="*70)

    display_cols = [
        'latent_dim', 'k', 'sparsity_pct', 'max_rpb_abs', 'n_significant_features',
        'max_significant_rpb', 'best_f1', 'test_reconstruction', 'composite_score'
    ]

    top3 = df_results.head(3)[display_cols].copy()
    top3['sparsity_pct'] = top3['sparsity_pct'].round(2)
    top3['max_rpb_abs'] = top3['max_rpb_abs'].round(3)
    top3['best_f1'] = top3['best_f1'].round(3)
    top3['composite_score'] = top3['composite_score'].round(3)
    top3['max_significant_rpb'] = top3['max_significant_rpb'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    top3['n_significant_features'] = top3['n_significant_features'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "N/A")

    print(top3.to_string(index=False))

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDED CONFIGURATION")
    print("="*70)
    print(f"\n  latent_dim = {int(best['latent_dim'])}")
    print(f"  k = {int(best['k'])}")
    print(f"  Sparsity: {best['sparsity_pct']:.2f}% ({best['description']})")
    print(f"\n  Key Metrics:")
    print(f"    • Max correlation: |rpb| = {best['max_rpb_abs']:.3f}")
    print(f"    • Best F1 score: {best['best_f1']:.3f}")
    if pd.notna(best.get('test_reconstruction')):
        print(f"    • Reconstruction loss: {best['test_reconstruction']:.2e}")
    if pd.notna(best.get('n_significant_features')):
        print(f"    • Significant features (FDR<0.05): {int(best['n_significant_features'])}")
        print(f"    • Max significant |rpb|: {best['max_significant_rpb']:.3f}")
    print(f"    • Active features: {int(best['n_active_features'])}/{int(best['latent_dim'])} ({100*(1-best['dead_feature_rate']):.1f}%)")
    print(f"    • Composite score: {best['composite_score']:.3f}")

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
  Two-Stage Selection Process:
    1. Stage 1: Fast screening of all configs using composite score
    2. Stage 2: Full permutation tests (1000 perms) on top 3 only
    3. Tiebreaker: If top scores within 0.05, choose lower reconstruction loss

  Composite Score Components:
    • 50% - Max correlation (effect size)
    • 35% - Best F1 score (predictive performance)
    • 15% - Capacity utilization (1 - dead feature rate)

  Good Configuration Indicators:
    ✓ Composite score > 0.5
    ✓ Max |rpb| > 0.3
    ✓ Best F1 > 0.3
    ✓ Significant features (FDR<0.05) > 20
    ✓ Low reconstruction loss (< 1e-6)
    ✓ Dead feature rate < 0.5

  To use the recommended configuration, update your code:
    LATENT_DIM = {best_latent}
    K = {best_k}
    """.format(best_latent=int(best['latent_dim']), best_k=int(best['k'])))

if __name__ == "__main__":
    main()
