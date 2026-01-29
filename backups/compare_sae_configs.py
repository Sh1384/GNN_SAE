#!/usr/bin/env python3
"""
SAE Hyperparameter Comparison Script with Topological Confound Controls

Runs motif analysis across different SAE configurations across all 4 variants
(TopK, Gated, JumpReLU, Switch) and summarizes key metrics to identify
optimal hyperparameters.

This version controls for degree and centrality confounds by computing partial
correlations between SAE features and motif membership, residualizing out
topological features (degree, betweenness centrality, closeness centrality).

Usage:
    python compare_sae_configs.py                    # Default: sort by max_rpb_abs
    python compare_sae_configs.py --metric max_rpb_abs
    python compare_sae_configs.py --metric composite_score

Output:
    - CSV file with comparison metrics
    - Summary table printed to console
    - Recommended configuration based on selected metric
    - Sanity check analysis showing topology-motif correlations (first config only)
"""

import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pointbiserialr
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
import torch
from sparse_autoencoder import BaseSAE, TopKSAE, GatedSAE, JumpReLUSAE, SwitchSAE
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_DIM = 64
N_PERMUTATIONS = 1000
SIGNIFICANCE_LEVEL = 0.05
SEED = 42
CONTROL_FOR_TOPO = True  # Control for degree + centrality confounds

# Hyperparameter grid to test for each variant
VARIANT_CONFIGS = {
    'topk': [
        # (latent_dim, k, description)
        (128, 4, "Low capacity, low sparsity"),
        (128, 8, "Low capacity, high sparsity"),
        (128, 16, "Low capacity, moderate sparsity"),
        (256, 4, "Medium capacity, very low sparsity"),
        (256, 8, "Medium capacity, high sparsity"),
        (256, 16, "Medium capacity, moderate sparsity"),
        (256, 32, "Medium capacity, low sparsity"),
        (512, 4, "High capacity, very low sparsity"),
        (512, 8, "High capacity, high sparsity"),
        (512, 16, "High capacity, moderate sparsity"),
        (512, 32, "High capacity, low sparsity"),
    ],
    'gated': [
        # (latent_dim, sparsity_coef, description)
        (128, 1e-4, "Low capacity, low sparsity penalty"),
        (128, 5e-4, "Low capacity, medium sparsity penalty"),
        (128, 1e-3, "Low capacity, high sparsity penalty"),
        (256, 1e-4, "Medium capacity, low sparsity penalty"),
        (256, 5e-4, "Medium capacity, medium sparsity penalty"),
        (256, 1e-3, "Medium capacity, high sparsity penalty"),
        (512, 1e-4, "High capacity, low sparsity penalty"),
        (512, 5e-4, "High capacity, medium sparsity penalty"),
        (512, 1e-3, "High capacity, high sparsity penalty"),
    ],
    'jumprelu': [
        # (latent_dim, threshold_init, description)
        (128, 0.01, "Low capacity, low threshold"),
        (128, 0.1, "Low capacity, high threshold"),
        (256, 0.01, "Medium capacity, low threshold"),
        (256, 0.1, "Medium capacity, high threshold"),
        (512, 0.01, "High capacity, low threshold"),
        (512, 0.1, "High capacity, high threshold"),
    ],
    'switch': [
        # (num_experts, latent_per_expert, k_per_expert, description)
        (4, 64, 8, "4 experts, 64D per expert, k=8"),
        (4, 128, 16, "4 experts, 128D per expert, k=16"),
        (8, 64, 8, "8 experts, 64D per expert, k=8"),
        (8, 128, 16, "8 experts, 128D per expert, k=16"),
    ],
}

def load_data_and_model(latent_dim, k):
    """Load TopK SAE model and prepare data."""
    # Load model
    checkpoint_path = f"checkpoints/sae_topk_latent{latent_dim}_k{k}_seed{SEED}.pt"
    if not Path(checkpoint_path).exists():
        return None, None, None

    model = TopKSAE(input_dim=INPUT_DIM, latent_dim=latent_dim, k=k)
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

    # CACHE LATENTS: Save to disk for Phase 4a to avoid re-encoding
    cache_dir = Path('outputs/latent_cache')
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f'latents_topk_latent{latent_dim}_k{k}.pkl'

    import pickle
    cache_data = {
        'variant': 'topk',
        'latent_dim': latent_dim,
        'k': k,
        'df': df,  # Complete dataframe: latents + motif labels
        'test_graph_ids': test_graph_ids,
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"\n✓ Cached latents to {cache_file}")
    print(f"  Size: {len(df)} nodes × {latent_dim} latent dims")
    print(f"  This avoids re-encoding during Phase 4a (latent caching optimization)")

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
        control_vars: List of control variable names (e.g., ['degree', 'betweenness_centrality'])
        df: DataFrame containing all variables

    Returns:
        Partial correlation coefficient and p-value
    """
    from sklearn.linear_model import LinearRegression

    # Check if control variables exist and have variance
    valid_controls = []
    for ctrl in control_vars:
        if ctrl in df.columns and df[ctrl].std() > 0:
            valid_controls.append(ctrl)

    if len(valid_controls) == 0:
        # No valid controls, return regular correlation
        return pointbiserialr(df[binary_var], df[continuous_var])

    # Create subset with required columns
    required_cols = valid_controls + [continuous_var, binary_var]
    df_subset = df[required_cols].copy()

    # Drop rows with NaN values
    df_clean = df_subset.dropna()

    # Check if we have enough valid data points
    if len(df_clean) < 10:
        df_clean_basic = df[[binary_var, continuous_var]].dropna()
        if len(df_clean_basic) >= 10:
            return pointbiserialr(df_clean_basic[binary_var], df_clean_basic[continuous_var])
        else:
            return np.nan, np.nan

    # Check if binary variable has variance
    if df_clean[binary_var].nunique() < 2:
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

            if partial_corr is not None and not np.isnan(partial_corr):
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
        print("  ⚠ No topological features found in metadata")
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

        # Breakdown by feature type
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

    # Compute correlations (with degree + centrality controls if enabled)
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
            precision_recall_results.append({
                'feature': feature,
                'motif': motif,
                'rpb': row['rpb'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

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
        'composite_score': composite_score,
    }

    print(f"  ✓ Max |rpb|: {max_rpb:.3f}")
    print(f"  ✓ Best F1: {best_f1:.3f}")
    print(f"  ✓ Composite score: {composite_score:.3f}")

    return results, df_corr


def analyze_topk_configuration(latent_dim, k, run_sanity_check=False):
    """Wrapper for TopK SAE configuration analysis. Returns (results_dict, correlation_dataframe)."""
    return analyze_configuration(latent_dim, k, run_sanity_check=run_sanity_check)


def analyze_gated_configuration(latent_dim, sparsity_coef, run_sanity_check=False):
    """Analyze Gated SAE configuration."""
    print(f"\nAnalyzing: Gated SAE - latent_dim={latent_dim}, sparsity_coef={sparsity_coef:.0e}")

    # Load model and data
    model, df, latent_dim = load_data_and_model_gated(latent_dim, sparsity_coef)
    if model is None or df is None:
        print(f"  ⚠ Checkpoint not found, skipping...")
        return None

    # Run sanity check (only for first config)
    if run_sanity_check:
        sanity_check_topo_confounds(df)

    # Compute correlations (with degree + centrality controls if enabled)
    df_corr = compute_correlations(df, latent_dim, control_for_topo=CONTROL_FOR_TOPO)
    if len(df_corr) == 0:
        print(f"  ⚠ No valid correlations, skipping...")
        return None

    # Determine which correlation column to use for metrics
    corr_col = 'rpb_partial_abs' if 'rpb_partial_abs' in df_corr.columns else 'rpb_abs'

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
            precision_recall_results.append({
                'feature': feature,
                'motif': motif,
                'rpb': row['rpb'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

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

    # Composite quality score
    capacity_utilization = 1 - dead_feature_rate

    composite_score = (
        0.50 * min(max_rpb / 0.5, 1.0) +
        0.35 * best_f1 +
        0.15 * min(capacity_utilization, 1.0)
    )

    results = {
        'latent_dim': latent_dim,
        'sparsity_coef': sparsity_coef,
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
        'composite_score': composite_score,
    }

    print(f"  ✓ Max |rpb|: {max_rpb:.3f}")
    print(f"  ✓ Best F1: {best_f1:.3f}")
    print(f"  ✓ Composite score: {composite_score:.3f}")

    return results, df_corr


def analyze_jumprelu_configuration(latent_dim, threshold_init, run_sanity_check=False):
    """Analyze JumpReLU SAE configuration."""
    print(f"\nAnalyzing: JumpReLU SAE - latent_dim={latent_dim}, threshold_init={threshold_init:.0e}")

    # Load model and data
    model, df, latent_dim = load_data_and_model_jumprelu(latent_dim, threshold_init)
    if model is None or df is None:
        print(f"  ⚠ Checkpoint not found, skipping...")
        return None

    # Run sanity check (only for first config)
    if run_sanity_check:
        sanity_check_topo_confounds(df)

    # Compute correlations (with degree + centrality controls if enabled)
    df_corr = compute_correlations(df, latent_dim, control_for_topo=CONTROL_FOR_TOPO)
    if len(df_corr) == 0:
        print(f"  ⚠ No valid correlations, skipping...")
        return None

    # Determine which correlation column to use for metrics
    corr_col = 'rpb_partial_abs' if 'rpb_partial_abs' in df_corr.columns else 'rpb_abs'

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
            precision_recall_results.append({
                'feature': feature,
                'motif': motif,
                'rpb': row['rpb'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

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

    # Composite quality score
    capacity_utilization = 1 - dead_feature_rate

    composite_score = (
        0.50 * min(max_rpb / 0.5, 1.0) +
        0.35 * best_f1 +
        0.15 * min(capacity_utilization, 1.0)
    )

    results = {
        'latent_dim': latent_dim,
        'threshold_init': threshold_init,
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
        'composite_score': composite_score,
    }

    print(f"  ✓ Max |rpb|: {max_rpb:.3f}")
    print(f"  ✓ Best F1: {best_f1:.3f}")
    print(f"  ✓ Composite score: {composite_score:.3f}")

    return results, df_corr


def analyze_switch_configuration(num_experts, latent_per_expert, k_per_expert, run_sanity_check=False):
    """Analyze Switch SAE configuration."""
    total_latent = num_experts * latent_per_expert
    print(f"\nAnalyzing: Switch SAE - experts={num_experts}, latent_per_expert={latent_per_expert}, k={k_per_expert}")

    # Load model and data
    model, df, latent_dim = load_data_and_model_switch(num_experts, latent_per_expert, k_per_expert)
    if model is None or df is None:
        print(f"  ⚠ Checkpoint not found, skipping...")
        return None

    # Run sanity check (only for first config)
    if run_sanity_check:
        sanity_check_topo_confounds(df)

    # Compute correlations (with degree + centrality controls if enabled)
    df_corr = compute_correlations(df, latent_dim, control_for_topo=CONTROL_FOR_TOPO)
    if len(df_corr) == 0:
        print(f"  ⚠ No valid correlations, skipping...")
        return None

    # Determine which correlation column to use for metrics
    corr_col = 'rpb_partial_abs' if 'rpb_partial_abs' in df_corr.columns else 'rpb_abs'

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
            precision_recall_results.append({
                'feature': feature,
                'motif': motif,
                'rpb': row['rpb'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

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

    # Compute effective sparsity for Switch SAE
    effective_sparsity_pct = 100 * k_per_expert / latent_per_expert

    # Composite quality score
    capacity_utilization = 1 - dead_feature_rate

    composite_score = (
        0.50 * min(max_rpb / 0.5, 1.0) +
        0.35 * best_f1 +
        0.15 * min(capacity_utilization, 1.0)
    )

    results = {
        'latent_dim': total_latent,
        'num_experts': num_experts,
        'latent_per_expert': latent_per_expert,
        'k_per_expert': k_per_expert,
        'sparsity_pct': effective_sparsity_pct,
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
        'composite_score': composite_score,
    }

    print(f"  ✓ Max |rpb|: {max_rpb:.3f}")
    print(f"  ✓ Best F1: {best_f1:.3f}")
    print(f"  ✓ Composite score: {composite_score:.3f}")

    return results, df_corr


def main():
    """Run analysis for all configurations and generate summary."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='SAE Hyperparameter Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_sae_configs.py                    # Default: sort by max_rpb_abs
  python compare_sae_configs.py --metric max_rpb_abs
  python compare_sae_configs.py --metric composite_score
        """
    )
    parser.add_argument(
        '--metric',
        type=str,
        choices=['max_rpb_abs', 'composite_score'],
        default='max_rpb_abs',
        help='Metric to use for ranking configurations (default: max_rpb_abs)'
    )
    args = parser.parse_args()

    print("="*70)
    print("SAE HYPERPARAMETER COMPARISON")
    print("="*70)

    # Count total configurations across all variants
    total_configs = sum(len(configs) for configs in VARIANT_CONFIGS.values())
    print(f"\nTesting {total_configs} configurations across all variants...")
    print(f"NOTE: Skipping significance testing for speed")
    print(f"RANKING METRIC: {args.metric}")

    results = []
    all_correlations = []  # Collect correlation data from all configs
    first_config_overall = True  # Track if this is the very first config

    # Iterate through all variants
    for variant, variant_configs in VARIANT_CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"VARIANT: {variant.upper()}")
        print(f"{'='*70}")

        for idx, config_tuple in enumerate(tqdm(variant_configs, desc=f"{variant} configs")):
            # Extract config parameters based on variant
            description = config_tuple[-1]  # Last element is always description

            # Run sanity check only on the very first config
            run_sanity = first_config_overall and CONTROL_FOR_TOPO

            if variant == 'topk':
                latent_dim, k = config_tuple[0], config_tuple[1]
                config_result, df_corr = analyze_topk_configuration(latent_dim, k, run_sanity_check=run_sanity)
                first_config_overall = False
                if config_result is not None:
                    config_result['variant'] = 'topk'
                    config_result['description'] = description
                    results.append(config_result)
                    # Collect correlation data
                    if df_corr is not None and len(df_corr) > 0:
                        df_corr['variant'] = variant
                        df_corr['latent_dim'] = latent_dim
                        df_corr['k'] = k
                        all_correlations.append(df_corr)

            elif variant == 'gated':
                latent_dim, sparsity_coef = config_tuple[0], config_tuple[1]
                config_result, df_corr = analyze_gated_configuration(latent_dim, sparsity_coef, run_sanity_check=run_sanity)
                first_config_overall = False
                if config_result is not None:
                    config_result['variant'] = 'gated'
                    config_result['sparsity_coef'] = sparsity_coef
                    config_result['description'] = description
                    results.append(config_result)
                    # Collect correlation data
                    if df_corr is not None and len(df_corr) > 0:
                        df_corr['variant'] = variant
                        df_corr['latent_dim'] = latent_dim
                        df_corr['sparsity_coef'] = sparsity_coef
                        all_correlations.append(df_corr)

            elif variant == 'jumprelu':
                latent_dim, threshold_init = config_tuple[0], config_tuple[1]
                config_result, df_corr = analyze_jumprelu_configuration(latent_dim, threshold_init, run_sanity_check=run_sanity)
                first_config_overall = False
                if config_result is not None:
                    config_result['variant'] = 'jumprelu'
                    config_result['threshold_init'] = threshold_init
                    config_result['bandwidth'] = 0.01  # Store bandwidth for reproducibility
                    config_result['description'] = description
                    results.append(config_result)
                    # Collect correlation data
                    if df_corr is not None and len(df_corr) > 0:
                        df_corr['variant'] = variant
                        df_corr['latent_dim'] = latent_dim
                        df_corr['threshold_init'] = threshold_init
                        df_corr['bandwidth'] = 0.01
                        all_correlations.append(df_corr)

            elif variant == 'switch':
                num_experts, latent_per_expert, k_per_expert = config_tuple[0], config_tuple[1], config_tuple[2]
                config_result, df_corr = analyze_switch_configuration(num_experts, latent_per_expert, k_per_expert, run_sanity_check=run_sanity)
                first_config_overall = False
                if config_result is not None:
                    config_result['variant'] = 'switch'
                    config_result['num_experts'] = num_experts
                    config_result['latent_per_expert'] = latent_per_expert
                    config_result['k_per_expert'] = k_per_expert
                    config_result['description'] = description
                    results.append(config_result)
                    # Collect correlation data
                    if df_corr is not None and len(df_corr) > 0:
                        df_corr['variant'] = variant
                        df_corr['num_experts'] = num_experts
                        df_corr['latent_per_expert'] = latent_per_expert
                        df_corr['k_per_expert'] = k_per_expert
                        all_correlations.append(df_corr)

    if len(results) == 0:
        print("\n⚠ No configurations completed successfully!")
        return

    # Create summary DataFrame
    df_results = pd.DataFrame(results)

    # Sort by selected metric
    df_results = df_results.sort_values(args.metric, ascending=False)
    print(f"✓ Sorted configurations by {args.metric}")

    # Save to CSV
    output_file = 'outputs/sae_config_comparison.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Saved detailed results to {output_file}")

    # Combine and save correlation data for use by run_interpretability_experiments.py
    if len(all_correlations) > 0:
        df_all_corr = pd.concat(all_correlations, ignore_index=True)

        # Add FDR-corrected significance columns (placeholder for now)
        # These can be computed with more rigorous permutation testing if needed
        df_all_corr['p_empirical'] = df_all_corr['pval']  # Use parametric p-value as proxy
        df_all_corr['p_fdr'] = df_all_corr['pval']  # Placeholder
        df_all_corr['significant_fdr'] = df_all_corr['pval'] < SIGNIFICANCE_LEVEL

        # Save correlation data
        corr_output_file = 'outputs/latent_correlations.csv'
        df_all_corr.to_csv(corr_output_file, index=True)
        print(f"✓ Saved correlation data to {corr_output_file}")
        print(f"  Total feature-motif pairs: {len(df_all_corr)}")
        print(f"  Significant pairs (p < {SIGNIFICANCE_LEVEL}): {df_all_corr['significant_fdr'].sum()}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Top 5 Configurations by Composite Score")
    print("="*70)

    display_cols = [
        'latent_dim', 'k', 'sparsity_pct', 'max_rpb_abs',
        'best_f1', 'dead_feature_rate', 'composite_score'
    ]

    top5 = df_results.head(5)[display_cols].copy()
    top5['sparsity_pct'] = top5['sparsity_pct'].round(2)
    top5['max_rpb_abs'] = top5['max_rpb_abs'].round(3)
    top5['best_f1'] = top5['best_f1'].round(3)
    top5['dead_feature_rate'] = top5['dead_feature_rate'].round(3)
    top5['composite_score'] = top5['composite_score'].round(3)

    print(top5.to_string(index=False))

    # Recommendations
    best = df_results.iloc[0]
    print("\n" + "="*70)
    print("RECOMMENDED CONFIGURATION")
    print("="*70)
    print(f"\n  latent_dim = {int(best['latent_dim'])}")
    print(f"  k = {int(best['k'])}")
    print(f"  Sparsity: {best['sparsity_pct']:.2f}% ({best['description']})")
    print(f"\n  Key Metrics:")
    print(f"    • Max correlation: |rpb| = {best['max_rpb_abs']:.3f}")
    print(f"    • Best F1 score: {best['best_f1']:.3f}")
    print(f"    • Active features: {int(best['n_active_features'])}/{int(best['latent_dim'])} ({100*(1-best['dead_feature_rate']):.1f}%)")
    print(f"    • Composite score: {best['composite_score']:.3f}")

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print(f"""
  Ranking Metric Used: {args.metric}

  Metric Descriptions:

    max_rpb_abs (Default):
      • Direct measure of feature-motif correlation strength
      • Simpler, more interpretable
      • Recommended for most use cases
      • Selects config with strongest feature-motif alignment

    composite_score:
      • Weighted combination: 50% max_rpb_abs + 35% F1 + 15% capacity
      • Balances effect size with predictive performance and efficiency
      • Use if you want to optimize for multiple factors simultaneously

  To use a different metric:
    python compare_sae_configs.py --metric composite_score

  Good Configuration Indicators:
    ✓ Max |rpb| > 0.3
    ✓ Best F1 > 0.3
    ✓ Dead feature rate < 0.5
    ✓ Composite score > 0.5 (if using composite_score metric)

  To use the recommended configuration, update your notebook:
    LATENT_DIM = {int(best['latent_dim'])}
    K = {int(best['k'])}
    """.format(best_latent=int(best['latent_dim']), best_k=int(best['k'])))

def load_data_and_model_gated(latent_dim, sparsity_coef):
    """Load Gated SAE model and prepare data."""
    checkpoint_path = f"checkpoints/sae_gated_latent{latent_dim}_lambda{sparsity_coef:.0e}_seed{SEED}.pt"
    if not Path(checkpoint_path).exists():
        return None, None, None

    model = GatedSAE(input_dim=INPUT_DIM, latent_dim=latent_dim, sparsity_coef=sparsity_coef)
    return _load_checkpoint_and_data(model, checkpoint_path, latent_dim)

def load_data_and_model_jumprelu(latent_dim, threshold_init):
    """Load JumpReLU SAE model and prepare data."""
    # Use dynamic bandwidth (current configs use 0.01)
    bandwidth = 0.01
    checkpoint_path = f"checkpoints/sae_jumprelu_latent{latent_dim}_thresh{threshold_init:.0e}_bw{bandwidth:.0e}_seed{SEED}.pt"
    if not Path(checkpoint_path).exists():
        return None, None, None

    model = JumpReLUSAE(input_dim=INPUT_DIM, latent_dim=latent_dim,
                       threshold_init=threshold_init, bandwidth=0.01)
    return _load_checkpoint_and_data(model, checkpoint_path, latent_dim)

def load_data_and_model_switch(num_experts, latent_per_expert, k_per_expert):
    """Load Switch SAE model and prepare data."""
    total_latent = num_experts * latent_per_expert
    checkpoint_path = f"checkpoints/sae_switch_experts{num_experts}_latent{total_latent}_k{k_per_expert}_seed{SEED}.pt"
    if not Path(checkpoint_path).exists():
        return None, None, None

    model = SwitchSAE(input_dim=INPUT_DIM, num_experts=num_experts,
                      latent_per_expert=latent_per_expert, k_per_expert=k_per_expert)
    return _load_checkpoint_and_data(model, checkpoint_path, total_latent)

def _load_checkpoint_and_data(model, checkpoint_path, latent_dim):
    """Helper to load checkpoint and extract data for any SAE model."""
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

if __name__ == "__main__":
    main()
