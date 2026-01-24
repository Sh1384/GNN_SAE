#!/usr/bin/env python3
"""
Batch Ablation Runner

Systematically ablate multiple SAE latent features and compare their impact on GNN performance.

This script runs individual ablations for multiple features, then aggregates and compares
the results to identify which latent features are most critical for downstream task performance.

Usage:
    # Ablate top 10 most correlated features (default)
    python run_batch_ablations.py --latent_dim 512 --k 32

    # Ablate top 20 most correlated features
    python run_batch_ablations.py --latent_dim 512 --k 32 --top_n 20

Output:
    - Individual ablation results in ablations/results/
    - Comparison summary: ablations/results/ablation_comparison.csv
    - Comparison plots: ablations/plots/ablation_comparison.png
"""

import argparse
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ABLATION_DIR = Path("ablations")

def run_single_ablation(latent_dim, k, feature_idx):
    """Run ablation for a single feature."""
    feature_name = f"z{feature_idx + 1}"
    experiment_name = f"latent{latent_dim}_k{k}_ablate_{feature_name}"

    cmd = [
        "python", "run_ablation.py",
        "--latent_dim", str(latent_dim),
        "--k", str(k),
        "--feature", feature_name,
        "--experiment_name", experiment_name
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running ablation for {feature_name}:")
        print(result.stderr)
        return None

    return experiment_name

def get_top_features_to_ablate(latent_dim, k, top_n):
    """Get list of top N features to ablate based on correlation analysis."""
    corr_file = Path("outputs/feature_motif_correlations_configurable.csv")
    if not corr_file.exists():
        print(f"Error: Correlation file not found at {corr_file}")
        print("Run the motif analysis notebook first.")
        return []

    df_corr = pd.read_csv(corr_file, index_col=0)
    max_corrs = df_corr.abs().max(axis=1).nlargest(top_n)

    # Convert feature names to indices
    feature_indices = [int(feat[1:]) - 1 for feat in max_corrs.index]

    return feature_indices

def compare_ablation_results(experiment_names):
    """Load and compare results from multiple ablation experiments."""
    results_list = []

    for exp_name in experiment_names:
        results_file = ABLATION_DIR / "results" / f"{exp_name}_results.csv"
        if not results_file.exists():
            print(f"Warning: Results file not found for {exp_name}")
            continue

        df = pd.read_csv(results_file)

        # Aggregate metrics per experiment
        summary = {
            'experiment': exp_name,
            'feature': df['ablated_features'].iloc[0],
            'mean_recon_mse': df['reconstruction_mse'].mean(),
            'std_recon_mse': df['reconstruction_mse'].std(),
            'mean_nodes_affected_pct': 100 * (df['n_affected_nodes'] / df['total_nodes']).mean(),
            'mean_gnn_output_mse': df['gnn_output_mse'].mean() if df['gnn_output_mse'].notna().any() else None,
            'std_gnn_output_mse': df['gnn_output_mse'].std() if df['gnn_output_mse'].notna().any() else None,
        }

        results_list.append(summary)

    df_comparison = pd.DataFrame(results_list)

    # Save comparison
    comparison_file = ABLATION_DIR / "results" / "ablation_comparison.csv"
    df_comparison.to_csv(comparison_file, index=False)
    print(f"\n✓ Saved comparison to {comparison_file}")

    return df_comparison

def plot_comparison(df_comparison):
    """Create comparison plots across ablation experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Reconstruction error by feature
    ax = axes[0, 0]
    df_sorted = df_comparison.sort_values('mean_recon_mse', ascending=False)
    colors = plt.cm.RdYlGn_r(df_sorted['mean_recon_mse'] / df_sorted['mean_recon_mse'].max())

    ax.barh(range(len(df_sorted)), df_sorted['mean_recon_mse'],
            color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['feature'])
    ax.set_xlabel('Mean Reconstruction MSE', fontsize=12)
    ax.set_ylabel('Ablated Feature', fontsize=12)
    ax.set_title('Reconstruction Error by Ablated Feature', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 2: Nodes affected
    ax = axes[0, 1]
    df_sorted = df_comparison.sort_values('mean_nodes_affected_pct', ascending=False)
    ax.barh(range(len(df_sorted)), df_sorted['mean_nodes_affected_pct'],
            color='coral', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['feature'])
    ax.set_xlabel('% Nodes Affected', fontsize=12)
    ax.set_ylabel('Ablated Feature', fontsize=12)
    ax.set_title('Node Impact by Ablated Feature', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 3: GNN output change
    ax = axes[1, 0]
    if df_comparison['mean_gnn_output_mse'].notna().any():
        df_sorted = df_comparison.sort_values('mean_gnn_output_mse', ascending=False)
        colors = plt.cm.Reds(df_sorted['mean_gnn_output_mse'] / df_sorted['mean_gnn_output_mse'].max())

        ax.barh(range(len(df_sorted)), df_sorted['mean_gnn_output_mse'],
                color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['feature'])
        ax.set_xlabel('Mean GNN Output MSE', fontsize=12)
        ax.set_ylabel('Ablated Feature', fontsize=12)
        ax.set_title('GNN Output Impact (Higher=More Important)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'GNN evaluation not available',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.grid(True, alpha=0.3, axis='x')

    # Plot 4: Scatter - Impact vs nodes affected
    ax = axes[1, 1]
    if df_comparison['mean_gnn_output_mse'].notna().any():
        scatter = ax.scatter(df_comparison['mean_nodes_affected_pct'],
                            df_comparison['mean_gnn_output_mse'],
                            s=200, alpha=0.6, c=df_comparison['mean_recon_mse'],
                            cmap='YlOrRd', edgecolors='black', linewidth=1.5)

        # Annotate points with feature names
        for _, row in df_comparison.iterrows():
            ax.annotate(row['feature'],
                       xy=(row['mean_nodes_affected_pct'], row['mean_gnn_output_mse']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, alpha=0.8)

        ax.set_xlabel('% Nodes Affected', fontsize=12)
        ax.set_ylabel('Mean GNN Output MSE', fontsize=12)
        ax.set_title('Impact vs Breadth (Color=Reconstruction Error)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Reconstruction MSE')
    else:
        ax.text(0.5, 0.5, 'GNN evaluation not available',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = ABLATION_DIR / "plots" / "ablation_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {plot_file}")
    plt.close()

def print_comparison_summary(df_comparison):
    """Print summary of all ablation experiments."""
    print(f"\n{'='*70}")
    print("ABLATION COMPARISON SUMMARY")
    print(f"{'='*70}")

    print(f"\nTotal experiments: {len(df_comparison)}")

    if df_comparison['mean_gnn_output_mse'].notna().any():
        # Rank by impact on GNN
        print("\nFeatures ranked by GNN output impact (descending):")
        df_sorted = df_comparison.sort_values('mean_gnn_output_mse', ascending=False)
        for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
            print(f"  {i}. {row['feature']:6s}: Output MSE = {row['mean_gnn_output_mse']:.6f}")

        print("\nMost critical features (largest output change when ablated):")
        most_critical = df_sorted.head(3)
        for _, row in most_critical.iterrows():
            print(f"  • {row['feature']}: MSE = {row['mean_gnn_output_mse']:.6f}")
            print(f"    Affects {row['mean_nodes_affected_pct']:.1f}% of nodes")
            print(f"    Reconstruction MSE: {row['mean_recon_mse']:.6f}")
    else:
        print("\nGNN output metrics not available.")

    print("\nFeatures with highest reconstruction error when ablated:")
    df_sorted = df_comparison.sort_values('mean_recon_mse', ascending=False)
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        print(f"  {i}. {row['feature']:6s}: MSE = {row['mean_recon_mse']:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Batch Ablation Runner')
    parser.add_argument('--latent_dim', type=int, required=True,
                        help='SAE latent dimension')
    parser.add_argument('--k', type=int, required=True,
                        help='SAE TopK sparsity parameter')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top features to ablate (default: 10)')

    args = parser.parse_args()

    print(f"{'='*70}")
    print("BATCH ABLATION ANALYSIS")
    print(f"{'='*70}")
    print(f"\nSAE config: latent_dim={args.latent_dim}, k={args.k}")
    print(f"Ablating top {args.top_n} most correlated features")

    # Get features to ablate
    feature_indices = get_top_features_to_ablate(args.latent_dim, args.k, args.top_n)

    if not feature_indices:
        print("Error: No features to ablate!")
        return

    print(f"\nFeatures to ablate: {[f'z{i+1}' for i in feature_indices]}")

    # Run ablations
    experiment_names = []
    for idx in feature_indices:
        exp_name = run_single_ablation(args.latent_dim, args.k, idx)
        if exp_name:
            experiment_names.append(exp_name)

    if not experiment_names:
        print("\nError: No ablations completed successfully!")
        return

    print(f"\n{'='*70}")
    print("COMPARING RESULTS")
    print(f"{'='*70}")

    # Compare results
    df_comparison = compare_ablation_results(experiment_names)

    # Create comparison plots
    plot_comparison(df_comparison)

    # Print summary
    print_comparison_summary(df_comparison)

    print(f"\n{'='*70}")
    print("BATCH ABLATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nComparison results: ablations/results/ablation_comparison.csv")
    print(f"Comparison plots: ablations/plots/ablation_comparison.png")

if __name__ == "__main__":
    main()
