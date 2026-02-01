"""
Plot Random Control Distributions

Creates histogram visualizations showing the distribution of random control ablation
impacts compared to motif-specific feature ablations. This helps visualize whether
motif-specific features are outliers compared to random feature sets.

Usage:
    python plot_random_control_distributions.py --results_dir ablations/interpretability_latent128_k8_rpb0.15_results --output_dir ablations/interpretability_latent128_k8_rpb0.15_plots
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

def load_results(results_dir: Path) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load motif-specific results and random trial results.

    Args:
        results_dir: Directory containing result CSVs

    Returns:
        Tuple of (motif_specific_df, random_trials_dict)
    """
    # Load motif-specific results
    motif_results_path = results_dir / "motif_specific_results.csv"
    if not motif_results_path.exists():
        raise FileNotFoundError(f"Motif results not found: {motif_results_path}")

    motif_df = pd.read_csv(motif_results_path)

    # Load random trial results
    random_trials = {}
    for random_file in results_dir.glob("random_*feat_trials.csv"):
        # Extract number of features from filename (e.g., "random_17feat_trials.csv")
        n_feat = int(random_file.stem.split('_')[1].replace('feat', ''))
        df = pd.read_csv(random_file)
        # Standardize column names to match motif_specific_results.csv
        df.columns = [col.replace('_', ' ').title().replace(' ', '_') for col in df.columns]
        random_trials[n_feat] = df

    if not random_trials:
        raise FileNotFoundError(f"No random trial files found in {results_dir}")

    return motif_df, random_trials


def plot_random_distributions(motif_df: pd.DataFrame,
                              random_trials: Dict[int, pd.DataFrame],
                              output_path: Path):
    """
    Create 2x2 subplot showing distributions of random control impacts for each motif type.

    Args:
        motif_df: DataFrame with motif-specific results
        random_trials: Dict mapping n_features -> random trial DataFrame
        output_path: Path to save plot
    """
    motif_types = ['Feedforward Loop', 'Feedback Loop', 'Single Input Module', 'Cascade']

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Color scheme
    colors = {
        'Feedforward Loop': '#3498db',  # Blue
        'Feedback Loop': '#e74c3c',     # Red
        'Single Input Module': '#2ecc71',  # Green
        'Cascade': '#f39c12'             # Orange
    }

    for idx, motif_type in enumerate(motif_types):
        ax = axes[idx]

        # Get all feature sets that have data for this motif
        feature_sets = motif_df['Feature_Set'].unique()

        # Collect random distributions for each feature set
        for feature_set in feature_sets:
            # Get the specific impact for this feature set -> motif
            motif_row = motif_df[(motif_df['Feature_Set'] == feature_set) &
                                 (motif_df['Motif'] == motif_type)]

            if len(motif_row) == 0:
                continue

            n_features = motif_row['N_Features'].values[0]
            specific_impact = motif_row['Mean_Impact'].values[0]

            # Get random trials for this number of features
            if n_features not in random_trials:
                print(f"Warning: No random trials found for {n_features} features")
                continue

            random_df = random_trials[n_features]

            # Filter random trials for this motif type
            random_impacts = random_df[random_df['Motif'] == motif_type]['Mean_Impact'].values

            # Plot histogram of random impacts
            ax.hist(random_impacts, bins=30, alpha=0.6,
                   color=colors[feature_set] if feature_set in colors else 'gray',
                   edgecolor='black', linewidth=0.5,
                   label=f'Random ({n_features} feat)')

            # Plot vertical line for motif-specific impact
            ax.axvline(specific_impact, color=colors[feature_set] if feature_set in colors else 'black',
                      linestyle='--', linewidth=2.5,
                      label=f'{feature_set} features ({specific_impact:.2e})')

        # Formatting
        ax.set_xlabel('Mean Ablation Impact\n(MSE Increase)', fontweight='bold')
        ax.set_ylabel('Frequency (across trials)', fontweight='bold')
        ax.set_title(f'{motif_type} Graphs', fontweight='bold',
                    color=colors.get(motif_type, 'black'))
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Add zero line
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    # Overall title
    plt.suptitle('Distribution of Random Control Impacts\n(Shows whether motif-specific features are outliers)',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_distributions(motif_df: pd.DataFrame,
                                random_trials: Dict[int, pd.DataFrame],
                                output_path: Path):
    """
    Create combined plot showing all distributions with clearer comparisons.
    Alternative visualization focusing on on-target vs off-target effects.

    Args:
        motif_df: DataFrame with motif-specific results
        random_trials: Dict mapping n_features -> random trial DataFrame
        output_path: Path to save plot
    """
    motif_types = ['Feedforward Loop', 'Feedback Loop', 'Single Input Module', 'Cascade']
    feature_sets = motif_df['Feature_Set'].unique()

    # Create subplots - one per feature set
    n_sets = len(feature_sets)
    fig, axes = plt.subplots(1, n_sets, figsize=(7*n_sets, 6))

    if n_sets == 1:
        axes = [axes]

    colors = {
        'Feedforward Loop': '#3498db',
        'Feedback Loop': '#e74c3c',
        'Single Input Module': '#2ecc71',
        'Cascade': '#f39c12'
    }

    for set_idx, feature_set in enumerate(feature_sets):
        ax = axes[set_idx]

        # Get number of features for this set
        n_features = motif_df[motif_df['Feature_Set'] == feature_set]['N_Features'].values[0]

        if n_features not in random_trials:
            continue

        random_df = random_trials[n_features]

        # Plot each motif type
        for motif_type in motif_types:
            # Random distribution
            random_impacts = random_df[random_df['Motif'] == motif_type]['Mean_Impact'].values

            # Specific impact
            motif_row = motif_df[(motif_df['Feature_Set'] == feature_set) &
                                 (motif_df['Motif'] == motif_type)]

            if len(motif_row) == 0:
                continue

            specific_impact = motif_row['Mean_Impact'].values[0]

            # Determine if on-target (feature set matches motif type)
            is_on_target = feature_set == motif_type

            # Plot histogram
            alpha = 0.7 if is_on_target else 0.3
            edge_width = 1.2 if is_on_target else 0.5

            ax.hist(random_impacts, bins=25, alpha=alpha,
                   color=colors[motif_type],
                   edgecolor='black', linewidth=edge_width,
                   label=f'{motif_type}')

            # Plot specific impact line
            line_style = '--' if is_on_target else ':'
            line_width = 2.5 if is_on_target else 1.5

            ax.axvline(specific_impact, color=colors[motif_type],
                      linestyle=line_style, linewidth=line_width,
                      label=f'{motif_type} ({specific_impact:.2e})')

        # Formatting
        ax.set_xlabel('Mean Ablation Impact', fontweight='bold')
        ax.set_ylabel('Frequency (across trials)', fontweight='bold')
        ax.set_title(f'{feature_set} Features vs Random Controls\n({n_features} features ablated)',
                    fontweight='bold')
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    plt.suptitle('Distribution of Random Control Impacts by Feature Set',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_statistics(motif_df: pd.DataFrame, random_trials: Dict[int, pd.DataFrame]):
    """Print summary statistics comparing specific vs random impacts."""

    print("\n" + "="*80)
    print("SUMMARY STATISTICS: Motif-Specific vs Random Control Impacts")
    print("="*80)

    for _, row in motif_df.iterrows():
        feature_set = row['Feature_Set']
        motif = row['Motif']
        n_features = row['N_Features']
        specific_impact = row['Mean_Impact']

        if n_features not in random_trials:
            continue

        random_df = random_trials[n_features]
        random_impacts = random_df[random_df['Motif'] == motif]['Mean_Impact'].values

        if len(random_impacts) == 0:
            continue

        random_mean = random_impacts.mean()
        random_std = random_impacts.std()

        # Compute z-score
        z_score = (specific_impact - random_mean) / random_std if random_std > 0 else 0

        # Compute percentile
        percentile = (random_impacts < specific_impact).mean() * 100

        print(f"\n{feature_set} Features -> {motif} Graphs:")
        print(f"  Specific Impact:     {specific_impact:.6e}")
        print(f"  Random Mean ± SD:    {random_mean:.6e} ± {random_std:.6e}")
        print(f"  Z-Score:             {z_score:.2f}")
        print(f"  Percentile:          {percentile:.1f}%")

        if z_score > 3:
            print(f"  *** HIGHLY SIGNIFICANT (specific >> random) ***")
        elif z_score < -3:
            print(f"  *** HIGHLY SIGNIFICANT (specific << random) ***")


def main():
    parser = argparse.ArgumentParser(description='Plot random control distributions')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing result CSVs')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save plots')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    motif_df, random_trials = load_results(results_dir)

    print(f"\nFound {len(motif_df)} motif-specific results")
    print(f"Found random trials for: {list(random_trials.keys())} features")

    # Print summary statistics
    print_summary_statistics(motif_df, random_trials)

    # Create plots
    print("\nGenerating plots...")

    # Main distribution plot (2x2 grid by motif type)
    output_path = output_dir / "random_control_distributions.png"
    plot_random_distributions(motif_df, random_trials, output_path)

    # Alternative plot (by feature set)
    output_path_alt = output_dir / "random_control_distributions_by_featureset.png"
    plot_combined_distributions(motif_df, random_trials, output_path_alt)

    print("\n" + "="*80)
    print("COMPLETE: Random control distribution plots generated")
    print("="*80)


if __name__ == "__main__":
    main()
