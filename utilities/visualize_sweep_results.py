"""
Visualization Script for Hyperparameter Sweep Results

Analyzes and visualizes results from the distributed multi-GPU hyperparameter sweep.
Can be run as standalone script or imported into notebook.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_sweep_results(sweep_dir: str) -> Dict:
    """
    Load all sweep results from directory.

    Args:
        sweep_dir: Path to sweep results directory

    Returns:
        Dictionary with trials_df, best_params, and study_info
    """
    sweep_path = Path(sweep_dir)

    # Load trials CSV
    trials_df = pd.read_csv(sweep_path / "trials.csv")

    # # Load best params
    # with open(sweep_path / "best_params.json", 'r') as f:
    #     best_params = json.load(f)

    # # Load study info
    # with open(sweep_path / "study_info.json", 'r') as f:
    #     study_info = json.load(f)

    # return {
    #     'trials_df': trials_df,
    #     'best_params': best_params,
    #     'study_info': study_info,
    #     'sweep_dir': sweep_path
    # }
    return trials_df


def plot_optimization_history(trials_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot optimization history showing best value over trials."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: All trial values
    ax = axes[0]
    ax.plot(trials_df['number'], trials_df['value'], 'o-', alpha=0.6, label='Trial value')

    # Cumulative best
    cum_best = trials_df['value'].cummin()
    ax.plot(trials_df['number'], cum_best, 'r-', linewidth=2, label='Best so far')

    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Optimization History', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Log scale for better visualization
    ax = axes[1]
    ax.plot(trials_df['number'], trials_df['value'], 'o-', alpha=0.6, label='Trial value')
    ax.plot(trials_df['number'], cum_best, 'r-', linewidth=2, label='Best so far')
    ax.set_yscale('log')
    ax.set_xlabel('Trial Number', fontsize=12)
    ax.set_ylabel('Validation Loss (log scale)', fontsize=12)
    ax.set_title('Optimization History (Log Scale)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.name}")

    plt.show()


def plot_parameter_distributions(trials_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot distributions of hyperparameters tried."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Hidden dim distribution
    ax = axes[0]
    ax.hist(trials_df['params_hidden_dim'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(trials_df.loc[trials_df['value'].idxmin(), 'params_hidden_dim'],
               color='red', linestyle='--', linewidth=2, label='Best')
    ax.set_xlabel('Hidden Dimension', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Hidden Dimension Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Dropout distribution
    ax = axes[1]
    ax.hist(trials_df['params_dropout'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(trials_df.loc[trials_df['value'].idxmin(), 'params_dropout'],
               color='red', linestyle='--', linewidth=2, label='Best')
    ax.set_xlabel('Dropout', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Dropout Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Learning rate distribution (log scale)
    ax = axes[2]
    ax.hist(np.log10(trials_df['params_learning_rate']), bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.log10(trials_df.loc[trials_df['value'].idxmin(), 'params_learning_rate']),
               color='red', linestyle='--', linewidth=2, label='Best')
    ax.set_xlabel('Learning Rate (log10)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Learning Rate Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.name}")

    plt.show()


def plot_parameter_importance(trials_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plot scatter plots showing parameter vs performance."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Color based on value (performance)
    colors = trials_df['value']

    # Hidden dim vs performance
    ax = axes[0]
    scatter = ax.scatter(trials_df['params_hidden_dim'], trials_df['value'],
                        c=colors, cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Hidden Dimension', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Hidden Dim vs Performance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Dropout vs performance
    ax = axes[1]
    ax.scatter(trials_df['params_dropout'], trials_df['value'],
              c=colors, cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel('Dropout', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Dropout vs Performance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Learning rate vs performance
    ax = axes[2]
    ax.scatter(trials_df['params_learning_rate'], trials_df['value'],
              c=colors, cmap='RdYlGn_r', s=100, alpha=0.6, edgecolors='black')
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate (log scale)', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Learning Rate vs Performance', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('Validation Loss', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.name}")

    plt.show()


def plot_top_trials(trials_df: pd.DataFrame, top_n: int = 10, save_path: Optional[Path] = None):
    """Plot top N trials as a heatmap."""
    top_trials = trials_df.nsmallest(top_n, 'value')[
        ['number', 'params_hidden_dim', 'params_dropout', 'params_learning_rate', 'value']
    ].copy()

    # Rename for display
    top_trials.columns = ['Trial', 'Hidden Dim', 'Dropout', 'Learning Rate', 'Val Loss']

    # Normalize params for visualization
    normalized = top_trials.copy()
    for col in ['Hidden Dim', 'Dropout', 'Learning Rate']:
        min_val = trials_df[f'params_{col.lower().replace(" ", "_")}'].min()
        max_val = trials_df[f'params_{col.lower().replace(" ", "_")}'].max()
        if max_val > min_val:
            normalized[col] = (top_trials[col] - min_val) / (max_val - min_val)

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4)))

    # Create heatmap
    sns.heatmap(
        normalized[['Hidden Dim', 'Dropout', 'Learning Rate']].T,
        annot=top_trials[['Hidden Dim', 'Dropout', 'Learning Rate']].T,
        fmt='.3f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Normalized Value'},
        xticklabels=[f"#{int(t)} ({v:.6f})" for t, v in zip(top_trials['Trial'], top_trials['Val Loss'])],
        yticklabels=['Hidden Dim', 'Dropout', 'Learning Rate'],
        ax=ax
    )

    ax.set_title(f'Top {top_n} Trials Hyperparameters (Best → Worst)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Trial (Validation Loss)', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path.name}")

    plt.show()


def print_summary(results: Dict):
    """Print summary statistics of the sweep."""
    trials_df = results['trials_df']
    best_params = results['best_params']
    study_info = results['study_info']

    print("=" * 80)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("=" * 80)
    print()

    print(f"Total trials: {len(trials_df)}")
    print(f"Completed trials: {study_info['n_complete_trials']}")
    print(f"Failed trials: {study_info.get('n_failed_trials', 0)}")
    print()

    print(f"Best trial: #{study_info['best_trial']}")
    print(f"Best validation loss: {study_info['best_value']:.6f}")
    print()

    print("Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()

    print("Performance statistics:")
    print(f"  Best: {trials_df['value'].min():.6f}")
    print(f"  Worst: {trials_df['value'].max():.6f}")
    print(f"  Mean: {trials_df['value'].mean():.6f}")
    print(f"  Std: {trials_df['value'].std():.6f}")
    print(f"  Median: {trials_df['value'].median():.6f}")
    print()

    # Top 5 trials
    print("Top 5 trials:")
    top5 = trials_df.nsmallest(5, 'value')
    for idx, row in top5.iterrows():
        print(f"  #{int(row['number'])}: val_loss={row['value']:.6f}, "
              f"hidden_dim={int(row['params_hidden_dim'])}, "
              f"dropout={row['params_dropout']:.2f}, "
              f"lr={row['params_learning_rate']:.6f}")
    print()

    if 'total_time' in study_info:
        print(f"Total time: {study_info['total_time']:.1f}s ({study_info['total_time']/60:.1f} min)")
        print(f"Avg time per trial: {study_info['avg_time_per_trial']:.1f}s")

    print("=" * 80)


def generate_all_visualizations(sweep_dir: str, output_dir: Optional[str] = None):
    """
    Generate all visualizations for a sweep.

    Args:
        sweep_dir: Path to sweep results directory
        output_dir: Optional path to save figures (defaults to sweep_dir/visualizations)
    """
    # Load results
    print("Loading sweep results...")
    # results = load_sweep_results(sweep_dir)
    # trials_df = results['trials_df']
    
    trials_df = load_sweep_results(sweep_dir)

    # Create output directory
    if output_dir is None:
        output_dir = Path(sweep_dir) / "visualizations" / "GAT"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving visualizations to: {output_dir}\n")

    # Print summary
    # print_summary(results)
    print()

    # Generate plots
    print("Generating visualizations...")
    print()

    plot_optimization_history(trials_df, save_path=output_dir / "optimization_history.png")
    plot_parameter_distributions(trials_df, save_path=output_dir / "parameter_distributions.png")
    plot_parameter_importance(trials_df, save_path=output_dir / "parameter_importance.png")
    plot_top_trials(trials_df, top_n=10, save_path=output_dir / "top_trials_heatmap.png")

    print()
    print(f"✓ All visualizations saved to: {output_dir}")
    print()

    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize_sweep_results.py <sweep_dir>")
        print("\nExample:")
        print("  python visualize_sweep_results.py outputs/sweep_distributed_20251201_194542")
        sys.exit(1)

    sweep_dir = sys.argv[1]
    generate_all_visualizations(sweep_dir)
