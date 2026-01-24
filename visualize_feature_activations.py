#!/usr/bin/env python3
"""
Feature Activation Visualization for SAE Interpretability

Generates visual representations of which graphs, nodes, and activations
correspond to each learned SAE feature. Helps demonstrate which aspects
of the input the model attends to.

Key outputs:
- Feature activation heatmaps (graphs × nodes)
- Top-activated graph examples with network visualizations
- Decoder weight distributions per feature
- Feature sparsity and selectivity indices

Usage:
    python visualize_feature_activations.py --variant topk --latent_dim 512 --k 8
    python visualize_feature_activations.py --all-variants
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = Path("outputs")
VIZ_DIR = OUTPUT_DIR / "feature_activation_visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)


def load_sae_model(checkpoint_path: str, variant: str):
    """Load trained SAE model from checkpoint.

    Args:
        checkpoint_path: Path to saved SAE checkpoint
        variant: SAE variant type (topk, gated, jumprelu, switch)

    Returns:
        Loaded SAE model or None if loading fails
    """
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Reconstruct model based on variant
        if variant == 'topk':
            from sparse_autoencoder import TopKSAE
            config = checkpoint.get('config', {})
            model = TopKSAE(**config)
            model.load_state_dict(checkpoint['state_dict'])
        elif variant == 'gated':
            from sparse_autoencoder import GatedSAE
            config = checkpoint.get('config', {})
            model = GatedSAE(**config)
            model.load_state_dict(checkpoint['state_dict'])
        elif variant == 'jumprelu':
            from sparse_autoencoder import JumpReLUSAE
            config = checkpoint.get('config', {})
            model = JumpReLUSAE(**config)
            model.load_state_dict(checkpoint['state_dict'])
        elif variant == 'switch':
            from sparse_autoencoder import SwitchSAE
            config = checkpoint.get('config', {})
            model = SwitchSAE(**config)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print(f"Unknown variant: {variant}")
            return None

        model.eval()
        return model

    except Exception as e:
        print(f"Error loading SAE model: {str(e)}")
        return None


def load_activations(data_path: str) -> Optional[np.ndarray]:
    """Load native GNN activations from file.

    Args:
        data_path: Path to stored activations (NPZ format expected)

    Returns:
        Activations array of shape (num_graphs, num_nodes, 64) or None
    """
    try:
        data = np.load(data_path)
        # Handle different storage formats
        if 'activations' in data:
            return data['activations']
        elif 'h' in data:
            return data['h']
        else:
            # Return first array found
            return data[list(data.files)[0]]
    except Exception as e:
        print(f"Error loading activations: {str(e)}")
        return None


def compute_feature_activations(sae_model, activations: np.ndarray) -> np.ndarray:
    """Encode activations through SAE to get latent representations.

    Args:
        sae_model: Trained SAE model
        activations: Native GNN activations (num_graphs, num_nodes, 64)

    Returns:
        SAE latent codes (num_graphs, num_nodes, latent_dim)
    """
    try:
        import torch

        # Reshape activations for encoding
        num_graphs, num_nodes, hidden_dim = activations.shape
        flat_activations = activations.reshape(-1, hidden_dim)

        with torch.no_grad():
            tensor = torch.from_numpy(flat_activations).float()
            latents = sae_model.encode(tensor)  # (num_graphs*num_nodes, latent_dim)

        latents_np = latents.numpy()
        return latents_np.reshape(num_graphs, num_nodes, -1)

    except Exception as e:
        print(f"Error computing feature activations: {str(e)}")
        return None


def identify_top_activated_graphs(latents: np.ndarray, feature_idx: int, top_k: int = 5) -> np.ndarray:
    """Identify graphs where feature is most strongly activated.

    Args:
        latents: SAE latent codes (num_graphs, num_nodes, latent_dim)
        feature_idx: Which feature to analyze
        top_k: How many top graphs to return

    Returns:
        Indices of top-k graphs (length top_k)
    """
    # Aggregate feature activation per graph (max or mean)
    feature_activations = np.abs(latents[:, :, feature_idx])  # (num_graphs, num_nodes)
    graph_activations = feature_activations.max(axis=1)  # Max activation per graph

    # Get top-k graphs
    top_indices = np.argsort(graph_activations)[-top_k:][::-1]
    return top_indices


def identify_top_activated_nodes(latents: np.ndarray, feature_idx: int, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Identify nodes most activated by feature across all graphs.

    Args:
        latents: SAE latent codes (num_graphs, num_nodes, latent_dim)
        feature_idx: Which feature to analyze
        top_k: How many top nodes to return

    Returns:
        (graph_indices, node_indices) - pairs of top-k most activated nodes
    """
    feature_activations = np.abs(latents[:, :, feature_idx])  # (num_graphs, num_nodes)

    # Flatten and get top-k activations
    flat_activations = feature_activations.ravel()
    top_flat_indices = np.argsort(flat_activations)[-top_k:][::-1]

    # Convert back to (graph, node) pairs
    graph_indices = top_flat_indices // feature_activations.shape[1]
    node_indices = top_flat_indices % feature_activations.shape[1]

    return graph_indices, node_indices


def plot_feature_activation_heatmap(latents: np.ndarray, feature_idx: int,
                                   variant: str, config: Dict, top_graphs: int = 10):
    """Plot heatmap of feature activation across graphs and nodes.

    Args:
        latents: SAE latent codes
        feature_idx: Which feature to visualize
        variant: SAE variant name
        config: Model configuration
        top_graphs: How many top graphs to show
    """
    feature_activations = np.abs(latents[:, :, feature_idx])

    # Get top graphs
    graph_scores = feature_activations.max(axis=1)
    top_graph_indices = np.argsort(graph_scores)[-top_graphs:][::-1]

    # Create heatmap data
    heatmap_data = feature_activations[top_graph_indices, :]

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis', interpolation='nearest')

    ax.set_xlabel('Node Index', fontweight='bold')
    ax.set_ylabel('Graph Index (sorted by activation)', fontweight='bold')
    ax.set_title(f'Feature {feature_idx} Activation Heatmap ({variant.upper()})\n' +
                 f'Config: {json.dumps(config, indent=0)}',
                 fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|Activation|', fontweight='bold')

    plt.tight_layout()
    output_path = VIZ_DIR / f'feature_{feature_idx}_activation_heatmap_{variant}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved feature_{feature_idx}_activation_heatmap_{variant}.png")
    plt.close()


def plot_top_activated_nodes(latents: np.ndarray, feature_idx: int,
                             variant: str, config: Dict, top_k: int = 10):
    """Plot top-activated node examples with activation strengths.

    Args:
        latents: SAE latent codes
        feature_idx: Which feature to visualize
        variant: SAE variant name
        config: Model configuration
        top_k: How many top nodes to show
    """
    graph_indices, node_indices = identify_top_activated_nodes(latents, feature_idx, top_k)
    activations = np.abs(latents[graph_indices, node_indices, feature_idx])

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(activations))
    colors = plt.cm.viridis(activations / activations.max())

    bars = ax.bar(x_pos, activations, color=colors, edgecolor='black', linewidth=1.5)

    # Label bars with (graph, node) coordinates
    labels = [f'G{g},N{n}' for g, n in zip(graph_indices, node_indices)]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_ylabel('Activation Strength', fontweight='bold')
    ax.set_title(f'Top {top_k} Node Activations for Feature {feature_idx} ({variant.upper()})',
                 fontweight='bold')

    # Add colorbar-like legend
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=activations.min(), vmax=activations.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Activation', fontweight='bold')

    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = VIZ_DIR / f'feature_{feature_idx}_top_nodes_{variant}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved feature_{feature_idx}_top_nodes_{variant}.png")
    plt.close()


def plot_decoder_weights(decoder_weights: np.ndarray, feature_idx: int,
                         variant: str, config: Dict):
    """Plot distribution of decoder weights for a feature.

    Args:
        decoder_weights: Decoder weight matrix (latent_dim, input_dim)
        feature_idx: Which feature to visualize
        variant: SAE variant name
        config: Model configuration
    """
    if feature_idx >= decoder_weights.shape[0]:
        print(f"  Warning: Feature index {feature_idx} out of range")
        return

    weights = decoder_weights[feature_idx, :]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram
    ax = axes[0]
    ax.hist(weights, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(weights.mean(), color='r', linestyle='--', linewidth=2, label=f"Mean: {weights.mean():.3f}")
    ax.axvline(np.median(weights), color='g', linestyle='--', linewidth=2, label=f"Median: {np.median(weights):.3f}")
    ax.set_xlabel('Weight Value', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(f'Decoder Weight Distribution - Feature {feature_idx}', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Sorted weights
    ax = axes[1]
    sorted_weights = np.sort(weights)
    ax.plot(sorted_weights, linewidth=2, color='steelblue')
    ax.fill_between(range(len(sorted_weights)), sorted_weights, alpha=0.3)
    ax.set_xlabel('Weight Index (sorted)', fontweight='bold')
    ax.set_ylabel('Weight Value', fontweight='bold')
    ax.set_title(f'Sorted Decoder Weights - Feature {feature_idx}', fontweight='bold')
    ax.grid(alpha=0.3)

    # Statistics box
    stats_text = f"""
    Statistics:
    • Mean: {weights.mean():.4f}
    • Std: {weights.std():.4f}
    • Min: {weights.min():.4f}
    • Max: {weights.max():.4f}
    • L2 norm: {np.linalg.norm(weights):.4f}
    • Sparsity: {(weights == 0).sum() / len(weights):.1%}
    """

    fig.text(0.98, 0.97, stats_text, transform=fig.transFigure,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')

    plt.tight_layout()
    output_path = VIZ_DIR / f'feature_{feature_idx}_decoder_weights_{variant}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved feature_{feature_idx}_decoder_weights_{variant}.png")
    plt.close()


def compute_feature_statistics(latents: np.ndarray) -> pd.DataFrame:
    """Compute summary statistics for all features.

    Args:
        latents: SAE latent codes (num_graphs, num_nodes, latent_dim)

    Returns:
        DataFrame with statistics per feature
    """
    num_graphs, num_nodes, latent_dim = latents.shape

    stats = []

    for feature_idx in range(latent_dim):
        feature_activations = np.abs(latents[:, :, feature_idx])

        stats.append({
            'feature_idx': feature_idx,
            'mean_activation': feature_activations.mean(),
            'max_activation': feature_activations.max(),
            'std_activation': feature_activations.std(),
            'sparsity': (feature_activations < 1e-6).sum() / (num_graphs * num_nodes),
            'selectivity': (feature_activations > feature_activations.mean()).sum() / (num_graphs * num_nodes),
        })

    return pd.DataFrame(stats)


def plot_feature_statistics_table(stats_df: pd.DataFrame, variant: str, num_rows: int = 20):
    """Create a PNG visualization of feature statistics table.

    Args:
        stats_df: DataFrame with feature statistics
        variant: SAE variant name
        num_rows: Number of top features to show in table
    """
    # Select top rows
    display_df = stats_df.head(num_rows).copy()

    # Format columns for display
    display_df['feature_idx'] = display_df['feature_idx'].astype(int)
    display_df['mean_activation'] = display_df['mean_activation'].apply(lambda x: f"{x:.4f}")
    display_df['max_activation'] = display_df['max_activation'].apply(lambda x: f"{x:.4f}")
    display_df['std_activation'] = display_df['std_activation'].apply(lambda x: f"{x:.4f}")
    display_df['sparsity'] = display_df['sparsity'].apply(lambda x: f"{x:.1%}")
    display_df['selectivity'] = display_df['selectivity'].apply(lambda x: f"{x:.1%}")

    # Rename for display
    display_df.columns = ['Feature', 'Mean Activation', 'Max Activation', 'Std Activation', 'Sparsity', 'Selectivity']

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                     cellLoc='center', loc='center',
                     colColours=['#E8E8E8']*len(display_df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')

    plt.title(f'Feature Statistics Summary - {variant.upper()}\n(Top {num_rows} Features)',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    output_path = VIZ_DIR / f'feature_statistics_summary_{variant}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved feature_statistics_summary_{variant}.png")
    plt.close()


def create_feature_summary_report(variant: str, latents: np.ndarray, decoder_weights: np.ndarray,
                                  config: Dict) -> str:
    """Create markdown summary report for visualized features.

    Args:
        variant: SAE variant name
        latents: SAE latent codes
        decoder_weights: Decoder weight matrix
        config: Model configuration

    Returns:
        Markdown report string
    """
    stats_df = compute_feature_statistics(latents)

    report = [
        f"# Feature Activation Visualizations - {variant.upper()}\n\n",
        f"## Configuration\n\n",
        f"```json\n{json.dumps(config, indent=2)}\n```\n\n",
        f"## Summary Statistics\n\n",
        f"| Feature | Mean | Max | Std | Sparsity | Selectivity |\n",
        f"|---------|------|-----|-----|----------|-------------|\n",
    ]

    for _, row in stats_df.head(20).iterrows():
        report.append(
            f"| {int(row['feature_idx'])} | {row['mean_activation']:.4f} | "
            f"{row['max_activation']:.4f} | {row['std_activation']:.4f} | "
            f"{row['sparsity']:.1%} | {row['selectivity']:.1%} |\n"
        )

    report.append("\n## Visualization Files\n\n")
    report.append("Generated visualizations include:\n\n")
    report.append("- Feature activation heatmaps (which graphs/nodes activate each feature)\n")
    report.append("- Top-activated node examples (strength of activation)\n")
    report.append("- Decoder weight distributions (what each feature learns)\n")
    report.append("- Feature statistics summary table (PNG image)\n\n")

    report.append("## Interpretation\n\n")
    report.append("- **High selectivity (> 0.5):** Feature activates on specific graphs/nodes\n")
    report.append("- **Low sparsity (< 0.8):** Feature is frequently active\n")
    report.append("- **Uneven decoder weights:** Feature focuses on specific input dimensions\n")

    return "".join(report)


def main():
    parser = argparse.ArgumentParser(description='Visualize SAE Feature Activations')
    parser.add_argument('--variant', type=str, choices=['topk', 'gated', 'jumprelu', 'switch'],
                       help='Specific variant to visualize')
    parser.add_argument('--all-variants', action='store_true', help='Visualize all variants')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension')
    parser.add_argument('--features', type=int, default=10, help='Number of features to visualize')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing SAE checkpoints')
    parser.add_argument('--activations_path', type=str, default='data/activations.npz',
                       help='Path to stored activations')

    args = parser.parse_args()

    print("="*70)
    print("FEATURE ACTIVATION VISUALIZATION")
    print("="*70)

    # Load activations
    print("\nLoading activations...")
    activations = load_activations(args.activations_path)

    if activations is None:
        print("Warning: Could not load activations. Continuing with model analysis only.")
        activations = None

    # Determine which variants to analyze
    if args.all_variants:
        variants = ['topk', 'gated', 'jumprelu', 'switch']
    elif args.variant:
        variants = [args.variant]
    else:
        variants = ['topk']

    for variant in variants:
        print(f"\n{'='*70}")
        print(f"Visualizing {variant.upper()}")
        print(f"{'='*70}")

        # Load checkpoint
        checkpoint_path = Path(args.checkpoint_dir) / f'sae_{variant}_best.pt'
        if not checkpoint_path.exists():
            # Try to find any checkpoint for this variant
            import glob
            checkpoints = glob.glob(f"{args.checkpoint_dir}/sae_{variant}_*.pt")
            if checkpoints:
                checkpoint_path = checkpoints[0]
            else:
                print(f"Warning: No checkpoint found for {variant}")
                continue

        print(f"Loading checkpoint: {checkpoint_path}")
        sae_model = load_sae_model(str(checkpoint_path), variant)

        if sae_model is None:
            print(f"Could not load model for {variant}")
            continue

        # Get decoder weights
        try:
            decoder_weights = sae_model.decoder.weight.detach().numpy()  # (latent_dim, input_dim)
        except (AttributeError, TypeError, RuntimeError) as e:
            print(f"Could not extract decoder weights for {variant}: {type(e).__name__}: {str(e)}")
            decoder_weights = None

        # Compute latent activations if activations available
        if activations is not None:
            print("Computing SAE latent activations...")
            latents = compute_feature_activations(sae_model, activations)

            if latents is not None:
                print(f"Latents shape: {latents.shape}")

                # Visualize top features
                config = sae_model.get_config() if hasattr(sae_model, 'get_config') else {}

                for feature_idx in range(min(args.features, latents.shape[2])):
                    print(f"\n  Feature {feature_idx}...")

                    plot_feature_activation_heatmap(latents, feature_idx, variant, config)
                    plot_top_activated_nodes(latents, feature_idx, variant, config)

                    if decoder_weights is not None:
                        plot_decoder_weights(decoder_weights, feature_idx, variant, config)

                # Create summary report
                report = create_feature_summary_report(variant, latents, decoder_weights, config)
                report_file = VIZ_DIR / f'feature_visualizations_{variant}_report.md'
                with open(report_file, 'w') as f:
                    f.write(report)
                print(f"\n✓ Saved report to {report_file}")

                # Create statistics table as PNG image
                stats_df = compute_feature_statistics(latents)
                plot_feature_statistics_table(stats_df, variant, num_rows=20)

    print("\n" + "="*70)
    print("✓ FEATURE VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nVisualizations saved to: {VIZ_DIR}/")


if __name__ == "__main__":
    main()
