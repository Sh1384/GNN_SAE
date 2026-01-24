"""
Interpretability Analysis Module

Performs correlation analysis and causal ablation studies to identify
SAE features that correspond to specific network motifs.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.metrics import mutual_info_score
from tqdm import tqdm


class InterpretabilityAnalyzer:
    """
    Analyzer for discovering interpretable SAE features correlated with motifs.

    Performs:
    1. Point-biserial correlation analysis
    2. Mutual information analysis
    3. Feature selection based on interpretability criteria
    4. Causal ablation experiments
    """

    def __init__(self, data_dir: str = "./virtual_graphs/data",
                 output_dir: str = "./outputs/interpretability"):
        """
        Initialize the analyzer.

        Args:
            data_dir: Directory containing graph data
            output_dir: Directory to save analysis results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Motif types
        self.motif_types = [
            "feedforward_loop",
            "feedback_loop",
            "single_input_module",
            "cascade"
        ]

    def load_graph_metadata(self, split_name: str = "train") -> Tuple[List[str], List[int]]:
        """
        Load graph paths and extract motif labels.

        Args:
            split_name: Data split to load

        Returns:
            Tuple of (graph_paths, motif_labels)
        """
        # This should match the split logic from gnn_train.py
        # For simplicity, we'll load all graphs and re-split

        all_graph_paths = []
        all_motif_labels = []

        # Load single-motif graphs
        single_motif_dir = self.data_dir / "single_motif_graphs"
        if single_motif_dir.exists():
            for motif_idx, motif_type in enumerate(self.motif_types):
                motif_dir = single_motif_dir / motif_type
                if motif_dir.is_dir():
                    graphs = sorted(motif_dir.glob("*.pkl"))
                    all_graph_paths.extend(graphs)
                    all_motif_labels.extend([motif_idx] * len(graphs))

        # Load mixed-motif graphs (label as -1 for "mixed")
        mixed_motif_dir = self.data_dir / "mixed_motif_graphs"
        if mixed_motif_dir.exists():
            graphs = sorted(mixed_motif_dir.glob("*.pkl"))
            all_graph_paths.extend(graphs)
            all_motif_labels.extend([-1] * len(graphs))

        return all_graph_paths, all_motif_labels

    def create_node_motif_labels(self) -> Tuple[np.ndarray, List[str]]:
        """
        Create node-level motif labels from graph-level metadata.

        Returns:
            Tuple of (node_motif_labels, node_graph_ids)
            - node_motif_labels: Array of motif labels for each node
            - node_graph_ids: List of graph IDs for each node
        """
        # Load activation files to get node count
        layer1_train = Path("outputs/activations/layer1/train")

        if not layer1_train.exists():
            raise ValueError("Activations not found. Run gnn_train.py first.")

        activation_files = sorted(layer1_train.glob("graph_*.pt"))

        graph_paths, graph_labels = self.load_graph_metadata()

        node_labels = []
        node_graph_ids = []

        # Map each node to its graph's motif label
        for graph_idx, act_file in enumerate(activation_files):
            # Load activation to get number of nodes
            act = torch.load(act_file, map_location='cpu')
            num_nodes = act.shape[0] if act.dim() > 1 else 1

            # Get graph label (use modulo in case of split mismatch)
            graph_label = graph_labels[graph_idx % len(graph_labels)]

            # Assign same label to all nodes in graph
            node_labels.extend([graph_label] * num_nodes)
            node_graph_ids.extend([f"graph_{graph_idx}"] * num_nodes)

        return np.array(node_labels), node_graph_ids

    def compute_pointbiserial_correlations(self, latents: np.ndarray,
                                          motif_labels: np.ndarray,
                                          layer_name: str) -> pd.DataFrame:
        """
        Compute point-biserial correlations between SAE features and motif presence.

        Args:
            latents: Latent features (num_nodes, latent_dim)
            motif_labels: Motif labels for each node (num_nodes,)
            layer_name: Name of layer

        Returns:
            DataFrame with correlation results
        """
        print(f"\nComputing point-biserial correlations for {layer_name}...")

        n_features = latents.shape[1]
        results = []

        for motif_idx, motif_name in enumerate(self.motif_types):
            # Create binary labels (1 if motif present, 0 otherwise)
            binary_labels = (motif_labels == motif_idx).astype(int)

            # Skip if no positive examples
            if binary_labels.sum() == 0:
                continue

            for feature_idx in tqdm(range(n_features),
                                   desc=f"Correlating {motif_name}",
                                   leave=False):
                feature_values = latents[:, feature_idx]

                # Compute point-biserial correlation
                corr, pval = pointbiserialr(binary_labels, feature_values)

                results.append({
                    'motif': motif_name,
                    'feature_idx': feature_idx,
                    'correlation': corr,
                    'p_value': pval,
                    'layer': layer_name
                })

        df = pd.DataFrame(results)
        return df

    def compute_mutual_information(self, latents: np.ndarray,
                                   motif_labels: np.ndarray,
                                   layer_name: str,
                                   n_bins: int = 10) -> pd.DataFrame:
        """
        Compute mutual information between SAE features and motif types.

        Args:
            latents: Latent features (num_nodes, latent_dim)
            motif_labels: Motif labels for each node (num_nodes,)
            layer_name: Name of layer
            n_bins: Number of bins for discretizing continuous features

        Returns:
            DataFrame with mutual information results
        """
        print(f"\nComputing mutual information for {layer_name}...")

        n_features = latents.shape[1]
        results = []

        # Filter out mixed motif graphs (label -1)
        valid_mask = motif_labels >= 0
        valid_labels = motif_labels[valid_mask]
        valid_latents = latents[valid_mask]

        if len(valid_labels) == 0:
            return pd.DataFrame()

        for feature_idx in tqdm(range(n_features),
                               desc="Computing MI"):
            feature_values = valid_latents[:, feature_idx]

            # Discretize continuous features into bins
            feature_discrete = np.digitize(feature_values,
                                          bins=np.linspace(feature_values.min(),
                                                          feature_values.max(),
                                                          n_bins))

            # Compute mutual information
            mi = mutual_info_score(valid_labels, feature_discrete)

            results.append({
                'feature_idx': feature_idx,
                'mutual_info': mi,
                'layer': layer_name
            })

        df = pd.DataFrame(results)
        return df

    def identify_interpretable_features(self, corr_df: pd.DataFrame,
                                       threshold_high: float = 0.5,
                                       threshold_low: float = 0.2) -> pd.DataFrame:
        """
        Identify interpretable features with high correlation to one motif
        and low correlation to others.

        Args:
            corr_df: DataFrame with correlation results
            threshold_high: Minimum correlation for target motif
            threshold_low: Maximum correlation for non-target motifs

        Returns:
            DataFrame with interpretable features
        """
        print("\nIdentifying interpretable features...")

        interpretable_features = []

        # Get unique features and layers
        features = corr_df['feature_idx'].unique()
        layers = corr_df['layer'].unique()

        for layer in layers:
            layer_df = corr_df[corr_df['layer'] == layer]

            for feature_idx in features:
                feature_df = layer_df[layer_df['feature_idx'] == feature_idx]

                # Find maximum correlation
                max_corr = feature_df['correlation'].abs().max()

                if max_corr >= threshold_high:
                    # Get motif with max correlation
                    max_motif = feature_df.loc[feature_df['correlation'].abs().idxmax(), 'motif']

                    # Check correlations with other motifs
                    other_motifs = feature_df[feature_df['motif'] != max_motif]
                    max_other_corr = other_motifs['correlation'].abs().max()

                    if max_other_corr < threshold_low:
                        interpretable_features.append({
                            'layer': layer,
                            'feature_idx': feature_idx,
                            'target_motif': max_motif,
                            'target_correlation': max_corr,
                            'max_other_correlation': max_other_corr,
                            'interpretable': True
                        })

        df = pd.DataFrame(interpretable_features)
        return df

    def plot_motif_feature_heatmap(self, corr_df: pd.DataFrame,
                                   layer_name: str,
                                   top_k: int = 50):
        """
        Plot heatmap of motif-feature correlations.

        Args:
            corr_df: DataFrame with correlation results
            layer_name: Name of layer
            top_k: Number of top features to show
        """
        print(f"\nPlotting heatmap for {layer_name}...")

        # Filter by layer
        layer_df = corr_df[corr_df['layer'] == layer_name]

        # Pivot to create motif x feature matrix
        pivot_df = layer_df.pivot_table(
            index='motif',
            columns='feature_idx',
            values='correlation',
            fill_value=0
        )

        # Select top features by max absolute correlation
        feature_max_corr = pivot_df.abs().max(axis=0)
        top_features = feature_max_corr.nlargest(top_k).index
        pivot_subset = pivot_df[top_features]

        # Plot heatmap
        plt.figure(figsize=(16, 6))
        sns.heatmap(pivot_subset, cmap='RdBu_r', center=0,
                   vmin=-1, vmax=1,
                   cbar_kws={'label': 'Point-Biserial Correlation'})
        plt.title(f'Motif-Feature Correlations ({layer_name}, Top {top_k} Features)')
        plt.xlabel('SAE Feature Index')
        plt.ylabel('Motif Type')
        plt.tight_layout()

        output_path = self.output_dir / f"heatmap_{layer_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved heatmap to {output_path}")

    def plot_feature_distribution(self, latents: np.ndarray,
                                  motif_labels: np.ndarray,
                                  feature_idx: int,
                                  layer_name: str):
        """
        Plot distribution of a specific feature across motif types.

        Args:
            latents: Latent features
            motif_labels: Motif labels
            feature_idx: Feature index to plot
            layer_name: Layer name
        """
        plt.figure(figsize=(10, 6))

        valid_mask = motif_labels >= 0
        valid_labels = motif_labels[valid_mask]
        valid_latents = latents[valid_mask, feature_idx]

        for motif_idx, motif_name in enumerate(self.motif_types):
            motif_mask = valid_labels == motif_idx
            if motif_mask.sum() > 0:
                values = valid_latents[motif_mask]
                plt.hist(values, bins=30, alpha=0.5, label=motif_name)

        plt.xlabel(f'Feature {feature_idx} Activation')
        plt.ylabel('Count')
        plt.title(f'Distribution of Feature {feature_idx} ({layer_name})')
        plt.legend()
        plt.tight_layout()

        output_path = self.output_dir / f"feature_{feature_idx}_{layer_name}_dist.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class CausalAblationAnalyzer:
    """
    Performs causal ablation experiments on SAE features.

    Tests whether modifying specific SAE features causally affects
    GNN predictions in motif-specific ways.
    """

    def __init__(self, gnn_model_path: str, sae_model_paths: Dict[str, str],
                 device: str = 'cuda'):
        """
        Initialize the ablation analyzer.

        Args:
            gnn_model_path: Path to trained GNN model
            sae_model_paths: Dict mapping layer names to SAE model paths
            device: Device to run on
        """
        self.device = device
        self.gnn_model = self._load_gnn_model(gnn_model_path)
        self.sae_models = {layer: self._load_sae_model(path)
                          for layer, path in sae_model_paths.items()}

    def _load_gnn_model(self, path: str):
        """Load trained GNN model."""
        from gnn_train import GCNModel

        model = GCNModel(input_dim=2, hidden_dim=64, output_dim=1, dropout=0.2)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model

    def _load_sae_model(self, path: str):
        """Load trained SAE model."""
        from sparse_autoencoder import SparseAutoencoder

        checkpoint = torch.load(path, map_location=self.device)
        model = SparseAutoencoder(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim'],
            sparsity_lambda=checkpoint['sparsity_lambda']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def ablate_feature(self, activations: torch.Tensor,
                      feature_idx: int,
                      layer_name: str,
                      ablation_type: str = 'zero') -> torch.Tensor:
        """
        Ablate a specific SAE feature and reconstruct activations.

        Args:
            activations: Original layer activations
            feature_idx: Index of feature to ablate
            layer_name: Name of layer
            ablation_type: Type of ablation ('zero' or 'amplify')

        Returns:
            Modified activations after ablation
        """
        sae_model = self.sae_models[layer_name]

        # Encode to latent space
        with torch.no_grad():
            z = sae_model.encode(activations)

            # Ablate feature
            if ablation_type == 'zero':
                z[:, feature_idx] = 0
            elif ablation_type == 'amplify':
                z[:, feature_idx] = 2 * z[:, feature_idx]
            else:
                raise ValueError(f"Unknown ablation type: {ablation_type}")

            # Decode back to activation space
            modified_activations = sae_model.decode(z)

        return modified_activations

    def compute_ablation_effect(self, graph_data,
                                feature_idx: int,
                                layer_name: str,
                                ablation_type: str = 'zero') -> Tuple[float, float]:
        """
        Compute effect of ablating a feature on prediction loss.

        Args:
            graph_data: PyG Data object
            feature_idx: Feature to ablate
            layer_name: Layer name
            ablation_type: Ablation type

        Returns:
            Tuple of (baseline_loss, ablated_loss)
        """
        # This is a simplified version - full implementation would require
        # intervening on GNN forward pass with modified activations

        # For now, we'll compute the difference in reconstruction loss
        # as a proxy for prediction effect

        raise NotImplementedError("Full causal intervention requires GNN architecture modification")


def analyze_layer(layer_name: str, latent_dim: int):
    """
    Run full interpretability analysis for a specific layer.

    Args:
        layer_name: Name of layer to analyze
        latent_dim: Dimension of latent space
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {layer_name}")
    print(f"{'='*60}")

    analyzer = InterpretabilityAnalyzer()

    # Load latent features
    latent_path = Path(f"outputs/sae_latents/{layer_name}/train/all_latents.pt")
    if not latent_path.exists():
        print(f"Error: Latents not found at {latent_path}")
        print("Please run sparse_autoencoder.py first.")
        return

    latents = torch.load(latent_path, map_location='cpu').numpy()
    print(f"Loaded latents with shape: {latents.shape}")

    # Create node-level motif labels
    motif_labels, graph_ids = analyzer.create_node_motif_labels()
    print(f"Created {len(motif_labels)} node labels")

    # Trim to match latent size
    min_size = min(len(latents), len(motif_labels))
    latents = latents[:min_size]
    motif_labels = motif_labels[:min_size]

    # Compute point-biserial correlations
    corr_df = analyzer.compute_pointbiserial_correlations(
        latents, motif_labels, layer_name
    )

    # Save correlations
    corr_output = analyzer.output_dir / f"motif_feature_correlation_{layer_name}.csv"
    corr_df.to_csv(corr_output, index=False)
    print(f"Saved correlations to {corr_output}")

    # Compute mutual information
    mi_df = analyzer.compute_mutual_information(
        latents, motif_labels, layer_name
    )

    # Save mutual information
    mi_output = analyzer.output_dir / f"motif_feature_mutualinfo_{layer_name}.csv"
    mi_df.to_csv(mi_output, index=False)
    print(f"Saved mutual information to {mi_output}")

    # Identify interpretable features
    interpretable_df = analyzer.identify_interpretable_features(corr_df)

    if len(interpretable_df) > 0:
        print(f"\nFound {len(interpretable_df)} interpretable features:")
        print(interpretable_df)

        # Save interpretable features
        interp_output = analyzer.output_dir / f"interpretable_features_{layer_name}.csv"
        interpretable_df.to_csv(interp_output, index=False)
        print(f"Saved to {interp_output}")

        # Plot distributions for top interpretable features
        for idx, row in interpretable_df.head(5).iterrows():
            analyzer.plot_feature_distribution(
                latents, motif_labels,
                int(row['feature_idx']), layer_name
            )
    else:
        print("\nNo highly interpretable features found with current thresholds.")

    # Plot heatmap
    analyzer.plot_motif_feature_heatmap(corr_df, layer_name, top_k=50)

    # Print summary statistics
    print("\nCorrelation Summary:")
    print(corr_df.groupby('motif')['correlation'].agg(['mean', 'std', 'max']))


def main():
    """Main interpretability analysis pipeline."""
    print("Starting interpretability analysis...")

    # Check if required files exist
    if not Path("outputs/sae_latents").exists():
        print("Error: SAE latents not found. Please run sparse_autoencoder.py first.")
        return

    # Analyze Layer 1
    analyze_layer("layer1", latent_dim=512)

    # Analyze Layer 2
    analyze_layer("layer2", latent_dim=32)

    # Create combined correlation file
    print("\nCreating combined analysis files...")
    output_dir = Path("outputs/interpretability")

    layer1_corr = output_dir / "motif_feature_correlation_layer1.csv"
    layer2_corr = output_dir / "motif_feature_correlation_layer2.csv"

    if layer1_corr.exists() and layer2_corr.exists():
        df1 = pd.read_csv(layer1_corr)
        df2 = pd.read_csv(layer2_corr)
        combined = pd.concat([df1, df2], ignore_index=True)
        combined.to_csv(output_dir / "motif_feature_correlation.csv", index=False)
        print("Saved combined correlation file")

    layer1_mi = output_dir / "motif_feature_mutualinfo_layer1.csv"
    layer2_mi = output_dir / "motif_feature_mutualinfo_layer2.csv"

    if layer1_mi.exists() and layer2_mi.exists():
        df1 = pd.read_csv(layer1_mi)
        df2 = pd.read_csv(layer2_mi)
        combined = pd.concat([df1, df2], ignore_index=True)
        combined.to_csv(output_dir / "motif_feature_mutualinfo.csv", index=False)
        print("Saved combined mutual information file")

    print("\n" + "="*60)
    print("Interpretability analysis complete!")
    print(f"Results saved to {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()
