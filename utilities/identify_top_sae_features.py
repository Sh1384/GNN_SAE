#!/usr/bin/env python3
"""
Identify Top SAE Features for Each Motif Type

This script loads the optimal SAE configuration and identifies
the top features for each motif type based on correlation analysis.

Output: JSON file with motif -> feature_idx mappings for use in comparison.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pointbiserialr
from sparse_autoencoder import SparseAutoencoder
from tqdm import tqdm


def identify_top_features(latent_dim: int = 128, k: int = 16, top_n: int = 3):
    """
    Identify top SAE features for each motif type.

    Args:
        latent_dim: SAE latent dimension
        k: SAE sparsity parameter
        top_n: Number of top features to return per motif

    Returns:
        Dictionary mapping motif type to list of (feature_idx, rpb, pval)
    """
    print(f"Analyzing SAE configuration: latent_dim={latent_dim}, k={k}")

    # Load SAE model
    INPUT_DIM = 64
    model = SparseAutoencoder(input_dim=INPUT_DIM, latent_dim=latent_dim, k=k)
    checkpoint_path = f"checkpoints/sae_latent{latent_dim}_k{k}.pt"

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test graph IDs
    with open('outputs/test_graph_ids.json', 'r') as f:
        test_graph_ids = json.load(f)['graph_ids']

    # Load activations and metadata
    activation_dir = Path("outputs/activations/layer3_new/test")
    metadata_dir = Path("virtual_graphs/data/all_graphs/graph_motif_metadata")

    all_latents = []
    all_motifs = []

    print("Loading activations and computing latent representations...")
    for graph_id in tqdm(test_graph_ids):
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

    # Compute correlations for each motif
    print("\nComputing correlations for each motif type...")
    latent_features = [f'z{i+1}' for i in range(latent_dim)]
    motif_types = ['in_feedforward_loop', 'in_feedback_loop',
                   'in_single_input_module', 'in_cascade']

    motif_to_features = {}

    for motif in motif_types:
        if motif not in df.columns:
            continue

        correlations = []
        for z_col in latent_features:
            if df[z_col].std() == 0:
                continue

            corr, pval = pointbiserialr(df[motif], df[z_col])
            correlations.append({
                'feature': z_col,
                'feature_idx': int(z_col[1:]),  # Extract number
                'rpb': corr,
                'rpb_abs': abs(corr),
                'pval': pval
            })

        df_corr = pd.DataFrame(correlations)

        # Get top N features
        top_features = df_corr.nlargest(top_n, 'rpb_abs')

        motif_short = motif.replace('in_', '')
        print(f"\n{motif_short}:")
        print(f"{'Feature':<10} {'rpb':>8} {'|rpb|':>8} {'p-value':>12}")
        print("-" * 42)

        feature_list = []
        for _, row in top_features.iterrows():
            print(f"{row['feature']:<10} {row['rpb']:>8.4f} {row['rpb_abs']:>8.4f} {row['pval']:>12.2e}")
            feature_list.append({
                'feature_idx': row['feature_idx'],
                'feature_name': row['feature'],
                'rpb': float(row['rpb']),
                'rpb_abs': float(row['rpb_abs']),
                'pval': float(row['pval'])
            })

        motif_to_features[motif_short] = feature_list

    return motif_to_features


def main():
    print("="*70)
    print("IDENTIFYING TOP SAE FEATURES FOR EACH MOTIF TYPE")
    print("="*70)
    print()

    # Use optimal config from compare_sae_configs.py
    LATENT_DIM = 128
    K = 16

    # Identify top features
    motif_features = identify_top_features(latent_dim=LATENT_DIM, k=K, top_n=5)

    if motif_features is None:
        print("Error: Could not identify features")
        return

    # Save to JSON
    output_path = Path('outputs/top_sae_features.json')
    with open(output_path, 'w') as f:
        json.dump(motif_features, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    print("\nTo use these in the comparison script, update the 'targets' dictionary:")
    print("\ntargets = {")
    for motif, features in motif_features.items():
        if len(features) > 0:
            top_feature = features[0]
            print(f"    '{motif}': {{")
            print(f"        'feature_idx': {top_feature['feature_idx']},")
            print(f"        'motif_type': '{motif}',")
            print(f"        'description': '{motif.replace('_', ' ').title()} detector (rpb={top_feature['rpb_abs']:.3f})'")
            print(f"    }},")
    print("}")


if __name__ == "__main__":
    main()
