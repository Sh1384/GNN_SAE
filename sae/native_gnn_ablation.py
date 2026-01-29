#!/usr/bin/env python3
"""
Native GNN Activation Space Ablation (Strategy 2: Activation Patching)

Performs ablations directly in native 80-dimensional GNN activation space,
validating mechanistic interpretation by avoiding SAE reconstruction error.

Key innovation: Instead of ablating SAE latent features and reconstructing,
directly patches (zeros out) activations for nodes most strongly activated by
a specific SAE feature.

Usage:
    python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --feature 0
    python native_gnn_ablation.py --all-features --variant gated
"""

import argparse
import json
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import networkx as nx
from scipy.stats import wilcoxon, ranksums, pointbiserialr
import warnings

warnings.filterwarnings('ignore')

# Configuration
INPUT_DIM = 80
OUTPUT_DIR = Path("outputs")
ABLATION_DIR = OUTPUT_DIR / "native_gnn_ablations"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)


def load_sae_and_activations(variant: str, config: Dict, graph_id: int, use_mixed_motifs: bool = False) -> Tuple:
    """Load SAE model and GNN activations for a graph.

    Args:
        variant: SAE variant name ('topk', 'gated', 'jumprelu', 'switch')
        config: Configuration dict for the variant
        graph_id: ID of the graph
        use_mixed_motifs: Load activations from mixed-motif graphs (4000-4999) instead of single-motif (0-3999)

    Returns:
        Tuple of (model, activations)

    Raises:
        ValueError: If variant is unknown or config is invalid
        FileNotFoundError: If SAE checkpoint or activation file not found
    """
    from sparse_autoencoder import TopKSAE, GatedSAE, JumpReLUSAE, SwitchSAE

    # Construct checkpoint path
    try:
        if variant == 'topk':
            ckpt_path = f"checkpoints/sae_topk_latent{config['latent_dim']}_k{config['k']}_seed42.pt"
            model = TopKSAE(input_dim=INPUT_DIM, **config)
        elif variant == 'gated':
            ckpt_path = f"checkpoints/sae_gated_latent{config['latent_dim']}_lambda{config['sparsity_coef']:.0e}_seed42.pt"
            model = GatedSAE(input_dim=INPUT_DIM, **config)
        elif variant == 'jumprelu':
            bandwidth = config.get('bandwidth', 0.01)
            ckpt_path = f"checkpoints/sae_jumprelu_latent{config['latent_dim']}_thresh{config['threshold_init']:.0e}_bw{bandwidth:.0e}_seed42.pt"
            model = JumpReLUSAE(input_dim=INPUT_DIM, **config)
        elif variant == 'switch':
            ckpt_path = f"checkpoints/sae_switch_experts{config['num_experts']}_latent{config['num_experts']*config['latent_per_expert']}_k{config['k_per_expert']}_seed42.pt"
            # SwitchSAE doesn't accept latent_dim, only num_experts, latent_per_expert, k_per_expert
            switch_config = {k: v for k, v in config.items() if k in ['num_experts', 'latent_per_expert', 'k_per_expert']}
            model = SwitchSAE(input_dim=INPUT_DIM, **switch_config)
        else:
            raise ValueError(f"Unknown SAE variant: {variant}")
    except KeyError as e:
        raise ValueError(f"Missing required config key for {variant}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to instantiate {variant} SAE model: {str(e)}")

    # Load SAE model
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    try:
        checkpoint = torch.load(ckpt_path, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load SAE checkpoint {ckpt_path}: {str(e)}")

    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint missing 'model_state_dict' key")

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        raise ValueError(f"Failed to load SAE state dict (shape mismatch?): {str(e)}")

    model.eval()

    # Load GNN activations - support both single-motif and mixed-motif graphs
    if use_mixed_motifs:
        # Use mixed-motif activations (4000-4999)
        activation_file = Path(f"outputs/activations/layer2_new/mixed/graph_{graph_id}.pt")
        if not activation_file.exists():
            raise FileNotFoundError(f"Mixed-motif activation file not found: {activation_file}\nPlease run: python generate_mixed_motif_activations.py")
    else:
        # Use single-motif test activations (0-3999)
        activation_file = Path(f"outputs/activations/layer2_new/test/graph_{graph_id}.pt")
        if not activation_file.exists():
            raise FileNotFoundError(f"Activation file not found: {activation_file}")

    try:
        activations = torch.load(activation_file, weights_only=True)
    except Exception as e:
        raise ValueError(f"Failed to load activations for graph {graph_id}: {str(e)}")

    # Validate activations shape
    if activations.ndim != 2 or activations.shape[1] != INPUT_DIM:
        raise ValueError(f"Invalid activation shape {activations.shape}, expected (N, {INPUT_DIM})")

    return model, activations


def patch_salient_nodes(activations: torch.Tensor, sae_model, feature_idx: int,
                        top_nodes_k: int = 5, patch_type: str = 'zero') -> Tuple[torch.Tensor, torch.Tensor]:
    """Patch (zero out) nodes most activated by a specific SAE feature.

    Args:
        activations: Native GNN activations (num_nodes, INPUT_DIM)
        sae_model: Trained SAE model
        feature_idx: Which SAE feature to ablate
        top_nodes_k: Number of top-activated nodes to patch
        patch_type: 'zero' | 'mean' | 'shuffle'

    Returns:
        (modified_activations, top_node_indices)

    Raises:
        ValueError: If patch_type is unknown or feature_idx is invalid
    """
    # Validate inputs
    if not isinstance(activations, torch.Tensor):
        raise ValueError("Activations must be a torch.Tensor")
    if activations.ndim != 2:
        raise ValueError(f"Activations must be 2D, got shape {activations.shape}")
    if feature_idx < 0:
        raise ValueError(f"Feature index cannot be negative: {feature_idx}")

    try:
        with torch.no_grad():
            # Encode activations through SAE
            latents = sae_model.encode(activations)  # (num_nodes, latent_dim)

            # Validate latent shape
            if latents.ndim != 2:
                raise ValueError(f"SAE output shape invalid: {latents.shape}")

            # Check feature index bounds
            if feature_idx >= latents.shape[1]:
                raise ValueError(f"Feature index {feature_idx} out of bounds [0, {latents.shape[1]-1}]")

            # Get activation strength for this feature
            feature_activations = latents[:, feature_idx]  # (num_nodes,)

            # Find top-k most activated nodes
            k_actual = min(top_nodes_k, len(feature_activations))
            if k_actual <= 0:
                raise ValueError(f"No nodes to patch (k_actual={k_actual})")

            top_node_indices = torch.topk(feature_activations, k_actual, dim=0)[1]
    except Exception as e:
        raise ValueError(f"Failed to compute top nodes for feature {feature_idx}: {str(e)}")

    # Apply patch
    try:
        modified = activations.clone()

        if patch_type == 'zero':
            modified[top_node_indices, :] = 0.0
        elif patch_type == 'mean':
            # Replace with dataset mean
            mean_act = activations.mean(dim=0, keepdim=True)
            modified[top_node_indices, :] = mean_act
        elif patch_type == 'shuffle':
            # Shuffle activations for these nodes
            perm = torch.randperm(len(top_node_indices))
            modified[top_node_indices, :] = modified[top_node_indices[perm], :]
        else:
            raise ValueError(f"Unknown patch type: {patch_type}. Expected 'zero', 'mean', or 'shuffle'")

        return modified, top_node_indices
    except Exception as e:
        raise ValueError(f"Failed to apply patch (type={patch_type}): {str(e)}")


def compute_native_rpb(test_graph_ids: List[int], motif: str) -> pd.DataFrame:
    """
    Compute point-biserial correlation between native GNN dimensions and motif presence.

    This bypasses the SAE encoder and directly measures which native neurons
    correlate with motif presence.

    Args:
        test_graph_ids: List of test graph IDs to analyze
        motif: Target motif (e.g., 'in_feedback_loop')

    Returns:
        DataFrame with columns: native_dim, rpb, rpb_abs, pval
    """
    print(f"\nComputing native r_pb for motif: {motif}")

    all_activations = []
    all_motif_labels = []

    for graph_id in tqdm(test_graph_ids[:200], desc="Loading activations"):
        # Load activations
        act_file = Path(f"outputs/activations/layer2_new/test/graph_{graph_id}.pt")
        if not act_file.exists():
            continue

        activations = torch.load(act_file, weights_only=True)

        # Load metadata
        meta_file = Path(f"virtual_graphs/data/all_graphs/graph_motif_metadata/graph_{graph_id}_metadata.csv")
        if not meta_file.exists():
            continue

        try:
            metadata = pd.read_csv(meta_file, index_col=0)
        except (pd.errors.ParserError, FileNotFoundError, OSError, ValueError) as e:
            print(f"    Warning: Could not read metadata for graph {graph_id}: {type(e).__name__}")
            continue

        if len(metadata) != activations.shape[0]:
            continue

        # Collect data
        for node_idx in range(activations.shape[0]):
            all_activations.append(activations[node_idx].cpu().numpy())

            # Get motif label (0 or 1)
            has_motif = int(metadata.iloc[node_idx].get(motif, 0))
            all_motif_labels.append(has_motif)

    if len(all_activations) == 0:
        print(f"  Warning: No data found for motif {motif}")
        return pd.DataFrame()

    activations_array = np.array(all_activations)  # (n_nodes, INPUT_DIM)
    motif_array = np.array(all_motif_labels)       # (n_nodes,)

    # Compute r_pb for each dimension
    results = []
    for dim in range(INPUT_DIM):
        if activations_array[:, dim].std() == 0:
            continue

        try:
            r_pb, p_val = pointbiserialr(motif_array, activations_array[:, dim])
            results.append({
                'native_dim': dim,
                'rpb': r_pb,
                'rpb_abs': abs(r_pb),
                'pval': p_val,
            })
        except (ValueError, RuntimeError, TypeError) as e:
            # Skip dimensions where r_pb computation fails (e.g., invalid input)
            print(f"    Warning: Failed to compute r_pb for dimension {dim}: {type(e).__name__}: {str(e)}")
            continue

    df = pd.DataFrame(results)
    df = df.sort_values('rpb_abs', ascending=False)

    print(f"  Max |r_pb|: {df['rpb_abs'].max():.3f}")
    print(f"  Dims with |r_pb| > 0.3: {(df['rpb_abs'] > 0.3).sum()}")

    return df


def patch_native_neurons_by_rpb(activations: torch.Tensor, motif: str,
                               test_graph_ids: List[int], top_dims_k: int = 5,
                               patch_type: str = 'zero') -> Tuple[torch.Tensor, List[int]]:
    """
    Patch (zero out) native GNN neurons that most strongly correlate with a motif.

    This is your proposed approach: identify native dimensions with high r_pb to motif,
    then directly ablate those dimensions (no SAE involved).

    Args:
        activations: Native GNN activations (num_nodes, INPUT_DIM)
        motif: Target motif (e.g., 'in_feedback_loop')
        test_graph_ids: Graph IDs for computing r_pb
        top_dims_k: Number of top correlated dimensions to patch
        patch_type: 'zero', 'mean', or 'shuffle'

    Returns:
        (modified_activations, patched_dim_indices)
    """
    # Compute r_pb for this motif
    df_rpb = compute_native_rpb(test_graph_ids, motif)

    if len(df_rpb) == 0:
        print(f"  Warning: Could not compute r_pb, returning unmodified activations")
        return activations, []

    # Get top-k dimensions by correlation strength
    top_dims = df_rpb.nlargest(top_dims_k, 'rpb_abs')['native_dim'].tolist()

    print(f"  Patching top {len(top_dims)} dimensions: {top_dims}")
    for _, row in df_rpb.nlargest(top_dims_k, 'rpb_abs').iterrows():
        print(f"    h_{int(row['native_dim'])}: |r_pb|={row['rpb_abs']:.3f}")

    # Apply patch
    modified = activations.clone()

    if patch_type == 'zero':
        for dim in top_dims:
            modified[:, dim] = 0.0
    elif patch_type == 'mean':
        for dim in top_dims:
            modified[:, dim] = activations[:, dim].mean()
    elif patch_type == 'shuffle':
        for dim in top_dims:
            perm = torch.randperm(len(modified))
            modified[:, dim] = modified[perm, dim]
    else:
        raise ValueError(f"Unknown patch type: {patch_type}")

    return modified, top_dims


def compute_gnn_loss(activations: torch.Tensor, gnn_model, graph_id: int) -> float:
    """
    Compute GNN loss (MSE on node imputation task).

    Mirrors the approach in run_ablation.py:
    1. Load graph structure and edge information
    2. Run GNN layer4 (conv4) using the given layer2 activations
    3. Compute MSE between GNN predictions and ground truth values on masked nodes

    Args:
        activations: Layer3 activations (num_nodes, INPUT_DIM) - output of conv3, either original or ablated
        gnn_model: Trained GNN model
        graph_id: Which graph to evaluate on

    Returns:
        MSE loss scalar or None if computation fails
    """
    if gnn_model is None:
        return None

    try:
        # Load graph data (same as run_ablation.py load_graph_data function)
        graph_path = Path(f"virtual_graphs/data/all_graphs/raw_graphs/graph_{graph_id}.pkl")
        if not graph_path.exists():
            return None

        import pickle
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        W = nx.to_numpy_array(G, weight='weight')
        edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
        edge_weight = torch.tensor(W[W != 0], dtype=torch.float32)

        # Load/simulate ground truth (same as run_ablation.py simulate_expression)
        local_seed = 42 + graph_id
        rng = np.random.default_rng(local_seed)
        n_nodes = W.shape[0]
        x = rng.uniform(0, 1, size=n_nodes)
        for _ in range(50):
            weighted_input = W @ x
            sigmoid_input = 1.0 / (1.0 + np.exp(-np.clip(weighted_input, -10, 10)))
            noise = rng.normal(0, 0.01, size=n_nodes)
            x = (1 - 0.3) * x + 0.3 * sigmoid_input + noise
            x = np.clip(x, 0, 1)
        y_true = torch.tensor(x, dtype=torch.float32)

        # Create evaluation mask (30% of nodes)
        mask = torch.tensor(rng.random(len(G.nodes())) < 0.3, dtype=torch.bool)

        # Run GNN layer4 (conv4) on the given layer2 activations to get final predictions
        with torch.no_grad():
            pred = gnn_model.conv4(activations, edge_index, edge_weight=edge_weight)
            pred = pred.squeeze(-1)

        # Compute MSE on masked nodes (same as run_ablation.py)
        loss = torch.mean(((pred - y_true)[mask]) ** 2).item()
        return loss

    except Exception as e:
        print(f"  Warning: Could not compute GNN loss for graph {graph_id}: {e}")
        return None


def compute_sae_rpb(variant: str, config: Dict, test_graph_ids: List[int], motif: str) -> pd.DataFrame:
    """
    Compute point-biserial correlation between SAE latent features and motif presence.

    This selects which SAE features are most correlated with the target motif,
    so we ablate the most interpretable features first.

    OPTIMIZATION: Tries to load cached latents from Phase 2 to avoid re-encoding.
    If cache not found, falls back to computing latents on-the-fly.

    Args:
        variant: SAE variant type (e.g., 'topk')
        config: SAE configuration dict with 'latent_dim', 'k', etc.
        test_graph_ids: List of test graph IDs
        motif: Target motif (e.g., 'in_feedback_loop')

    Returns:
        DataFrame with columns: feature_idx, rpb, rpb_abs, pval (sorted by |rpb|)
    """
    print(f"\nComputing SAE r_pb for motif: {motif}")
    print(f"  Variant: {variant}, Latent dim: {config.get('latent_dim')}")

    latents_array = None
    motif_array = None

    # TRY 1: Load cached latents from Phase 2 (FAST)
    if variant == 'topk':
        # Use actual config values, don't default to wrong values
        latent_dim = config.get('latent_dim')
        k = config.get('k')
        if latent_dim is None or k is None:
            print(f"  ⚠️  Missing latent_dim or k in config, cannot load cache")
            latents_array = None
        else:
            cache_file = Path(f'outputs/latent_cache/latents_topk_latent{latent_dim}_k{k}.pkl')
            
            if cache_file.exists():
                print(f"  ✓ Loading cached latents from {cache_file.name}")
                try:
                    import pickle
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    df_cached = cache_data['df']

                    # Extract latent features (z1, z2, ... zN) and motif labels
                    latent_cols = [col for col in df_cached.columns if col.startswith('z')]
                    if latent_cols and motif in df_cached.columns:
                        latents_array = df_cached[latent_cols].values
                        motif_array = df_cached[motif].values
                        print(f"  ✓ Loaded {len(latents_array)} nodes × {len(latent_cols)} latent dims (CACHED)")
                except Exception as e:
                    print(f"  ⚠️  Could not load cache: {e}, falling back to on-the-fly encoding")

    # TRY 2: Compute latents on-the-fly if cache not available
    if latents_array is None:
        print(f"  Loading SAE model and encoding activations...")
        try:
            sae_model, _ = load_sae_and_activations(variant, config, test_graph_ids[0])
        except Exception as e:
            print(f"  ⚠️  Could not load SAE model: {e}")
            return pd.DataFrame()

        all_latents = []
        all_motif_labels = []

        for graph_id in tqdm(test_graph_ids[:200], desc="Encoding activations"):
            act_file = Path(f"outputs/activations/layer2_new/test/graph_{graph_id}.pt")
            if not act_file.exists():
                continue

            activations = torch.load(act_file, weights_only=True)
            meta_file = Path(f"virtual_graphs/data/all_graphs/graph_motif_metadata/graph_{graph_id}_metadata.csv")
            if not meta_file.exists():
                continue

            try:
                metadata = pd.read_csv(meta_file, index_col=0)
            except (pd.errors.ParserError, FileNotFoundError, OSError, ValueError) as e:
                print(f"    Warning: Could not read metadata for graph {graph_id}: {type(e).__name__}")
                continue

            if len(metadata) != activations.shape[0]:
                continue

            # Encode through SAE
            with torch.no_grad():
                latents = sae_model.encode(activations)

            # Collect data
            for node_idx in range(latents.shape[0]):
                all_latents.append(latents[node_idx].cpu().numpy())
                has_motif = int(metadata.iloc[node_idx].get(motif, 0))
                all_motif_labels.append(has_motif)

        if len(all_latents) == 0:
            print(f"  ⚠️  No data found for motif {motif}")
            return pd.DataFrame()

        latents_array = np.array(all_latents)
        motif_array = np.array(all_motif_labels)
        print(f"  ✓ Loaded {len(latents_array)} nodes × {latents_array.shape[1]} latent dims (on-the-fly)")

    # Compute r_pb for each SAE feature
    results = []
    latent_dim = latents_array.shape[1]

    for feature_idx in range(latent_dim):
        if latents_array[:, feature_idx].std() == 0:
            continue

        try:
            r_pb, p_val = pointbiserialr(motif_array, latents_array[:, feature_idx])
            results.append({
                'feature_idx': feature_idx,
                'rpb': r_pb,
                'rpb_abs': abs(r_pb),
                'pval': p_val,
            })
        except (ValueError, RuntimeError, TypeError) as e:
            # Skip features where r_pb computation fails (e.g., invalid input)
            print(f"    Warning: Failed to compute r_pb for feature {feature_idx}: {type(e).__name__}: {str(e)}")
            continue

    df = pd.DataFrame(results)
    df = df.sort_values('rpb_abs', ascending=False)

    print(f"  Max |r_pb|: {df['rpb_abs'].max():.3f}")
    print(f"  Features with |r_pb| > 0.3: {(df['rpb_abs'] > 0.3).sum()}")

    return df


def run_native_ablation(variant: str, config: Dict, feature_idx: int,
                       graph_ids: List[int], motif: str = 'in_feedback_loop',
                       top_nodes_k: int = 5, use_mixed_motifs: bool = False) -> pd.DataFrame:
    """Run native activation patching ablation for a specific feature.

    Args:
        variant: SAE variant name ('topk', 'gated', 'jumprelu', 'switch')
        config: Configuration dict
        feature_idx: Which SAE feature to ablate
        graph_ids: List of test graph IDs
        motif: Which motif to check for presence (e.g., 'in_feedback_loop')
        top_nodes_k: Number of top-activated nodes to patch
        use_mixed_motifs: Use mixed-motif graphs (4000-4999) instead of single-motif (0-3999)

    Returns:
        DataFrame with columns: graph_id, has_motif, loss_original, loss_patched, delta_loss
    """
    results = []
    skipped_count = 0
    error_count = 0

    # Validate inputs
    if not graph_ids:
        print("Error: Empty graph ID list")
        return pd.DataFrame()
    if feature_idx < 0:
        print(f"Error: Invalid feature index: {feature_idx}")
        return pd.DataFrame()

    # Load SAE and GNN models once
    try:
        sae_model, _ = load_sae_and_activations(variant, config, graph_ids[0], use_mixed_motifs=use_mixed_motifs)
    except Exception as e:
        print(f"Error: Could not load SAE model: {str(e)}")
        return pd.DataFrame()

    # Load GNN model for loss computation
    gnn_model = None
    try:
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        from gnn_train_copy import GCNModel
        gnn_checkpoint_path = "checkpoints/gnn_model.pt"
        if Path(gnn_checkpoint_path).exists():
            try:
                state_dict = torch.load(gnn_checkpoint_path, weights_only=True)
                if 'conv1.bias' not in state_dict or 'conv2.bias' not in state_dict:
                    print(f"Error: GNN checkpoint missing required keys")
                    gnn_model = None
                else:
                    hidden_dim = state_dict['conv1.bias'].shape[0]
                    gnn_model = GCNModel(input_dim=2, hidden_dim=hidden_dim, output_dim=1, dropout=0.5)
                    gnn_model.load_state_dict(state_dict)
                    gnn_model.eval()
            except Exception as e:
                print(f"Error: Failed to load GNN model: {str(e)}")
                gnn_model = None
        else:
            print(f"Warning: GNN model not found at {gnn_checkpoint_path}")
            gnn_model = None
    except ImportError as e:
        print(f"Error: Could not import GCNModel: {str(e)}")
        gnn_model = None
    except Exception as e:
        print(f"Warning: Could not load GNN model: {str(e)}")
        gnn_model = None

    print(f"Running native ablation for feature {feature_idx}, motif={motif}")

    for graph_id in tqdm(graph_ids, desc=f"Ablating feature {feature_idx}"):
        try:
            # Load activations - support both single-motif and mixed-motif
            if use_mixed_motifs:
                activation_file = Path(f"outputs/activations/layer2_new/mixed/graph_{graph_id}.pt")
            else:
                activation_file = Path(f"outputs/activations/layer2_new/test/graph_{graph_id}.pt")

            if not activation_file.exists():
                skipped_count += 1
                continue

            try:
                activations = torch.load(activation_file, weights_only=True)
            except Exception as e:
                print(f"Error: Failed to load activations for graph {graph_id}: {str(e)}")
                error_count += 1
                continue

            # Get original loss
            try:
                loss_original = compute_gnn_loss(activations, gnn_model, graph_id)
                if loss_original is None:
                    error_count += 1
                    continue
            except Exception as e:
                print(f"Error: Failed to compute original loss for graph {graph_id}: {str(e)}")
                error_count += 1
                continue

            # Patch nodes and get patched loss
            try:
                activations_patched, patched_nodes = patch_salient_nodes(
                    activations, sae_model, feature_idx, top_nodes_k=top_nodes_k, patch_type='zero'
                )
                loss_patched = compute_gnn_loss(activations_patched, gnn_model, graph_id)
                if loss_patched is None:
                    error_count += 1
                    continue
            except Exception as e:
                print(f"Error: Patching/loss computation failed for graph {graph_id}: {str(e)}")
                error_count += 1
                continue

            # Load if graph has target motif from metadata
            has_motif = False
            try:
                meta_file = Path(f"virtual_graphs/data/all_graphs/graph_motif_metadata/graph_{graph_id}_metadata.csv")
                if meta_file.exists():
                    try:
                        metadata = pd.read_csv(meta_file, index_col=0)
                        # Check if ANY node in the graph has the target motif
                        # (using first node as representative)
                        if len(metadata) > 0:
                            has_motif = int(metadata.iloc[0].get(motif, 0))
                    except Exception as e:
                        print(f"Warning: Failed to load motif metadata for graph {graph_id}: {str(e)}")
            except Exception as e:
                print(f"Warning: Error processing motif metadata for graph {graph_id}: {str(e)}")

            # Validate loss values
            if not all(np.isfinite(v) for v in [loss_original, loss_patched]):
                print(f"Error: Non-finite loss values for graph {graph_id}")
                error_count += 1
                continue

            results.append({
                'graph_id': graph_id,
                'feature_idx': feature_idx,
                'has_motif': has_motif,
                'loss_original': loss_original,
                'loss_patched': loss_patched,
                'delta_loss': loss_patched - loss_original,
                'n_patched_nodes': len(patched_nodes),
            })

        except Exception as e:
            print(f"Error: Unexpected error processing graph {graph_id}: {str(e)}")
            error_count += 1
            continue

    print(f"\nProcessing Summary:")
    print(f"  Successfully processed: {len(results)} graphs")
    print(f"  Skipped (missing files): {skipped_count} graphs")
    print(f"  Errors: {error_count} graphs")
    print()

    return pd.DataFrame(results)


def run_native_dimension_ablation(motif: str, native_dims: List[int],
                                  graph_ids: List[int]) -> pd.DataFrame:
    """
    Run native dimension ablation across multiple graphs.

    Strategy 2: Ablate native GNN dimensions that correlate with a motif.
    Measures GNN loss impact when these dimensions are zeroed out.

    Args:
        motif: Target motif (e.g., 'in_feedback_loop')
        native_dims: List of native dimension indices to ablate
        graph_ids: List of test graph IDs

    Returns:
        DataFrame with columns: graph_id, has_motif, loss_original, loss_ablated, delta_loss
    """
    results = []

    # Load GNN model once
    try:
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        from gnn_train_copy import GCNModel
        gnn_checkpoint_path = "checkpoints/gnn_model.pt"
        if Path(gnn_checkpoint_path).exists():
            state_dict = torch.load(gnn_checkpoint_path, weights_only=True)
            hidden_dim = state_dict['conv1.bias'].shape[0]
            gnn_model = GCNModel(input_dim=2, hidden_dim=hidden_dim, output_dim=1, dropout=0.5)
            gnn_model.load_state_dict(state_dict)
            gnn_model.eval()
        else:
            print(f"Warning: GNN model not found at {gnn_checkpoint_path}")
            gnn_model = None
    except Exception as e:
        print(f"Warning: Could not load GNN model: {e}")
        gnn_model = None

    for graph_id in tqdm(graph_ids, desc="Strategy 2 Ablation"):
        try:
            # Load activations
            activation_file = Path(f"outputs/activations/layer2_new/test/graph_{graph_id}.pt")
            if not activation_file.exists():
                continue

            activations = torch.load(activation_file, weights_only=True)

            # Get original loss
            loss_original = compute_gnn_loss(activations, gnn_model, graph_id)
            if loss_original is None:
                continue

            # Ablate by zeroing native dimensions
            activations_ablated = activations.clone()
            for dim in native_dims:
                activations_ablated[:, dim] = 0.0

            loss_ablated = compute_gnn_loss(activations_ablated, gnn_model, graph_id)
            if loss_ablated is None:
                continue

            # Load graph motif information
            has_motif = False
            try:
                meta_file = Path(f"virtual_graphs/data/all_graphs/graph_motif_metadata/graph_{graph_id}_metadata.csv")
                if meta_file.exists():
                    metadata = pd.read_csv(meta_file, index_col=0)
                    if len(metadata) > 0:
                        # Map motif column name
                        motif_col_map = {
                            'in_feedforward_loop': 'feedforward_loop',
                            'in_feedback_loop': 'feedback_loop',
                            'in_single_input_module': 'single_input_module',
                            'in_cascade': 'cascade'
                        }
                        col_name = motif_col_map.get(motif, motif)
                        if col_name in metadata.columns:
                            has_motif = int(metadata.iloc[0].get(col_name, 0)) > 0
            except (pd.errors.ParserError, FileNotFoundError, OSError, ValueError, KeyError) as e:
                # Could not load motif metadata; will treat as no motif
                pass

            results.append({
                'graph_id': graph_id,
                'motif': motif,
                'has_motif': has_motif,
                'loss_original': loss_original,
                'loss_ablated': loss_ablated,
                'delta_loss': loss_ablated - loss_original,
                'n_ablated_dims': len(native_dims),
                'ablated_dims': str(native_dims),
            })

        except Exception as e:
            continue

    return pd.DataFrame(results)


def analyze_conditional_effects(ablation_results: pd.DataFrame) -> Dict:
    """
    Analyze ablation effects conditioned on motif presence.

    Returns:
        Dict with Wilcoxon test results and effect sizes
    """
    with_motif = ablation_results[ablation_results['has_motif'] == True]['delta_loss']
    without_motif = ablation_results[ablation_results['has_motif'] == False]['delta_loss']

    result = {
        'with_motif_mean': with_motif.mean(),
        'with_motif_std': with_motif.std(),
        'without_motif_mean': without_motif.mean(),
        'without_motif_std': without_motif.std(),
        'n_with_motif': len(with_motif),
        'n_without_motif': len(without_motif),
    }

    # Statistical test
    if len(with_motif) > 0 and len(without_motif) > 0:
        try:
            stat, p_value = wilcoxon(with_motif, without_motif)
            result['wilcoxon_stat'] = stat
            result['wilcoxon_p'] = p_value
        except ValueError as e:
            # Wilcoxon requires paired samples; use unpaired test instead
            stat, p_value = ranksums(with_motif, without_motif)
            result['ranksums_stat'] = stat
            result['ranksums_p'] = p_value

        # Cohen's d
        cohens_d = (with_motif.mean() - without_motif.mean()) / \
                  np.sqrt((with_motif.std()**2 + without_motif.std()**2) / 2)
        result['cohens_d'] = cohens_d
        result['effect_size'] = 'large' if abs(cohens_d) > 0.8 else \
                               'medium' if abs(cohens_d) > 0.5 else 'small'

    return result


def plot_native_ablation_results(results_by_feature: Dict[int, pd.DataFrame]):
    """Plot native ablation results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Δ Loss distributions
    ax = axes[0]
    feature_ids = []
    delta_losses_with = []
    delta_losses_without = []

    for feature_idx, df in results_by_feature.items():
        if len(df) > 0:
            with_motif = df[df['has_motif'] == True]['delta_loss']
            without_motif = df[df['has_motif'] == False]['delta_loss']

            # Only add if we have data for both categories
            if len(with_motif) > 0 and len(without_motif) > 0:
                feature_ids.append(str(feature_idx))
                delta_losses_with.append(with_motif.mean())
                delta_losses_without.append(without_motif.mean())

    if len(feature_ids) > 0:
        x_pos = np.arange(len(feature_ids))
        width = 0.35

        ax.bar(x_pos - width/2, delta_losses_with, width, label='With motif', alpha=0.7, edgecolor='black')
        ax.bar(x_pos + width/2, delta_losses_without, width, label='Without motif', alpha=0.7, edgecolor='black')

        ax.set_xlabel('Feature Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Δ Loss (Patched - Original)', fontsize=12, fontweight='bold')
        ax.set_title('Native Activation Patching: Selective Degradation by Motif',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_ids)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # Plot 2: Comparison with SAE latent ablation (if available)
    ax = axes[1]
    ax.text(0.5, 0.5, 'Placeholder: SAE latent vs native comparison\n(requires run_ablation.py output)',
           ha='center', va='center', transform=ax.transAxes, fontsize=11)
    ax.set_title('Comparison: SAE Latent vs Native Ablation', fontweight='bold')

    plt.tight_layout()
    plt.savefig(ABLATION_DIR / 'native_ablation_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved native_ablation_results.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Native GNN Activation Space Ablation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy 1 (Original - SAE Feature Guided):
  python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --feature 0
  → Identifies nodes most activated by SAE feature, patches their 64D activations

Strategy 1 Enhanced (SAE r_pb Guided):
  python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_feedback_loop --top-features 5
  → Uses r_pb to rank SAE features by motif correlation, ablates top-5 most correlated features
  → This combines SAE interpretability with native activation patching

Strategy 2 (NEW - Native r_pb Guided):
  python native_gnn_ablation.py --native-rpb --motif in_feedback_loop --top-dims 5
  → Identifies native dimensions most correlated with motif, patches those dimensions
  → This bypasses SAE encoder entirely and directly measures native neuron importance
        """
    )

    # Original arguments for SAE-guided ablation
    parser.add_argument('--variant', type=str, choices=['topk', 'gated', 'jumprelu', 'switch'],
                       default='topk', help='SAE variant to analyze (for Strategy 1)')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension')
    parser.add_argument('--k', type=int, default=8, help='TopK sparsity (for TopK variant)')
    parser.add_argument('--feature', type=int, help='Specific SAE feature to ablate (0-indexed, Strategy 1)')
    parser.add_argument('--all-features', action='store_true', help='Ablate all SAE features (Strategy 1)')
    parser.add_argument('--top-nodes-k', type=int, default=5, help='Number of top nodes to patch (Strategy 1)')
    parser.add_argument('--patch-type', type=str, choices=['zero', 'mean', 'shuffle'],
                       default='zero', help='Type of patch to apply')
    parser.add_argument('--use-rpb', action='store_true',
                       help='Use SAE r_pb to select which features to ablate (Strategy 1 Enhanced)')
    parser.add_argument('--top-features', type=int, default=5,
                       help='Number of top r_pb-ranked features to ablate (Strategy 1 Enhanced)')

    # New arguments for native r_pb-guided ablation
    parser.add_argument('--native-rpb', action='store_true',
                       help='Use native r_pb for dimension selection (Strategy 2: NEW)')
    parser.add_argument('--motif', type=str, choices=['in_feedforward_loop', 'in_feedback_loop',
                                                       'in_single_input_module', 'in_cascade'],
                       help='Target motif for native r_pb analysis (Strategy 2 or Strategy 1 Enhanced)')
    parser.add_argument('--top-dims', type=int, default=5,
                       help='Number of top native dimensions to patch (Strategy 2)')

    # Mixed-motif support
    parser.add_argument('--use-mixed-motifs', action='store_true',
                       help='Run ablations on mixed-motif graphs (4000-4999) instead of single-motif (0-3999). Requires running generate_mixed_motif_activations.py first.')

    args = parser.parse_args()

    # Determine which strategy
    if args.native_rpb:
        print("="*70)
        print("NATIVE GNN ABLATION - STRATEGY 2 (Native r_pb Guided)")
        print("="*70)
        print(f"\nAblating native dimensions with highest r_pb to motif")
        print(f"This bypasses SAE encoder and directly measures native neuron importance")

        # Validate arguments
        if args.motif is None:
            print("\nERROR: --motif is required when using --native-rpb")
            return

        # Load test graph IDs
        try:
            with open(OUTPUT_DIR / 'test_graph_ids.json', 'r') as f:
                graph_ids = json.load(f)['graph_ids']
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load test graph IDs ({type(e).__name__}), using first 100 graphs")
            graph_ids = list(range(100))

        # Run native r_pb-guided ablation
        print(f"\nTarget motif: {args.motif}")
        print(f"Top dimensions to patch: {args.top_dims}")

        # Step 1: Identify top native dimensions by r_pb
        try:
            act_file = Path(f"outputs/activations/layer2_new/test/graph_{graph_ids[0]}.pt")
            activations_example = torch.load(act_file, weights_only=True)

            # Patch by native r_pb to identify which dimensions to ablate
            modified, patched_dims = patch_native_neurons_by_rpb(
                activations_example,
                args.motif,
                graph_ids[:100],  # Use first 100 graphs for r_pb computation
                top_dims_k=args.top_dims,
                patch_type=args.patch_type
            )

            print(f"\n✓ Successfully identified native dimensions by r_pb")
            print(f"  Dimensions to ablate: {patched_dims}")

        except Exception as e:
            print(f"\nERROR during dimension identification: {e}")
            import traceback
            traceback.print_exc()
            return

        # Step 2: Run ablation across all test graphs
        print(f"\nRunning ablations across {len(graph_ids[:200])} graphs...")
        try:
            results = run_native_dimension_ablation(args.motif, patched_dims, graph_ids[:200])

            if len(results) == 0:
                print("ERROR: No ablation results generated")
                return

            # Conditional analysis
            with_motif = results[results['has_motif'] == True]['delta_loss']
            without_motif = results[results['has_motif'] == False]['delta_loss']

            print(f"\n  Ablations with motif: {len(with_motif)} graphs")
            print(f"    Mean Δ Loss: {with_motif.mean():.6f} ± {with_motif.std():.6f}")
            print(f"  Ablations without motif: {len(without_motif)} graphs")
            print(f"    Mean Δ Loss: {without_motif.mean():.6f} ± {without_motif.std():.6f}")

            if len(with_motif) > 1 and len(without_motif) > 1:
                stat, pval = wilcoxon(with_motif, without_motif) if len(with_motif) == len(without_motif) else (np.nan, np.nan)
                if not np.isnan(pval):
                    print(f"    Wilcoxon p-value: {pval:.2e}")

            # Save ablation results
            results_file = ABLATION_DIR / f"native_ablation_strategy2_{args.motif}_top{args.top_dims}.csv"
            results.to_csv(results_file, index=False)
            print(f"\n✓ Saved ablation results to {results_file}")

            # Save dimension rankings
            rankings_file = ABLATION_DIR / f"native_rpb_rankings_{args.motif}.csv"
            rpb_df = compute_native_rpb(graph_ids[:100], args.motif)
            rpb_df.to_csv(rankings_file, index=False)
            print(f"✓ Saved native dimension r_pb rankings to {rankings_file}")

        except Exception as e:
            print(f"\nERROR during ablation: {e}")
            import traceback
            traceback.print_exc()
            return

        print("\n" + "="*70)
        print("✓ STRATEGY 2 (Native r_pb Guided) ABLATION COMPLETE")
        print("="*70)
        return

    # Strategy 1 Enhanced: SAE r_pb-guided ablation
    if args.use_rpb:
        print("="*70)
        print("NATIVE GNN ABLATION - STRATEGY 1 ENHANCED (SAE r_pb Guided)")
        print("="*70)

        if args.motif is None:
            print("\nERROR: --motif is required when using --use-rpb")
            return

        print(f"\nUsing r_pb to rank SAE features by motif correlation")
        print(f"Target motif: {args.motif}")
        print(f"Top features to ablate: {args.top_features}")

        # Construct variant-specific config
        # Try to auto-load from Phase 2 results if available
        config = {'latent_dim': args.latent_dim}

        config_csv = Path('outputs/sae_config_comparison.csv')
        if config_csv.exists():
            try:
                df_config = pd.read_csv(config_csv)
                # Filter by variant and latent_dim
                variant_configs = df_config[
                    (df_config['variant'] == args.variant) &
                    (df_config['latent_dim'] == args.latent_dim)
                ]

                if len(variant_configs) > 0:
                    # Use the top-ranked config for this variant/latent_dim combo
                    best_config = variant_configs.iloc[0]

                    if args.variant == 'topk':
                        config['k'] = int(best_config['k']) if pd.notna(best_config['k']) else args.k
                    elif args.variant == 'gated':
                        config['sparsity_coef'] = float(best_config['sparsity_coef']) if pd.notna(best_config['sparsity_coef']) else 1e-3
                    elif args.variant == 'jumprelu':
                        config['threshold_init'] = float(best_config['threshold_init']) if pd.notna(best_config['threshold_init']) else 0.01
                        config['bandwidth'] = float(best_config['bandwidth']) if pd.notna(best_config['bandwidth']) else 0.01
                    elif args.variant == 'switch':
                        config['num_experts'] = int(best_config['num_experts']) if pd.notna(best_config['num_experts']) else 8
                        config['latent_per_expert'] = int(best_config['latent_per_expert']) if pd.notna(best_config['latent_per_expert']) else 64
                        config['k_per_expert'] = int(best_config['k_per_expert']) if pd.notna(best_config['k_per_expert']) else 8

                    print(f"  Loaded config from Phase 2 results: {config}")
                else:
                    print(f"  No matching config found in CSV, using defaults/args")
                    # Fall back to defaults
                    if args.variant == 'topk':
                        config['k'] = args.k
                    elif args.variant == 'gated':
                        config['sparsity_coef'] = 1e-3
                    elif args.variant == 'jumprelu':
                        config['threshold_init'] = 0.01
                        config['bandwidth'] = 0.01
                    elif args.variant == 'switch':
                        config['num_experts'] = 8
                        config['latent_per_expert'] = 64
                        config['k_per_expert'] = 8
            except Exception as e:
                print(f"  Warning: Could not load config from CSV ({e}), using defaults/args")
                # Fall back to defaults
                if args.variant == 'topk':
                    config['k'] = args.k
                elif args.variant == 'gated':
                    config['sparsity_coef'] = 1e-3
                elif args.variant == 'jumprelu':
                    config['threshold_init'] = 0.01
                    config['bandwidth'] = 0.01
                elif args.variant == 'switch':
                    config['num_experts'] = 8
                    config['latent_per_expert'] = 64
                    config['k_per_expert'] = 8
        else:
            print(f"  Config CSV not found, using defaults/args")
            # Fall back to defaults
            if args.variant == 'topk':
                config['k'] = args.k
            elif args.variant == 'gated':
                config['sparsity_coef'] = 1e-3
            elif args.variant == 'jumprelu':
                config['threshold_init'] = 0.01
                config['bandwidth'] = 0.01
            elif args.variant == 'switch':
                config['num_experts'] = 8
                config['latent_per_expert'] = 64
                config['k_per_expert'] = 8

        # Load test graph IDs
        try:
            with open(OUTPUT_DIR / 'test_graph_ids.json', 'r') as f:
                graph_ids = json.load(f)['graph_ids']
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load test graph IDs ({type(e).__name__}), using first 100 graphs")
            graph_ids = list(range(100))

        # Compute SAE r_pb to select features by motif correlation
        print(f"\nComputing r_pb for all SAE features...")
        rpb_df = compute_sae_rpb(args.variant, config, graph_ids[:200], args.motif)

        if len(rpb_df) == 0:
            print(f"ERROR: Could not compute r_pb for variant {args.variant}")
            return

        # Select top features by |r_pb|
        top_features_df = rpb_df.head(args.top_features)
        top_features = top_features_df['feature_idx'].tolist()
        print(f"\nTop {args.top_features} features by |r_pb| to {args.motif}:")
        for idx, row in enumerate(top_features_df.itertuples()):
            print(f"  {idx+1}. Feature {row.feature_idx}: |r_pb| = {row.rpb_abs:.3f}")

        # Ablate top features
        results_by_feature = {}
        for feature_idx in tqdm(top_features, desc="Features"):
            results = run_native_ablation(args.variant, config, feature_idx, graph_ids[:100],
                                         top_nodes_k=args.top_nodes_k, use_mixed_motifs=args.use_mixed_motifs)
            if len(results) > 0:
                results_by_feature[feature_idx] = results

        # Plot results
        if len(results_by_feature) > 0:
            plot_native_ablation_results(results_by_feature)

            # Save combined results
            all_results = pd.concat(results_by_feature.values(), ignore_index=True)
            results_file = ABLATION_DIR / f"native_ablation_{args.variant}_rpb_{args.motif}.csv"
            all_results.to_csv(results_file, index=False)
            print(f"\n✓ Saved r_pb-guided ablation results to {results_file}")

            # Save feature rankings
            rankings_file = ABLATION_DIR / f"sae_rpb_rankings_{args.motif}.csv"
            rpb_df.to_csv(rankings_file, index=False)
            print(f"✓ Saved SAE feature r_pb rankings to {rankings_file}")

        print("\n" + "="*70)
        print("✓ STRATEGY 1 ENHANCED (SAE r_pb Guided) COMPLETE")
        print("="*70)
        return

    # Original Strategy 1: SAE-feature-guided ablation
    print("="*70)
    print("NATIVE GNN ABLATION - STRATEGY 1 (SAE Feature Guided)")
    print("="*70)
    print(f"\nAblating nodes most activated by SAE features")

    # Construct variant-specific config
    config = {'latent_dim': args.latent_dim}

    if args.variant == 'topk':
        config['k'] = args.k
    elif args.variant == 'gated':
        config['sparsity_coef'] = 1e-3
    elif args.variant == 'jumprelu':
        config['threshold_init'] = 0.01
        config['bandwidth'] = 0.01
    elif args.variant == 'switch':
        config['num_experts'] = 8
        config['latent_per_expert'] = 64
        config['k_per_expert'] = 8

    # Load test graph IDs
    try:
        with open(OUTPUT_DIR / 'test_graph_ids.json', 'r') as f:
            graph_ids = json.load(f)['graph_ids']
    except:
        print("Warning: Could not load test graph IDs, using first 100 graphs")
        graph_ids = list(range(100))

    # Run ablations
    if args.feature is not None:
        # Single feature ablation
        print(f"\nAblating feature {args.feature}...")
        results = run_native_ablation(args.variant, config, args.feature, graph_ids[:100],
                                     top_nodes_k=args.top_nodes_k, use_mixed_motifs=args.use_mixed_motifs)

        if len(results) > 0:
            # Conditional analysis
            conditional = analyze_conditional_effects(results)
            print(f"\n  With motif Δ Loss: {conditional['with_motif_mean']:.6f} ± {conditional['with_motif_std']:.6f}")
            print(f"  Without motif Δ Loss: {conditional['without_motif_mean']:.6f} ± {conditional['without_motif_std']:.6f}")

            if 'cohens_d' in conditional:
                print(f"  Effect size (Cohen's d): {conditional['cohens_d']:.3f} ({conditional['effect_size']})")

            # Save results
            results_file = ABLATION_DIR / f"native_ablation_{args.variant}_feature{args.feature}.csv"
            results.to_csv(results_file, index=False)
            print(f"  ✓ Saved results to {results_file}")

    elif args.all_features:
        # Ablate all features
        print(f"\nAblating all {args.latent_dim} features...")
        results_by_feature = {}

        for feature_idx in tqdm(range(min(args.latent_dim, 20)), desc="Features"):  # Limit to first 20
            results = run_native_ablation(args.variant, config, feature_idx, graph_ids[:100],
                                         top_nodes_k=args.top_nodes_k, use_mixed_motifs=args.use_mixed_motifs)
            if len(results) > 0:
                results_by_feature[feature_idx] = results

        # Plot results
        if len(results_by_feature) > 0:
            plot_native_ablation_results(results_by_feature)

            # Save combined results
            all_results = pd.concat(results_by_feature.values(), ignore_index=True)
            results_file = ABLATION_DIR / f"native_ablation_{args.variant}_all_features.csv"
            all_results.to_csv(results_file, index=False)
            print(f"\n✓ Saved combined results to {results_file}")

    print("\n" + "="*70)
    print("✓ NATIVE ABLATION ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to: {ABLATION_DIR}/")
    print("\nNext steps:")
    print("1. Run SAE latent ablations: python run_ablation.py --variant {variant} --latent_dim {latent_dim}")
    print("2. Compare strategies: python compare_ablation_strategies.py --variant {variant}")


if __name__ == "__main__":
    main()
