#!/usr/bin/env python3
"""
SAE Feature Ablation Analysis (3-Way Comparison)

Compares the distribution of GNN errors (MSE) across test graphs in three scenarios:
1. Original: GNN inference using original layer2 activations (no SAE).
2. Full SAE: GNN inference using full SAE reconstruction.
3. Ablated: GNN inference using SAE reconstruction with specific features zeroed out.

The "Error" is measured as MSE between GNN predictions and GROUND TRUTH expression values
on the masked nodes.

Feature:
- Lines are colored by the dominant motif in the graph.
- Line thickness is proportional to the deviation in MSE (thicker = bigger change).
- Legend included at the top.

Usage:
    python run_ablation.py --latent_dim 512 --k 16 --feature z496
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns
from tqdm import tqdm
import torch
import pickle
import networkx as nx
from scipy import stats

# Import your models
from sparse_autoencoder import BaseSAE, TopKSAE, GatedSAE, JumpReLUSAE, SwitchSAE
# Ensure GCNModel is importable
try:
    from gnn_train import GCNModel
except ImportError as e:
    print(f"Warning: Could not import GCNModel from gnn_train: {str(e)}")
    GCNModel = None

# Setup Directories
ABLATION_DIR = Path("ablations")
ABLATION_DIR.mkdir(exist_ok=True)
(ABLATION_DIR / "results").mkdir(exist_ok=True)
(ABLATION_DIR / "plots").mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Model Loading & Helpers
# -----------------------------------------------------------------------------

def detect_variant_from_path(checkpoint_path: str) -> str:
    """Auto-detect SAE variant from checkpoint filename."""
    path_str = checkpoint_path.lower()
    if 'topk' in path_str:
        return 'topk'
    elif 'gated' in path_str:
        return 'gated'
    elif 'jumprelu' in path_str:
        return 'jumprelu'
    elif 'switch' in path_str:
        return 'switch'
    else:
        # Default to TopK for backward compatibility with old naming
        return 'topk'

def load_sae_model(variant=None, checkpoint_path=None, latent_dim=None, **kwargs):
    """
    Load trained SAE model with auto-detection of variant.

    Args:
        variant: One of 'topk', 'gated', 'jumprelu', 'switch' (auto-detected if None)
        checkpoint_path: Explicit path to checkpoint (if None, constructed from params)
        latent_dim: Latent dimension (required for TopK and Gated)
        **kwargs: Additional variant-specific parameters (k, sparsity_coef, threshold_init, etc.)

    Returns:
        Loaded SAE model (subclass of BaseSAE)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        ValueError: If variant is unknown or checkpoint is corrupted
    """
    # Auto-construct checkpoint path if not provided
    if checkpoint_path is None:
        if variant == 'topk':
            checkpoint_path = f"checkpoints/sae_topk_latent{latent_dim}_k{kwargs.get('k')}_seed42.pt"
        elif variant == 'gated':
            sparsity_coef = kwargs.get('sparsity_coef', 1e-3)
            checkpoint_path = f"checkpoints/sae_gated_latent{latent_dim}_lambda{sparsity_coef:.0e}_seed42.pt"
        elif variant == 'jumprelu':
            threshold_init = kwargs.get('threshold_init', 0.01)
            checkpoint_path = f"checkpoints/sae_jumprelu_latent{latent_dim}_thresh{threshold_init:.0e}_bw1e-02_seed42.pt"
        elif variant == 'switch':
            num_experts = kwargs.get('num_experts', 8)
            latent_per_expert = kwargs.get('latent_per_expert', 64)
            k_per_expert = kwargs.get('k_per_expert', 8)
            checkpoint_path = f"checkpoints/sae_switch_experts{num_experts}_latent{num_experts*latent_per_expert}_k{k_per_expert}_seed42.pt"
        else:
            # Fallback to TopK default
            checkpoint_path = f"checkpoints/sae_latent{latent_dim}_k{kwargs.get('k')}.pt"

    # Auto-detect variant if not provided
    if variant is None:
        variant = detect_variant_from_path(checkpoint_path)

    # Check file exists
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")

    # Create appropriate model instance
    try:
        if variant == 'topk':
            model = TopKSAE(input_dim=64, latent_dim=latent_dim, k=kwargs.get('k'))
        elif variant == 'gated':
            model = GatedSAE(input_dim=64, latent_dim=latent_dim, sparsity_coef=kwargs.get('sparsity_coef', 1e-3))
        elif variant == 'jumprelu':
            model = JumpReLUSAE(input_dim=64, latent_dim=latent_dim, threshold_init=kwargs.get('threshold_init', 0.01))
        elif variant == 'switch':
            model = SwitchSAE(input_dim=64, num_experts=kwargs.get('num_experts', 8),
                             latent_per_expert=kwargs.get('latent_per_expert', 64),
                             k_per_expert=kwargs.get('k_per_expert', 8))
        else:
            raise ValueError(f"Unknown SAE variant: {variant}")
    except Exception as e:
        raise ValueError(f"Failed to instantiate SAE model (variant={variant}): {str(e)}")

    # Load checkpoint with error handling
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except Exception as e:
        raise ValueError(f"Failed to load SAE checkpoint file {checkpoint_path}: {str(e)}")

    # Validate checkpoint structure
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint missing 'model_state_dict' key at {checkpoint_path}")

    # Load state dict with error handling
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        raise ValueError(f"Failed to load SAE state dict (shape mismatch?): {str(e)}")

    model.eval()
    return model

def load_gnn_model():
    """Load trained GNN model.

    Returns:
        GNN model on evaluation mode, or None if loading failed.
    """
    try:
        from gnn_train import GCNModel
    except ImportError as e:
        print(f"Error: Could not import GCNModel from gnn_train: {str(e)}")
        return None

    gnn_checkpoint_path = "checkpoints/gnn_model.pt"
    if not Path(gnn_checkpoint_path).exists():
        print(f"Warning: GNN checkpoint not found at {gnn_checkpoint_path}")
        return None

    try:
        # Load state dict to infer architecture
        state_dict = torch.load(gnn_checkpoint_path, weights_only=True)
    except Exception as e:
        print(f"Error: Failed to load GNN checkpoint file {gnn_checkpoint_path}: {str(e)}")
        return None

    # Validate required keys
    required_keys = ['conv1.bias', 'conv2.bias']
    missing_keys = [k for k in required_keys if k not in state_dict]
    if missing_keys:
        print(f"Error: GNN checkpoint missing keys: {missing_keys}")
        return None

    try:
        hidden_dim1 = state_dict['conv1.bias'].shape[0]
        hidden_dim2 = state_dict['conv2.bias'].shape[0]

        gnn = GCNModel(input_dim=2, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=1, dropout=0.5)
        gnn.load_state_dict(state_dict)
        gnn.eval()
        return gnn
    except RuntimeError as e:
        print(f"Error: Failed to load GNN state dict (architecture mismatch?): {str(e)}")
        return None
    except Exception as e:
        print(f"Error: Unexpected error loading GNN model: {str(e)}")
        return None

def get_feature_indices(feature_spec, latent_dim):
    """Parse feature specification string and validate indices.

    Args:
        feature_spec: Feature specification (e.g., 'z496' or 'z496,z200')
        latent_dim: Maximum valid feature index

    Returns:
        List of valid feature indices (0-based)

    Raises:
        ValueError: If feature_spec is empty or contains invalid indices
    """
    features = []
    if not feature_spec or not feature_spec.strip():
        raise ValueError("Feature specification cannot be empty")

    for feat in feature_spec.split(','):
        feat = feat.strip()
        if not feat:
            continue
        if not feat.startswith('z'):
            raise ValueError(f"Invalid feature format: {feat}. Expected format: z123")
        try:
            idx = int(feat[1:]) - 1  # Convert z-indexed to 0-based
            if idx < 0 or idx >= latent_dim:
                raise ValueError(f"Feature index {idx + 1} out of range [1, {latent_dim}]")
            features.append(idx)
        except ValueError as e:
            # Re-raise with context
            raise ValueError(f"Failed to parse feature {feat}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error parsing feature {feat}: {str(e)}")

    if not features:
        raise ValueError(f"No valid features parsed from specification: {feature_spec}")

    return features

def load_graph_motif_metadata(graph_id):
    """Load motif metadata for a specific graph.

    Args:
        graph_id: ID of the graph

    Returns:
        Dict mapping motif names to node counts, or {} if file not found
    """
    metadata_path = Path(f"virtual_graphs/data/all_graphs/graph_motif_metadata/graph_{graph_id}_metadata.csv")
    if not metadata_path.exists():
        return {}

    try:
        df = pd.read_csv(metadata_path, index_col=0)
        # Count nodes in each motif
        motif_counts = df.sum(axis=0).to_dict()
        return motif_counts
    except Exception as e:
        print(f"Warning: Failed to load motif metadata for graph {graph_id}: {str(e)}")
        return {}

def get_dominant_motif(graph_id):
    """Determine the majority motif for a graph."""
    counts = load_graph_motif_metadata(graph_id)
    
    # Map raw column names to display names
    name_map = {
        'feedforward_loop': 'Feedforward Loop',
        'feedback_loop': 'Feedback Loop',
        'single_input_module': 'Single Input Module',
        'cascade': 'Cascade'
    }
    
    # Filter for known motifs and non-zero counts
    valid_counts = {name_map.get(k, k): v for k, v in counts.items() if v > 0 and k in name_map}
    
    if not valid_counts:
        return "Other"
    
    # Return motif with max count
    return max(valid_counts, key=valid_counts.get)

def simulate_expression(W, graph_id, steps=50, gamma=0.3, noise_std=0.01):
    local_seed = 42 + graph_id
    rng = np.random.default_rng(local_seed)
    n_nodes = W.shape[0]
    x = rng.uniform(0, 1, size=n_nodes)
    for _ in range(steps):
        weighted_input = W @ x
        sigmoid_input = 1.0 / (1.0 + np.exp(-np.clip(weighted_input, -10, 10)))
        noise = rng.normal(0, noise_std, size=n_nodes)
        x = (1 - gamma) * x + gamma * sigmoid_input + noise
        x = np.clip(x, 0, 1)
    return x

def load_graph_data(graph_id):
    """Load graph structure and compute edge information.

    Args:
        graph_id: ID of the graph

    Returns:
        Tuple of (edge_index, edge_weight, y_true, mask) or None if loading fails

    Note:
        Returns None silently on missing file (not uncommon in partial datasets),
        but raises on corrupted files to alert user.
    """
    path = Path(f"virtual_graphs/data/all_graphs/raw_graphs/graph_{graph_id}.pkl")
    if not path.exists():
        return None

    try:
        with open(path, 'rb') as f:
            G = pickle.load(f)
    except Exception as e:
        print(f"Error: Failed to load graph {graph_id} from {path}: {str(e)}")
        return None

    try:
        W = nx.to_numpy_array(G, weight='weight')
        edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
        edge_weight = torch.tensor(W[W != 0], dtype=torch.float32)
        y_true = torch.tensor(simulate_expression(W, graph_id), dtype=torch.float32)
        rng = np.random.default_rng(42 + graph_id)
        mask = torch.tensor(rng.random(len(G.nodes())) < 0.3, dtype=torch.bool)
        return edge_index, edge_weight, y_true, mask
    except Exception as e:
        print(f"Error: Failed to process graph {graph_id} (corrupt data?): {str(e)}")
        return None

def evaluate_gnn_output(gnn_model, layer2_activations, edge_index, edge_weight):
    """Evaluate GNN output for given layer2 activations.

    Args:
        gnn_model: GNN model (or None)
        layer2_activations: Tensor of layer2 activations
        edge_index: Edge indices
        edge_weight: Edge weights

    Returns:
        Predictions tensor or None if gnn_model is None or inference fails
    """
    if gnn_model is None:
        return None

    try:
        with torch.no_grad():
            h3 = gnn_model.conv3(layer2_activations, edge_index, edge_weight=edge_weight)
            pred = h3.squeeze(-1)

        # Validate output
        if pred.shape[0] != layer2_activations.shape[0]:
            print(f"Error: GNN output shape mismatch: expected {layer2_activations.shape[0]} predictions, got {pred.shape[0]}")
            return None
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("Error: GNN output contains NaN or Inf values")
            return None

        return pred
    except Exception as e:
        print(f"Error: GNN inference failed: {str(e)}")
        return None

# -----------------------------------------------------------------------------
# Main Analysis Logic
# -----------------------------------------------------------------------------

def run_ablation_experiment(latent_dim, k, ablate_indices, experiment_name, motif_type_filter=None, use_mixed_motifs=False):
    """Run ablation experiment comparing original, full SAE, and ablated SAE reconstructions.

    Args:
        latent_dim: SAE latent dimension
        k: TopK parameter
        ablate_indices: List of feature indices to ablate
        experiment_name: Name for this experiment (used in output filenames)
        motif_type_filter: Optional filter for specific motif types
        use_mixed_motifs: Use mixed-motif test set instead of single-motif

    Returns:
        DataFrame with ablation results (columns: graph_id, Motif, Loss (Original), Loss (Full SAE), Loss (Ablated), ...)
    """
    print(f"Running Ablation: {experiment_name}")
    print(f"Ablating {len(ablate_indices)} features: {ablate_indices}")

    # Load SAE model
    try:
        sae_model = load_sae_model(variant='topk', latent_dim=latent_dim, k=k)
    except Exception as e:
        print(f"Error: Failed to load SAE model: {str(e)}")
        return pd.DataFrame()

    # Load GNN model (optional, continue if not available)
    gnn_model = load_gnn_model()

    # Load graph IDs based on mode
    try:
        if use_mixed_motifs:
            # Use ALL mixed-motif graphs (4000-4999)
            # These should be in outputs/activations/layer2/mixed/
            mixed_dir = Path('outputs/activations/layer2/mixed')
            if not mixed_dir.exists():
                print(f"Error: Mixed-motif activations not found at {mixed_dir}")
                print("Please run: python generate_mixed_motif_activations.py")
                return pd.DataFrame()

            graph_ids = []
            for act_file in mixed_dir.glob('graph_*.pt'):
                try:
                    graph_id = int(act_file.stem.replace('graph_', ''))
                    graph_ids.append(graph_id)
                except ValueError:
                    print(f"Warning: Could not parse graph ID from filename: {act_file.stem}")

            if len(graph_ids) == 0:
                print(f"Error: No activation files found in {mixed_dir}")
                print("Please run: python generate_mixed_motif_activations.py")
                return pd.DataFrame()

            graph_ids = sorted(graph_ids)
            print(f"Found {len(graph_ids)} mixed-motif graphs (range: {min(graph_ids)}-{max(graph_ids)})")
        else:
            # Use single-motif test graphs
            test_graph_ids_file = Path('outputs/test_graph_ids.json')
            if not test_graph_ids_file.exists():
                print(f"Error: Test graph IDs file not found at {test_graph_ids_file}")
                return pd.DataFrame()

            try:
                with open(test_graph_ids_file, 'r') as f:
                    graph_ids = json.load(f)['graph_ids']
            except Exception as e:
                print(f"Error: Failed to load test graph IDs: {str(e)}")
                return pd.DataFrame()

            print(f"Loaded {len(graph_ids)} test graphs")
    except Exception as e:
        print(f"Error: Failed to load graph ID list: {str(e)}")
        return pd.DataFrame()

    results = []
    skipped_count = 0
    error_count = 0
    print(f"Processing {len(graph_ids)} graphs...")

    for graph_id in tqdm(graph_ids):
        try:
            # 1. Determine Motif
            motif_label = get_dominant_motif(graph_id)

            # Apply optional filter
            if motif_type_filter and motif_type_filter.lower() != 'all':
                # Simple check if filter string is part of label
                if motif_type_filter.lower() not in motif_label.lower().replace(" ", "_"):
                    continue

            # 2. Load Activations
            if use_mixed_motifs:
                # Mixed motifs are in layer2/mixed
                act_file = Path(f"outputs/activations/layer2/mixed/graph_{graph_id}.pt")
            else:
                # Single motifs use layer2/test
                act_file = Path(f"outputs/activations/layer2/test/graph_{graph_id}.pt")

            if not act_file.exists():
                skipped_count += 1
                continue

            try:
                original_acts = torch.load(act_file, weights_only=True)
            except Exception as e:
                print(f"Error: Failed to load activations for graph {graph_id}: {str(e)}")
                error_count += 1
                continue

            # Validate activations shape
            if original_acts.shape[1] != 64:
                print(f"Error: Graph {graph_id} has wrong activation dim {original_acts.shape[1]}, expected 64")
                error_count += 1
                continue

            # 3. SAE Reconstructions
            try:
                with torch.no_grad():
                    latents_full = sae_model.encode(original_acts)
                    reconstructed_full = sae_model.decoder(latents_full)

                    latents_ablated = latents_full.clone()
                    latents_ablated[:, ablate_indices] = 0.0
                    reconstructed_ablated = sae_model.decoder(latents_ablated)
            except Exception as e:
                print(f"Error: SAE processing failed for graph {graph_id}: {str(e)}")
                error_count += 1
                continue

            # 4. GNN Inference
            graph_data = load_graph_data(graph_id)
            if graph_data is None:
                skipped_count += 1
                continue

            if gnn_model:
                try:
                    edge_index, edge_weight, y_true, mask = graph_data

                    out_original = evaluate_gnn_output(gnn_model, original_acts, edge_index, edge_weight)
                    out_full_sae = evaluate_gnn_output(gnn_model, reconstructed_full, edge_index, edge_weight)
                    out_ablated = evaluate_gnn_output(gnn_model, reconstructed_ablated, edge_index, edge_weight)

                    if out_original is None or out_full_sae is None or out_ablated is None:
                        error_count += 1
                        continue

                    # Compute losses
                    try:
                        loss_original = torch.mean(((out_original - y_true)[mask]) ** 2).item()
                        loss_full_sae = torch.mean(((out_full_sae - y_true)[mask]) ** 2).item()
                        loss_ablated = torch.mean(((out_ablated - y_true)[mask]) ** 2).item()

                        # Validate loss values
                        if any(not np.isfinite(v) for v in [loss_original, loss_full_sae, loss_ablated]):
                            print(f"Error: Non-finite loss values for graph {graph_id}: {loss_original}, {loss_full_sae}, {loss_ablated}")
                            error_count += 1
                            continue

                        results.append({
                            'graph_id': graph_id,
                            'Motif': motif_label,
                            'Loss (Original)': loss_original,
                            'Loss (Full SAE)': loss_full_sae,
                            'Loss (Ablated)': loss_ablated,
                            'SAE Degradation': loss_full_sae - loss_original,
                            'Ablation Impact': loss_ablated - loss_full_sae
                        })
                    except Exception as e:
                        print(f"Error: Loss computation failed for graph {graph_id}: {str(e)}")
                        error_count += 1
                except Exception as e:
                    print(f"Error: GNN inference failed for graph {graph_id}: {str(e)}")
                    error_count += 1
        except Exception as e:
            print(f"Error: Unexpected error processing graph {graph_id}: {str(e)}")
            error_count += 1
            continue

    # Print summary stats
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"  Successfully processed: {len(results)} graphs")
    print(f"  Skipped (missing files): {skipped_count} graphs")
    print(f"  Errors: {error_count} graphs")
    print(f"  Total: {len(results) + skipped_count + error_count} / {len(graph_ids)}")
    print(f"{'='*60}\n")

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_boxplots(df, experiment_name):
    """
    Plot 3-way paired plot colored by motif with top legend.
    FIXED: Uses high zorder, thick black borders, and robust data handling to ensure visibility.
    """
    if df.empty:
        print("Dataframe is empty. Skipping plot.")
        return

    # Define Color Palette (Colorblind friendly)
    motif_palette = {
        'Feedforward Loop': '#377eb8',      # Blue
        'Feedback Loop': '#ff7f00',         # Orange
        'Single Input Module': '#4daf4a',   # Green
        'Cascade': '#e41a1c',               # Red
        'Other': '#999999'                  # Grey
    }

    loss_cols = ['Loss (Original)', 'Loss (Full SAE)', 'Loss (Ablated)']
    
    # 1. Validation & Cleaning
    for col in loss_cols:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in dataframe.")
            return

    # Filter outliers
    df_clean = df.copy()
    threshold = 3.0
    for col in loss_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            # Filter out points outside 3 IQR range
            df_clean = df_clean[~((df_clean[col] < (Q1 - threshold * IQR)) | (df_clean[col] > (Q3 + threshold * IQR)))]

    # Drop rows with NaNs in the loss columns to ensure lines plot correctly
    df_clean = df_clean.dropna(subset=loss_cols)
    print(f"Plotting {len(df_clean)} graphs after cleaning...")

    fig, ax = plt.subplots(figsize=(12, 8))

    positions = [0, 1, 2]
    x_labels = ['Original', 'Full SAE', 'Ablated']

    # 2. Draw Connected Lines (Bottom Layer)
    # zorder=2 ensures they are behind the points
    segments = []
    colors = []
    linewidths = []
    
    for _, row in df_clean.iterrows():
        motif = row.get('Motif', 'Other')
        c = motif_palette.get(motif, '#999999')
        
        y_orig = row['Loss (Original)']
        y_full = row['Loss (Full SAE)']
        y_abl = row['Loss (Ablated)']
        
        # Calculate differences for line thickness
        diff1 = abs(y_full - y_orig)
        diff2 = abs(y_abl - y_full)
        
        segments.append([(0, y_orig), (1, y_full)])
        colors.append(c)
        linewidths.append(diff1)
        
        segments.append([(1, y_full), (2, y_abl)])
        colors.append(c)
        linewidths.append(diff2)

    # Normalize linewidths: Ensure minimum thickness is visible (1.0)
    lw_array = np.array(linewidths)
    if len(lw_array) > 0:
        max_diff = np.percentile(lw_array, 95)
        if max_diff < 1e-9: max_diff = 1.0
        # Scale: Min 1.0, Max 3.5
        normalized_lws = 1.0 + 2.5 * np.clip(lw_array / max_diff, 0, 1)
    else:
        normalized_lws = np.ones(len(segments))

    # alpha=0.4 gives good visibility for the paired lines
    lc = LineCollection(segments, colors=colors, linewidths=normalized_lws, alpha=0.4, zorder=2)
    ax.add_collection(lc)

    # 3. Draw Scatter Points (Middle Layer)
    # zorder=3 sits on top of lines
    for i, col in enumerate(loss_cols):
        point_colors = [motif_palette.get(m, '#999999') for m in df_clean['Motif']]
        ax.scatter([positions[i]] * len(df_clean), df_clean[col], 
                   c=point_colors, alpha=0.5, s=25, zorder=3, edgecolors='white', linewidth=0.3)

    # 4. Draw Boxplots (Top Layer)
    # zorder=10 forces this to the very front, guaranteeing visibility.
    # facecolor=(0,0,0,0) or 'none' ensures the box is transparent.
    # We use .dropna() on the series passed to boxplot for robustness.
    plot_data = [df_clean[col].dropna() for col in loss_cols]
    
    bp = ax.boxplot(plot_data,
                     positions=positions,
                     widths=0.4,
                     patch_artist=True,
                     # Increased linewidth to 2.5 for clarity, guaranteed transparent fill
                     boxprops=dict(facecolor=(0,0,0,0), edgecolor='black', linewidth=2.5), 
                     whiskerprops=dict(color='black', linewidth=2),
                     capprops=dict(color='black', linewidth=2),
                     # Bold red median line
                     medianprops=dict(color='red', linewidth=3), 
                     showfliers=False,
                     zorder=10) # <-- CRITICAL: High zorder for visibility

    # 5. Legend & Aesthetics
    legend_handles = []
    unique_motifs = df_clean['Motif'].unique() if 'Motif' in df_clean.columns else ['Other']
    for motif, color in motif_palette.items():
        if motif in unique_motifs:
            patch = mpatches.Patch(color=color, label=motif)
            legend_handles.append(patch)

    ax.legend(handles=legend_handles, 
              loc='lower center', 
              bbox_to_anchor=(0.5, 1.02), 
              ncol=min(len(legend_handles), 5), 
              frameon=False,
              fontsize=11)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('GNN MSE Loss', fontsize=12)
    ax.set_title(f'Feature Ablation Impact by Graph Motif\n({experiment_name})', fontsize=14, pad=40)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    
    save_path = ABLATION_DIR / "plots" / f"{experiment_name}_colored_boxplot.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {save_path}")

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    """Main entry point for ablation analysis.

    Command-line interface:
        python run_ablation.py --latent_dim 512 --k 16 --feature z496
    """
    parser = argparse.ArgumentParser(description="SAE Feature Ablation Analysis")
    parser.add_argument('--latent_dim', type=int, required=True, help='SAE latent dimension')
    parser.add_argument('--k', type=int, required=True, help='TopK parameter')
    parser.add_argument('--feature', type=str, required=True, help='Feature(s) to ablate (e.g., z496 or z496,z200)')
    parser.add_argument('--motif_type', type=str, default='all', help='Optional filter for specific motif types')
    parser.add_argument('--experiment_name', type=str, default=None, help='Optional experiment name override')
    parser.add_argument('--use_mixed_motifs', action='store_true',
                       help='Run ablations on mixed-motif graphs (4000+) using dominant motif labels')
    args = parser.parse_args()

    try:
        # Parse feature indices with validation
        print(f"Feature specification: {args.feature}")
        ablate_indices = get_feature_indices(args.feature, args.latent_dim)
        print(f"Parsed ablation indices: {ablate_indices}")
    except ValueError as e:
        print(f"Error: {str(e)}")
        return 1

    # Name experiment
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        feat_str = "multi" if "," in args.feature else args.feature
        experiment_name = f"ablate_{feat_str}"

    try:
        # Run Analysis
        df = run_ablation_experiment(
            args.latent_dim, args.k, ablate_indices, experiment_name,
            args.motif_type, args.use_mixed_motifs
        )
    except Exception as e:
        print(f"Error: Ablation experiment failed: {str(e)}")
        return 1

    # Save results to CSV
    try:
        results_file = ABLATION_DIR / "results" / f"{experiment_name}_results.csv"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_file, index=False)
        print(f"\nSaved results to: {results_file}")
    except Exception as e:
        print(f"Error: Failed to save results CSV: {str(e)}")
        return 1

    # Calculate Stats
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    if df.empty:
        print("Warning: No results generated. DataFrame is empty.")
        print("This may occur if:")
        print("  - No activation files were found")
        print("  - All graphs were filtered out")
        print("  - All graphs produced errors during processing")
        return 1

    if 'Ablation Impact' not in df.columns:
        print("Error: DataFrame missing 'Ablation Impact' column.")
        print(f"Available columns: {df.columns.tolist()}")
        return 1

    # Overall Impact Stats
    try:
        mean_imp = df['Ablation Impact'].mean()
        std_imp = df['Ablation Impact'].std()
        print(f"\nMean Ablation Impact: {mean_imp:.4e} ± {std_imp:.4e}")

        # Statistical test
        try:
            p_val = stats.wilcoxon(df['Loss (Full SAE)'], df['Loss (Ablated)'])[1]
            print(f"Wilcoxon signed-rank test p-value: {p_val:.4e}")
        except Exception as e:
            print(f"Warning: Could not compute Wilcoxon test: {str(e)}")

        # Breakdown by Motif
        print("\nImpact by Motif Type:")
        motif_stats = df.groupby('Motif')['Ablation Impact'].agg(['count', 'mean', 'std'])
        print(motif_stats)
    except Exception as e:
        print(f"Error: Failed to compute statistics: {str(e)}")
        return 1

    # Plot results
    try:
        plot_boxplots(df, experiment_name)
    except Exception as e:
        print(f"Warning: Failed to generate plot: {str(e)}")
        # Don't fail if plotting fails, as results are already saved

    print("\n✓ Ablation analysis completed successfully!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())