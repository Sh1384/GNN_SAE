#!/usr/bin/env python3
"""
SAE-based Interpretation vs GNNExplainer Comparison Analysis (Multi-Variant)

Rigorous comparison of graph explanation methods across all SAE variants:
1. Baseline: GNNExplainer (explains GNN predictions)
2. Novel: SAE Gradient Saliency (explains SAE feature activations)

Evaluation metric: Localization accuracy (AUROC, AUPRC) against ground truth motif edges.

Usage:
    python compare_sae_vs_gnnexplainer.py --variant topk
    python compare_sae_vs_gnnexplainer.py --variant jumprelu
    python compare_sae_vs_gnnexplainer.py --all  # Run all variants
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import SAE models
sys.path.insert(0, str(Path(__file__).parent))
from sparse_autoencoder import TopKSAE, GatedSAE, JumpReLUSAE, SwitchSAE


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class GCNModel(nn.Module):
    """Four-layer GCN with intermediate activation extraction."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 80, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, 64, normalize=False)  # Bottleneck layer
        self.conv4 = GCNConv(64, output_dim, normalize=False)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        if isinstance(x, Data):
            data = x
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_weight = edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv4(x, edge_index, edge_weight)
        return x

    def get_intermediate_activations(self, x, edge_index=None, edge_attr=None):
        """
        Get layer 2 activations for SAE analysis.

        Returns layer 2 (80-dim) activations BEFORE the bottleneck layer.
        This matches the activations used to train the SAE models.
        """
        if isinstance(x, Data):
            data = x
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_weight = edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=False)

        x = F.relu(self.conv2(x, edge_index, edge_weight))
        # Return layer 2 activations (80-dim) BEFORE conv3 bottleneck

        return x  # [num_nodes, 80]


# ============================================================================
# GROUND TRUTH EDGE MASK GENERATION
# ============================================================================

def get_ground_truth_edge_mask(data: Data, motif_type: str) -> torch.Tensor:
    """
    Generate binary ground truth edge mask for a specific motif type.

    Args:
        data: PyG Data object with edge_index
        motif_type: One of 'in_feedback_loop', 'in_feedforward_loop', 'in_single_input_module', 'in_cascade'

    Returns:
        Binary tensor of shape [num_edges] indicating motif edges
    """
    edge_index = data.edge_index.cpu().numpy()
    num_edges = edge_index.shape[1]
    num_nodes = data.x.shape[0]

    # Build NetworkX graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    edge_list = [(edge_index[0, i], edge_index[1, i]) for i in range(num_edges)]
    G.add_edges_from(edge_list)

    motif_edges = set()

    # Map motif_type to detection logic
    if motif_type == 'in_feedback_loop':
        # Bidirectional edges
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if G.has_edge(i, j) and G.has_edge(j, i):
                    motif_edges.add((i, j))
                    motif_edges.add((j, i))

    elif motif_type == 'in_feedforward_loop':
        # Triangles: A→B, A→C, B→C
        for a in G.nodes():
            for b in G.nodes():
                if a == b or not G.has_edge(a, b):
                    continue
                for c in G.nodes():
                    if c in (a, b):
                        continue
                    if G.has_edge(a, c) and G.has_edge(b, c):
                        motif_edges.add((a, b))
                        motif_edges.add((a, c))
                        motif_edges.add((b, c))

    elif motif_type == 'in_single_input_module':
        # Hub node with out-degree ≥ 3
        best_hub = None
        max_targets = 0

        for node in G.nodes():
            successors = list(G.successors(node))
            if len(successors) >= 3:
                is_pure = all(not G.has_edge(t, node) for t in successors)
                if is_pure and len(successors) > max_targets:
                    best_hub = node
                    max_targets = len(successors)

        if best_hub is not None:
            for target in G.successors(best_hub):
                motif_edges.add((best_hub, target))

    elif motif_type == 'in_cascade':
        # Linear chains of length ≥ 4
        for source in G.nodes():
            for target in G.nodes():
                if source == target:
                    continue

                try:
                    paths = list(nx.all_simple_paths(G, source, target, cutoff=10))
                except nx.NetworkXNoPath:
                    continue

                for path in paths:
                    if len(path) >= 4:
                        is_linear = True
                        for i, node in enumerate(path):
                            if i == 0 or i == len(path) - 1:
                                continue
                            in_path_preds = [p for p in G.predecessors(node) if p in path]
                            in_path_succs = [s for s in G.successors(node) if s in path]
                            if len(in_path_preds) != 1 or len(in_path_succs) != 1:
                                is_linear = False
                                break

                        if is_linear:
                            for i in range(len(path) - 1):
                                motif_edges.add((path[i], path[i + 1]))

    else:
        raise ValueError(f"Unknown motif type: {motif_type}")

    # Convert to binary mask
    mask = torch.zeros(num_edges, dtype=torch.bool)
    for edge_idx in range(num_edges):
        src, dst = edge_index[0, edge_idx], edge_index[1, edge_idx]
        if (src, dst) in motif_edges:
            mask[edge_idx] = True

    return mask


# ============================================================================
# EXPLANATION METHODS
# ============================================================================

def explain_with_gnnexplainer(
    gnn: nn.Module,
    data: Data,
    node_idx: int,
    device: str = 'cuda'
) -> np.ndarray:
    """Explain GNN prediction using GNNExplainer."""
    gnn.eval()
    data = data.to(device)

    explainer = Explainer(
        model=gnn,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='node',
            return_type='raw'
        )
    )

    explanation = explainer(data.x, data.edge_index, index=node_idx, edge_attr=data.edge_attr)
    edge_mask = explanation.edge_mask.cpu().detach().numpy()

    # Normalize to [0, 1]
    if edge_mask.max() > edge_mask.min():
        edge_mask = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min())
    else:
        edge_mask = np.ones_like(edge_mask) * 0.5

    return edge_mask


def explain_with_sae_gradient(
    gnn: nn.Module,
    sae: nn.Module,
    data: Data,
    feature_idx: int,
    device: str = 'cuda'
) -> np.ndarray:
    """Explain SAE feature activation using gradient-based saliency on edge weights."""
    gnn.eval()
    sae.eval()

    data = data.to(device)
    data.edge_attr = data.edge_attr.clone().requires_grad_(True)

    with torch.enable_grad():
        h_bottleneck = gnn.get_intermediate_activations(data)  # [num_nodes, 80] - Layer 2 activations

        # Get SAE activations (pre-sparsity)
        if hasattr(sae, 'encoder'):
            z = sae.encoder(h_bottleneck)
            z = F.relu(z)
        else:
            # For switch SAE, use forward pass
            z = sae.encode(h_bottleneck)

        # Target: activation of specific feature
        feature_activations = z[:, feature_idx]
        target = feature_activations.sum()

        # Compute gradients
        if target.item() == 0:
            grads = torch.zeros_like(data.edge_attr)
        else:
            grads = torch.autograd.grad(target, data.edge_attr, create_graph=False)[0]

    edge_importance = grads.abs().cpu().detach().numpy().squeeze()

    # Normalize to [0, 1]
    if edge_importance.max() > edge_importance.min():
        edge_importance = (edge_importance - edge_importance.min()) / (edge_importance.max() - edge_importance.min())
    else:
        edge_importance = np.ones_like(edge_importance) * 0.5

    return edge_importance.flatten()


# ============================================================================
# COMPARISON EXPERIMENT
# ============================================================================

def load_top_features_from_phase2(variant: str, config: Dict) -> Dict[str, int]:
    """
    Load top SAE features for each motif from Phase 2 correlation results.

    Args:
        variant: SAE variant name (topk, gated, jumprelu, switch)
        config: Config dict with hyperparameters (latent_dim, k, etc.)

    Returns:
        Dict mapping motif names to feature indices (0-indexed)
    """
    corr_file = Path('outputs/latent_correlations.csv')

    if not corr_file.exists():
        print(f"WARNING: Correlation file not found: {corr_file}")
        print(f"  Falling back to default features")
        return {
            'in_feedback_loop': 0,
            'in_feedforward_loop': 0,
            'in_single_input_module': 0,
            'in_cascade': 0
        }

    df = pd.read_csv(corr_file)

    # Filter by variant
    df = df[df['variant'] == variant]

    # Filter by config parameters
    if 'latent_dim' in config:
        df = df[df['latent_dim'] == config['latent_dim']]

    if variant == 'topk' and 'k' in config:
        df = df[df['k'] == config['k']]
    elif variant == 'gated' and 'sparsity_coef' in config:
        df = df[df['sparsity_coef'] == config['sparsity_coef']]
    elif variant == 'jumprelu':
        if 'threshold_init' in config:
            df = df[df['threshold_init'] == config['threshold_init']]
        if 'bandwidth' in config:
            df = df[df['bandwidth'] == config['bandwidth']]
    elif variant == 'switch':
        if 'num_experts' in config:
            df = df[df['num_experts'] == config['num_experts']]
        if 'latent_per_expert' in config:
            df = df[df['latent_per_expert'] == config['latent_per_expert']]
        if 'k_per_expert' in config:
            df = df[df['k_per_expert'] == config['k_per_expert']]

    if len(df) == 0:
        print(f"WARNING: No features found for variant={variant}, config={config}")
        print(f"  Falling back to default features")
        return {
            'in_feedback_loop': 0,
            'in_feedforward_loop': 0,
            'in_single_input_module': 0,
            'in_cascade': 0
        }

    top_features = {}
    for motif in ['in_feedback_loop', 'in_feedforward_loop', 'in_single_input_module', 'in_cascade']:
        df_motif = df[df['motif'] == motif]

        if len(df_motif) > 0:
            # Get feature with highest |rpb|
            best_idx = df_motif['rpb_abs'].idxmax()
            feature_name = df_motif.loc[best_idx, 'feature']

            # Extract feature index (e.g., "z126" -> 126)
            if isinstance(feature_name, str) and feature_name.startswith('z'):
                feature_idx = int(feature_name[1:])
            else:
                feature_idx = int(feature_name)

            rpb_value = df_motif.loc[best_idx, 'rpb']
            top_features[motif] = feature_idx
            print(f"  {motif}: Feature z{feature_idx} (rpb={rpb_value:.3f})")
        else:
            print(f"  {motif}: No features found, using z0")
            top_features[motif] = 0

    return top_features


def load_test_graphs_with_features(max_graphs: int = 200, min_motif_ratio: float = 0.2, max_motif_ratio: float = 0.8, use_mixed_motif: bool = True) -> List[Data]:
    """
    Load test graphs with proper node features, filtering for graphs with mixed motif/non-motif edges.

    Args:
        max_graphs: Maximum number of graphs to load
        min_motif_ratio: Minimum ratio of motif edges (e.g., 0.2 = at least 20% motif edges)
        max_motif_ratio: Maximum ratio of motif edges (e.g., 0.8 = at most 80% motif edges)
        use_mixed_motif: If True, use mixed-motif graphs (IDs 4000+) instead of test set

    Returns:
        List of PyG Data objects
    """
    print(f"Loading test graphs (filtering for {min_motif_ratio:.0%}-{max_motif_ratio:.0%} motif edge ratio)...")

    # Choose graph IDs
    if use_mixed_motif:
        # Use mixed-motif graphs (IDs 4000-4999)
        # These naturally have "mixed edges" for each motif type
        print("  → Using mixed-motif graphs (IDs 4000-4999)")
        test_graph_ids = list(range(4000, 4000 + max_graphs))
    else:
        # Use original test set (single-motif graphs)
        print("  → Using test set from test_graph_ids.json")
        with open('outputs/test_graph_ids.json', 'r') as f:
            test_graph_ids = json.load(f)['graph_ids']

    test_graphs = []
    graph_dir = Path("virtual_graphs/data/all_graphs/raw_graphs")

    for graph_id in tqdm(test_graph_ids[:max_graphs], desc="Loading graphs"):
        graph_path = graph_dir / f"graph_{graph_id}.pkl"
        if not graph_path.exists():
            continue

        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        # Convert to PyG Data
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges()]).unsqueeze(1).float()

        num_nodes = len(G.nodes())

        # Create node features (use degree centrality + in-degree)
        in_degrees = [G.in_degree(n) for n in range(num_nodes)]
        out_degrees = [G.out_degree(n) for n in range(num_nodes)]

        x = torch.tensor([[in_degrees[i], out_degrees[i]] for i in range(num_nodes)], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.graph_id = graph_id

        # Check if graph has mixed edges for at least one motif
        has_mixed = False
        for motif in ['in_feedback_loop', 'in_feedforward_loop', 'in_single_input_module', 'in_cascade']:
            try:
                gt_mask = get_ground_truth_edge_mask(data, motif)
                num_motif_edges = gt_mask.sum().item()
                num_total_edges = len(gt_mask)

                if num_total_edges > 0:
                    motif_ratio = num_motif_edges / num_total_edges

                    if min_motif_ratio <= motif_ratio <= max_motif_ratio:
                        has_mixed = True
                        break
            except Exception:
                continue

        if has_mixed:
            test_graphs.append(data)

    print(f"Loaded {len(test_graphs)} test graphs with mixed edges")
    return test_graphs


def run_comparison(
    gnn: nn.Module,
    sae: nn.Module,
    test_graphs: List[Data],
    top_features: Dict[str, int],
    variant: str,
    device: str = 'cuda'
) -> pd.DataFrame:
    """Run comparison experiment."""
    results = []

    for motif_name, feature_idx in top_features.items():
        print(f"\n{'='*70}")
        print(f"Analyzing Motif: {motif_name}")
        print(f"SAE Feature: z{feature_idx} (Variant: {variant})")
        print(f"{'='*70}\n")

        # Filter graphs with this motif (and mixed edges)
        motif_graphs = []
        for data in test_graphs:
            try:
                gt_mask = get_ground_truth_edge_mask(data, motif_name)
                num_motif = gt_mask.sum().item()
                num_total = len(gt_mask)

                # Require mixed edges (both motif and non-motif)
                if num_motif > 0 and num_motif < num_total:
                    motif_graphs.append(data)

                if len(motif_graphs) >= 20:
                    break
            except Exception:
                continue

        print(f"Found {len(motif_graphs)} graphs with mixed edges for {motif_name}")

        if len(motif_graphs) == 0:
            print(f"⚠ No suitable graphs found, skipping...")
            continue

        # Run comparison
        for graph_idx, data in enumerate(tqdm(motif_graphs, desc=f"{motif_name}")):
            try:
                gt_mask = get_ground_truth_edge_mask(data, motif_name).numpy()

                if gt_mask.sum() == 0 or gt_mask.sum() == len(gt_mask):
                    continue  # Skip if all or none are motif edges

                # Select a node involved in motif edges
                motif_edge_indices = np.where(gt_mask)[0]
                edge_index = data.edge_index.cpu().numpy()
                motif_node = edge_index[0, motif_edge_indices[0]]

                # Run GNNExplainer
                gnn_scores = explain_with_gnnexplainer(gnn, data, motif_node, device)

                # Run SAE Gradient Saliency
                sae_scores = explain_with_sae_gradient(gnn, sae, data, feature_idx, device)

                # Compute metrics
                if len(np.unique(gt_mask)) > 1:
                    gnn_auroc = roc_auc_score(gt_mask, gnn_scores)
                    sae_auroc = roc_auc_score(gt_mask, sae_scores)
                else:
                    gnn_auroc = np.nan
                    sae_auroc = np.nan

                gnn_auprc = average_precision_score(gt_mask, gnn_scores)
                sae_auprc = average_precision_score(gt_mask, sae_scores)

                results.append({
                    'variant': variant,
                    'motif': motif_name,
                    'feature_idx': feature_idx,
                    'graph_id': data.graph_id,
                    'graph_idx': graph_idx,
                    'num_edges': len(gt_mask),
                    'num_motif_edges': gt_mask.sum(),
                    'motif_ratio': gt_mask.sum() / len(gt_mask),
                    'gnn_auroc': gnn_auroc,
                    'sae_auroc': sae_auroc,
                    'gnn_auprc': gnn_auprc,
                    'sae_auprc': sae_auprc,
                })

            except Exception as e:
                print(f"  ERROR on graph {graph_idx}: {e}")
                continue

    return pd.DataFrame(results)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_sae_model(variant: str, config: Dict, device: str) -> nn.Module:
    """Load SAE model from checkpoint."""
    input_dim = 80

    # Construct checkpoint path
    if variant == 'topk':
        latent_dim = config['latent_dim']
        k = config['k']
        ckpt_path = f"checkpoints/sae_topk_latent{latent_dim}_k{k}_seed42.pt"
        sae = TopKSAE(input_dim=input_dim, latent_dim=latent_dim, k=k)

    elif variant == 'gated':
        latent_dim = config['latent_dim']
        sparsity_coef = config['sparsity_coef']
        ckpt_path = f"checkpoints/sae_gated_latent{latent_dim}_lambda{sparsity_coef:.0e}_seed42.pt"
        sae = GatedSAE(input_dim=input_dim, latent_dim=latent_dim, sparsity_coef=sparsity_coef)

    elif variant == 'jumprelu':
        latent_dim = config['latent_dim']
        threshold_init = config['threshold_init']
        bandwidth = config['bandwidth']
        ckpt_path = f"checkpoints/sae_jumprelu_latent{latent_dim}_thresh{threshold_init:.0e}_bw{bandwidth:.0e}_seed42.pt"
        sae = JumpReLUSAE(input_dim=input_dim, latent_dim=latent_dim, threshold_init=threshold_init, bandwidth=bandwidth)

    elif variant == 'switch':
        num_experts = config['num_experts']
        latent_per_expert = config['latent_per_expert']
        k_per_expert = config['k_per_expert']
        total_latent = num_experts * latent_per_expert
        ckpt_path = f"checkpoints/sae_switch_experts{num_experts}_latent{total_latent}_k{k_per_expert}_seed42.pt"
        sae = SwitchSAE(input_dim=input_dim, num_experts=num_experts, latent_per_expert=latent_per_expert, k_per_expert=k_per_expert)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location=device)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()

    print(f"   ✓ Loaded SAE from {ckpt_path}")
    return sae


def run_variant_comparison(variant: str, device: str = 'cuda') -> pd.DataFrame:
    """Run comparison for a single SAE variant."""
    print(f"\n{'='*70}")
    print(f"RUNNING COMPARISON FOR VARIANT: {variant.upper()}")
    print(f"{'='*70}\n")

    # Load GNN
    print("1. Loading GNN model...")
    gnn = GCNModel(input_dim=2, hidden_dim=80, output_dim=1, dropout=0.2)
    checkpoint = torch.load('checkpoints/gnn_model.pt', map_location=device, weights_only=True)
    gnn.load_state_dict(checkpoint)
    gnn.to(device)
    gnn.eval()
    print("   ✓ GNN loaded")

    # Load SAE config from Phase 2 results
    print("\n2. Loading SAE configuration...")
    config_csv = Path('outputs/sae_config_comparison.csv')
    if not config_csv.exists():
        raise FileNotFoundError(f"Phase 2 results not found: {config_csv}")

    df_config = pd.read_csv(config_csv)
    variant_config = df_config[df_config['variant'] == variant]

    if len(variant_config) == 0:
        raise ValueError(f"No configuration found for variant {variant}")

    # Use top-ranked config
    best_config = variant_config.iloc[0]

    config = {'latent_dim': int(best_config['latent_dim'])}

    if variant == 'topk':
        config['k'] = int(best_config['k'])
    elif variant == 'gated':
        config['sparsity_coef'] = float(best_config['sparsity_coef'])
    elif variant == 'jumprelu':
        config['threshold_init'] = float(best_config['threshold_init'])
        config['bandwidth'] = float(best_config['bandwidth'])
    elif variant == 'switch':
        config['num_experts'] = int(best_config['num_experts'])
        config['latent_per_expert'] = int(best_config['latent_per_expert'])
        config['k_per_expert'] = int(best_config['k_per_expert'])

    print(f"   Using config: {config}")

    # Load SAE model
    print("\n3. Loading SAE model...")
    sae = load_sae_model(variant, config, device)

    # Load top features from Phase 2
    print("\n4. Loading top features from Phase 2...")
    top_features = load_top_features_from_phase2(variant, config)

    # Load test graphs
    print("\n5. Loading test graphs...")
    test_graphs = load_test_graphs_with_features(max_graphs=200, min_motif_ratio=0.2, max_motif_ratio=0.8, use_mixed_motif=True)

    # Run comparison
    print("\n6. Running comparison...")
    df_results = run_comparison(gnn, sae, test_graphs, top_features, variant, device)

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Compare SAE vs GNNExplainer for motif edge localization")
    parser.add_argument('--variant', type=str, choices=['topk', 'gated', 'jumprelu', 'switch'],
                        help='SAE variant to use')
    parser.add_argument('--all', action='store_true',
                        help='Run comparison for all variants')
    parser.add_argument('--single-motif', action='store_true',
                        help='Use single-motif test graphs instead of mixed-motif graphs (default: mixed-motif)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.all:
        variants = ['jumprelu', 'topk', 'switch', 'gated']
    elif args.variant:
        variants = [args.variant]
    else:
        print("ERROR: Must specify --variant or --all")
        return

    all_results = []

    for variant in variants:
        try:
            df_variant = run_variant_comparison(variant, device)
            all_results.append(df_variant)
        except Exception as e:
            print(f"\nERROR running variant {variant}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_results) == 0:
        print("\nERROR: No results generated!")
        return

    # Combine results
    df_all = pd.concat(all_results, ignore_index=True)

    # Save results
    output_dir = Path('outputs/gnnexplainer_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'comparison_all_variants.csv'
    df_all.to_csv(results_path, index=False)
    print(f"\n✓ Saved results to {results_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    summary = df_all.groupby(['variant', 'motif']).agg({
        'gnn_auroc': ['mean', 'std', 'count'],
        'sae_auroc': ['mean', 'std'],
        'gnn_auprc': ['mean', 'std'],
        'sae_auprc': ['mean', 'std']
    }).round(3)

    print("\n", summary.to_string())

    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE (Paired t-test)")
    print("="*70)

    from scipy.stats import ttest_rel

    for variant in df_all['variant'].unique():
        print(f"\n{variant.upper()}:")
        df_var = df_all[df_all['variant'] == variant]

        for motif in df_var['motif'].unique():
            df_motif = df_var[df_var['motif'] == motif]

            valid = ~(df_motif['gnn_auroc'].isna() | df_motif['sae_auroc'].isna())
            df_valid = df_motif[valid]

            if len(df_valid) > 1:
                t_auroc, p_auroc = ttest_rel(df_valid['sae_auroc'], df_valid['gnn_auroc'])
                t_auprc, p_auprc = ttest_rel(df_valid['sae_auprc'], df_valid['gnn_auprc'])

                print(f"  {motif}:")
                print(f"    AUROC: SAE vs GNN, t={t_auroc:.3f}, p={p_auroc:.4f}")
                print(f"    AUPRC: SAE vs GNN, t={t_auprc:.3f}, p={p_auprc:.4f}")

                if p_auroc < 0.05:
                    winner = "SAE" if t_auroc > 0 else "GNN"
                    print(f"    → {winner} is significantly better (AUROC, p<0.05)")

    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
