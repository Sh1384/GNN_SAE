#!/usr/bin/env python3
"""
SAE-based Interpretation vs GNNExplainer Comparison Analysis

This script rigorously compares two graph explanation methods:
1. Baseline: GNNExplainer (explains GNN predictions)
2. Novel: SAE Gradient Saliency (explains SAE feature activations)

Evaluation metric: Localization accuracy (AUROC, AUPRC) against ground truth motif edges.
"""

import json
import pickle
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

# Import SAE model
import sys
sys.path.append('.')
from sparse_autoencoder import SparseAutoencoder


# ============================================================================
# MODEL DEFINITIONS (with activation extraction capability)
# ============================================================================

class GCNModel(nn.Module):
    """Four-layer GCN with intermediate activation extraction."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 88, output_dim: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, normalize=False)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=False)
        self.conv3 = GCNConv(hidden_dim, 64, normalize=False)  # Bottleneck layer
        self.conv4 = GCNConv(64, output_dim, normalize=False)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass compatible with both Data objects and GNNExplainer.

        Args:
            x: Node features [num_nodes, input_dim] OR Data object
            edge_index: Edge indices [2, num_edges] (if x is features)
            edge_attr: Edge weights [num_edges, 1] (if x is features)
        """
        # Handle both calling conventions
        if isinstance(x, Data):
            data = x
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Use edge_attr as edge_weight
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
        Get layer 3 (bottleneck) activations for SAE analysis.

        Args:
            x: Node features [num_nodes, input_dim] OR Data object
            edge_index: Edge indices [2, num_edges] (if x is features)
            edge_attr: Edge weights [num_edges, 1] (if x is features)

        Returns:
            Layer 3 activations [num_nodes, 64]
        """
        # Handle both calling conventions
        if isinstance(x, Data):
            data = x
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        edge_weight = edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=False)  # No dropout during analysis

        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.dropout, training=False)

        x = F.relu(self.conv3(x, edge_index, edge_weight))

        return x  # [num_nodes, 64]


# ============================================================================
# GROUND TRUTH EDGE MASK GENERATION
# ============================================================================

def get_ground_truth_edge_mask(data: Data, motif_type: str) -> torch.Tensor:
    """
    Generate binary ground truth edge mask for a specific motif type.

    Args:
        data: PyG Data object with edge_index
        motif_type: One of 'feedback_loop', 'feedforward_loop', 'single_input_module', 'cascade'

    Returns:
        Binary tensor of shape [num_edges] indicating motif edges
    """
    edge_index = data.edge_index.cpu().numpy()
    num_edges = edge_index.shape[1]
    num_nodes = data.x.shape[0]

    # Build NetworkX graph for motif detection
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    edge_list = [(edge_index[0, i], edge_index[1, i]) for i in range(num_edges)]
    G.add_edges_from(edge_list)

    # Initialize mask
    motif_edges = set()

    if motif_type == 'feedback_loop':
        # Find bidirectional edges (X↔Y)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if G.has_edge(i, j) and G.has_edge(j, i):
                    motif_edges.add((i, j))
                    motif_edges.add((j, i))

    elif motif_type == 'feedforward_loop':
        # Find triangles: A→B, A→C, B→C
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

    elif motif_type == 'single_input_module':
        # Find hub node (highest out-degree ≥ 3) with pure fan-out
        best_hub = None
        max_targets = 0

        for node in G.nodes():
            successors = list(G.successors(node))
            if len(successors) >= 3:
                # Check pure fan-out (no feedback from targets)
                is_pure = all(not G.has_edge(t, node) for t in successors)
                if is_pure and len(successors) > max_targets:
                    best_hub = node
                    max_targets = len(successors)

        if best_hub is not None:
            for target in G.successors(best_hub):
                motif_edges.add((best_hub, target))

    elif motif_type == 'cascade':
        # Find linear chains of length ≥ 4
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
                        # Check if it's truly linear (internal nodes have in=1, out=1)
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
    """
    Explain GNN prediction for a target node using GNNExplainer.

    Args:
        gnn: Trained GNN model
        data: PyG Data object
        node_idx: Target node index
        device: Device to run on

    Returns:
        Edge importance scores (0-1), shape [num_edges]
    """
    gnn.eval()
    data = data.to(device)

    # Create explainer
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

    # Generate explanation
    explanation = explainer(data.x, data.edge_index, index=node_idx, edge_attr=data.edge_attr)

    # Extract edge mask
    edge_mask = explanation.edge_mask.cpu().detach().numpy()

    # Debug: Check edge mask values
    if edge_mask.std() < 1e-6:
        print(f"      WARNING: GNNExplainer edge_mask is constant (std={edge_mask.std():.2e})")
        print(f"               Values: min={edge_mask.min():.4f}, max={edge_mask.max():.4f}, mean={edge_mask.mean():.4f}")

    # Normalize to [0, 1]
    if edge_mask.max() > edge_mask.min():
        edge_mask = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min())
    else:
        # If all values are the same, return uniform importance
        edge_mask = np.ones_like(edge_mask) * 0.5

    return edge_mask


def explain_with_sae_gradient(
    gnn: nn.Module,
    sae: SparseAutoencoder,
    data: Data,
    feature_idx: int,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Explain SAE feature activation using gradient-based saliency on edge weights.

    Args:
        gnn: Trained GNN model
        sae: Trained SAE model
        data: PyG Data object
        feature_idx: Target SAE feature index
        device: Device to run on

    Returns:
        Edge importance scores (absolute gradients), shape [num_edges]
    """
    gnn.eval()
    sae.eval()

    # Move to device
    data = data.to(device)

    # Enable gradients for edge weights
    data.edge_attr = data.edge_attr.clone().requires_grad_(True)

    # Forward pass through GNN to get bottleneck activations
    with torch.enable_grad():
        h_bottleneck = gnn.get_intermediate_activations(data)  # [num_nodes, 64]

        # Debug: Check GNN activations
        h_mean = h_bottleneck.mean().item()
        h_std = h_bottleneck.std().item()
        h_max = h_bottleneck.max().item()

        if h_mean < 1e-6 and h_std < 1e-6:
            print(f"      WARNING: GNN bottleneck activations are all near-zero!")
            print(f"               h_bottleneck stats: mean={h_mean:.2e}, std={h_std:.2e}, max={h_max:.2e}")

        # Get pre-TopK activations (before sparsity is applied)
        # This is important because TopK sets most features to zero
        z_pre_topk = sae.encoder(h_bottleneck)
        z_pre_topk = F.relu(z_pre_topk)  # Apply ReLU as in encode()

        # Debug: Check how many features are active
        active_features = (z_pre_topk.max(dim=0)[0] > 1e-6).sum().item()

        # Target: activation of specific feature BEFORE TopK (sum across all nodes)
        feature_activations = z_pre_topk[:, feature_idx]
        target = feature_activations.sum()

        # Debug: Check feature activation
        if target.item() < 1e-6:
            print(f"      WARNING: SAE feature z{feature_idx} has near-zero activation ({target.item():.2e})")
            print(f"               Active features (>1e-6): {active_features}/{z_pre_topk.shape[1]}")
            print(f"               Feature activations (pre-TopK): {feature_activations.detach().cpu().numpy()[:5]}")
            # Show which features ARE active
            active_idx = (z_pre_topk.max(dim=0)[0] > 0.01).nonzero(as_tuple=True)[0]
            if len(active_idx) > 0:
                print(f"               Features with activation > 0.01: {active_idx[:10].cpu().numpy()}")

        # Compute gradients w.r.t. edge weights
        if target.item() == 0:
            # If target is exactly zero, gradients will be zero
            grads = torch.zeros_like(data.edge_attr)
        else:
            grads = torch.autograd.grad(target, data.edge_attr, create_graph=False)[0]

    # Take absolute value and normalize
    edge_importance = grads.abs().cpu().detach().numpy().squeeze()

    # Debug: Check gradient values
    if edge_importance.std() < 1e-6:
        print(f"      WARNING: SAE gradients are constant (std={edge_importance.std():.2e})")
        print(f"               Gradient values: min={edge_importance.min():.4f}, max={edge_importance.max():.4f}")

    # Normalize to [0, 1]
    if edge_importance.max() > edge_importance.min():
        edge_importance = (edge_importance - edge_importance.min()) / (edge_importance.max() - edge_importance.min())
    else:
        # If all gradients are the same, return uniform importance
        edge_importance = np.ones_like(edge_importance) * 0.5

    return edge_importance.flatten()  # Ensure 1D array


# ============================================================================
# COMPARISON EXPERIMENT
# ============================================================================

def run_comparison(
    gnn: nn.Module,
    sae: SparseAutoencoder,
    test_graphs: List[Data],
    motif_to_node_map: Dict[str, List[int]],
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Run comparison experiment between GNNExplainer and SAE gradient saliency.

    Args:
        gnn: Trained GNN model
        sae: Trained SAE model
        test_graphs: List of test graphs
        motif_to_node_map: Mapping from graph_id to motif node indices
        device: Device

    Returns:
        DataFrame with comparison results
    """
    # Define targets based on interpretability findings
    # These should be updated based on your actual compare_sae_configs.py results
    targets = {
        'feedforward_loop': {
            'feature_idx': 112,
            'motif_type': 'feedforward_loop',
            'description': 'Feedforward Loop detector (rpb=0.290)'
        },
        'feedback_loop': {
            'feature_idx': 112,
            'motif_type': 'feedback_loop',
            'description': 'Feedback Loop detector (rpb=0.727)'
        },
        'single_input_module': {
            'feature_idx': 112,
            'motif_type': 'single_input_module',
            'description': 'Single Input Module detector (rpb=0.290)'
        },
        'cascade': {
            'feature_idx': 120,
            'motif_type': 'cascade',
            'description': 'Cascade detector (rpb=0.149)'
        },
    }

    results = []

    for target_name, target_config in targets.items():
        print(f"\n{'='*70}")
        print(f"Analyzing Target: {target_name}")
        print(f"SAE Feature: z{target_config['feature_idx']}")
        print(f"Motif Type: {target_config['motif_type']}")
        print(f"{'='*70}\n")

        motif_type = target_config['motif_type']
        feature_idx = target_config['feature_idx'] - 1  # Convert to 0-indexed

        # Filter graphs with this motif
        motif_graphs = []
        print(f"DEBUG: Checking {len(test_graphs)} test graphs for {motif_type}...")
        for i, data in enumerate(test_graphs):
            # Check if graph contains motif
            try:
                gt_mask = get_ground_truth_edge_mask(data, motif_type)
                if gt_mask.sum() > 0:  # Has motif edges
                    motif_graphs.append(data)
                    if i < 5:  # Debug first few
                        print(f"  Graph {i} (id={data.graph_id}): {gt_mask.sum()} motif edges found")
                if len(motif_graphs) >= 20:
                    break
            except Exception as e:
                print(f"  ERROR checking graph {i}: {e}")
                if i < 3:
                    import traceback
                    traceback.print_exc()

        print(f"Found {len(motif_graphs)} graphs with {motif_type}")

        if len(motif_graphs) == 0:
            print(f"⚠ No graphs found with {motif_type}, skipping...")
            continue

        # Run comparison on each graph
        print(f"DEBUG: Starting comparison on {len(motif_graphs)} graphs...")
        successful_comparisons = 0
        for graph_idx, data in enumerate(tqdm(motif_graphs, desc=f"{target_name}")):
            try:
                # 1. Compute ground truth
                gt_mask = get_ground_truth_edge_mask(data, motif_type).numpy()

                if gt_mask.sum() == 0:
                    if graph_idx < 3:
                        print(f"  DEBUG: Graph {graph_idx} skipped - no motif edges after recompute")
                    continue  # Skip if no motif edges

                # 2. Select a node that's part of the motif
                # Find first node involved in motif edges
                motif_edge_indices = np.where(gt_mask)[0]
                if len(motif_edge_indices) == 0:
                    if graph_idx < 3:
                        print(f"  DEBUG: Graph {graph_idx} skipped - no motif edge indices")
                    continue

                edge_index = data.edge_index.cpu().numpy()
                motif_node = edge_index[0, motif_edge_indices[0]]  # Source of first motif edge

                if graph_idx < 3:
                    print(f"  DEBUG: Graph {graph_idx} - motif_node={motif_node}, {len(motif_edge_indices)} motif edges")

                # 3. Run GNNExplainer
                gnn_scores = explain_with_gnnexplainer(gnn, data, motif_node, device)
                if graph_idx < 3:
                    print(f"    GNNExplainer scores: min={gnn_scores.min():.4f}, max={gnn_scores.max():.4f}")

                # 4. Run SAE Gradient Saliency
                sae_scores = explain_with_sae_gradient(gnn, sae, data, feature_idx, device)
                if graph_idx < 3:
                    print(f"    SAE scores: min={sae_scores.min():.4f}, max={sae_scores.max():.4f}")

                # 5. Compute metrics
                # AUROC
                if len(np.unique(gt_mask)) > 1:  # Need both classes
                    gnn_auroc = roc_auc_score(gt_mask, gnn_scores)
                    sae_auroc = roc_auc_score(gt_mask, sae_scores)
                else:
                    gnn_auroc = np.nan
                    sae_auroc = np.nan

                # AUPRC (Average Precision)
                gnn_auprc = average_precision_score(gt_mask, gnn_scores)
                sae_auprc = average_precision_score(gt_mask, sae_scores)

                # Sparsity (how concentrated are the scores?)
                gnn_sparsity = (gnn_scores > 0.5).sum() / len(gnn_scores)
                sae_sparsity = (sae_scores > 0.5).sum() / len(sae_scores)

                results.append({
                    'target': target_name,
                    'motif_type': motif_type,
                    'feature_idx': target_config['feature_idx'],
                    'graph_idx': graph_idx,
                    'num_edges': len(gt_mask),
                    'num_motif_edges': gt_mask.sum(),
                    'gnn_auroc': gnn_auroc,
                    'sae_auroc': sae_auroc,
                    'gnn_auprc': gnn_auprc,
                    'sae_auprc': sae_auprc,
                    'gnn_sparsity': gnn_sparsity,
                    'sae_sparsity': sae_sparsity,
                    'data': data,  # Store for visualization
                    'gt_mask': gt_mask,
                    'gnn_scores': gnn_scores,
                    'sae_scores': sae_scores,
                    'motif_node': motif_node
                })
                successful_comparisons += 1
                if graph_idx < 3:
                    print(f"    ✓ Result added (AUROC: GNN={gnn_auroc:.3f}, SAE={sae_auroc:.3f})")

            except Exception as e:
                print(f"  ERROR on graph {graph_idx}: {e}")
                if graph_idx < 3:
                    import traceback
                    traceback.print_exc()
                continue

        print(f"DEBUG: Completed {successful_comparisons} successful comparisons for {target_name}")
        print(f"DEBUG: Total results collected so far: {len(results)}")

    print(f"\nDEBUG: run_comparison() returning {len(results)} total results")
    if len(results) == 0:
        print("WARNING: No results collected!")
        print("  Check if graphs contain motifs")
        print("  Check if ground truth detection is working")
        print("  Check for errors in the inner loop above")
    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_comparison(
    data: Data,
    gt_mask: np.ndarray,
    gnn_scores: np.ndarray,
    sae_scores: np.ndarray,
    motif_type: str,
    target_name: str,
    save_path: Optional[str] = None
):
    """
    Create side-by-side visualization of ground truth vs explanations.

    Args:
        data: PyG Data object
        gt_mask: Ground truth binary mask
        gnn_scores: GNNExplainer scores
        sae_scores: SAE gradient scores
        motif_type: Motif type
        target_name: Target name for title
        save_path: Optional path to save figure
    """
    # Convert to NetworkX
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.x.shape[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))

    # Add edges with attributes
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst,
                   gt=gt_mask[i],
                   gnn_score=gnn_scores[i],
                   sae_score=sae_scores[i])

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

    # Custom colormap (white -> red)
    cmap = LinearSegmentedColormap.from_list('importance', ['lightgray', 'red'])

    # Plot 1: Ground Truth
    ax = axes[0]
    edge_colors = ['red' if G[u][v]['gt'] else 'lightgray' for u, v in G.edges()]
    edge_widths = [3.0 if G[u][v]['gt'] else 1.0 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           arrows=True, arrowsize=15, ax=ax, connectionstyle='arc3,rad=0.1')
    ax.set_title(f'Ground Truth: {motif_type}\n(Red = Motif Edges)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Plot 2: GNNExplainer
    ax = axes[1]
    edge_colors = [G[u][v]['gnn_score'] for u, v in G.edges()]
    edge_widths = [1 + 4 * G[u][v]['gnn_score'] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                                    edge_cmap=cmap, arrows=True, arrowsize=15, ax=ax,
                                    connectionstyle='arc3,rad=0.1', edge_vmin=0, edge_vmax=1)
    ax.set_title(f'GNNExplainer\n(Baseline Method)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Plot 3: SAE Gradient Saliency
    ax = axes[2]
    edge_colors = [G[u][v]['sae_score'] for u, v in G.edges()]
    edge_widths = [1 + 4 * G[u][v]['sae_score'] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    edges = nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                                    edge_cmap=cmap, arrows=True, arrowsize=15, ax=ax,
                                    connectionstyle='arc3,rad=0.1', edge_vmin=0, edge_vmax=1)
    ax.set_title(f'SAE Gradient Saliency\n(Novel Method)', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1:], orientation='horizontal', pad=0.05, aspect=30)
    cbar.set_label('Edge Importance Score', fontsize=11)

    plt.suptitle(f'Explanation Comparison: {target_name}', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def plot_roc_and_pr_curves(df_results: pd.DataFrame, save_dir: str = 'outputs/comparison_plots'):
    """
    Plot ROC and Precision-Recall curves for all targets.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for target in df_results['target'].unique():
        df_target = df_results[df_results['target'] == target]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Aggregate predictions across all graphs
        all_gt = []
        all_gnn = []
        all_sae = []

        for _, row in df_target.iterrows():
            all_gt.extend(row['gt_mask'])
            all_gnn.extend(row['gnn_scores'])
            all_sae.extend(row['sae_scores'])

        all_gt = np.array(all_gt)
        all_gnn = np.array(all_gnn)
        all_sae = np.array(all_sae)

        # Check if ground truth has both classes
        if len(np.unique(all_gt)) < 2:
            print(f"  ⚠ Skipping curves for {target}: Ground truth has only one class")
            plt.close(fig)
            continue

        # ROC Curve
        ax = axes[0]
        fpr_gnn, tpr_gnn, _ = roc_curve(all_gt, all_gnn)
        fpr_sae, tpr_sae, _ = roc_curve(all_gt, all_sae)

        try:
            auc_gnn = roc_auc_score(all_gt, all_gnn)
            auc_sae = roc_auc_score(all_gt, all_sae)
            ax.plot(fpr_gnn, tpr_gnn, label=f'GNNExplainer (AUC={auc_gnn:.3f})', linewidth=2)
            ax.plot(fpr_sae, tpr_sae, label=f'SAE Saliency (AUC={auc_sae:.3f})', linewidth=2)
        except ValueError as e:
            print(f"  ⚠ Could not compute AUC for {target}: {e}")
            ax.plot(fpr_gnn, tpr_gnn, label='GNNExplainer', linewidth=2)
            ax.plot(fpr_sae, tpr_sae, label='SAE Saliency', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Precision-Recall Curve
        ax = axes[1]
        precision_gnn, recall_gnn, _ = precision_recall_curve(all_gt, all_gnn)
        precision_sae, recall_sae, _ = precision_recall_curve(all_gt, all_sae)

        ax.plot(recall_gnn, precision_gnn, label=f'GNNExplainer (AP={average_precision_score(all_gt, all_gnn):.3f})', linewidth=2)
        ax.plot(recall_sae, precision_sae, label=f'SAE Saliency (AP={average_precision_score(all_gt, all_sae):.3f})', linewidth=2)
        ax.axhline(y=all_gt.mean(), color='k', linestyle='--', label=f'Random (AP={all_gt.mean():.3f})', linewidth=1)
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'Target: {target}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = Path(save_dir) / f'curves_{target}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC/PR curves to {save_path}")
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_test_graphs(device: str = 'cuda') -> List[Data]:
    """Load test graphs from activation directory."""
    print("Loading test graphs...")

    # Load test graph IDs
    with open('outputs/test_graph_ids.json', 'r') as f:
        test_graph_ids = json.load(f)['graph_ids']

    test_graphs = []
    graph_dir = Path("virtual_graphs/data/all_graphs/raw_graphs")

    for graph_id in test_graph_ids[:100]:  # Limit for speed
        graph_path = graph_dir / f"graph_{graph_id}.pkl"
        if not graph_path.exists():
            continue

        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        # Convert to PyG Data
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges()]).unsqueeze(1).float()

        # Create node features (degree-based or random)
        num_nodes = len(G.nodes())
        x = torch.randn(num_nodes, 2)  # Simple random features

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.graph_id = graph_id

        test_graphs.append(data)

    print(f"Loaded {len(test_graphs)} test graphs")
    return test_graphs


def main():
    """Main comparison pipeline."""
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}\n")

    # ========================================================================
    # 1. Load Models
    # ========================================================================
    print("="*70)
    print("LOADING MODELS")
    print("="*70)

    # Load GNN
    print("\n1. Loading GNN model...")
    gnn = GCNModel(input_dim=2, hidden_dim=80, output_dim=1, dropout=0.3)
    checkpoint = torch.load('checkpoints/gnn_model.pt', map_location=DEVICE, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        gnn.load_state_dict(checkpoint['model_state_dict'])
    else:
        gnn.load_state_dict(checkpoint)  # Checkpoint is the state dict directly

    gnn.to(DEVICE)
    gnn.eval()
    print("   ✓ GNN loaded")

    # Load SAE (optimal config from compare_sae_configs.py)
    print("\n2. Loading SAE model (latent_dim=128, k=16)...")
    sae = SparseAutoencoder(input_dim=64, latent_dim=128, k=16)
    sae_checkpoint = torch.load('checkpoints/sae_latent128_k16.pt', map_location=DEVICE, weights_only=False)
    sae.load_state_dict(sae_checkpoint['model_state_dict'])
    sae.to(DEVICE)
    sae.eval()
    print("   ✓ SAE loaded")

    # ========================================================================
    # 2. Load Test Graphs
    # ========================================================================
    print("\n" + "="*70)
    print("LOADING TEST DATA")
    print("="*70)
    test_graphs = load_test_graphs(DEVICE)

    # ========================================================================
    # 3. Run Comparison
    # ========================================================================
    print("\n" + "="*70)
    print("RUNNING COMPARISON EXPERIMENT")
    print("="*70)

    df_results = run_comparison(gnn, sae, test_graphs, {}, device=DEVICE)

    # ========================================================================
    # 4. Print Summary Statistics
    # ========================================================================
    print("\n" + "="*70)
    print("QUANTITATIVE RESULTS")
    print("="*70)

    print(f"DEBUG: df_results shape: {df_results.shape}")
    print(f"DEBUG: df_results columns: {list(df_results.columns)}")
    if len(df_results) == 0:
        print("ERROR: No results to analyze! Exiting...")
        print("\nPossible issues:")
        print("  1. Ground truth detection found no motifs in test graphs")
        print("  2. Model architecture mismatch between trained and loaded")
        print("  3. All comparisons failed due to errors")
        print("\nCheck the DEBUG output above for details.")
        return

    summary_metrics = df_results.groupby('target').agg({
        'gnn_auroc': ['mean', 'std'],
        'sae_auroc': ['mean', 'std'],
        'gnn_auprc': ['mean', 'std'],
        'sae_auprc': ['mean', 'std'],
        'gnn_sparsity': ['mean', 'std'],
        'sae_sparsity': ['mean', 'std']
    }).round(3)

    print("\nMean ± Std across all test graphs:\n")
    print(summary_metrics.to_string())

    # Statistical significance test
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE (Paired t-test)")
    print("="*70)

    from scipy.stats import ttest_rel

    for target in df_results['target'].unique():
        df_target = df_results[df_results['target'] == target]

        # Remove NaN values
        valid_mask = ~(df_target['gnn_auroc'].isna() | df_target['sae_auroc'].isna())
        df_valid = df_target[valid_mask]

        if len(df_valid) > 1:
            t_stat_auroc, p_val_auroc = ttest_rel(df_valid['sae_auroc'], df_valid['gnn_auroc'])
            t_stat_auprc, p_val_auprc = ttest_rel(df_valid['sae_auprc'], df_valid['gnn_auprc'])

            print(f"\nTarget: {target}")
            print(f"  AUROC: SAE vs GNN, t={t_stat_auroc:.3f}, p={p_val_auroc:.4f}")
            print(f"  AUPRC: SAE vs GNN, t={t_stat_auprc:.3f}, p={p_val_auprc:.4f}")

            if p_val_auroc < 0.05:
                winner = "SAE" if t_stat_auroc > 0 else "GNN"
                print(f"  → {winner} is significantly better (AUROC, p<0.05)")
            else:
                print(f"  → No significant difference (AUROC)")

    # ========================================================================
    # 5. Save Results
    # ========================================================================
    output_dir = Path('outputs/comparison_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_path = output_dir / 'comparison_summary.csv'
    summary_metrics.to_csv(summary_path)
    print(f"\n✓ Saved summary to {summary_path}")

    # Save detailed results
    df_save = df_results.drop(columns=['data', 'gt_mask', 'gnn_scores', 'sae_scores'])
    details_path = output_dir / 'comparison_detailed.csv'
    df_save.to_csv(details_path, index=False)
    print(f"✓ Saved detailed results to {details_path}")

    # ========================================================================
    # 6. Generate Visualizations
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Plot ROC and PR curves
    plot_roc_and_pr_curves(df_results, save_dir='outputs/comparison_plots')

    # Visualize representative examples
    viz_dir = Path('outputs/comparison_plots')
    viz_dir.mkdir(parents=True, exist_ok=True)

    for target in df_results['target'].unique():
        df_target = df_results[df_results['target'] == target]

        # Skip if all AUROC values are NaN
        if df_target['sae_auroc'].isna().all():
            print(f"  ⚠ Skipping visualizations for {target}: All AUROC values are NaN")
            continue

        # Select best example (highest SAE AUROC)
        valid_auroc = df_target.dropna(subset=['sae_auroc'])
        if len(valid_auroc) == 0:
            print(f"  ⚠ Skipping visualizations for {target}: No valid AUROC values")
            continue

        best_idx = valid_auroc['sae_auroc'].idxmax()
        best_row = df_target.loc[best_idx]

        save_path = viz_dir / f'visualization_{target}_best.png'
        visualize_comparison(
            best_row['data'],
            best_row['gt_mask'],
            best_row['gnn_scores'],
            best_row['sae_scores'],
            best_row['motif_type'],
            target,
            save_path=str(save_path)
        )

        # Select worst example (lowest SAE AUROC)
        worst_idx = valid_auroc['sae_auroc'].idxmin()
        worst_row = df_target.loc[worst_idx]

        save_path = viz_dir / f'visualization_{target}_worst.png'
        visualize_comparison(
            worst_row['data'],
            worst_row['gt_mask'],
            worst_row['gnn_scores'],
            worst_row['sae_scores'],
            worst_row['motif_type'],
            target,
            save_path=str(save_path)
        )

        # Select median example
        median_val = valid_auroc['sae_auroc'].median()
        median_idx = (valid_auroc['sae_auroc'] - median_val).abs().idxmin()
        median_row = df_target.loc[median_idx]

        save_path = viz_dir / f'visualization_{target}_median.png'
        visualize_comparison(
            median_row['data'],
            median_row['gt_mask'],
            median_row['gnn_scores'],
            median_row['sae_scores'],
            median_row['motif_type'],
            target,
            save_path=str(save_path)
        )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    print("\nKey findings:")
    print("  - Check comparison_summary.csv for aggregate metrics")
    print("  - Check comparison_detailed.csv for per-graph results")
    print("  - Check visualization_*.png for qualitative comparisons")


if __name__ == "__main__":
    main()
