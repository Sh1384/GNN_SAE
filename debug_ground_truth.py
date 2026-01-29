#!/usr/bin/env python3
"""
Debug script to check if ground truth detection is working.
"""

import json
import pickle
import torch
from pathlib import Path
from torch_geometric.data import Data
import networkx as nx
import sys

# Add the ground truth function from compare_sae_vs_gnnexplainer.py
def get_ground_truth_edge_mask(data: Data, motif_type: str) -> torch.Tensor:
    """Generate ground truth edge mask for a specific motif type."""
    edge_index = data.edge_index.cpu().numpy()
    num_edges = edge_index.shape[1]
    gt_mask = torch.zeros(num_edges, dtype=torch.bool)

    # Build adjacency dict
    adj = {}
    for i in range(num_edges):
        src, tgt = edge_index[0, i], edge_index[1, i]
        if src not in adj:
            adj[src] = []
        adj[src].append(tgt)

    # Detect motif
    if motif_type == 'feedback_loop':
        # Bidirectional edges: (i,j) and (j,i) both exist
        for i in range(num_edges):
            src, tgt = edge_index[0, i], edge_index[1, i]
            if src in adj and tgt in adj:
                if tgt in adj.get(src, []) and src in adj.get(tgt, []):
                    gt_mask[i] = True

    elif motif_type == 'feedforward_loop':
        # Triangle: A->B, A->C, B->C
        for a in adj:
            for b in adj[a]:
                for c in adj.get(b, []):
                    if c in adj.get(a, []):
                        # Found triangle a->b->c and a->c
                        for i in range(num_edges):
                            src, tgt = edge_index[0, i], edge_index[1, i]
                            if (src == a and tgt == b) or (src == a and tgt == c) or (src == b and tgt == c):
                                gt_mask[i] = True

    elif motif_type == 'single_input_module':
        # Hub with out-degree >= 3 with no back edges
        for node in adj:
            targets = adj[node]
            if len(targets) >= 3:
                # Check no back edges
                has_back_edge = any(node in adj.get(t, []) for t in targets)
                if not has_back_edge:
                    # Mark edges from hub to targets
                    for i in range(num_edges):
                        src, tgt = edge_index[0, i], edge_index[1, i]
                        if src == node and tgt in targets:
                            gt_mask[i] = True

    elif motif_type == 'cascade':
        # Linear path of length >= 4
        G = nx.DiGraph()
        for i in range(num_edges):
            src, tgt = edge_index[0, i], edge_index[1, i]
            G.add_edge(src, tgt)

        # Find simple paths
        for source in G.nodes():
            for target in G.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(G, source, target, cutoff=10))
                        for path in paths:
                            if len(path) >= 4:
                                # Check linearity
                                is_linear = True
                                for k in range(1, len(path)-1):
                                    node = path[k]
                                    in_deg = sum(1 for p in path if p in G.predecessors(node))
                                    out_deg = sum(1 for p in path if p in G.successors(node))
                                    if in_deg != 1 or out_deg != 1:
                                        is_linear = False
                                        break

                                if is_linear:
                                    # Mark edges in path
                                    for k in range(len(path)-1):
                                        for i in range(num_edges):
                                            src, tgt = edge_index[0, i], edge_index[1, i]
                                            if src == path[k] and tgt == path[k+1]:
                                                gt_mask[i] = True
                    except nx.NetworkXNoPath:
                        continue

    return gt_mask


def main():
    print("="*70)
    print("DEBUGGING GROUND TRUTH DETECTION")
    print("="*70)

    # Load test graph IDs
    with open('outputs/test_graph_ids.json', 'r') as f:
        test_graph_ids = json.load(f)['graph_ids']

    print(f"\nTotal test graphs: {len(test_graph_ids)}")
    print(f"Checking first 100 graphs...\n")

    # Statistics
    motif_counts = {
        'feedback_loop': 0,
        'feedforward_loop': 0,
        'single_input_module': 0,
        'cascade': 0
    }

    graph_dir = Path("virtual_graphs/data/all_graphs/raw_graphs")

    # Check first 100 graphs
    for graph_id in test_graph_ids[:100]:
        graph_path = graph_dir / f"graph_{graph_id}.pkl"
        if not graph_path.exists():
            continue

        # Load graph
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        # Convert to PyG Data
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        edge_attr = torch.tensor([G[u][v]['weight'] for u, v in G.edges()]).unsqueeze(1).float()
        num_nodes = len(G.nodes())
        x = torch.randn(num_nodes, 2)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.graph_id = graph_id

        # Check each motif type
        for motif_type in motif_counts.keys():
            try:
                gt_mask = get_ground_truth_edge_mask(data, motif_type)
                if gt_mask.sum() > 0:
                    motif_counts[motif_type] += 1
                    if motif_counts[motif_type] <= 3:  # Show first 3 examples
                        print(f"  Graph {graph_id}: Found {motif_type} ({gt_mask.sum()} edges)")
            except Exception as e:
                print(f"  Error on graph {graph_id}, motif {motif_type}: {e}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nGraphs containing each motif type (out of first 100):\n")
    for motif_type, count in motif_counts.items():
        print(f"  {motif_type:25s}: {count:3d} graphs")

    if all(count == 0 for count in motif_counts.values()):
        print("\n⚠ WARNING: NO MOTIFS DETECTED IN ANY GRAPH!")
        print("\nPossible issues:")
        print("  1. Ground truth detection logic doesn't match graph structure")
        print("  2. Graphs don't actually contain these motifs")
        print("  3. Graph format is different than expected")
        print("\nDebugging suggestions:")
        print("  - Check a sample graph structure manually")
        print("  - Verify metadata files have motif labels")
        print("  - Compare ground truth logic with actual graph patterns")
    else:
        print("\n✓ Ground truth detection is working!")
        print(f"  At least {sum(1 for c in motif_counts.values() if c > 0)} motif types detected")


if __name__ == "__main__":
    main()
