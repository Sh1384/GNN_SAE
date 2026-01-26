#!/usr/bin/env python3
"""Test script to verify motif detection in run_ablation.py matches actual graph data."""

import pickle
import pandas as pd
from pathlib import Path

def load_graph_motif_metadata(graph_id):
    """Load motif metadata for a specific graph (from run_ablation.py)."""
    metadata_path = Path(f"virtual_graphs/data/all_graphs/graph_motif_metadata/graph_{graph_id}_metadata.csv")
    if not metadata_path.exists():
        return {}

    df = pd.read_csv(metadata_path, index_col=0)
    # Count nodes in each motif
    motif_counts = df.sum(axis=0).to_dict()
    return motif_counts

def get_dominant_motif(graph_id):
    """Determine the majority motif for a graph (from run_ablation.py)."""
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

def get_actual_motif_from_graph(graph_id):
    """Load the graph and get the actual motif type from graph attributes."""
    graph_path = Path(f"virtual_graphs/data/all_graphs/raw_graphs/graph_{graph_id}.pkl")
    if not graph_path.exists():
        return None

    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    if 'motif_type' in G.graph:
        # Single motif graph
        name_map = {
            'feedforward_loop': 'Feedforward Loop',
            'feedback_loop': 'Feedback Loop',
            'single_input_module': 'Single Input Module',
            'cascade': 'Cascade'
        }
        return name_map.get(G.graph['motif_type'], G.graph['motif_type'])
    elif 'motif_composition' in G.graph:
        # Mixed motif graph
        return f"Mixed: {G.graph['motif_composition']}"
    else:
        return "Unknown"

# Test on various graph IDs
test_ids = [0, 1, 50, 100, 500, 999,     # Should be feedforward_loop
            1000, 1001, 1500, 1999,       # Should be feedback_loop
            2000, 2001, 2500, 2999,       # Should be single_input_module
            3000, 3001, 3500, 3999,       # Should be cascade
            4000, 4001, 4500, 4999]       # Should be mixed

print("=" * 80)
print("MOTIF DETECTION VERIFICATION")
print("=" * 80)
print(f"{'Graph ID':<10} {'Actual Motif':<30} {'Detected Motif':<30} {'Match?':<10}")
print("-" * 80)

mismatches = []

for graph_id in test_ids:
    actual = get_actual_motif_from_graph(graph_id)
    detected = get_dominant_motif(graph_id)

    # For mixed graphs, detected will be the dominant single motif
    if actual and actual.startswith("Mixed"):
        match = "N/A (Mixed)"
    else:
        match = "✓" if actual == detected else "✗ MISMATCH"
        if actual != detected and actual is not None:
            mismatches.append((graph_id, actual, detected))

    print(f"{graph_id:<10} {str(actual):<30} {detected:<30} {match:<10}")

print("=" * 80)

if mismatches:
    print(f"\n⚠️  FOUND {len(mismatches)} MISMATCHES:")
    for graph_id, actual, detected in mismatches:
        print(f"  Graph {graph_id}: Expected '{actual}', Got '{detected}'")
else:
    print("\n✓ All single-motif graphs correctly detected!")

print("\nNote: Mixed-motif graphs show the dominant motif by node count, which is expected behavior.")
