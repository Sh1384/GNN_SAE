"""
Compute Node-Level Topological Features

This script computes degree and other topological features for all nodes
in all graphs and adds them to the metadata CSV files.

This addresses the confound control issue where SAE features may correlate
with degree patterns rather than motif semantics.

Usage:
    python compute_node_features.py

Output:
    Updates all graph_motif_metadata CSV files with additional columns:
    - degree: Total degree (in-degree + out-degree)
    - in_degree: Number of incoming edges
    - out_degree: Number of outgoing edges
    - degree_ratio: Ratio of in-degree to out-degree (or NaN if out-degree=0)
"""

import pickle
from pathlib import Path
from typing import Dict
import networkx as nx
import pandas as pd
from tqdm import tqdm


def compute_node_features(G: nx.DiGraph) -> Dict[int, Dict[str, float]]:
    """
    Compute topological features for all nodes in a directed graph.

    Args:
        G: Directed graph

    Returns:
        Dictionary mapping node_id to feature dictionary
        {
            node_id: {
                'degree': total_degree,
                'in_degree': in_degree,
                'out_degree': out_degree,
                'degree_ratio': in_degree / out_degree (or NaN)
            }
        }
    """
    features = {}

    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        total_deg = in_deg + out_deg

        # Compute ratio (handle division by zero)
        if out_deg > 0:
            deg_ratio = in_deg / out_deg
        else:
            deg_ratio = float('nan')

        features[node] = {
            'degree': total_deg,
            'in_degree': in_deg,
            'out_degree': out_deg,
            'degree_ratio': deg_ratio
        }

    return features


def update_metadata_with_features(graph_id: int,
                                  graphs_dir: Path,
                                  metadata_dir: Path) -> bool:
    """
    Load a graph, compute features, and update its metadata CSV.

    Args:
        graph_id: Graph ID
        graphs_dir: Directory containing graph pickle files
        metadata_dir: Directory containing metadata CSV files

    Returns:
        True if successful, False otherwise
    """
    graph_path = graphs_dir / f"graph_{graph_id}.pkl"
    metadata_path = metadata_dir / f"graph_{graph_id}_metadata.csv"

    # Check if files exist
    if not graph_path.exists():
        print(f"  ⚠ Graph file not found: {graph_path}")
        return False

    if not metadata_path.exists():
        print(f"  ⚠ Metadata file not found: {metadata_path}")
        return False

    try:
        # Load graph
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        # Compute features
        node_features = compute_node_features(G)

        # Load existing metadata
        df_metadata = pd.read_csv(metadata_path, index_col=0)

        # Add feature columns
        for feature_name in ['degree', 'in_degree', 'out_degree', 'degree_ratio']:
            df_metadata[feature_name] = [
                node_features[i][feature_name] for i in range(len(df_metadata))
            ]

        # Save updated metadata
        df_metadata.to_csv(metadata_path)

        return True

    except Exception as e:
        print(f"  ✗ Error processing graph {graph_id}: {e}")
        return False


def main():
    """Process all graphs and update metadata files."""

    base_dir = Path("virtual_graphs/data/all_graphs")
    graphs_dir = base_dir / "raw_graphs"
    metadata_dir = base_dir / "graph_motif_metadata"

    print("=" * 70)
    print("Computing Node-Level Topological Features")
    print("=" * 70)
    print(f"\nGraphs directory: {graphs_dir}")
    print(f"Metadata directory: {metadata_dir}")

    # Check directories exist
    if not graphs_dir.exists():
        print(f"\n✗ Error: Graphs directory not found: {graphs_dir}")
        return

    if not metadata_dir.exists():
        print(f"\n✗ Error: Metadata directory not found: {metadata_dir}")
        return

    # Find all graph files
    graph_files = sorted(graphs_dir.glob("graph_*.pkl"))
    print(f"\nFound {len(graph_files)} graph files")

    # Process each graph
    print("\nProcessing graphs...")
    success_count = 0
    fail_count = 0

    for graph_file in tqdm(graph_files, desc="Computing features"):
        # Extract graph ID from filename
        graph_id = int(graph_file.stem.split('_')[1])

        success = update_metadata_with_features(graph_id, graphs_dir, metadata_dir)

        if success:
            success_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"✓ Successfully processed: {success_count} graphs")
    if fail_count > 0:
        print(f"✗ Failed: {fail_count} graphs")
    print(f"\nMetadata files updated with 4 new columns:")
    print("  - degree: Total degree (in + out)")
    print("  - in_degree: Number of incoming edges")
    print("  - out_degree: Number of outgoing edges")
    print("  - degree_ratio: in_degree / out_degree")

    # Show example
    if success_count > 0:
        example_path = metadata_dir / "graph_0_metadata.csv"
        if example_path.exists():
            print(f"\nExample (graph_0_metadata.csv):")
            df_example = pd.read_csv(example_path, index_col=0)
            print(df_example.head(10).to_string())


if __name__ == "__main__":
    main()
