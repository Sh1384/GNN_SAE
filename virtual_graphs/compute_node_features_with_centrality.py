"""
Compute Node-Level Topological Features Including Centrality

This script computes degree AND centrality features for all nodes
in all graphs and adds them to the metadata CSV files.

This provides additional confound controls beyond degree:
- Betweenness centrality (how often node is on shortest paths)
- Closeness centrality (how close node is to all other nodes)
- PageRank (importance based on incoming connections)
- Clustering coefficient (how connected node's neighbors are)

Usage:
    python compute_node_features_with_centrality.py

Output:
    Updates all graph_motif_metadata CSV files with columns:
    - degree, in_degree, out_degree, degree_ratio (existing)
    - betweenness_centrality
    - in_closeness_centrality, out_closeness_centrality
    - pagerank
    - clustering_coefficient
"""

import pickle
from pathlib import Path
from typing import Dict
import networkx as nx
import pandas as pd
from tqdm import tqdm
import numpy as np


def compute_node_features_with_centrality(G: nx.DiGraph) -> Dict[int, Dict[str, float]]:
    """
    Compute topological features including centrality for all nodes in a directed graph.

    Args:
        G: Directed graph

    Returns:
        Dictionary mapping node_id to feature dictionary
    """
    features = {}

    # Basic degree features
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

    # Centrality measures (computed for all nodes at once, more efficient)
    print(f"    Computing betweenness centrality...", end='', flush=True)
    try:
        betweenness = nx.betweenness_centrality(G)
        print(" ✓")
    except:
        betweenness = {node: 0.0 for node in G.nodes()}
        print(" (failed, using zeros)")

    print(f"    Computing closeness centrality...", end='', flush=True)
    try:
        # For directed graphs, we compute both in and out closeness
        in_closeness = nx.closeness_centrality(G.reverse())  # Reverse for in-closeness
        out_closeness = nx.closeness_centrality(G)
        print(" ✓")
    except:
        in_closeness = {node: 0.0 for node in G.nodes()}
        out_closeness = {node: 0.0 for node in G.nodes()}
        print(" (failed, using zeros)")

    print(f"    Computing PageRank...", end='', flush=True)
    try:
        pagerank = nx.pagerank(G, max_iter=100)
        print(" ✓")
    except:
        pagerank = {node: 1.0 / G.number_of_nodes() for node in G.nodes()}
        print(" (failed, using uniform)")

    print(f"    Computing clustering coefficient...", end='', flush=True)
    try:
        # For directed graphs, use directed clustering
        clustering = nx.clustering(G)
        print(" ✓")
    except:
        clustering = {node: 0.0 for node in G.nodes()}
        print(" (failed, using zeros)")

    # Add centrality measures to features
    for node in G.nodes():
        features[node].update({
            'betweenness_centrality': betweenness.get(node, 0.0),
            'in_closeness_centrality': in_closeness.get(node, 0.0),
            'out_closeness_centrality': out_closeness.get(node, 0.0),
            'pagerank': pagerank.get(node, 0.0),
            'clustering_coefficient': clustering.get(node, 0.0)
        })

    return features


def update_metadata_with_features(graph_id: int,
                                  graphs_dir: Path,
                                  metadata_dir: Path,
                                  verbose: bool = False) -> bool:
    """
    Load a graph, compute features, and update its metadata CSV.

    Args:
        graph_id: Graph ID
        graphs_dir: Directory containing graph pickle files
        metadata_dir: Directory containing metadata CSV files
        verbose: Print detailed progress

    Returns:
        True if successful, False otherwise
    """
    graph_path = graphs_dir / f"graph_{graph_id}.pkl"
    metadata_path = metadata_dir / f"graph_{graph_id}_metadata.csv"

    # Check if files exist
    if not graph_path.exists():
        if verbose:
            print(f"  ⚠ Graph file not found: {graph_path}")
        return False

    if not metadata_path.exists():
        if verbose:
            print(f"  ⚠ Metadata file not found: {metadata_path}")
        return False

    try:
        # Load graph
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        if verbose:
            print(f"  Processing graph {graph_id} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

        # Compute features
        node_features = compute_node_features_with_centrality(G)

        # Load existing metadata
        df_metadata = pd.read_csv(metadata_path, index_col=0)

        # Add all feature columns
        feature_names = [
            'degree', 'in_degree', 'out_degree', 'degree_ratio',
            'betweenness_centrality', 'in_closeness_centrality',
            'out_closeness_centrality', 'pagerank', 'clustering_coefficient'
        ]

        for feature_name in feature_names:
            df_metadata[feature_name] = [
                node_features[i][feature_name] for i in range(len(df_metadata))
            ]

        # Save updated metadata
        df_metadata.to_csv(metadata_path)

        return True

    except Exception as e:
        if verbose:
            print(f"  ✗ Error processing graph {graph_id}: {e}")
        return False


def main():
    """Process all graphs and update metadata files."""

    base_dir = Path("virtual_graphs/data/all_graphs")
    graphs_dir = base_dir / "raw_graphs"
    metadata_dir = base_dir / "graph_motif_metadata"

    print("=" * 70)
    print("Computing Node-Level Features (Degree + Centrality)")
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

    print("\nFeatures to compute:")
    print("  Degree-based:")
    print("    - degree, in_degree, out_degree, degree_ratio")
    print("  Centrality-based:")
    print("    - betweenness_centrality")
    print("    - in_closeness_centrality, out_closeness_centrality")
    print("    - pagerank")
    print("    - clustering_coefficient")

    # Process each graph
    print("\nProcessing graphs...")
    success_count = 0
    fail_count = 0

    # Show detailed output for first graph
    for i, graph_file in enumerate(tqdm(graph_files, desc="Computing features")):
        # Extract graph ID from filename
        graph_id = int(graph_file.stem.split('_')[1])

        verbose = (i == 0)  # Verbose for first graph only
        success = update_metadata_with_features(graph_id, graphs_dir, metadata_dir, verbose=verbose)

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
    print(f"\nMetadata files updated with 9 new columns:")
    print("\n  Degree features:")
    print("    - degree: Total degree (in + out)")
    print("    - in_degree: Number of incoming edges")
    print("    - out_degree: Number of outgoing edges")
    print("    - degree_ratio: in_degree / out_degree")
    print("\n  Centrality features:")
    print("    - betweenness_centrality: How often node is on shortest paths")
    print("    - in_closeness_centrality: How accessible node is from others")
    print("    - out_closeness_centrality: How quickly node can reach others")
    print("    - pagerank: Node importance (Google PageRank algorithm)")
    print("    - clustering_coefficient: How connected node's neighbors are")

    # Show example
    if success_count > 0:
        example_path = metadata_dir / "graph_0_metadata.csv"
        if example_path.exists():
            print(f"\nExample (first 5 nodes of graph_0):")
            print("-" * 70)
            df_example = pd.read_csv(example_path, index_col=0)

            # Show motif columns + all topological features
            motif_cols = ['feedforward_loop', 'feedback_loop', 'single_input_module', 'cascade']
            feature_cols = ['degree', 'betweenness_centrality', 'pagerank', 'clustering_coefficient']
            cols_to_show = [c for c in motif_cols + feature_cols if c in df_example.columns]

            print(df_example[cols_to_show].head(5).to_string())
            print("\n(Showing subset of columns for readability)")

    print("\n" + "=" * 70)
    print("NEXT STEP")
    print("=" * 70)
    print("\nNow run the updated SAE comparison script:")
    print("  python compare_sae_configs_with_centrality.py")
    print("\nThis will use ALL topological features (degree + centrality)")
    print("as control variables when computing partial correlations.")


if __name__ == "__main__":
    main()
