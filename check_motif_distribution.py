#!/usr/bin/env python3
"""Check distribution of motif types in test graphs."""

import pickle
import json
from pathlib import Path
import networkx as nx
from tqdm import tqdm

# Load test graph IDs
with open('outputs/test_graph_ids.json', 'r') as f:
    test_graph_ids = json.load(f)['graph_ids'][:200]

print(f'Checking first {len(test_graph_ids)} test graphs for motif types with mixed edges...\n')

graph_dir = Path('virtual_graphs/data/all_graphs/raw_graphs')

motif_stats = {
    'in_feedback_loop': {'total': 0, 'mixed': 0, 'all_motif': 0, 'no_motif': 0, 'examples': []},
    'in_feedforward_loop': {'total': 0, 'mixed': 0, 'all_motif': 0, 'no_motif': 0, 'examples': []},
    'in_single_input_module': {'total': 0, 'mixed': 0, 'all_motif': 0, 'no_motif': 0, 'examples': []},
    'in_cascade': {'total': 0, 'mixed': 0, 'all_motif': 0, 'no_motif': 0, 'examples': []}
}

def get_motif_edges(G, motif_type):
    """Get motif edges for a graph."""
    motif_edges = set()
    num_nodes = len(G.nodes())

    if motif_type == 'in_feedback_loop':
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if G.has_edge(i, j) and G.has_edge(j, i):
                    motif_edges.add((i, j))
                    motif_edges.add((j, i))

    elif motif_type == 'in_feedforward_loop':
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
        for source in G.nodes():
            for target in G.nodes():
                if source == target:
                    continue
                try:
                    paths = list(nx.all_simple_paths(G, source, target, cutoff=10))
                except:
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

    return motif_edges

for graph_id in tqdm(test_graph_ids, desc="Checking graphs"):
    graph_path = graph_dir / f'graph_{graph_id}.pkl'
    if not graph_path.exists():
        continue

    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    num_edges = len(G.edges())

    for motif_type in motif_stats.keys():
        motif_edges = get_motif_edges(G, motif_type)
        num_motif = len(motif_edges)

        if num_edges > 0:
            motif_ratio = num_motif / num_edges

            motif_stats[motif_type]['total'] += 1

            if num_motif == 0:
                motif_stats[motif_type]['no_motif'] += 1
            elif num_motif == num_edges:
                motif_stats[motif_type]['all_motif'] += 1
            elif 0.2 <= motif_ratio <= 0.8:
                motif_stats[motif_type]['mixed'] += 1
                if len(motif_stats[motif_type]['examples']) < 5:
                    motif_stats[motif_type]['examples'].append(
                        (graph_id, num_edges, num_motif, f'{motif_ratio:.1%}')
                    )

print('\n' + '='*70)
print('MOTIF STATISTICS')
print('='*70)
for motif, stats in motif_stats.items():
    print(f'\n{motif}:')
    print(f'  Total graphs checked: {stats["total"]}')
    print(f'  No motif (0%): {stats["no_motif"]}')
    print(f'  All motif (100%): {stats["all_motif"]}')
    print(f'  Mixed (20-80%): {stats["mixed"]} ← USABLE FOR COMPARISON')

    if stats['examples']:
        print(f'  Example mixed graphs:')
        for graph_id, n_edges, n_motif, ratio in stats['examples']:
            print(f'    Graph {graph_id}: {n_motif}/{n_edges} edges ({ratio})')

print('\n' + '='*70)
print('DIAGNOSIS')
print('='*70)
print('\nIf any motif has 0 mixed graphs, the comparison cannot run for that motif.')
print('This is because AUROC requires both positive (motif) and negative (non-motif) examples.')
print('\nPossible solutions:')
print('  1. Use training graphs instead of test graphs (they may have better distribution)')
print('  2. Adjust the mixed ratio threshold (currently 20-80%)')
print('  3. Generate synthetic graphs with controlled motif ratios')
print('  4. Accept that some motifs cannot be compared with this approach')
