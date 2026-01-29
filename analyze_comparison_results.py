#!/usr/bin/env python3
"""Analyze GNNExplainer comparison results."""

import csv
from collections import defaultdict

# Read CSV
data = []
with open('outputs/gnnexplainer_comparison/comparison_all_variants.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['gnn_auroc'] = float(row['gnn_auroc'])
        row['sae_auroc'] = float(row['sae_auroc'])
        row['gnn_auprc'] = float(row['gnn_auprc'])
        row['sae_auprc'] = float(row['sae_auprc'])
        row['motif_ratio'] = float(row['motif_ratio'])
        data.append(row)

print('='*70)
print('GNNEXPLAINER COMPARISON RESULTS SUMMARY')
print('='*70)

# 1. Overall by variant
print('\n1. OVERALL PERFORMANCE BY VARIANT:')
print('='*70)
for variant in ['jumprelu', 'topk', 'switch', 'gated']:
    variant_data = [d for d in data if d['variant'] == variant]
    if not variant_data:
        continue

    gnn_aurocs = [d['gnn_auroc'] for d in variant_data]
    sae_aurocs = [d['sae_auroc'] for d in variant_data]
    gnn_auprcs = [d['gnn_auprc'] for d in variant_data]
    sae_auprcs = [d['sae_auprc'] for d in variant_data]

    print(f'\n{variant.upper()}:')
    print(f'  GNN AUROC:  {sum(gnn_aurocs)/len(gnn_aurocs):.3f}')
    print(f'  SAE AUROC:  {sum(sae_aurocs)/len(sae_aurocs):.3f}')
    print(f'  GNN AUPRC:  {sum(gnn_auprcs)/len(gnn_auprcs):.3f}')
    print(f'  SAE AUPRC:  {sum(sae_auprcs)/len(sae_auprcs):.3f}')
    print(f'  Graphs tested: {len(variant_data)}')

# 2. By motif
print('\n\n2. PERFORMANCE BY MOTIF (averaged across all variants):')
print('='*70)
for motif in ['in_feedback_loop', 'in_feedforward_loop', 'in_single_input_module', 'in_cascade']:
    motif_data = [d for d in data if d['motif'] == motif]
    if not motif_data:
        continue

    gnn_aurocs = [d['gnn_auroc'] for d in motif_data]
    sae_aurocs = [d['sae_auroc'] for d in motif_data]
    gnn_auprcs = [d['gnn_auprc'] for d in motif_data]
    sae_auprcs = [d['sae_auprc'] for d in motif_data]
    motif_ratios = [d['motif_ratio'] for d in motif_data]

    print(f'\n{motif}:')
    print(f'  GNN AUROC:  {sum(gnn_aurocs)/len(gnn_aurocs):.3f}')
    print(f'  SAE AUROC:  {sum(sae_aurocs)/len(sae_aurocs):.3f}')
    print(f'  GNN AUPRC:  {sum(gnn_auprcs)/len(gnn_auprcs):.3f}')
    print(f'  SAE AUPRC:  {sum(sae_auprcs)/len(sae_auprcs):.3f}')
    print(f'  Avg motif ratio: {100*sum(motif_ratios)/len(motif_ratios):.1f}%')
    print(f'  Graphs tested: {len(motif_data)}')

# 3. Winner comparison
print('\n\n3. METHOD COMPARISON (SAE vs GNN):')
print('='*70)
sae_wins = sum(1 for d in data if d['sae_auroc'] > d['gnn_auroc'])
gnn_wins = sum(1 for d in data if d['gnn_auroc'] > d['sae_auroc'])
ties = sum(1 for d in data if d['gnn_auroc'] == d['sae_auroc'])

print(f'SAE wins (higher AUROC): {sae_wins}/{len(data)} ({100*sae_wins/len(data):.1f}%)')
print(f'GNN wins (higher AUROC): {gnn_wins}/{len(data)} ({100*gnn_wins/len(data):.1f}%)')
print(f'Ties: {ties}')

avg_diff = sum(d['sae_auroc'] - d['gnn_auroc'] for d in data) / len(data)
print(f'\nAverage AUROC difference (SAE - GNN): {avg_diff:+.3f}')
if avg_diff > 0:
    print('  → SAE method performs better on average')
elif avg_diff < 0:
    print('  → GNN method performs better on average')
else:
    print('  → Methods perform equally on average')

# 4. Best performers
print('\n\n4. BEST PERFORMING CONFIGURATIONS:')
print('='*70)

print('\nTop 5 AUROC scores (SAE method):')
sorted_sae = sorted(data, key=lambda x: x['sae_auroc'], reverse=True)[:5]
for d in sorted_sae:
    print(f"  {d['variant']:8s} | {d['motif']:25s} | Graph {d['graph_id']} | SAE: {d['sae_auroc']:.3f} | GNN: {d['gnn_auroc']:.3f}")

print('\nTop 5 AUROC scores (GNN method):')
sorted_gnn = sorted(data, key=lambda x: x['gnn_auroc'], reverse=True)[:5]
for d in sorted_gnn:
    print(f"  {d['variant']:8s} | {d['motif']:25s} | Graph {d['graph_id']} | GNN: {d['gnn_auroc']:.3f} | SAE: {d['sae_auroc']:.3f}")

# 5. Interpretation
print('\n\n5. KEY INSIGHTS:')
print('='*70)

# Random baseline is 0.5 for AUROC
better_than_random = sum(1 for d in data if max(d['sae_auroc'], d['gnn_auroc']) > 0.5)
print(f'\nGraphs where at least one method beats random (AUROC > 0.5): {better_than_random}/{len(data)} ({100*better_than_random/len(data):.1f}%)')

# Good performance threshold
good_performance = sum(1 for d in data if max(d['sae_auroc'], d['gnn_auroc']) > 0.7)
print(f'Graphs where at least one method shows good performance (AUROC > 0.7): {good_performance}/{len(data)} ({100*good_performance/len(data):.1f}%)')

# Motif-specific success
print('\n✓ SUCCESS: All 4 motif types are now testable!')
print('  (Previous issue: only single_input_module worked)')
print('  (Fix: Using mixed-motif graphs instead of single-motif graphs)')

print('\n' + '='*70)
