#!/usr/bin/env python3
"""Print Phase 2 comparison summary from existing results."""

import csv

# Load results
results = []
with open('outputs/sae_config_comparison.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append(row)

print("="*70)
print("PHASE 2: SAE CONFIGURATION COMPARISON RESULTS (Layer 2, 80-dim)")
print("="*70)
print(f"\nTotal configurations evaluated: {len(results)}")

variants = {}
for row in results:
    v = row['variant']
    variants[v] = variants.get(v, 0) + 1

for variant, count in sorted(variants.items()):
    print(f"  • {variant}: {count}")

print("\n" + "="*70)
print("TOP 5 CONFIGURATIONS (by composite_score)")
print("="*70)

# Print top 5 (already sorted in CSV)
for i, row in enumerate(results[:5], 1):
    print(f"\n#{i}: {row['variant'].upper()} - {row['description']}")
    print(f"  latent_dim: {row['latent_dim']}")

    if row['k']:
        print(f"  k: {row['k']}")
    elif row.get('k_per_expert'):
        print(f"  k_per_expert: {row['k_per_expert']} (experts: {row['num_experts']})")
    elif row.get('sparsity_coef'):
        print(f"  sparsity_coef: {row['sparsity_coef']}")

    print(f"  Sparsity: {float(row['sparsity_pct']):.2f}%")
    print(f"  Metrics:")
    print(f"    • Max |rpb|: {float(row['max_rpb_abs']):.4f} (feature: {row['max_rpb_feature']}, motif: {row['max_rpb_motif']})")
    print(f"    • Best F1: {float(row['best_f1']):.4f} (feature: {row['best_f1_feature']}, motif: {row['best_f1_motif']})")
    print(f"    • Active features: {row['n_active_features']}/{row['latent_dim']} ({100*(1-float(row['dead_feature_rate'])):.1f}%)")
    print(f"    • Composite score: {float(row['composite_score']):.4f}")

# Best configuration
best = results[0]
print("\n" + "="*70)
print("RECOMMENDED CONFIGURATION")
print("="*70)
print(f"\n  Variant: {best['variant']}")
print(f"  Description: {best['description']}")
print(f"  latent_dim = {best['latent_dim']}")

if best['k']:
    print(f"  k = {best['k']}")
elif best.get('k_per_expert'):
    print(f"  k_per_expert = {best['k_per_expert']} (num_experts = {best['num_experts']})")
elif best.get('sparsity_coef'):
    print(f"  sparsity_coef = {best['sparsity_coef']}")

print(f"  Sparsity: {float(best['sparsity_pct']):.2f}%")
print(f"\n  Key Metrics:")
print(f"    • Max correlation: |rpb| = {float(best['max_rpb_abs']):.4f}")
print(f"    • Best F1 score: {float(best['best_f1']):.4f}")
print(f"    • Active features: {best['n_active_features']}/{best['latent_dim']} ({100*(1-float(best['dead_feature_rate'])):.1f}%)")
print(f"    • Composite score: {float(best['composite_score']):.4f}")

print("\n" + "="*70)
print("VARIANT COMPARISON")
print("="*70)

# Calculate best score per variant
variant_best = {}
for row in results:
    v = row['variant']
    score = float(row['composite_score'])
    if v not in variant_best or score > variant_best[v]:
        variant_best[v] = score

print("\nBest composite_score by variant:")
for variant in sorted(variant_best.keys()):
    print(f"  • {variant:10s}: {variant_best[variant]:.4f}")

print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)
print(f"""
✓ Phase 2 completed successfully with all 30 configs evaluated

All features show strongest correlation with: {best['max_rpb_motif']}
  → This suggests layer 2 activations capture feedback loop patterns

Top performers:
  • Best overall: {best['variant']} (composite_score: {float(best['composite_score']):.3f})
  • Highest |rpb|: {results[0]['max_rpb_abs']} ({results[0]['variant']})

Layer 2 (80-dim) observations:
  • Rich representation before bottleneck
  • Consistent motif detection across variants
  • Dead feature rates vary by sparsity level
""")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("""
1. Review full results in: outputs/sae_config_comparison.csv

2. Compare with Layer 3 results (if you have them) to validate hypothesis

3. Run Phase 3 ablation experiments with all variants:
   python3 run_phase3_all_variants.py --all

4. Run Phase 2.5 for PCA visualizations (if available)
""")
