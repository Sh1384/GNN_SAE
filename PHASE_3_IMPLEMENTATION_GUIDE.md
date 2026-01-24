# Phase 3 Implementation Guide: Single Best Config Architecture

## Summary of Changes

This document outlines all changes made to implement the corrected Phase 3 architecture:
- **Phase 3a**: Selects best config by max_rpb_abs (not composite_score), includes variant in filenames
- **Phase 3b**: Reads Phase 3a metadata, uses correct variant and parameters
- **Phase 3c**: Loads results for single best variant only (not all 4 variants)

---

## Python Scripts - COMPLETED ✅

### 1. `run_interpretability_experiments.py` ✅

**Modified to:**
- Accept `--variant` parameter
- Include variant in directory names
- Include variant in experiment names (output filenames)
- Pass variant to subprocess calls

**New Output Format:**
```
ablations/results/feedback_loop_topk_l512_k8_results.csv
ablations/results/cascade_topk_l512_k8_results.csv
ablations/results/feedforward_loop_topk_l512_k8_results.csv
ablations/results/single_input_module_topk_l512_k8_results.csv
```

---

### 2. `compare_ablation_strategies.py` ✅

**Modified to:**
- `load_sae_motif_results()` accepts optional `variant` parameter
- Tries new format with variant first, falls back to old format
- Passes variant when loading in motif-mode

**Backward Compatible:** Works with both old (no variant) and new (with variant) filenames

---

## Notebook Updates - MANUAL REQUIRED ⚠️

### Phase 3a Cell - Select Best Config

**Current Implementation (WRONG):**
```python
best = df.iloc[0]  # ❌ First row (sorted by composite_score)
```

**Required Implementation:**
```python
# Select best config by MAX_RPB_ABS (most directly relevant metric)
best_idx = df['max_rpb_abs'].idxmax()
best = df.loc[best_idx]

best_variant = best['variant']
best_latent = int(best['latent_dim'])

# Extract variant-specific parameters
if best_variant == 'topk':
    best_k = int(best['k'])
elif best_variant == 'gated':
    best_sparsity_coef = float(best['sparsity_coef'])
elif best_variant == 'jumprelu':
    best_threshold = float(best['threshold_init'])
elif best_variant == 'switch':
    best_num_experts = int(best['num_experts'])
    best_latent_per_expert = int(best['latent_per_expert'])
    best_k_per_expert = int(best['k_per_expert'])

# Save metadata for Phase 3b/3c
import json
metadata = {
    'phase': '3a',
    'selected_by': 'max_rpb_abs',
    'best_config': {
        'variant': best_variant,
        'latent_dim': best_latent,
        'max_rpb_abs': float(best['max_rpb_abs']),
        # Include variant-specific params...
    }
}
Path('ablations').mkdir(exist_ok=True)
with open('ablations/phase_3a_config.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Pass --variant to run_interpretability_experiments.py
result = subprocess.run([
    sys.executable, 'run_interpretability_experiments.py',
    '--variant', best_variant,
    '--latent_dim', str(best_latent),
    '--min_rpb', '0.05',
    '--n_random_trials', '20'
], capture_output=False)
```

**Key Points:**
- Uses `max_rpb_abs` for selection (most directly relevant)
- Extracts ALL variant-specific parameters (k, sparsity_coef, threshold_init, num_experts, etc.)
- Saves metadata JSON for Phase 3b/3c to use
- Passes `--variant` to run_interpretability_experiments.py

---

### Phase 3b Cell - Read Metadata, Use Correct Parameters

**Current Implementation (WRONG):**
```python
best = df.iloc[0]  # ❌ Re-selects independently from Phase 3a
```

**Required Implementation:**
```python
import json

# Load Phase 3a metadata (ensures same config as 3a)
metadata_file = Path('ablations/phase_3a_config.json')
if not metadata_file.exists():
    print("ERROR: Phase 3a metadata not found. Run Phase 3a first.")
    sys.exit(1)

with open(metadata_file, 'r') as f:
    metadata = json.load(f)

best_config = metadata['best_config']
best_variant = best_config['variant']
best_latent = best_config['latent_dim']

# Extract variant-specific params from metadata
if best_variant == 'topk':
    best_k = int(best_config['k'])
elif best_variant == 'gated':
    best_k = int(best_config.get('k', 8))
elif best_variant == 'jumprelu':
    best_k = 8
elif best_variant == 'switch':
    best_k = int(best_config['k_per_expert'])

# Run for EACH motif with SAME variant and parameters
motifs = ['in_feedback_loop', 'in_cascade', 'in_feedforward_loop', 'in_single_input_module']
for motif in motifs:
    result = subprocess.run([
        sys.executable, 'native_gnn_ablation.py',
        '--variant', best_variant,
        '--latent_dim', str(best_latent),
        '--use-rpb',
        '--motif', motif
    ], capture_output=False)
```

**Key Points:**
- Reads metadata from Phase 3a (same config guaranteed)
- Extracts variant-specific parameters from metadata
- Passes `--variant` to native_gnn_ablation.py
- Runs exactly 4 motif-specific calls

---

### Phase 3c Cell - Load Single Variant Results Only

**Current Implementation (WRONG):**
```python
result = subprocess.run([
    script,
    '--all-variants',      # ❌ Tries all 4 variants
    '--motif-mode'
], capture_output=False)
```

**Required Implementation:**
```python
import json

# Load best variant from Phase 3a metadata
metadata_file = Path('ablations/phase_3a_config.json')
if not metadata_file.exists():
    print("ERROR: Phase 3a metadata not found. Run Phase 3a first.")
    sys.exit(1)

with open(metadata_file, 'r') as f:
    metadata = json.load(f)

best_config = metadata['best_config']
best_variant = best_config['variant']
best_latent = best_config['latent_dim']

# Run SINGLE variant (not --all-variants)
result = subprocess.run([
    script,
    '--variant', best_variant,    # ✅ Single variant
    '--latent_dim', str(best_latent),
    '--motif-mode'                # Motif-grouped comparison
], capture_output=False)
```

**Key Points:**
- Reads metadata to determine which variant to use
- Passes `--variant` (single) instead of `--all-variants`
- Uses `--motif-mode` for grouped analysis
- Loads Phase 3a/3b results for that one variant

---

## File Structure After Phase 3

```
ablations/
├── phase_3a_config.json              ← ✅ NEW: Metadata file
├── results/
│   ├── feedback_loop_topk_l512_k8_results.csv         ← ✅ NEW: variant in filename
│   ├── cascade_topk_l512_k8_results.csv               ← ✅ NEW: variant in filename
│   ├── feedforward_loop_topk_l512_k8_results.csv      ← ✅ NEW: variant in filename
│   └── single_input_module_topk_l512_k8_results.csv   ← ✅ NEW: variant in filename
├── interpretability_topk_l512_k8_rpb0.05_results/
│   ├── motif_specific_results.csv
│   ├── statistical_tests.csv
│   └── feature_motif_mapping.json
└── interpretability_topk_l512_k8_rpb0.05_plots/
    └── interpretability_vs_random_controls.png

outputs/
├── native_gnn_ablations/
│   ├── native_ablation_topk_rpb_in_feedback_loop.csv
│   ├── native_ablation_topk_rpb_in_cascade.csv
│   ├── native_ablation_topk_rpb_in_feedforward_loop.csv
│   └── native_ablation_topk_rpb_in_single_input_module.csv
└── ablation_strategy_comparison/
    ├── motif_agreement_summary.csv
    └── comparison_plots/
        └── strategy_comparison_*.png
```

---

## Execution Flow

1. **Phase 3a**:
   - Loads Phase 2 CSV
   - Selects best by max_rpb_abs
   - Extracts variant-specific params
   - Saves metadata JSON
   - Runs run_interpretability_experiments.py with --variant

2. **Phase 3b**:
   - Loads Phase 3a metadata JSON
   - Extracts variant and params
   - Runs native_gnn_ablation.py 4 times (once per motif)
   - All 4 runs use SAME variant/params

3. **Phase 3c**:
   - Loads Phase 3a metadata JSON
   - Extracts variant
   - Runs compare_ablation_strategies.py with --variant (single)
   - Compares Phase 3a vs Phase 3b for that variant

---

## Design Rationale

### Why Select by max_rpb_abs?
- Directly measures feature-motif correlation strength
- Most relevant metric for Phase 3 (mechanistic interpretability)
- Different from Phase 2 composite_score (intentional - different goals)

### Why Include Variant in Filenames?
- Self-documents which variant produced each result
- Enables Phase 3c to load variant-specific files
- Future-proofs if architecture changes

### Why Metadata JSON?
- Single source of truth across Phase 3a/3b/3c
- Prevents inconsistencies (what if phases select different best configs?)
- Enables reproducibility and debugging

### Why Single Variant, Not All 4?
- Focuses interpretability analysis on best overall performer
- Avoids 4×4=16 comparisons
- Multi-variant robustness testing is Phase 4's job
- Cleaner narrative: "We found feature X causes Y via mechanism Z" (singular best, not multiple options)

---

## Backward Compatibility

If you have Phase 3 results from before these changes:

1. The code is backward compatible - it tries new format (with variant) first, then falls back to old format
2. Phase 3b and 3c can work with old filenames if no variant in name
3. However, to use full functionality, you should rename files to include variant:

```bash
# Example: update old Phase 3a files
mv ablations/results/feedback_loop_l512_k8_results.csv \
   ablations/results/feedback_loop_topk_l512_k8_results.csv

# Create metadata file
cat > ablations/phase_3a_config.json << 'EOF'
{
  "phase": "3a",
  "selected_by": "max_rpb_abs",
  "best_config": {
    "variant": "topk",
    "latent_dim": 512,
    "k": 8
  }
}
EOF
```

---

## Code References

**Complete Implementation:**
- [run_interpretability_experiments.py](run_interpretability_experiments.py) - Python script with --variant support ✅
- [compare_ablation_strategies.py](compare_ablation_strategies.py) - Updated load_sae_motif_results() ✅
- [update_notebook_cells.py](update_notebook_cells.py) - Reference code for notebook cells (see above)

**Architecture Documentation:**
- [MULTI_SEED_ARCHITECTURE.md](MULTI_SEED_ARCHITECTURE.md) - Phase 1/2/2b/3/4 seed strategy
- [INTERPRETABILITY_PIPELINE_GUIDE.md](INTERPRETABILITY_PIPELINE_GUIDE.md) - Complete pipeline overview

---

## Next Steps

1. ✅ Update run_interpretability_experiments.py
2. ✅ Update compare_ablation_strategies.py
3. ⚠️ **Manually update sae_colab_pipeline.ipynb notebook cells** (see code above)
4. Test Phase 3a/3b/3c with actual data
5. Verify metadata JSON is created and used correctly
6. Check output filenames include variant
7. Validate Phase 3c loads single variant results

---

## Troubleshooting

**Phase 3b fails with "Metadata not found"**
→ Run Phase 3a first to generate `ablations/phase_3a_config.json`

**Phase 3c loads wrong variant results**
→ Check that `phase_3a_config.json` exists and contains correct variant

**Filenames don't include variant**
→ Ensure Phase 3a passes `--variant` parameter to run_interpretability_experiments.py

**Phase 3b uses different config than Phase 3a**
→ Phase 3b must read metadata.json, not re-select from CSV
