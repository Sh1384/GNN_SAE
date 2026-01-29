# GNNExplainer Comparison - Complete Fix Summary

## Problem Identified

Your comparison was only finding testable graphs for `single_input_module` because:

1. **Test set uses single-motif graphs** (IDs 1-3997)
   - feedback_loop graphs: 100% of edges are bidirectional → no non-motif edges
   - feedforward_loop graphs: 100% of edges are triangles → no non-motif edges
   - cascade graphs: 100% of edges are chains → no non-motif edges
   - single_input_module graphs: Hub edges (motif) + other edges (non-motif) ✓

2. **AUROC requires "mixed edges"**
   - Needs BOTH motif edges (positive samples) AND non-motif edges (negative samples)
   - Single-motif graphs with 100% coverage can't be evaluated with AUROC

## Solution Applied

**Now uses your mixed-motif graphs (IDs 4000-4999) by default** ✓

Mixed-motif graphs naturally have "mixed edges" for ALL motif types:
- Each graph contains 2-3 different motif types
- Plus random interconnections between motifs
- So for any specific motif, some edges are "in motif" and some are "not in motif"

Example mixed-motif graph:
```
Graph 4000:
  - 2 edges form a feedback loop (motif for feedback_loop detection)
  - 3 edges form a feedforward loop (motif for feedforward_loop detection)
  - 4 edges form a cascade (motif for cascade detection)
  - 3 random edges connecting motifs (non-motif for all detections)

For feedback_loop detection:
  - Positive samples: 2 feedback edges
  - Negative samples: 10 other edges
  - AUROC can be calculated ✓
```

## Changes Made

### 1. Updated test graph loading ([sae/compare_sae_vs_gnnexplainer.py:346](sae/compare_sae_vs_gnnexplainer.py#L346))

**Before**:
```python
def load_test_graphs_with_features(max_graphs: int = 200):
    # Load test graph IDs from test_graph_ids.json
    with open('outputs/test_graph_ids.json', 'r') as f:
        test_graph_ids = json.load(f)['graph_ids']
```

**After**:
```python
def load_test_graphs_with_features(max_graphs: int = 200, use_mixed_motif: bool = True):
    if use_mixed_motif:
        # Use mixed-motif graphs (IDs 4000-4999)
        test_graph_ids = list(range(4000, 4000 + max_graphs))
    else:
        # Use original test set
        with open('outputs/test_graph_ids.json', 'r') as f:
            test_graph_ids = json.load(f)['graph_ids']
```

### 2. Set default to use mixed-motif graphs ([sae/compare_sae_vs_gnnexplainer.py:658](sae/compare_sae_vs_gnnexplainer.py#L658))

```python
test_graphs = load_test_graphs_with_features(
    max_graphs=200,
    min_motif_ratio=0.2,
    max_motif_ratio=0.8,
    use_mixed_motif=True  # Now uses mixed-motif graphs by default
)
```

## Expected Results After Fix

**All 4 motif types should now be testable:**
- ✓ feedback_loop: Should find ~20 graphs with mixed edges
- ✓ feedforward_loop: Should find ~20 graphs with mixed edges
- ✓ single_input_module: Should find ~20 graphs with mixed edges
- ✓ cascade: Should find ~20 graphs with mixed edges

**Performance metrics:**
- AUROC > 0.5: Better than random
- AUROC ≈ 0.7-0.8: Good edge localization
- Compare SAE vs GNN: See which method is better at identifying motif edges

## How to Run

```bash
# Run comparison for all SAE variants (RECOMMENDED)
python3 sae/compare_sae_vs_gnnexplainer.py --all

# Or run single variant
python3 sae/compare_sae_vs_gnnexplainer.py --variant topk
```

## Output Files

- `outputs/gnnexplainer_comparison/comparison_all_variants.csv`
  - Detailed results for each graph and motif
  - Columns: variant, motif, feature_idx, graph_id, gnn_auroc, sae_auroc, etc.

## What Changed from Original Codebase

**Summary of ALL fixes:**

1. ✅ **Layer 2 (80-dim) activations** - Fixed dimension mismatch
2. ✅ **Phase 2 feature loading** - Fixed to use correct correlation file
3. ✅ **Mixed-motif graph usage** - Now uses graphs with testable "mixed edges"

All 3 issues are now resolved. The comparison is ready to run!
