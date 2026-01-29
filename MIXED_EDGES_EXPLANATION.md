# Mixed Edges vs Mixed Graphs - Explanation

## The Confusion

**"Mixed Edges"** (what the comparison code needs):
- A graph where SOME edges are part of a specific motif and SOME edges are NOT
- Example: Graph has 10 edges total
  - 3 edges are feedback_loop edges (bidirectional)
  - 7 edges are NOT feedback_loop edges
- This gives: positive samples (motif edges) + negative samples (non-motif edges)

**"Mixed Graphs"** (what you generated):
- A graph that contains MULTIPLE DIFFERENT MOTIF TYPES
- Example: One graph contains:
  - A feedback loop (2 bidirectional edges)
  - A feedforward loop (3 edges forming a triangle)
  - A cascade (4 edges in a chain)
  - Random interconnections between motifs
- Generated via `generate_mixed_motif_graph()` in your generator

## Why AUROC Requires "Mixed Edges"

**AUROC (Area Under ROC Curve)** is a binary classification metric:
- Requires: **Positive class** (edges in motif) AND **Negative class** (edges not in motif)
- If ALL edges are motif edges (100%), AUROC is undefined (no negatives)
- If NO edges are motif edges (0%), AUROC is undefined (no positives)

**Example of why it fails:**
```python
# Graph with ONLY motif edges
gt_mask = [1, 1, 1, 1, 1]  # All edges are motif edges
scores = [0.8, 0.7, 0.9, 0.6, 0.5]

# roc_auc_score fails!
# ValueError: Only one class present in y_true
```

## The Actual Problem

Your test set uses **single-motif graphs only**:

```python
# From graph generation:
# Single-motif graphs: IDs 0-3999 (1000 per motif type)
# Mixed-motif graphs: IDs 4000-4999 (1000 mixed)

# Your test set:
# test_graph_ids.json: IDs 1-3997
# ↓ These are ALL single-motif graphs!
```

**Why only single_input_module works:**
- Single-input-module graphs: Hub with 3+ outgoing edges (motif) + other random edges (non-motif) = Mixed edges ✓
- Feedback_loop graphs: Only bidirectional edges, no other edges = 100% motif coverage = No mixed edges ✗
- Feedforward_loop graphs: Only triangle edges, no other edges = 100% motif coverage = No mixed edges ✗
- Cascade graphs: Only chain edges, no other edges = 100% motif coverage = No mixed edges ✗

## The Solution

**Option 1: Use your mixed-motif graphs (RECOMMENDED)**
- Mixed-motif graphs naturally have "mixed edges" for each motif
- Example: A graph with feedback_loop + feedforward_loop + random edges
  - For feedback_loop detection: 2 edges are feedback (motif), 8 are not (non-motif) ✓
  - For feedforward_loop detection: 3 edges are feedforward (motif), 7 are not (non-motif) ✓

**Option 2: Generate single-motif graphs with noise edges**
- Add random edges to single-motif graphs
- Ensures each graph has both motif and non-motif edges

**Option 3: Remove AUROC requirement (NOT RECOMMENDED)**
- Only use AUPRC metric
- But AUROC is the standard metric for this type of evaluation

## Recommended Fix

Update the comparison to use mixed-motif graphs from your test set:
- Change test IDs to include mixed-motif graphs (IDs 4000+)
- Or create a new test split that includes mixed-motif graphs
