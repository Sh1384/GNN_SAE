# Debugging Status: SAE vs GNNExplainer Comparison

## Problem Summary
The comparison script `compare_sae_vs_gnnexplainer.py` runs but returns an empty DataFrame, causing a `KeyError: 'target'` when trying to group results.

## Root Cause Analysis

The issue is that `run_comparison()` is returning zero results, which means:
1. Either no graphs contain the target motifs (ground truth detection finds nothing)
2. Or all comparisons are failing with exceptions
3. Or there's a model compatibility issue

## Changes Made

### 1. Fixed Checkpoint Loading (Already Done)
- **File**: `compare_sae_vs_gnnexplainer.py` line 655-659
- **Fix**: Added conditional logic to handle both checkpoint formats:
  ```python
  if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
      gnn.load_state_dict(checkpoint['model_state_dict'])
  else:
      gnn.load_state_dict(checkpoint)  # State dict directly
  ```

### 2. Added Extensive Debug Logging
- **File**: `compare_sae_vs_gnnexplainer.py`
- **Changes**:
  - Line ~360: Debug output for ground truth detection loop
  - Line ~380: Debug output for first few graphs in comparison loop
  - Line ~457: Track successful comparisons and show first few results
  - Line ~468: Show total results collected after each target
  - Line ~471: Warning if no results returned
  - Line ~708: Check DataFrame before aggregation

### 3. Created Test Scripts
- **`debug_ground_truth.py`**: Test if ground truth motif detection works (needs Python env)
- **`test_basic_loading.py`**: Verify basic data loading works (simpler test)

## Diagnostic Steps to Run

### Step 1: Verify Basic Loading
```bash
cd /data/users/goodarzilab/shervin/182-GNN_SAE
python test_basic_loading.py
```

This checks:
- Test graph IDs load
- Graph files can be opened
- Checkpoint files exist and can be loaded
- Model architecture dimensions

### Step 2: Run Comparison with Debug Output
```bash
python compare_sae_vs_gnnexplainer.py
```

Look for these DEBUG messages:
```
DEBUG: Checking 100 test graphs for feedback_loop...
  Graph 0 (id=X): N motif edges found      <- Are motifs being detected?
DEBUG: Starting comparison on M graphs...
  DEBUG: Graph 0 - motif_node=X, N motif edges    <- Are comparisons running?
  ✓ Result added (AUROC: GNN=X.XXX, SAE=Y.YYY)   <- Are results collected?
DEBUG: Completed N successful comparisons        <- How many succeeded?
DEBUG: Total results collected so far: N         <- Final count
```

### Step 3: Analyze the Output

**Case A: No motifs detected**
```
DEBUG: Checking 100 test graphs for feedback_loop...
Found 0 graphs with feedback_loop
```
**Meaning**: Ground truth detection logic doesn't match the actual graph structure

**Solution**:
1. Check if graphs actually contain these motifs
2. Verify ground truth logic in `get_ground_truth_edge_mask()` (line 88-197)
3. Look at metadata files to see which motifs are labeled
4. May need to adjust motif detection patterns

**Case B: Motifs detected but comparisons fail**
```
Found 20 graphs with feedback_loop
DEBUG: Starting comparison on 20 graphs...
  ERROR on graph 0: [error message]
  ERROR on graph 1: [error message]
DEBUG: Completed 0 successful comparisons
```
**Meaning**: Comparison loop is hitting errors

**Common errors**:
- Model architecture mismatch (hidden_dim)
- Edge attributes missing or wrong shape
- GNNExplainer incompatibility
- SAE feature index out of bounds

**Case C: Model architecture mismatch**
```
RuntimeError: Error(s) in loading state_dict for GCNModel:
  size mismatch for conv1.weight: copying a param with shape...
```
**Meaning**: Trained model has different architecture than code

**Solution**: Check `gnn_train_copy.py` for actual trained architecture:
```bash
grep "hidden_dim" gnn_train_copy.py
```
Then update line 652 in `compare_sae_vs_gnnexplainer.py` to match.

## Known Issues to Check

### Issue 1: Hidden Dimension Mismatch
- **Script uses**: `hidden_dim=80` (line 652)
- **Trained model**: Need to verify from `gnn_train_copy.py`
- **Impact**: Would cause immediate loading error

### Issue 2: SAE Feature Indices
Current configuration (line 324-345):
```python
'feedforward_loop': feature_idx=112
'feedback_loop': feature_idx=112  # Same as feedforward!
'single_input_module': feature_idx=112  # Same as feedforward!
'cascade': feature_idx=120
```

**Problem**: Three motifs use the same feature (z112). This might be intentional if one feature correlates with multiple motifs, but needs verification.

**Action**: Run `identify_top_sae_features.py` to get proper feature assignments:
```bash
python identify_top_sae_features.py
```

### Issue 3: Layer 3 Activations
- SAE expects 64-dim input (layer 3 activations)
- GNN architecture: layer3 output is 64-dim (line 50: `GCNConv(hidden_dim, 64)`)
- This should be correct, but verify activations were saved correctly

## Expected Debug Output (Successful Run)

```
LOADING MODELS
✓ GNN loaded
✓ SAE loaded

LOADING TEST DATA
Loaded 100 test graphs

RUNNING COMPARISON EXPERIMENT

Analyzing Target: feedback_loop
SAE Feature: z112
Motif Type: feedback_loop

DEBUG: Checking 100 test graphs for feedback_loop...
  Graph 0 (id=5): 4 motif edges found
  Graph 1 (id=12): 0 motif edges found
  ...
Found 20 graphs with feedback_loop
DEBUG: Starting comparison on 20 graphs...
  DEBUG: Graph 0 - motif_node=3, 4 motif edges
    GNNExplainer scores: min=0.0234, max=0.8765
    SAE scores: min=0.0123, max=0.9234
    ✓ Result added (AUROC: GNN=0.723, SAE=0.856)
  DEBUG: Graph 1 - motif_node=2, 6 motif edges
    ...
DEBUG: Completed 20 successful comparisons for feedback_loop
DEBUG: Total results collected so far: 20

[Repeat for other targets...]

DEBUG: run_comparison() returning 80 total results

QUANTITATIVE RESULTS
DEBUG: df_results shape: (80, 16)
DEBUG: df_results columns: ['target', 'motif_type', ...]
...
```

## Next Steps

1. **Run test_basic_loading.py** to verify data access
2. **Run compare_sae_vs_gnnexplainer.py** with debug output
3. **Analyze the DEBUG messages** to identify failure point:
   - No motifs detected → Fix ground truth logic
   - Motifs detected but errors → Fix model compatibility
   - Comparisons succeed but wrong indices → Run identify_top_sae_features.py
4. **Update feature indices** based on identify_top_sae_features.py output
5. **Re-run comparison** and verify results

## Files Modified

1. `compare_sae_vs_gnnexplainer.py` - Added extensive debug logging
2. `debug_ground_truth.py` - Created standalone test for motif detection
3. `test_basic_loading.py` - Created basic loading verification
4. `DEBUG_STATUS.md` - This file

## Questions to Answer

From the debug output, determine:

1. **Are test graphs loading?** (Should see "Loaded N test graphs")
2. **Are motifs being detected?** (Should see "Found N graphs with [motif_type]")
3. **Are comparisons running?** (Should see individual graph debug messages)
4. **Are results being collected?** (Should see "✓ Result added")
5. **What is the final count?** (Should see "returning N total results")

Once you have this debug output, the failure point will be clear and we can fix it.
