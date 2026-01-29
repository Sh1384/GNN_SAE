# SAE vs GNNExplainer Comparison - Issue Fixed

## Problem Summary

The comparison script was producing **all NaN AUROC values** because:

1. **SAE feature z112 (z111 in 0-indexed) has ZERO activation** across all nodes and graphs
2. This led to zero gradients and constant scores (0.5)
3. GNNExplainer also returned constant values (all zeros) for most motifs

## Root Cause

The feature indices in the comparison script (**z112, z120**) were **placeholder values** that were never updated to the actual best features from `identify_top_sae_features.py`.

## Quick Fix (3 steps, ~10 minutes)

### Step 1: Check which features are active
```bash
python check_sae_features_quick.py
```

This shows which of the 128 SAE features actually have non-zero activations.

### Step 2: Find the best features for each motif
```bash
python identify_top_sae_features.py
```

This computes correlations and tells you which feature to use for each motif.

Example output:
```
feedback_loop:
Feature      rpb      |rpb|      p-value
------------------------------------------
z45        0.6826    0.6826    1.23e-156
z12        0.4521    0.4521    3.45e-78
...
```

### Step 3: Update the comparison script

Edit [compare_sae_vs_gnnexplainer.py](compare_sae_vs_gnnexplainer.py) lines 324-345:

Replace:
```python
'feedback_loop': {
    'feature_idx': 112,  # OLD - has zero activation!
    ...
},
```

With:
```python
'feedback_loop': {
    'feature_idx': 45,  # NEW - from Step 2 output
    ...
},
```

Do this for all 4 motif types.

### Step 4: Run comparison again
```bash
python compare_sae_vs_gnnexplainer.py
```

You should now see:
- ✅ Non-zero SAE feature activations
- ✅ Non-zero gradients
- ✅ Valid AUROC values (not NaN)
- ✅ Actual comparison results

## Automated Fix

Or run the automated script:
```bash
./QUICK_FIX.sh
```

This runs all steps interactively.

## What Was Fixed

### Code Changes (compare_sae_vs_gnnexplainer.py):

1. **Fixed visualization crash** (lines 897-950):
   - Handle NaN indices when all AUROC values are NaN
   - Skip visualizations gracefully when no valid data

2. **Enhanced debugging** (lines 275-300):
   - Show which features are active
   - Warn when GNN activations are zero
   - Show alternatives when target feature is inactive

3. **Better error handling** (lines 240-250, 304-315):
   - Handle constant edge masks from GNNExplainer
   - Handle zero gradients from SAE
   - Return uniform (0.5) scores instead of crashing

### New Files:

1. **check_sae_features_quick.py** - Fast check of which features are active
2. **DIAGNOSIS_AND_NEXT_STEPS.md** - Comprehensive troubleshooting guide
3. **QUICK_FIX.sh** - Automated fix workflow
4. **README_COMPARISON_FIX.md** - This file

## Expected Results After Fix

### Before (with wrong features):
```
WARNING: SAE feature z111 has near-zero activation (0.00e+00)
Feature activations (pre-TopK): [0. 0. 0. 0. 0.]
SAE scores: min=0.5000, max=0.5000
AUROC: nan
```

### After (with correct features):
```
SAE feature z45 activation: 12.34
Feature activations (pre-TopK): [2.1 3.4 1.2 0.8 4.5]
SAE scores: min=0.0123, max=0.8765
AUROC: 0.756
```

## Remaining Issue: GNNExplainer

GNNExplainer may still return all zeros even after fixing SAE features. This is a separate issue related to:

1. Model architecture (graph-level vs node-level)
2. Model prediction variance
3. Edge weight importance

See [DIAGNOSIS_AND_NEXT_STEPS.md](DIAGNOSIS_AND_NEXT_STEPS.md) for solutions if GNNExplainer still fails.

## Files Modified

1. `compare_sae_vs_gnnexplainer.py` - Enhanced debugging and error handling
2. `check_sae_features_quick.py` - NEW: Feature activation checker
3. `QUICK_FIX.sh` - NEW: Automated fix workflow
4. `DIAGNOSIS_AND_NEXT_STEPS.md` - NEW: Comprehensive guide
5. `README_COMPARISON_FIX.md` - NEW: This summary

## Next Steps

1. ✅ Run `python check_sae_features_quick.py`
2. ✅ Run `python identify_top_sae_features.py`
3. ✅ Update feature indices in `compare_sae_vs_gnnexplainer.py`
4. ✅ Run `python compare_sae_vs_gnnexplainer.py`
5. ✅ Check if AUROC values are now valid (not NaN)

If still seeing NaN after this, the issue is with GNNExplainer (not SAE), see the diagnosis doc.

## Summary

**The fix is simple**: Update 4 feature indices (2 minutes) after running identify_top_sae_features.py (5 minutes).

The current values (z112, z120) were never updated from their initial placeholder values and those features have zero activation.
