# Diagnosis: Zero Activations and NaN AUROC

## Current Status

The comparison script runs but produces **all NaN AUROC values** because both explanation methods return constant scores.

## Root Causes

### 1. SAE Feature z111 (z112 in 1-indexed) has ZERO activation

**Evidence**:
```
WARNING: SAE feature z111 has near-zero activation (0.00e+00)
Feature activations (pre-TopK): [0. 0. 0. 0. 0.]
```

**What this means**:
- ALL nodes in ALL graphs have exactly zero activation for this feature
- Even the pre-TopK (dense) activations are zero
- This feature was either never learned or is permanently inactive

**Why this causes NaN AUROC**:
```
Zero activation → Zero gradients → Constant SAE scores (0.5) → NaN AUROC
```

### 2. GNNExplainer returns all zeros for most motifs

**Evidence**:
```
WARNING: GNNExplainer edge_mask is constant (std=0.00e+00)
Values: min=0.0000, max=0.0000, mean=0.0000
```

**Why**:
- GNN model's predictions are likely constant across nodes
- Model doesn't differentiate based on local structure
- GNNExplainer needs prediction variance to assign edge importance

**Result**:
```
All zeros → Normalized to 0.5 (uniform) → NaN AUROC
```

## Immediate Fix: Find Active Features

The current feature indices (z112, z120) were likely **never updated** after initial setup. They're placeholder values.

### Step 1: Check which features are actually active

```bash
cd /data/users/goodarzilab/shervin/182-GNN_SAE
python check_sae_features_quick.py
```

This will show:
- How many features have non-zero activation
- Which features are most active
- Whether z112 and z120 are active at all

### Step 2: Identify correct feature-motif mapping

```bash
python identify_top_sae_features.py
```

This will:
- Compute point-biserial correlations between each SAE feature and each motif
- Show the top 5 features for each motif type
- Output the feature indices to use

**Expected output**:
```
feedback_loop:
Feature      rpb      |rpb|      p-value
------------------------------------------
z45        0.6826    0.6826    1.23e-156  ← Use this, not z112!
z12        0.4521    0.4521    3.45e-78
...
```

### Step 3: Update comparison script with correct features

Edit [compare_sae_vs_gnnexplainer.py](compare_sae_vs_gnnexplainer.py) line 324-345:

```python
targets = {
    'feedforward_loop': {
        'feature_idx': 45,  # UPDATE from identify_top_sae_features.py
        'motif_type': 'feedforward_loop',
        'description': 'Feedforward Loop detector'
    },
    'feedback_loop': {
        'feature_idx': 12,  # UPDATE from identify_top_sae_features.py
        'motif_type': 'feedback_loop',
        'description': 'Feedback Loop detector'
    },
    # ... etc
}
```

### Step 4: Re-run comparison

```bash
python compare_sae_vs_gnnexplainer.py
```

You should now see:
- **Non-zero SAE activations**
- **Non-zero gradients**
- **Non-constant SAE scores**
- **Valid AUROC values** (not NaN)

## Deeper Issue: GNNExplainer Still Returning Zeros

Even with correct SAE features, GNNExplainer may still return all zeros. This suggests:

### Possible causes:

1. **Model trained for graph-level task**
   - Your GNN might make graph-level predictions (one output per graph)
   - GNNExplainer expects node-level predictions
   - Check `gnn_train_copy.py` to see if model output is `[num_graphs, 1]` vs `[num_nodes, 1]`

2. **Model predictions are constant**
   - Test: Check if model outputs vary across nodes
   ```python
   with torch.no_grad():
       preds = gnn(data)
       print(f"Prediction variance: {preds.std().item()}")
   ```
   - If std ~ 0, model makes same prediction for all nodes

3. **Edge weights not used in prediction**
   - If model ignores edge_attr, GNNExplainer can't attribute importance
   - GNNExplainer perturbs edge weights to measure importance

### Solutions for GNNExplainer:

**Option A**: Use graph-level GNNExplainer
```python
explainer = Explainer(
    model=gnn,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    task_level='graph',  # Change from 'node' to 'graph'
    return_type='raw'
)
```

**Option B**: Focus on SAE-only comparison
- Skip GNNExplainer entirely
- Compare SAE gradient saliency against other baselines:
  - Random edge importance
  - Degree-based importance
  - GradCAM on GNN activations

**Option C**: Use a different baseline
- Try Integrated Gradients
- Try GradCAM
- Try attention-based explanations

## What to Do Now

### Immediate (Required):
1. ✅ Run `check_sae_features_quick.py` to verify z112/z120 are inactive
2. ✅ Run `identify_top_sae_features.py` to get correct feature indices
3. ✅ Update `compare_sae_vs_gnnexplainer.py` with new feature indices
4. ✅ Re-run comparison

### After getting correct features:

If SAE now works but GNNExplainer still fails:

1. **Debug GNN predictions**:
   ```python
   # Add to comparison script
   with torch.no_grad():
       preds = gnn(data)
       print(f"GNN predictions: min={preds.min():.4f}, max={preds.max():.4f}, std={preds.std():.4f}")
   ```

2. **Try graph-level GNNExplainer** (if model is graph-level)

3. **Consider alternative baselines** (if GNNExplainer fundamentally incompatible)

## Expected Timeline

- **5 minutes**: Run feature check scripts
- **5 minutes**: Update feature indices in comparison script
- **3 minutes**: Re-run comparison
- **Result**: Should see non-NaN AUROC for SAE

If GNNExplainer still fails after this:
- **30 minutes**: Debug GNN prediction behavior
- **1 hour**: Implement alternative baseline

## Success Criteria

### Minimum Success:
- ✅ SAE gradients are non-zero
- ✅ SAE AUROC is valid (not NaN)
- ✅ Can compare SAE against random/degree baseline

### Full Success:
- ✅ Above +
- ✅ GNNExplainer works and produces non-constant scores
- ✅ Statistical comparison between SAE and GNNExplainer

## Files to Check

1. **identify_top_sae_features.py** - Get correct feature indices
2. **compare_sae_vs_gnnexplainer.py** - Update line 324-345 with new indices
3. **gnn_train_copy.py** - Check if model is graph-level or node-level
4. **check_sae_features_quick.py** - Quick diagnostic of feature activation

## Questions to Answer

1. ✅ Are z112 and z120 actually active features? → Run check_sae_features_quick.py
2. ✅ What are the correct feature indices? → Run identify_top_sae_features.py
3. ❓ Is the GNN model graph-level or node-level? → Check gnn_train_copy.py
4. ❓ Do GNN predictions vary across nodes? → Add debug logging

Let me know the output of steps 1-2 and we can proceed from there!
