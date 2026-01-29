# Fixes Applied to SAE vs GNNExplainer Comparison

## Issues Found and Fixed

### 1. Model Signature Incompatibility (FIXED ✓)
**Problem**: GNNExplainer calls models with `(x, edge_index, edge_attr)` but GCNModel expected a Data object.

**Error**: `TypeError: GCNModel.forward() got an unexpected keyword argument 'edge_attr'`

**Fix**: Updated `forward()` and `get_intermediate_activations()` to accept both calling conventions:
```python
def forward(self, x, edge_index, edge_attr=None):
    # Handle both Data objects and individual arguments
    if isinstance(x, Data):
        data = x
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    # ... rest of forward pass
```

### 2. SAE TopK Sparsity Causing Zero Gradients (FIXED ✓)
**Problem**: SAE uses TopK sparsity (k=16), meaning only top 16 features are non-zero. If target feature (z112, z120) is not in top 16, its activation is zero → zero gradients.

**Symptoms**:
- SAE scores: min=0.0000, max=0.0000 (all zeros)
- AUROC = NaN (can't compute ROC AUC with constant predictions)

**Fix**: Use **pre-TopK activations** instead of post-TopK:
```python
# Before: z_latent = sae.encode(h_bottleneck)  # Post-TopK, sparse
# After:
z_pre_topk = sae.encoder(h_bottleneck)
z_pre_topk = F.relu(z_pre_topk)  # Pre-TopK, dense
```

This allows gradients to flow even if the feature isn't in the top K.

### 3. GNNExplainer Returning Constant Values (PARTIALLY FIXED ⚠)
**Problem**: GNNExplainer returns all zeros for some motifs.

**Symptoms**:
- GNNExplainer scores: min=0.0000, max=0.0000
- Only works for feedback_loop (min=0, max=1)

**Partial Fix**: Added handling for constant edge masks:
```python
if edge_mask.max() > edge_mask.min():
    edge_mask = (edge_mask - edge_mask.min()) / (edge_mask.max() - edge_mask.min())
else:
    # Return uniform importance if all values are the same
    edge_mask = np.ones_like(edge_mask) * 0.5
```

**Note**: This doesn't fix the root cause (why GNNExplainer returns constant values). Possible reasons:
- GNN predictions are constant for those nodes
- GNNExplainer configuration needs adjustment
- Model was trained for graph-level tasks, not node-level

### 4. Visualization Error with Constant Ground Truth (FIXED ✓)
**Problem**: ROC AUC cannot be computed when ground truth has only one class.

**Error**: `ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.`

**Fix**: Added check before plotting:
```python
if len(np.unique(all_gt)) < 2:
    print(f"  ⚠ Skipping curves for {target}: Ground truth has only one class")
    continue
```

## Debug Features Added

1. **GNNExplainer debug**:
   - Warns if edge_mask is constant
   - Shows min/max/mean values

2. **SAE gradient debug**:
   - Warns if feature has near-zero activation
   - Shows first 5 feature activation values
   - Warns if gradients are constant
   - Shows gradient min/max values

3. **Comparison loop debug**:
   - Shows first 3 graphs with detailed scores
   - Tracks successful comparisons
   - Shows total results collected

## Expected Output After Fixes

```
DEBUG: Graph 0 - motif_node=1, 9 motif edges
  WARNING: SAE feature z111 has near-zero activation (1.23e-05)
  GNNExplainer scores: min=0.0234, max=0.8765
  SAE scores: min=0.0123, max=0.6543  # Non-zero now!
  ✓ Result added (AUROC: GNN=0.723, SAE=0.652)
```

## Remaining Issues to Investigate

1. **GNNExplainer still returning zeros**: Need to investigate why it works for feedback_loop but not other motifs
2. **Feature selection**: Verify that z112, z120 are actually the best features for these motifs (run `identify_top_sae_features.py`)
3. **Model architecture**: Verify hidden_dim=80 matches the trained model

## Next Steps

1. Run the comparison again to see if SAE gradients are now non-zero
2. If GNNExplainer still returns zeros, try:
   - Different GNNExplainer configuration (more epochs, different mode)
   - Check if model predictions vary for different nodes
   - Consider using graph-level explanations instead
3. Update feature indices based on `identify_top_sae_features.py` output
