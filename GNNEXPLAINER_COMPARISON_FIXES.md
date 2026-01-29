# GNNExplainer Comparison Fixes - Layer 2 Migration

## Summary
Fixed the GNNExplainer comparison code to work with the new layer 2 (80-dim) SAE models instead of the old layer 3 (64-dim) models.

## Changes Made

### 1. Updated Activation Extraction (Line 79-95)
**Before**: Extracted layer 3 (64-dim) activations
```python
x = F.relu(self.conv3(x, edge_index, edge_weight))
return x  # [num_nodes, 64]
```

**After**: Extract layer 2 (80-dim) activations
```python
x = F.relu(self.conv1(x, edge_index, edge_weight))
x = F.dropout(x, p=self.dropout, training=False)
x = F.relu(self.conv2(x, edge_index, edge_weight))
return x  # [num_nodes, 80] - Layer 2 activations BEFORE bottleneck
```

**Rationale**: SAE models are now trained on layer 2 (80-dim) activations, which are extracted BEFORE the conv3 bottleneck layer that compresses to 64-dim.

### 2. Updated SAE Input Dimension (Line 515)
Already correctly set to 80-dim:
```python
input_dim = 80
```

### 3. Fixed Feature Loading Function (Line 298-372)
**Before**: Looked for non-existent per-variant correlation files
```python
corr_file = Path(f'outputs/feature_analysis_{variant}/latent_correlations.csv')
```

**After**: Uses Phase 2 global correlation file with config filtering
```python
corr_file = Path('outputs/latent_correlations.csv')
# Filter by variant AND specific config parameters
df = df[df['variant'] == variant]
df = df[df['latent_dim'] == config['latent_dim']]
# ... filter by k, sparsity_coef, etc.
```

**Rationale**: Phase 2 generates a single `latent_correlations.csv` file (2.6MB) containing all features for all 30 configs. The function now filters by specific config to get the correct top features.

### 4. Updated Function Signature and Call
Added `config` parameter to pass hyperparameters:
```python
def load_top_features_from_phase2(variant: str, config: Dict) -> Dict[str, int]:
    ...

# Call site updated:
top_features = load_top_features_from_phase2(variant, config)
```

## Comparison Logic (Verified Sound)

### Methodology
The comparison evaluates two methods for identifying motif edges:

1. **GNNExplainer (Baseline)**
   - Explains which edges are important for GNN predictions
   - Uses perturbation + gradient-based approach
   - Standard interpretability method

2. **SAE Gradient Saliency (Novel)**
   - Explains which edges activate specific SAE features
   - Computes: ∂(SAE_feature_activation) / ∂(edge_weights)
   - Leverages interpretable SAE features identified in Phase 2

### Evaluation Metric
Both methods produce edge importance scores. Performance measured by:
- **AUROC**: Area Under ROC Curve (requires mixed edges)
- **AUPRC**: Area Under Precision-Recall Curve
- **Ground Truth**: Edges that are part of the motif

### "Mixed Edges" Requirement
Only graphs with 20-80% motif edge ratio are used because:
- AUROC requires both positive (motif) and negative (non-motif) edges
- Graphs with 0% or 100% motif edges would have undefined AUROC
- This explains why previous runs only found testable `single_input_module` graphs

### Process Flow
```
For each SAE variant (topk, gated, jumprelu, switch):
  1. Load best config from Phase 2 results
  2. Load trained SAE model
  3. Load top features for each motif (from Phase 2 correlations)
  4. Load test graphs with mixed edges (20-80% motif ratio)
  5. For each motif:
     a. Find graphs containing that motif with mixed edges
     b. For each graph:
        - Compute ground truth edge mask
        - Run GNNExplainer → edge scores
        - Run SAE gradient saliency → edge scores
        - Compare both to ground truth (AUROC, AUPRC)
  6. Aggregate results and statistical tests
```

## Expected Behavior

### If Comparison Works Well
- AUROC > 0.5: Better than random at identifying motif edges
- AUROC ≈ 0.7-0.8: Good performance
- SAE > GNN: Novel method outperforms baseline

### If Results Show AUROC ≈ 0.5 (Random)
Possible causes:
1. **Motif detection issue**: Ground truth labeling might be incorrect
2. **Feature selection**: Top features from Phase 2 might not be predictive
3. **Mixed edges scarcity**: Not enough testable graphs for some motifs
4. **Method limitations**: Neither method can identify these motifs from edges alone

## How to Run

```bash
# Single variant
python3 sae/compare_sae_vs_gnnexplainer.py --variant topk

# All variants
python3 sae/compare_sae_vs_gnnexplainer.py --all
```

## Output Files
- `outputs/gnnexplainer_comparison/comparison_all_variants.csv`: Detailed results
- Console: Summary statistics and statistical significance tests

## Next Steps After Running
1. Review AUROC/AUPRC scores by variant and motif
2. If all results are random (≈0.5), investigate ground truth labeling
3. Compare results across SAE variants to identify best method
4. Generate visualization plots for successful detections
