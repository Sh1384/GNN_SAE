# Migration Guide: Layer 3 → Layer 2 Activations

## Summary

This guide explains the changes needed to switch the SAE pipeline from training on **layer 3 activations (64-dim)** to **layer 2 activations (80-dim)**.

---

## Why Switch to Layer 2?

### GNN Architecture
```
Layer 1: Conv(2 → 80)   + ReLU + Dropout
Layer 2: Conv(80 → 80)  + ReLU + Dropout  ← **NEW TARGET**
Layer 3: Conv(80 → 64)  + ReLU + Dropout  ← **OLD TARGET (bottleneck)**
Layer 4: Conv(64 → 1)   (final prediction)
```

### Rationale
- **Layer 2**: 80-dim, full representation before bottleneck
  - More information preserved
  - Aggregates 2-hop neighborhoods
  - No dimensionality reduction yet

- **Layer 3**: 64-dim, compressed bottleneck
  - Information loss due to compression (80→64)
  - Already applying dimensionality reduction
  - SAE then applies further compression (redundant)

**Hypothesis**: Layer 2 may provide richer, more interpretable features since it's pre-compression.

---

## What Changed

### 1. Activation Path
```bash
# OLD
outputs/activations/layer3_new/{train,val,test}/

# NEW
outputs/activations/layer2_new/{train,val,test}/
```

### 2. Input Dimension
```python
# OLD
INPUT_DIM = 64

# NEW
INPUT_DIM = 80
```

### 3. All Model Instantiations
```python
# OLD
TopKSAE(input_dim=64, latent_dim=256, k=16)
GatedSAE(input_dim=64, latent_dim=512, sparsity_coef=1e-4)
# etc.

# NEW
TopKSAE(input_dim=80, latent_dim=256, k=16)
GatedSAE(input_dim=80, latent_dim=512, sparsity_coef=1e-4)
# etc.
```

---

## Files Modified

| File | Changes |
|------|---------|
| `run_sae_pipeline_multi_gpu.py` | INPUT_DIM: 64→80, paths: layer3→layer2 |
| `sae/sparse_autoencoder.py` | INPUT_DIM: 64→80, default args: 64→80, paths: layer3→layer2 |
| `sae/analyze_feature_significance.py` | INPUT_DIM: 64→80, paths: layer3→layer2 |
| `sae/native_gnn_ablation.py` | INPUT_DIM: 64→80, paths: layer3→layer2 |
| `sae/analyze_sae_reconstruction_fidelity.py` | input_dim: 64→80, paths: layer3→layer2 |
| `sae/retrain_best_configs.py` | INPUT_DIM: 64→80, paths: layer3→layer2 |
| `sae/compare_sae_configs.py` | INPUT_DIM: 64→80, paths: layer3→layer2 |
| `sae/compare_sae_vs_gnnexplainer.py` | input_dim: 64→80, paths: layer3→layer2 |
| `sae/run_ablation.py` | input_dim: 64→80, paths: layer3→layer2 |
| `sae/generate_mixed_motif_activations.py` | layer3_activations→layer2_activations, paths: layer3→layer2 |

---

## Step-by-Step Instructions

### Step 1: Verify Layer 2 Activations Exist

```bash
# Check if activations already saved (they should be!)
ls outputs/activations/layer2_new/train/ | wc -l
# Expected: 3200 files (one per training graph)

ls outputs/activations/layer2_new/val/ | wc -l
# Expected: 400 files

ls outputs/activations/layer2_new/test/ | wc -l
# Expected: 400 files
```

**If activations DON'T exist**, you need to run GNN training first:
```bash
python3 gnn_train_copy.py
```

This will save layer 1, 2, and 3 activations automatically.

### Step 2: Run the Migration Script

```bash
cd /data/users/goodarzilab/shervin/182-GNN_SAE
./switch_to_layer2.sh
```

This will:
- Create backups of all modified files in `backups/`
- Update INPUT_DIM from 64 to 80 in all files
- Change activation paths from layer3_new to layer2_new
- Update all documentation strings

### Step 3: Verify Changes

```bash
# Check that INPUT_DIM changed
grep "INPUT_DIM = 80" run_sae_pipeline_multi_gpu.py
grep "INPUT_DIM = 80" sae/sparse_autoencoder.py

# Check that paths changed
grep "layer2_new" run_sae_pipeline_multi_gpu.py
grep "layer2_new" sae/sparse_autoencoder.py
```

### Step 4: Run Phase 1 (Train SAEs on Layer 2)

```bash
python3 run_sae_pipeline_multi_gpu.py --phase 1
```

This will:
- Load layer 2 activations (80-dim) instead of layer 3 (64-dim)
- Train all 30 SAE configurations with 80-dim input
- Save new checkpoints: `checkpoints/sae_{variant}_latent{dim}_*_seed42.pt`

**Important**: Old SAE checkpoints (trained on layer 3, 64-dim) will be **incompatible** with the new setup. Phase 1 creates new 80-dim checkpoints.

### Step 5: Run Remaining Phases

Once Phase 1 completes, run the rest of the pipeline:

```bash
# Phase 2: Configuration comparison
python3 run_sae_pipeline_multi_gpu.py --phase 2

# Phase 3: Ablation experiments
python3 run_phase3_all_variants.py --all

# Or run everything:
python3 run_sae_pipeline_multi_gpu.py --all
```

---

## What Stays the Same

### SAE Architectures
- TopK, Gated, JumpReLU, Switch variants unchanged
- Latent dimensions unchanged (128, 256, 512)
- Sparsity parameters unchanged (k=4,8,16,32, etc.)

### Training Hyperparameters
- Learning rate: 5e-4
- Batch size: 1024
- Epochs: 200
- Loss function: MSE reconstruction

### Evaluation Metrics
- Point-biserial correlation (rpb)
- F1 score
- AUROC/AUPRC
- Ablation impact (Δ loss)

---

## Expected Results Comparison

### Layer 3 (64-dim, bottleneck)
✅ **Pros**:
- More compressed representation
- Forces features to be essential
- Already dimensionality-reduced by GNN

❌ **Cons**:
- Information loss from 80→64 compression
- May miss patterns that require higher-dimensional space
- Bottleneck might force polysemantic features

### Layer 2 (80-dim, pre-bottleneck)
✅ **Pros**:
- Richer representation (25% more dimensions)
- No GNN-imposed compression yet
- Potentially more monosemantic features
- Better for interpretability

❌ **Cons**:
- Slightly higher computational cost (80 vs 64)
- More parameters in encoder/decoder
- May learn redundant features

**Hypothesis to test**: Does layer 2 produce more interpretable (monosemantic) SAE features than layer 3?

---

## Reverting Changes

If you want to revert to layer 3 activations:

```bash
# Restore backups
cp backups/run_sae_pipeline_multi_gpu.py ./
cp backups/sparse_autoencoder.py sae/
# etc. for all files

# Or manually edit
sed -i 's/INPUT_DIM = 80/INPUT_DIM = 64/g' run_sae_pipeline_multi_gpu.py
sed -i 's/layer2_new/layer3_new/g' run_sae_pipeline_multi_gpu.py
# etc.
```

---

## Troubleshooting

### Issue: "Activation files not found"
**Solution**: Run `python3 gnn_train_copy.py` to generate layer 2 activations

### Issue: "Dimension mismatch" errors
**Cause**: Mixing old (64-dim) checkpoints with new (80-dim) code
**Solution**: Delete old checkpoints and retrain with Phase 1

### Issue: "Out of memory" errors
**Cause**: 80-dim activations use ~25% more memory than 64-dim
**Solution**: Reduce batch size in pipeline config (e.g., 1024 → 768)

---

## Next Steps After Migration

1. **Compare results**: Run Phase 2 to see if layer 2 produces:
   - Higher rpb correlations with motifs
   - Better F1 scores
   - More monosemantic features

2. **Analyze ablations**: Run Phase 3 to see if:
   - Ablations have stronger effects
   - Features are more specific to motifs
   - Native vs SAE comparison changes

3. **Visualize differences**:
   - Compare PCA plots (layer 2 vs layer 3)
   - Compare feature activation patterns
   - Compare reconstruction quality

---

## Contact

If you encounter issues with the migration, check:
1. Activation file dimensions match INPUT_DIM
2. All checkpoint paths updated consistently
3. Old checkpoints cleared before retraining
