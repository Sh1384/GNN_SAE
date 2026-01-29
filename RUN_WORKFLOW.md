# GNN-SAE Analysis Workflow with Degree Confound Controls

This document outlines the complete workflow for training your GNN, SAE, and performing interpretability analysis **with degree confound controls**.

## Critical Updates

1. **Degree controls added**: compare_sae_configs.py now computes partial correlations controlling for node degree
2. **Sanity checks added**: Automatically checks if motif labels correlate with degree (confounding indicator)
3. **Layer 3 activations**: All analysis now uses layer 3 (64-dim) hidden activations

---

## Step 0: Compute Node Degree Features (ONE-TIME SETUP)

**Purpose**: Add degree features (in-degree, out-degree, total degree) to all graph metadata files.

**Command**:
```bash
cd /data/users/goodarzilab/shervin/182-GNN_SAE
python virtual_graphs/compute_node_features.py
```

**Expected Output**:
```
Computing Node-Level Topological Features
Found 5000 graph files
Processing graphs...
✓ Successfully processed: 5000 graphs
Metadata files updated with 4 new columns:
  - degree: Total degree (in + out)
  - in_degree: Number of incoming edges
  - out_degree: Number of outgoing edges
  - degree_ratio: in_degree / out_degree
```

**What it does**:
- Loads each graph from `virtual_graphs/data/all_graphs/raw_graphs/`
- Computes degree metrics for all 10 nodes
- Updates `virtual_graphs/data/all_graphs/graph_motif_metadata/graph_{ID}_metadata.csv`
- Adds 4 new columns to each CSV

**⚠ IMPORTANT**: Run this BEFORE the rest of the pipeline. Only needs to be run once (unless you regenerate graphs).

---

## Step 1: Hyperparameter Tuning (Optional, if you need to retune)

**Purpose**: Find optimal GNN hyperparameters (learning rate, hidden dim, dropout, etc.)

**Command**:
```bash
python hyperparameter_sweep_multi_gpu.py
```

**Expected Duration**: ~30 minutes - 2 hours (depending on GPU count)

**What it does**:
- Tests different GNN configurations
- Saves best hyperparameters
- Creates checkpoint files

**Output Files**:
- `outputs/hyperparameter_sweep_results.csv`
- `checkpoints/best_config.pt` (or similar)

---

## Step 2: Train GNN on Single-Motif Graphs

**Purpose**: Train GNN and extract layer 1, 2, and 3 hidden activations

**Command**:
```bash
python gnn_train_copy.py
```

**Expected Duration**: ~10-30 minutes

**What it does**:
- Trains 4-layer GCN on single-motif graphs only (IDs 0-3999)
- Extracts activations from all three hidden layers:
  - Layer 1: 128-dim (1-hop neighborhoods)
  - Layer 2: 128-dim (2-hop neighborhoods)
  - Layer 3: 64-dim (3-hop neighborhoods) ← **We use this layer**
- Saves activations for train/val/test splits

**Output Files**:
- `checkpoints/gnn_model.pt`
- `outputs/activations/layer1_new/{train,val,test}/graph_*.pt`
- `outputs/activations/layer2_new/{train,val,test}/graph_*.pt`
- `outputs/activations/layer3_new/{train,val,test}/graph_*.pt`
- `outputs/training_metrics.json`
- `outputs/motif_metrics.json`

**Verification**:
```bash
# Check that layer 3 activations exist
ls outputs/activations/layer3_new/train/ | wc -l  # Should show ~3200 files
ls outputs/activations/layer3_new/test/ | wc -l   # Should show ~600 files
```

---

## Step 3: Train Sparse Autoencoders

**Purpose**: Train SAE on layer 3 activations to learn interpretable features

**Command**:
```bash
python sparse_autoencoder.py
```

**Expected Duration**: ~5-20 minutes per configuration

**What it does**:
- Loads layer 3 activations (64-dim)
- Trains SAE with TopK activation
- Tests multiple configurations:
  - latent_dim ∈ {128, 256, 512}
  - k ∈ {4, 8, 16, 32}
- Saves each trained model

**Output Files**:
- `checkpoints/sae_latent{dim}_k{k}.pt` (for each configuration)
- `outputs/sae_metrics_latent{dim}_k{k}.json` (reconstruction losses)

**Verification**:
```bash
# Check trained SAE models
ls checkpoints/sae_*.pt | wc -l  # Should show ~11 files (one per config)
```

---

## Step 4: Compare SAE Configurations (WITH DEGREE CONTROLS)

**Purpose**: Identify optimal SAE hyperparameters using controlled correlations

**Command**:
```bash
python compare_sae_configs.py
```

**Expected Duration**: ~10-30 minutes

**What it does**:
- **Stage 1** (Fast screening):
  - Loads each SAE configuration
  - Computes **partial correlations** (controlling for degree, in_degree, out_degree)
  - Computes bivariate correlations (for comparison)
  - Ranks configs by composite score

- **Sanity Check** (on first config):
  - Computes correlations between motif labels and degree features
  - Warns if strong confounding detected
  - Example output:
    ```
    SANITY CHECK: Degree-Motif Correlations
    High correlations indicate potential confounding:

    single_input_module  ~ out_degree  : rpb= 0.872, p=3.14e-156
    cascade              ~ degree      : rpb=-0.423, p=2.15e-32
    feedforward_loop     ~ degree      : rpb= 0.301, p=1.22e-15
    ...

    Interpretation:
      ⚠ Strong confounding detected (max |rpb|=0.872)
        → Partial correlations are CRITICAL for valid interpretation
    ```

- **Stage 2** (Deep analysis):
  - Runs full permutation tests (1000 perms) on top 3 configs
  - Applies FDR correction
  - Uses reconstruction loss as tiebreaker

**Output Files**:
- `outputs/sae_config_comparison.csv`

**Key Metrics**:
- `max_rpb_abs`: Maximum bivariate correlation
- `max_rpb_partial_abs`: Maximum partial correlation (degree-controlled)
- `composite_score`: 50% correlation + 35% F1 + 15% capacity
- `test_reconstruction`: Reconstruction loss (tiebreaker)

**Interpretation**:
```
If partial correlations are MUCH LOWER than bivariate correlations:
  → Features were confounded with degree (not motif-specific)
  → Use partial correlations for valid interpretation

If partial correlations remain strong:
  → Features are robustly motif-specific (survived controls)
  → Safe to claim motif interpretation
```

**Verification**:
```bash
# Check output
cat outputs/sae_config_comparison.csv | column -t -s, | head -15
```

**Example Expected Output**:
```
RECOMMENDED CONFIGURATION
  latent_dim = 256
  k = 16

  Key Metrics:
    • Max correlation (bivariate): |rpb| = 0.487
    • Max correlation (partial): |rpb| = 0.312  ← Degree-controlled
    • Best F1 score: 0.421
    • Significant features (FDR<0.05): 34
    • Reconstruction loss: 8.45e-07
    • Composite score: 0.612
```

---

## Step 5: Detailed Analysis in Notebook

**Purpose**: Visualize results, compare before/after degree controls

**Command**:
Open `analysis_notebook.ipynb` in Jupyter or VSCode

**What to examine**:
1. **Load comparison results**:
   ```python
   import pandas as pd
   df_results = pd.read_csv('outputs/sae_config_comparison.csv')

   # Compare bivariate vs partial correlations
   df_results[['latent_dim', 'k', 'max_rpb_abs', 'max_rpb_partial_abs', 'composite_score']]
   ```

2. **Analyze specific features**:
   - Which features survive degree controls?
   - How much do correlations drop when controlling for degree?
   - Are any features purely degree-driven?

3. **Visualizations**:
   - Correlation heatmaps (before/after controls)
   - Feature activation distributions
   - Motif-specific feature identification

---

## Summary of Complete Workflow

```bash
# ONE-TIME SETUP (only run once)
cd /data/users/goodarzilab/shervin/182-GNN_SAE
python virtual_graphs/compute_node_features.py

# MAIN PIPELINE (run in sequence)
# Step 1: (Optional) Hyperparameter tuning
python hyperparameter_sweep_multi_gpu.py

# Step 2: Train GNN and extract activations
python gnn_train_copy.py

# Step 3: Train SAE on layer 3 activations
python sparse_autoencoder.py

# Step 4: Compare SAE configs with degree controls
python compare_sae_configs.py

# Step 5: Analyze in notebook
jupyter notebook analysis_notebook.ipynb
```

---

## Files Modified for Degree Controls

### New Files:
1. **`virtual_graphs/compute_node_features.py`** - Computes degree features
2. **`RUN_WORKFLOW.md`** (this file) - Workflow documentation

### Updated Files:
1. **`compare_sae_configs.py`**:
   - Added `CONTROL_FOR_DEGREE = True` flag
   - Added `compute_partial_correlation()` function
   - Added `sanity_check_degree_confounds()` function
   - Updated `compute_correlations()` to optionally compute partial correlations
   - Updated metrics to use `rpb_partial_abs` when available
   - Changed INPUT_DIM from 80 → 64 (layer 3 activations)
   - Changed activation_dir from layer1_new → layer3_new

2. **`gnn_train_copy.py`** (already updated):
   - Now extracts activations from layers 1, 2, AND 3

### Updated Metadata:
- All 5000 `graph_motif_metadata/graph_{ID}_metadata.csv` files now have 4 extra columns:
  - degree, in_degree, out_degree, degree_ratio

---

## Configuration Options

### To disable degree controls (not recommended):
Edit `compare_sae_configs.py`:
```python
CONTROL_FOR_DEGREE = False  # Change to False
```

### To use different layer activations:
Edit `compare_sae_configs.py`:
```python
INPUT_DIM = 128  # For layer 1 or 2
activation_dir = Path("outputs/activations/layer2_new/test")  # Change layer
```

---

## Troubleshooting

### Error: "No degree features found in data"
**Solution**: Run `python virtual_graphs/compute_node_features.py` first

### Error: "Checkpoint not found"
**Solution**: Ensure you ran previous steps in sequence (GNN training before SAE training)

### Low partial correlations (much lower than bivariate)
**Interpretation**: This is EXPECTED if features were degree-confounded. This is the control working correctly. Only features with high partial correlations are truly motif-specific.

### Sanity check shows strong confounding
**Expected**: Single input modules REQUIRE high out-degree by definition (rpb ~ 0.87)
**Action**: This confirms partial correlations are critical. Proceed with confidence in controlled results.

---

## Questions?

See the detailed analysis in `/data/users/goodarzilab/shervin/.claude/plans/groovy-napping-engelbart.md` for:
- Complete explanation of degree confounds
- Evidence from code exploration
- Future recommendations
- References to specific code sections
