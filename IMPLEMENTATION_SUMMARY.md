# Implementation Summary: Degree Confound Controls

## What Was Implemented

I've implemented degree confound controls for your GNN-SAE interpretability analysis. This addresses the critical issue where SAE features might correlate with degree patterns (e.g., "high out-degree") rather than true motif semantics (e.g., "single input module").

---

## Files Created

### 1. **virtual_graphs/compute_node_features.py**
- Computes node-level degree features for all 5000 graphs
- Adds 4 columns to each metadata CSV:
  - `degree`: Total degree (in + out)
  - `in_degree`: Number of incoming edges
  - `out_degree`: Number of outgoing edges
  - `degree_ratio`: Ratio of in/out degree
- Run once as setup: `python virtual_graphs/compute_node_features.py`

### 2. **RUN_WORKFLOW.md**
- Complete documentation of the analysis pipeline
- Explains each step, expected outputs, and interpretation
- Includes troubleshooting guide

### 3. **COMMANDS_TO_RUN.sh**
- Executable script to run the full pipeline
- Interactive prompts between steps
- Automatic verification checks

### 4. **IMPLEMENTATION_SUMMARY.md** (this file)
- Quick reference for what was changed

---

## Files Modified

### 1. **compare_sae_configs.py** (Major updates)

**Configuration changes:**
- `INPUT_DIM`: 80 → 64 (now uses layer 3 activations)
- `activation_dir`: layer1_new → layer3_new
- Added `CONTROL_FOR_DEGREE = True` flag

**New functions:**
- `compute_partial_correlation()`: Computes partial rpb controlling for degree
- `sanity_check_degree_confounds()`: Checks motif-degree correlations
  - Warns if strong confounding detected
  - Example: SIM ~ out_degree has rpb ≈ 0.87 (expected!)

**Updated functions:**
- `compute_correlations()`: Now optionally computes partial correlations
  - Adds columns: `rpb_partial`, `pval_partial`, `rpb_partial_abs`
  - Uses residualization approach (regress out degree effects)
- `analyze_configuration()`: Uses partial correlations when available
  - Runs sanity check on first config
  - Metrics use `rpb_partial_abs` instead of `rpb_abs` when controls enabled

**Output changes:**
- Results now include both bivariate and partial correlations
- Sanity check shows degree-motif correlations at start
- Prints whether using "bivariate" or "partial" correlations

### 2. **sparse_autoencoder.py**

**Changes:**
- `INPUT_DIM`: 80 → 64
- `train_dir`: layer1_new → layer3_new
- `val_dir`: layer1_new → layer3_new
- `test_dir`: layer1_new → layer3_new
- Print message updated to "Layer3 Activations"

### 3. **gnn_train_copy.py** (Already updated in previous session)
- Now extracts activations from all three hidden layers (1, 2, 3)

---

## Exact Commands to Run (In Order)

### Quick Start (Interactive):
```bash
cd /data/users/goodarzilab/shervin/182-GNN_SAE
./COMMANDS_TO_RUN.sh
```

### Manual Step-by-Step:

```bash
cd /data/users/goodarzilab/shervin/182-GNN_SAE

# STEP 0: ONE-TIME SETUP (Compute degree features)
python virtual_graphs/compute_node_features.py

# STEP 1 (Optional): Hyperparameter tuning
python hyperparameter_sweep_multi_gpu.py

# STEP 2: Train GNN and extract layer 1, 2, 3 activations
python gnn_train_copy.py

# STEP 3: Train SAE on layer 3 activations
python sparse_autoencoder.py

# STEP 4: Compare SAE configs with degree controls
python compare_sae_configs.py

# STEP 5: Analyze results
jupyter notebook analysis_notebook.ipynb
```

---

## What You'll See

### Step 0 Output (compute_node_features.py):
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

Example (graph_0_metadata.csv):
                feedforward_loop  feedback_loop  ...  degree  in_degree  out_degree
node_0                         1              0  ...       2          1           1
node_1                         1              0  ...       2          1           1
...
```

### Step 4 Output (compare_sae_configs.py):

**Sanity Check (appears once at start):**
```
SANITY CHECK: Degree-Motif Correlations
High correlations indicate potential confounding:
(SAE features may be detecting degree rather than motif semantics)

  single_input_module  ~ out_degree  : rpb= 0.872, p=3.14e-156
  cascade              ~ degree      : rpb=-0.423, p=2.15e-32
  feedforward_loop     ~ degree      : rpb= 0.301, p=1.22e-15
  feedback_loop        ~ in_degree   : rpb= 0.156, p=4.33e-08
  ...

Interpretation:
  ⚠ Strong confounding detected (max |rpb|=0.872)
    → Partial correlations are CRITICAL for valid interpretation
```

**Config Analysis (per config):**
```
Analyzing: latent_dim=256, k=16 (6.25% active)
  Using partial correlations (controlling for degree)
  Skipping significance testing...
  ✓ Max |rpb|: 0.312  (partial correlation, degree-controlled)
  ✓ Best F1: 0.421
  ✓ Composite score: 0.612
```

**Final Recommendation:**
```
RECOMMENDED CONFIGURATION
  latent_dim = 256
  k = 16
  Sparsity: 6.25% (Medium capacity, moderate sparsity)

  Key Metrics:
    • Max correlation (bivariate): |rpb| = 0.487
    • Max correlation (partial): |rpb| = 0.312  ← Degree-controlled!
    • Best F1 score: 0.421
    • Significant features (FDR<0.05): 34
    • Reconstruction loss: 8.45e-07
    • Active features: 243/256 (94.9%)
    • Composite score: 0.612
```

---

## Interpretation Guide

### Understanding the Results

**High bivariate correlation but LOW partial correlation:**
```
Feature z42:
  rpb (bivariate):  0.678
  rpb (partial):    0.134

Interpretation: This feature is CONFOUNDED with degree.
  It's detecting "high out-degree" not "SIM motif semantics".
  NOT safe to claim motif-specific interpretation.
```

**High bivariate AND high partial correlation:**
```
Feature z17:
  rpb (bivariate):  0.543
  rpb (partial):    0.487

Interpretation: This feature is ROBUST to degree controls.
  It represents true motif structure beyond just degree.
  Safe to claim motif-specific interpretation.
```

### Sanity Check Interpretation

**Expected confounds (these are NORMAL):**
- `single_input_module ~ out_degree`: HIGH (rpb ≈ 0.87)
  - SIM requires out-degree ≥ 3 by definition
- `cascade ~ degree`: NEGATIVE (rpb ≈ -0.42)
  - Cascade nodes have low degree (in=1, out=1 for internal nodes)
- `feedforward_loop ~ degree`: MODERATE (rpb ≈ 0.30)
  - Feedforward requires 3 nodes with specific connections

**What to worry about:**
- If partial correlations remain strong (rpb > 0.3) → Features are robustly motif-specific ✓
- If partial correlations drop to near-zero (rpb < 0.1) → Features were just detecting degree ✗

---

## Configuration Options

### To disable degree controls (not recommended):
Edit `compare_sae_configs.py`:
```python
CONTROL_FOR_DEGREE = False  # Change line 33
```

### To use different layer activations:
Edit both `compare_sae_configs.py` and `sparse_autoencoder.py`:
```python
# For layer 2 (128-dim, 2-hop neighborhoods):
INPUT_DIM = 128
activation_dir = Path("outputs/activations/layer2_new/test")
```

### To change hyperparameter sweep:
Edit `compare_sae_configs.py`:
```python
CONFIGS = [
    # (latent_dim, k, description)
    (256, 16, "Your custom config"),
    ...
]
```

---

## Verification Checklist

After running the pipeline, verify:

- [ ] Degree features added to metadata:
  ```bash
  head -2 virtual_graphs/data/all_graphs/graph_motif_metadata/graph_0_metadata.csv
  # Should show: feedforward_loop,feedback_loop,...,degree,in_degree,out_degree,degree_ratio
  ```

- [ ] Layer 3 activations extracted:
  ```bash
  ls outputs/activations/layer3_new/train/*.pt | wc -l  # ~3200
  ls outputs/activations/layer3_new/test/*.pt | wc -l   # ~600
  ```

- [ ] SAE models trained:
  ```bash
  ls checkpoints/sae_*.pt | wc -l  # Should be 11
  ```

- [ ] Comparison results generated:
  ```bash
  wc -l outputs/sae_config_comparison.csv  # Should have ~12 rows (header + 11 configs)
  ```

- [ ] Results include partial correlations:
  ```bash
  head -1 outputs/sae_config_comparison.csv | grep "rpb_partial"
  # Should find "rpb_partial" column
  ```

---

## Next Steps

1. **Review sanity check output**: Confirm expected confounds (SIM ~ out_degree ≈ 0.87)

2. **Compare correlations**: Load `outputs/sae_config_comparison.csv` and compare bivariate vs partial

3. **Identify robust features**: Find features with high partial correlations (survived controls)

4. **Update claims**: Only claim motif-specificity for features with strong partial correlations

5. **Visualize**: Create plots showing before/after degree controls in `analysis_notebook.ipynb`

---

## Documentation References

1. **Detailed workflow**: `RUN_WORKFLOW.md`
2. **Deep dive analysis**: `~/.claude/plans/groovy-napping-engelbart.md`
3. **Code changes**: This file (IMPLEMENTATION_SUMMARY.md)

---

## Technical Details

### Partial Correlation Method

We use **residualization** to control for degree:

1. Regress SAE feature on degree controls:
   ```
   z_i = β₀ + β₁·degree + β₂·in_degree + β₃·out_degree + ε
   ```

2. Extract residuals:
   ```
   z_i^residual = z_i - z_i^predicted
   ```

3. Compute point-biserial correlation between motif label and residuals:
   ```
   rpb_partial = corr(motif_label, z_i^residual)
   ```

This isolates the motif-specific signal after removing degree effects.

### Why This Matters

**Without controls:**
- Can't distinguish "SIM detector" from "high out-degree detector"
- Claims of motif-specificity are unreliable
- Confounding undermines interpretability conclusions

**With controls:**
- Explicitly test if features survive degree removal
- Confidence in motif-specific interpretations
- Rigorous, defensible analysis

---

## Questions?

See `RUN_WORKFLOW.md` for:
- Troubleshooting guide
- Expected outputs for each step
- Detailed interpretation guidelines
- Configuration options
