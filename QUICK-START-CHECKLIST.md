# Quick-Start Checklist: GNN-SAE Causal Ablation Pipeline

**For**: First-time execution, quick reference, verification steps
**See**: [SAE-CAUSAL-ABLATION-PIPELINE.md](SAE-CAUSAL-ABLATION-PIPELINE.md) for comprehensive reference
- Full variant architectures: [SAE Variants Architecture](SAE-CAUSAL-ABLATION-PIPELINE.md#sae-variants-architecture)
- Design patterns & implementation details: See same section

---

## Overview: SAE Variants

Phase 1 trains **30 SAE configurations** across **4 distinct variants**:

| Variant | Configs | Key Feature | Loss |
|---------|---------|---|---|
| **TopK** | 11 | Fixed top-K sparsity | MSE only |
| **Gated** | 9 | Gate + magnitude networks | MSE + L1 gate + aux loss |
| **JumpReLU** | 6 | Per-feature learnable thresholds | MSE + direct L0 |
| **Switch** | 4 | Mixture-of-experts routing | MSE + load balancing |

**All variants** implement a common Abstract Base Class interface enabling polymorphic training and consistent checkpoint formats.

See [SAE Variants Architecture](SAE-CAUSAL-ABLATION-PIPELINE.md#sae-variants-architecture) for detailed variant descriptions, hyperparameter ranges, and design pattern explanations.

---

## ✅ Pre-Pipeline Checklist

Before running any Python scripts, verify:

- [ ] **Virtual graphs generated** - `virtual_graphs/data/all_graphs/raw_graphs/` has 5000 .pkl files
- [ ] **GNN trained locally** - `checkpoints/gnn_model.pt` exists
- [ ] **Activations extracted** - `outputs/activations/layer2/{train,val,test,mixed}/` has .pt files
  - [ ] train: ~3000 files
  - [ ] val: ~500 files
  - [ ] test: ~500 files
  - [ ] mixed: ~1000 files
- [ ] **Test graph IDs saved** - `outputs/test_graph_ids.json` exists
- [ ] **Motif metadata saved** - `virtual_graphs/data/all_graphs/graph_motif_metadata/` has CSVs
- [ ] **All uploaded to Colab** - Data and scripts in Google Drive (for Colab notebook)

---

## 📋 Standard Pipeline (Recommended for Publication)

### Phase 1: Train All 30 SAE Configs (~5 hours)
```bash
python sparse_autoencoder.py
```
**Verify**:
- [ ] `ls checkpoints/sae_*.pt` → 30 files
- [ ] All files have `seed42` in name

---

### Phase 2: Rank Configs & Extract Features (~30 min)
```bash
python compare_sae_configs.py
```
**Verify**:
- [ ] `outputs/sae_config_comparison.csv` exists
- [ ] `outputs/latent_correlations.csv` exists
- [ ] `outputs/test_graph_ids.json` exists
- [ ] Open CSV, check for 30 rows (all configs ranked)

---

### Phase 2.5a: Feature Significance (Optional, ~1 hour)
```bash
# Run once per variant (4 separate commands)
python analyze_feature_significance.py --variant topk
python analyze_feature_significance.py --variant gated
python analyze_feature_significance.py --variant jumprelu
python analyze_feature_significance.py --variant switch
```
**Verify**:
- [ ] Statistical metrics saved for each variant

---

### Phase 2.5b: Cross-Variant Comparison (~5 min)
```bash
python compare_sae_variants.py
```
**Verify**:
- [ ] `outputs/sae_variant_comparison.csv` exists
- [ ] `outputs/variant_comparison_plots/pareto_frontier.png` created

---

### Phase 2b: Multi-Seed Retraining (~2.7 hours, REQUIRED for publication)
```bash
python retrain_best_configs.py
```
**Verify**:
- [ ] `ls checkpoints/sae_*_seed*.pt` → 16 additional files
- [ ] Files have seed123, seed456, seed789, seed1011 in names
- [ ] `outputs/retrain_summary.json` exists

---

### Phase 3a: SAE Latent Ablations (~2 hours, via Colab notebook or command)

**Via Notebook** (sae_colab_pipeline.ipynb):
- Cell automatically:
  1. Selects best config by max_rpb_abs
  2. Creates `ablations/phase_3a_config.json`
  3. Runs `run_interpretability_experiments.py`

**Or via Command Line**:
```bash
# Extract best variant from Phase 2
# (Inspect sae_config_comparison.csv for max_rpb_abs row)
python run_interpretability_experiments.py \
  --variant topk \
  --latent_dim 512 \
  --min_rpb 0.05 \
  --n_random_trials 20
```

**Verify**:
- [ ] `ablations/phase_3a_config.json` exists
- [ ] `ablations/results/` has 4 CSV files (one per motif)
- [ ] Filenames include variant name (e.g., `feedback_loop_topk_*.csv`)

---

### Phase 3b: Native GNN Ablations (~2 hours, 4 runs)

**Run exactly 4 times** (once per motif):
```bash
# Load best config from Phase 3a metadata
# (Check ablations/phase_3a_config.json)

python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_feedback_loop
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_cascade
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_feedforward_loop
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_single_input_module
```

**Verify**:
- [ ] `outputs/native_gnn_ablations/` has 4 CSV files
- [ ] One file per motif (feedback_loop, cascade, feedforward_loop, single_input_module)

---

### Phase 3c: Ablation Strategy Comparison (~20 min for Option A)

**OPTION A: Single Best Variant (RECOMMENDED)** ✅
```bash
# Use variant from Phase 3a
python compare_ablation_strategies.py --variant topk --latent_dim 512 --motif-mode
```

**OPTION B: All 4 Variants (Extended, ~80 min)**
```bash
python compare_ablation_strategies.py --all-variants --latent_dim 512 --motif-mode
```

**Verify**:
- [ ] `ablation_strategy_comparison/` exists
- [ ] At least 4 motif_agreement_*.csv files

---

### Phase 3d: Mixed-Motif Generalization (~45 min, REQUIRED for publication)

**Step 1: Generate mixed-motif activations** (one-time)
```bash
python generate_mixed_motif_activations.py
```

**Step 2: SAE ablation on mixed-motif graphs**
```bash
# Get best features from Phase 2
# (Top rows from latent_correlations.csv sorted by rpb_abs)

python run_ablation.py \
  --variant topk \
  --latent_dim 512 \
  --use_mixed_motifs \
  --feature z1,z2,z3,z10,z50 \
  --experiment_name mixed_motifs_topk
```

**Step 3: Native ablation on mixed-motif graphs**
```bash
python native_gnn_ablation.py \
  --variant topk \
  --latent_dim 512 \
  --use_mixed_motifs \
  --feature z1,z2,z3,z10,z50
```

**Verify**:
- [ ] `ablations/results/ablation*mixed_motifs*.csv` exists
- [ ] `outputs/native_gnn_ablations/*mixed_motifs*.csv` exists

---

### Phase 4: Statistical Validation (~30 min)
```bash
# Requires Phase 2b multi-seed checkpoints
python statistical_analysis_suite.py --seed-analysis
```

**Verify**:
- [ ] `outputs/statistical_analysis/feature_stability.png` created
- [ ] `outputs/statistical_analysis/correlation_distributions_*.png` created
- [ ] `outputs/statistical_analysis/redundancy_heatmap.png` created

---

### Phase 5: Visualization (~20 min)
```bash
# Get best variant from Phase 3a
python visualize_feature_activations.py --variant topk --latent_dim 512 --features 20
python analyze_sae_reconstruction_fidelity.py --variant topk --latent-dim 512 --k 8
```

**Verify**:
- [ ] `outputs/feature_activation_visualizations/` created
- [ ] `outputs/sae_reconstruction_fidelity/pca_histograms_*.png` created

---

## ⚡ Quick Pipeline (NOT for publication, ~8 hours)

```bash
# Skip Phase 2b, Phase 4 --seed-analysis
python sparse_autoencoder.py              # 5h
python compare_sae_configs.py             # 30m
python run_interpretability_experiments.py --variant topk --latent_dim 512  # 2h
python visualize_feature_activations.py --variant topk --latent_dim 512 --features 20  # 20m
```

---

## 🔧 Common Commands Reference

### Get Best Config from Phase 2
```bash
head -2 outputs/sae_config_comparison.csv | tail -1
# Read columns: variant, latent_dim, k, sparsity_coef, threshold_init, num_experts, max_rpb_abs
```

### Get Top Features for Phase 3d
```bash
# Top 10 features by |rpb| for best variant
grep "topk" outputs/latent_correlations.csv | sort -t',' -k4 -nr | head -10
```

### Check All Checkpoints
```bash
ls -lh checkpoints/sae_*.pt | wc -l    # Should show 46 after Phase 2b
ls -lh checkpoints/sae_*seed42.pt      # Phase 1: 30 files
ls -lh checkpoints/sae_*seed{123,456,789,1011}.pt  # Phase 2b: 16 files
```

### Verify Metadata JSON
```bash
cat ablations/phase_3a_config.json | jq '.'
# Check: variant, latent_dim, k/sparsity_coef/threshold_init values
```

---

## ❌ Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| Using `--all-features` flag | Flag doesn't exist | Use `--feature z1,z2,z3` instead |
| Wrong variant in Phase 3b | Phase 3b uses different config than 3a | Load metadata from `phase_3a_config.json` |
| Skipping Phase 2b | Can't run Phase 4 --seed-analysis | Phase 2b is REQUIRED for publication |
| Phase 3b with wrong motif | 4 motifs required, missing results | Run exactly 4 times, once per motif |
| Phase 3d without preprocessing | Activations not found | Run `generate_mixed_motif_activations.py` first |
| Mixing layer2 and layer2_new | Wrong activation data used | ALL scripts use `layer2` (not layer2_new) |

---

## 📊 Expected Outputs Summary

| Phase | Key Output | Location | Used By |
|-------|-----------|----------|---------|
| 1 | 30 SAE checkpoints | `checkpoints/sae_*_seed42.pt` | Phases 2, 3, 4 |
| 2 | Config ranking + features | `sae_config_comparison.csv`, `latent_correlations.csv` | Phases 3, 4 |
| 2b | Multi-seed checkpoints | `checkpoints/sae_*_seed{123,456,789,1011}.pt` | Phase 4 |
| 3a | SAE ablation results | `ablations/results/*.csv` | Phase 3c, 4 |
| 3b | Native ablation results | `outputs/native_gnn_ablations/*.csv` | Phase 3c, 4 |
| 3c | Agreement metrics | `ablation_strategy_comparison/*.csv` | Publication |
| 3d | Mixed-motif results | `ablations/results/*_mixed_motifs.csv` | Publication |
| 4 | Stability analysis | `outputs/statistical_analysis/*.png` | Publication |
| 5 | Visualizations | `outputs/feature_activation_visualizations/*` | Publication |

---

## 🚨 Emergency Restart Points

If pipeline fails partway:

**Restart from Phase 2**:
```bash
python compare_sae_configs.py  # Assumes Phase 1 completed
```

**Restart from Phase 3a**:
```bash
python run_interpretability_experiments.py --variant topk --latent_dim 512 --min_rpb 0.05
# Assumes Phases 1-2 completed
```

**Restart from Phase 4**:
```bash
python statistical_analysis_suite.py --seed-analysis
# Assumes Phases 1-2b-3 completed
```

**Full restart** (if unsure):
```bash
python sparse_autoencoder.py  # Start from Phase 1
```

---

## 📞 Need Help?

See comprehensive guide: [SAE-CAUSAL-ABLATION-PIPELINE.md](SAE-CAUSAL-ABLATION-PIPELINE.md)

Common issues covered:
- Phase X fails with "file not found"
- Checkpoint naming issues
- Multi-seed checkpoint problems
- Mixed-motif activation not found
- Flag/argument errors (--all-features vs --feature)

---

## ✅ Final Verification Checklist

After completing full pipeline:

- [ ] All Phases 1-5 completed without errors
- [ ] Total checkpoint count: 46 (30 seed42 + 16 multi-seed)
- [ ] Directory structure matches expected output locations
- [ ] Phase 3a metadata JSON created and contains correct variant
- [ ] Phase 3 results include variant in filenames
- [ ] Phase 4 statistical plots generated (feature_stability.png, etc.)
- [ ] Ready for publication analysis ✅

---

## ☁️ Google Colab Execution Checklist

### Before Running on Colab

1. **Complete Local Prerequisites** (MUST be done locally, not in Colab)
   - [ ] Run `graph_motif_generator.py` → generates 5000 graphs
   - [ ] Run `gnn_train.py` → generates GNN model + activations
   - [ ] Verify all prerequisite data exists (see Pre-Upload Verification below)

2. **Pre-Upload Verification** (run locally before uploading)

```bash
# Check activation counts (should be ~5000 total across all splits)
ls outputs/activations/layer2/train/ | wc -l        # ~3000
ls outputs/activations/layer2/val/ | wc -l          # ~500
ls outputs/activations/layer2/test/ | wc -l         # ~500
ls outputs/activations/layer2/mixed/ | wc -l        # ~1000 (for Phase 3d)

# Check required files exist
ls -la outputs/test_graph_ids.json
ls -la checkpoints/gnn_model.pt
ls virtual_graphs/data/all_graphs/graph_motif_metadata/ | wc -l  # ~5000

# Check all scripts present
ls *.py | grep -E "sparse_autoencoder|compare_sae|retrain|run_ablation|native_gnn|statistical|visualize|analyze_sae"
```

3. **Upload to Google Drive** (`/My Drive/182-GNN_SAE/`)
   - [ ] All `*.py` scripts (13 files)
   - [ ] `outputs/activations/layer2/` directory (train + val + test + mixed splits, ~5000 .pt files total)
   - [ ] `outputs/test_graph_ids.json`
   - [ ] `virtual_graphs/data/all_graphs/graph_motif_metadata/` (all CSV files)
   - [ ] `checkpoints/gnn_model.pt` (optional: only needed if Phase 5 uses it)

4. **Colab Runtime Setup**
   - [ ] Enable GPU: Runtime → Change runtime type → GPU (T4 or V100)
   - [ ] Verify GPU memory available
   - [ ] Note: Full pipeline is ~15.8 hours (Colab timeout is 12 hours)

### Phase-Specific Prerequisites

Before running each phase in Colab, verify previous outputs exist:

- [ ] **Phase 1:** GPU is enabled
- [ ] **Phase 2:** `checkpoints/` has 30 .pt files
- [ ] **Phase 2.5:** Phase 1 completed
- [ ] **Phase 2b:** `outputs/sae_config_comparison.csv` exists
- [ ] **Phase 3a:** `outputs/latent_correlations.csv` exists + Phase 2 metric choice (max_rpb_abs or composite_score)
- [ ] **Phase 3b:** `checkpoints/` has 30 files (Phase 1 output)
- [ ] **Phase 3c:** Both Phase 3a AND 3b outputs exist
- [ ] **Phase 4:** Phase 2b preferred (for multi-seed reproducibility analysis)
- [ ] **Phase 5:** Optional visualization phase

### Timeout Strategy ⚠️

Full pipeline (~15.8 hours) exceeds Colab 12-hour timeout.

**Recommended approach: Split into 2 Colab sessions**

**Session 1** (~9 hours): Phases 1-3c
```bash
# Run Phases 1, 2, 2.5, 2b, 3a, 3b, 3c
# After completion, export results to Google Drive
```

**Session 2** (~2 hours): Load Phase 1-3c outputs, run Phases 4-5
```bash
# Load outputs from Google Drive
# Run Phases 4 and 5 (statistical analysis + visualization)
```

**Alternative:** If timeout occurs mid-phase, Colab can resume from the last completed cell.

---
