# GNN-SAE Interpretability Pipeline Guide

## 🚨 CRITICAL PREREQUISITES

**Before running the SAE training/ablation pipeline, you MUST complete these steps first:**

1. **Generate Virtual Graphs** (run locally)
   - Script: `graph_motif_generator.py`
   - Creates: `virtual_graphs/data/all_graphs/raw_graphs/*.pkl` (5000 graphs)

2. **Train GNN Model & Extract Activations** (run locally)
   - Script: `gnn_train.py` (NOT included in Colab notebook)
   - Creates:
     - `checkpoints/gnn_model.pt` (trained GNN model)
     - `outputs/activations/layer2/{train,val,test}/` (3000+500+500 .pt files)
     - `outputs/test_graph_ids.json` (test set definition)
     - `virtual_graphs/data/all_graphs/graph_motif_metadata/*.csv` (motif labels)

3. **Upload to Google Drive** (before running Colab)
   - All Python scripts
   - Activation data (`outputs/activations/layer2/`)
   - Test graph IDs (`outputs/test_graph_ids.json`)
   - Motif metadata

**The Colab notebook assumes this prerequisite data already exists.**

---

**IMPORTANT**: All analysis scripts use consistent activation directory: `outputs/activations/layer2/{train,val,test,mixed}/`

## Data Consistency Guarantee

✅ **All layer2_new references have been replaced with layer2**
✅ **All 30+ analysis scripts use the same activation data across phases**
✅ **Features identified in Phase 2 and tested in Phase 3 use identical activation sources**
✅ **gnn_train.py creates ALL prerequisite data needed by the SAE/interpretability pipeline**

### Unified Activation Directory Structure
```
outputs/activations/layer2/
├── train/          (Training set activations, graphs 0-2999)
├── val/            (Validation set activations, graphs 3000-3499)
├── test/           (Test set activations, graphs 3500-3999)
└── mixed/          (Mixed-motif activations, graphs 4000-4999, REQUIRED for robustness validation)
```

---

## Quick Reference: Pipeline Architecture

### Core Pipeline Flow

```
RAW DATA
├─ Virtual Graphs (0-3999 single-motif, 4000-4999 mixed)
├─ GNN Activations (64-dim Layer 2) → outputs/activations/layer2/
└─ Motif Metadata (FFL, FBL, SIM, CASC)
        ↓
PHASE 1: SAE TRAINING
├─ sparse_autoencoder.py
├─ Input: outputs/activations/layer2/{train,val,test}/
├─ 4 variants: TopK, Gated, JumpReLU, Switch (30 configs total)
└─ Output: checkpoints/sae_*.pt + outputs/sae_metrics_*.json
        ↓
PHASE 2: CONFIGURATION COMPARISON
├─ compare_sae_configs.py
├─ Input: SAE checkpoints + outputs/activations/layer2/test/
├─ Feature-motif correlation (point-biserial r_pb)
├─ Within-variant ranking by composite score
└─ Output: sae_config_comparison.csv + latent_correlations.csv + test_graph_ids.json
        ↓
PHASE 2.5: CROSS-VARIANT COMPARISON
├─ compare_sae_variants.py
├─ Input: All 30 SAE checkpoints + metrics
├─ Pareto frontiers: reconstruction vs sparsity
├─ Cross-variant trade-off analysis
└─ Output: sae_variant_comparison.csv + variant_comparison_plots/
        ↓
PHASE 2b: MULTI-SEED RETRAINING (REQUIRED FOR PUBLICATION)
├─ retrain_best_configs.py
├─ Input: Best config per variant (from Phase 2)
├─ Retrain with 5 seeds: [42, 123, 456, 789, 1011]
└─ Output: Multi-seed checkpoints + retrain_summary.json (stability metrics)
        ↓
PHASE 3: ABLATION STUDIES (Can run in parallel)
├─ Path 3a: SAE Latent Space Ablations (Single-Motif Test Set)
│  ├─ run_ablation.py
│  ├─ Input: SAE checkpoints + outputs/activations/layer2/test/
│  ├─ Three-way comparison: original vs full SAE vs ablated
│  └─ Output: ablations/results/ablation_*.csv
├─ Path 3b: Native GNN Activation Space Validation (Single-Motif Test Set)
│  ├─ native_gnn_ablation.py --use-rpb --motif {motif} (run once per motif)
│  ├─ Input: SAE checkpoints + outputs/activations/layer2/test/ + Phase 2.5a correlation data
│  ├─ Process: For each motif, ablate top features correlated with that motif via native GNN
│  ├─ Direct activation patching on same feature groups as Phase 3a
│  └─ Output: outputs/native_gnn_ablations/native_ablation_{variant}_rpb_{motif}.csv (one per motif)
├─ Path 3c: Motif-Group Strategy Comparison
│  ├─ Input:
│  │  ├─ Phase 3a outputs: ablations/results/{motif}_l{latent_dim}_k{k}_results.csv (one per motif)
│  │  └─ Phase 3b outputs: outputs/native_gnn_ablations/native_ablation_{variant}_rpb_{motif}.csv (one per motif)
│  ├─ Process: For each motif group, compare SAE vs native ablation impacts at graph level
│  ├─ Metric: Pearson/Spearman correlation of ablation impacts per motif
│  ├─ Script: compare_ablation_strategies.py (motif-group mode)
│  └─ Output: ablation_strategy_comparison/ (motif_agreement_*.csv + comparison_plots/)
└─ Path 3d: MIXED-MOTIF GENERALIZATION TEST (REQUIRED for Publication)
   ├─ Preprocessing: python generate_mixed_motif_activations.py (one-time, inference only)
   ├─ Step 1: run_ablation.py --use_mixed_motifs --feature [features_from_Phase_2.5a]
   │  ├─ Input: SAE checkpoints (trained on single-motif) + outputs/activations/layer2/mixed/
   │  ├─ Tests if single-motif-discovered features generalize to mixed-motif graphs
   │  └─ Output: ablations/results/ablation_*_mixed_motifs.csv
   ├─ Step 2: native_gnn_ablation.py --use_mixed_motifs --feature [features_from_Phase_2.5a]
   │  ├─ Validates SAE via direct activation patching on mixed-motif
   │  └─ Output: outputs/native_gnn_ablations/*_mixed_motifs/
   └─ Interpretation: Compare single-motif vs mixed-motif ablation impacts → Robustness assessment
        ↓
PHASE 4: STATISTICAL VALIDATION (Requires Phase 2b multi-seed models)
├─ statistical_analysis_suite.py --seed-analysis
├─ Feature stability across seeds (decoder cosine similarity)
├─ Correlation distributions per variant & motif
├─ Feature redundancy analysis
├─ Permutation testing with FDR correction
├─ Ablation conditional effects (Wilcoxon tests)
└─ Output: outputs/statistical_analysis/ (plots + CSV with feature_stability.png)
        ↓
PHASE 5: VISUALIZATION & RECONSTRUCTION ANALYSIS
├─ visualize_feature_activations.py
├─ analyze_sae_reconstruction_fidelity.py
├─ Feature selectivity heatmaps + PCA analysis
└─ Output: Feature visualizations + reconstruction metrics
        ↓
PUBLICATION OUTPUTS
└─ Comprehensive reports, statistical tables, publication-ready plots
```

---

## File-by-File Responsibility

### Phase 1: SAE Training
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `sparse_autoencoder.py` | `outputs/activations/layer2/{train,val,test}/` | 30 checkpoints + metrics | Train TopK, Gated, JumpReLU, Switch (11+9+6+4 configs) |
| `gnn_train.py` | Virtual graphs | `outputs/activations/layer2/{train,val,test}/` | Extract 64D layer2 activations |

### Phase 2: Configuration Comparison
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `compare_sae_configs.py` | SAE checkpoints + `layer2/test/` | `sae_config_comparison.csv` + `latent_correlations.csv` | Rank configs within each variant by composite score; save feature-motif correlations |

### Phase 2.5: Cross-Variant Comparison
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `compare_sae_variants.py` | All 30 SAE checkpoints + metrics | `sae_variant_comparison.csv` + Pareto plots | Cross-variant analysis: TopK vs Gated vs JumpReLU vs Switch trade-offs |

**What Phase 2.5 does:**
1. Analyzes all 30 configurations (same data as Phase 2, different perspective)
2. Computes cross-variant reconstruction, sparsity, interpretability metrics
3. Generates Pareto frontier plots: reconstruction MSE vs L0 sparsity per variant
4. Creates comparative visualizations and summary report
5. Justifies choice of best variant for subsequent ablation studies

**Outputs:**
- `outputs/sae_variant_comparison.csv` - Comparison table for all 30 configs
- `outputs/variant_comparison_plots/pareto_frontier.png` - Trade-off curves per variant
- `outputs/variant_comparison_plots/interpretability_heatmap.png` - Cross-variant interpretability comparison
- `outputs/variant_comparison_plots/training_efficiency.png` - Convergence speed comparison
- `outputs/variant_comparison_report.md` - Summary report with recommendations

### Phase 2b: Multi-Seed Training (REQUIRED for Publication-Ready Results)
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `retrain_best_configs.py` | Best config per variant (from Phase 2) | Multi-seed checkpoints (5 seeds each) | Retrain best TopK, Gated, JumpReLU, Switch with seeds [42, 123, 456, 789, 1011] for feature stability analysis |

**What multi-seed training enables:**
1. Feature stability analysis across random initializations
2. Dictionary convergence metrics (decoder cosine similarity across seeds)
3. Confidence intervals on correlation measurements
4. Publication-ready reproducibility claims

**Output**: `checkpoints/sae_{variant}_{params}_seed{seed}.pt` + `outputs/retrain_summary.json`
- **CRITICAL DEPENDENCY**: Phase 4 statistical analysis requires these multi-seed models

### Phase 3a: SAE Latent Space Ablations
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `compare_sae_configs.py` (Phase 2) | SAE checkpoints + `layer2/test/` | `outputs/latent_correlations.csv` | Save feature-motif correlations from Phase 2 for Phase 3a |
| `run_interpretability_experiments.py` | SAE models + `layer2/{test,mixed}/` + `latent_correlations.csv` | `ablations/interpretability_*/` | Motif-guided feature ablation with statistical controls |

**What run_interpretability_experiments.py does**:
1. Loads feature-motif correlations and significance testing results from `latent_correlations.csv`
2. Filters features by: FDR significance (< 0.05) AND effect size (|rpb| >= min_rpb threshold)
3. Groups filtered features by motif type (feedback loop, feedforward loop, cascade, single input module)
4. For each motif-specific feature group, runs ablation via `run_ablation.py`
5. Runs parallel random control trials (20+ trials per feature count) for statistical comparison
6. Computes z-scores, percentiles, and p-values vs. random controls
7. Generates monosemanticity analysis: reports % features specific to 1 motif vs polysemantic features

**What run_ablation.py measures** (called by run_interpretability_experiments.py):
1. `Loss(Original)` - GNN loss on native 64D activations
2. `Loss(Full SAE)` - GNN loss on SAE reconstruction
3. `Loss(Ablated)` - GNN loss on SAE reconstruction with features zeroed
4. `SAE Degradation` = Loss(Full SAE) - Loss(Original) [SAE reconstruction cost]
5. `Ablation Impact` = Loss(Ablated) - Loss(Full SAE) [causality measure for ablated features]

### Phase 3b: Native GNN Activation Space Validation
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `native_gnn_ablation.py` | SAE models + `layer2/test/` + Phase 2.5a correlations | `outputs/native_gnn_ablations/native_ablation_{variant}_rpb_{motif}.csv` (per motif) | Direct activation patching of top features per motif |

**CRITICAL:** Phase 3b must run ONCE PER MOTIF to match Phase 3a grouping:
```bash
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_feedback_loop
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_cascade
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_feedforward_loop
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_single_input_module
```

### Phase 3c: Motif-Group Strategy Comparison
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `compare_ablation_strategies.py` | Phase 3a motif results + Phase 3b motif results | `ablation_strategy_comparison/motif_agreement_*.csv` | Compare SAE vs native ablation impacts per motif |

### Phase 4: Statistical Validation (Requires Phase 2b Multi-Seed Models)
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `statistical_analysis_suite.py` | Multi-seed checkpoints (Phase 2b) + Phase 2-3 outputs | `outputs/statistical_analysis/` | Correlation distributions, feature stability across seeds, redundancy analysis, FDR-corrected p-values |

**CRITICAL PREREQUISITES:**
- Phase 2b must be completed first (retrain_best_configs.py)
- Multi-seed checkpoints: `checkpoints/sae_{variant}_{params}_seed{seed}.pt` with seeds [42, 123, 456, 789, 1011]

**What Phase 4 computes with multi-seed data:**
1. `--seed-analysis`: Feature stability (% features with cosine similarity > 0.8 across seeds)
2. Correlation distributions (per variant, per motif)
3. Feature redundancy analysis (decoder similarity between features)
4. Ablation conditional effects (Wilcoxon tests with Cohen's d effect sizes)
5. Sparsity-interpretability trade-off curves

### Phase 5: Visualization & Analysis
| File | Input | Output | Purpose |
|------|-------|--------|---------|
| `visualize_feature_activations.py` | SAE models + `layer2/test/` | Feature activation heatmaps | Feature selectivity analysis |
| `analyze_sae_reconstruction_fidelity.py` | SAE models + `layer2/{test,mixed}/` | PCA histograms | Reconstruction quality validation |

---

## Execution Sequences

### Standard Pipeline with Multi-Seed (RECOMMENDED for Publication, 11 steps, ~17 hours)
```bash
1. sparse_autoencoder.py                              # 5 hours (train 30 SAE configs, seed=42)
2. compare_sae_configs.py                             # 30 min (rank within variants + save latent_correlations.csv)
2.5a. analyze_feature_significance.py (per variant)   # 1 hour (permutation testing + FDR for feature-motif significance)
2.5b. compare_sae_variants.py                         # 3 min (cross-variant Pareto frontiers)
2b. retrain_best_configs.py                           # 2.7 hours (REQUIRED: retrain best configs with 5 seeds)
3a. run_interpretability_experiments.py               # 2 hours (motif-guided SAE ablations, groups features by motif)
3b. native_gnn_ablation.py --use-rpb per motif        # 2 hours (native patching of top features per motif, 4 runs)
3c. compare_ablation_strategies.py --motif-mode       # 20 min (motif-by-motif agreement metrics)
3d. Mixed-motif generalization test (REQUIRED)        # 45 min
    - generate_mixed_motif_activations.py
    - run_ablation.py --use_mixed_motifs [features_from_2.5a]
    - native_gnn_ablation.py --use_mixed_motifs [features_from_2.5a]
4. statistical_analysis_suite.py                      # 30 min (INCLUDES --seed-analysis for multi-seed stability)
5. Visualization & Reconstruction Analysis            # 20 min (Phase 5)
```

**Why Phase 2b is mandatory:**
- Generates multi-seed checkpoints needed for reproducibility claims
- Enables feature stability analysis (Phase 4)
- Provides confidence intervals on all correlation metrics
- Required for publication-ready results addressing reviewer concerns

**Why Phase 3d (Mixed-Motif Generalization) is mandatory:**
- Validates that features identified in single-motif context generalize to realistic mixed-motif settings
- Tests robustness: Do SAE-learned features still work when 2-3 motifs interact?
- Strengthens mechanistic interpretation claims
- Provides evidence that interpretability is not an artifact of overly-simple graphs

### Quick Pipeline - Single-Seed Only (~8 hours, NOT RECOMMENDED for publication)
```bash
1. sparse_autoencoder.py                    # 5 hours (seed=42 only)
2. compare_sae_configs.py                   # 30 min (saves latent_correlations.csv)
2.5. compare_sae_variants.py (optional)     # 3 min (cross-variant comparison)
3. run_interpretability_experiments.py      # 2 hours (motif-guided ablations)
4. statistical_analysis_suite.py            # 30 min (limited to correlation/redundancy/tradeoff analysis)
   NOTE: --seed-analysis will be SKIPPED (requires Phase 2b multi-seed models)
```

### Individual Experiments
```bash
# Motif-guided ablation with statistical controls (requires Phase 2 correlation data)
python run_interpretability_experiments.py --latent_dim 512 --k 8 --min_rpb 0.05 --n_random_trials 20

# Strict threshold (only strong feature-motif correlations)
python run_interpretability_experiments.py --latent_dim 512 --k 8 --min_rpb 0.15 --n_random_trials 50

# Test on mixed-motif graphs using dominant motif labels
python run_interpretability_experiments.py --latent_dim 512 --k 8 --min_rpb 0.08 --use_mixed_motifs

# Single feature ablation (quick exploration - useful for debugging)
python run_ablation.py --latent_dim 512 --k 8 --feature z496

# Multiple specific features in one call
python run_ablation.py --latent_dim 512 --k 8 --feature z1,z10,z50,z100

# Native validation for one variant
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --all-features

# Feature visualization
python visualize_feature_activations.py --variant topk --latent_dim 512 --features 20

# Reconstruction analysis
python analyze_sae_reconstruction_fidelity.py --variant topk --latent-dim 512 --k 8
```

---

## Key Data Files

### Input Requirements
| File | Purpose | Generated By |
|------|---------|--------------|
| `outputs/activations/layer2/test/*.pt` | 64D node activations | gnn_train.py |
| `checkpoints/gnn_model.pt` | GNN model for loss computation | gnn_train.py |
| `checkpoints/sae_*.pt` | Trained SAE models | sparse_autoencoder.py |
| `virtual_graphs/data/all_graphs/raw_graphs/*.pkl` | Graph structures | graph_motif_generator.py |
| `virtual_graphs/data/all_graphs/graph_motif_metadata/*.csv` | Node motif labels | graph_motif_generator.py |

### Key Intermediate Outputs
| File | Used By | Purpose |
|------|---------|---------|
| `outputs/sae_config_comparison.csv` | Many downstream scripts | Best configs per variant |
| `outputs/latent_correlations.csv` | run_interpretability_experiments.py | Feature-motif correlations + significance from Phase 2 |
| `outputs/latent_cache/*.pkl` | compare_sae_configs.py | Cached SAE latents (optimization) |
| `outputs/test_graph_ids.json` | run_ablation.py, native_gnn_ablation.py, run_interpretability_experiments.py | Test set definition |
| `ablations/results/ablation_*.csv` | compare_ablation_strategies.py, run_interpretability_experiments.py | Ablation impact results |
| `ablations/interpretability_*/` | Phase 4 statistical analysis | Motif-guided ablation results with statistical controls |

---

## Three-Level Analysis Strategy

### Level 1: Feature Characterization
**Purpose**: Identify which features matter for which motifs

**Implemented By**:
- `compare_sae_configs.py`: Point-biserial correlation r_pb
- **Output**: Correlation matrix, feature rankings per motif

**Key Metric**:
```
r_pb(feature_i, motif_j) = correlation between feature activation and motif presence
```

### Level 2: Causality Validation
**Purpose**: Confirm features actually cause changes in GNN behavior

**Implemented By**:
- `run_ablation.py`: SAE latent space ablations
- `native_gnn_ablation.py`: Native activation ablations
- `compare_ablation_strategies.py`: Agreement metrics

**Key Tests**:
```
GNN_loss(original) < GNN_loss(ablated_feature)  →  Feature is causal
SAE_ablation ≈ Native_ablation  →  SAE is valid
```

### Level 3: Statistical Rigor
**Purpose**: Ensure results are reproducible and statistically significant

**Implemented By**:
- Multi-seed training: 5 seeds [42, 123, 456, 789, 1011]
- Permutation testing: FDR-corrected p-values
- Effect sizes: Cohen's d, rank-biserial correlation
- Stability: Decoder weight cosine similarity across seeds

**Key Statistics**:
```
Wilcoxon signed-rank test  →  Statistical significance
Cohen's d                  →  Effect size
95% Bootstrap CI           →  Confidence bounds
```

---

## Common Tasks & Where to Find Them

### "Which features encode motif X?"
→ `compare_sae_configs.py` output: `sae_config_comparison.csv` → filter by motif → rank by r_pb

### "How much does feature Y matter?"
→ `run_ablation.py --feature Y` → look at ablation_impact column

### "Is SAE valid?"
→ `compare_ablation_strategies.py` → look at correlation between SAE and native ablation results

### "Are features stable across runs?"
→ Phase 2b: `retrain_best_configs.py` (retrains best config with 5 seeds)
→ Phase 4: `statistical_analysis_suite.py --seed-analysis` (computes feature similarity across seeds)
→ Output: `outputs/statistical_analysis/feature_stability.png` (% stable features per variant)

### "What's the best configuration?"
→ `compare_sae_configs.py` output: `sae_config_comparison.csv` → highest composite_score row

### "How redundant are the features?"
→ `statistical_analysis_suite.py --redundancy` → look at feature_redundancy_heatmap

### "What's the sparsity-interpretability trade-off?"
→ `statistical_analysis_suite.py --tradeoff` → varies K and measures both metrics

---

## Critical Checkpoints

Before proceeding to next phase, verify:

| Phase | Checkpoint | How to Verify |
|-------|-----------|--------------|
| 1 | SAE training completed | `ls checkpoints/sae_*.pt` → should have 30 files (seed=42) |
| 2 | Config ranking + correlations saved | `ls outputs/sae_config_comparison.csv` `outputs/latent_correlations.csv` |
| 2b | Multi-seed training complete | `ls checkpoints/sae_*_seed*.pt` → should have 20 files (4 variants × 5 seeds) |
| 2b | Stability summary generated | `cat outputs/retrain_summary.json` → verify MSE statistics per variant |
| 3a | Motif-guided ablations ready | `ls ablations/interpretability_*/*.csv` → motif-specific results |
| 3b | Native ablations complete | `ls outputs/native_gnn_ablations/*.csv` → direct patching results |
| 4 | Statistics computed | `ls outputs/statistical_analysis/*.png` → visualizations + feature_stability.png |

---

## Known Issues & Resolutions (ALREADY FIXED)

### ✅ Issue 1: Activation Directory Consistency (RESOLVED)
**Status**: All layer2_new references have been replaced with layer2
- All scripts use `outputs/activations/layer2/` consistently
- Phases 2 and 3 use identical activation sources
- Python files: ✅ Fixed
- Jupyter notebooks: ✅ Fixed (sae_activations_motif_new.ipynb, sae_colab_pipeline.ipynb)

### ✅ Issue 2: test_graph_ids.json Location
**Status**: Correctly created by compare_sae_configs.py during Phase 2
- Location: `outputs/test_graph_ids.json`
- Generated: Automatically during Phase 2 execution
- Usage: Loaded by Phase 3 scripts (run_ablation.py, native_gnn_ablation.py)

### ✅ Issue 3: Feature Indexing Convention
**Status**: Features are 1-indexed in output (z1, z2, ..., z512) but 0-indexed internally (0-511)
- Command line: Use feature indices 0-511 for --feature-idx arguments
- CSV output: Shows z1-z512 column headers (1-indexed for readability)

---

## Troubleshooting Guide

### Problem: "Checkpoint not found"
**Solution**: Verify Phase 1 (sparse_autoencoder.py) completed successfully
```bash
ls -la checkpoints/sae_*.pt  # Should have 30 files
```

### Problem: "No data found for variant/motif"
**Solution**: Check if config_comparison.csv was generated
```bash
python compare_sae_configs.py  # Re-run Phase 2
```

### Problem: "Activation directory not found"
**Solution**: Verify activations are in unified directory structure
```bash
ls -la outputs/activations/layer2/test/  # Should have ~500 .pt files
# NOT outputs/activations/layer2_new/test/
```

### Problem: Ablation results incomplete
**Solution**: Run with all features instead of single feature
```bash
python run_ablation.py --variant topk --latent_dim 512 --k 8 --all-features
```

### Problem: "Multi-seed checkpoints not found"
**Solution**: Run retrain_best_configs.py first
```bash
python retrain_best_configs.py --variant all --seeds 5
```

---

## Output Locations Summary

```
checkpoints/
├── sae_topk_*.pt              (TopK SAE models)
├── sae_gated_*.pt             (Gated SAE models)
├── sae_jumprelu_*.pt          (JumpReLU SAE models)
├── sae_switch_*.pt            (Switch SAE models)
└── gnn_model.pt               (GNN model for loss computation)

outputs/
├── sae_config_comparison.csv         (Phase 2: Rankings)
├── sae_variant_comparison.csv        (Phase 2: Cross-variant)
├── test_graph_ids.json               (Phase 2: Test set)
├── latent_cache/                     (Phase 2: Cached activations)
├── native_gnn_ablations/             (Phase 3b: Native ablation results)
├── statistical_analysis/             (Phase 4: Statistics + visualizations)
├── sae_reconstruction_fidelity/      (Phase 5: PCA analysis)
└── feature_activation_visualizations/ (Phase 5: Feature plots)

ablations/
├── results/
│   └── ablation_*.csv               (Phase 3a: Ablation results)
└── plots/
    └── ablation_*.png               (Phase 3a: Visualizations)
```

---

## Publication-Ready Outputs

After completing full pipeline:

**For Methods Section**:
- `outputs/sae_config_comparison.csv` → SAE hyperparameter selection
- `outputs/native_gnn_ablations/agreement.png` → SAE validation

**For Results Section**:
- `outputs/statistical_analysis/correlation_distributions.png` → Feature-motif associations
- `ablations/plots/ablation_*.png` → Feature causality
- `outputs/statistical_analysis/feature_stability.png` → Reproducibility
- `outputs/statistical_analysis/redundancy_heatmaps.png` → Model efficiency

**For Appendix**:
- `outputs/sae_reconstruction_fidelity/*.png` → Reconstruction quality
- `outputs/feature_activation_visualizations/*.png` → Feature selectivity

---

## Key Research Questions Answered

| Question | Component | Answer Location |
|----------|-----------|-----------------|
| Which features encode motifs? | Phase 2 | sae_config_comparison.csv (r_pb column) |
| Are the features causal? | Phase 3 | ablation_*.csv (ablation_impact) |
| Is the SAE valid? | Phase 3b | agreement metrics (correlation) |
| Are findings reproducible? | Phase 4 | stability plots (% stable features) |
| How efficient is the model? | Phase 4 | redundancy analysis (% redundant) |
| What's the best tradeoff? | Phase 4 | sparsity_tradeoff.png |
| How good is reconstruction? | Phase 5 | pca_histograms.png (distribution match) |

