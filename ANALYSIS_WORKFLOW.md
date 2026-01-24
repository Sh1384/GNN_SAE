# Complete Analysis Workflow Guide

A step-by-step guide to run the entire SAE variants analysis pipeline from training through comprehensive comparisons.

---

## Quick Start (8 Steps - Complete Pipeline)

```bash
# Phase 1: Train all variants (30 configs, ~5 hours)
python sparse_autoencoder.py

# Phase 2: Within-variant comparison (auto-identifies best configs)
python compare_sae_configs.py

# Phase 2b: Retrain best configs with multiple seeds (~2.7 hours)
python retrain_best_configs.py

# Phase 3: Cross-variant comparison
python compare_sae_variants.py

# Phase 4: Run statistical analysis
python statistical_analysis_suite.py --variant all --redundancy --tradeoff

# Phase 5: Validate with native ablations
python native_gnn_ablation.py --all-features --variant topk

# Phase 6: Compare ablation strategies
python compare_ablation_strategies.py --all-variants

# Phase 8: Feature stability analysis (requires multi-seed training)
python statistical_analysis_suite.py --seed-analysis
```

---

## Detailed Workflow

### Phase 1: Model Training (5 hours)

#### Step 1: Train All SAE Variants
```bash
python sparse_autoencoder.py
```

**What happens:**
- Trains 30 configurations sequentially with seed=42
  - TopK: 11 configs (latent_dim × k combinations)
  - Gated: 9 configs (latent_dim × sparsity_coef combinations)
  - JumpReLU: 6 configs (latent_dim × threshold_init combinations)
  - Switch: 4 configs (num_experts × latent_per_expert × k combinations)

**Outputs:**
- `checkpoints/sae_topk_latent*_k*_seed42.pt` (11 files)
- `checkpoints/sae_gated_latent*_lambda*_seed42.pt` (9 files)
- `checkpoints/sae_jumprelu_latent*_thresh*_seed42.pt` (6 files)
- `checkpoints/sae_switch_experts*_latent*_seed42.pt` (4 files)
- `outputs/sae_metrics_*.json` (30 files with training metrics)

**Estimated time:** ~5 hours
- 10 minutes per config × 30 configs

**Monitor progress:**
```bash
tail -f training_log.txt  # Real-time output
```

---

### Phase 2: Configuration Analysis (30 minutes)

#### Step 2: Within-Variant Comparison
```bash
python compare_sae_configs.py
```

**What happens:**
- Compares hyperparameters within each variant
- Identifies best configuration per variant

**Outputs:**
- `outputs/sae_config_comparison.csv`
- Console: Top 5 configs ranked by composite score
- Console: Recommended config per variant

**Key findings:**
- Best TopK config (latent_dim, k)
- Best Gated config (latent_dim, sparsity_coef)
- Best JumpReLU config (latent_dim, threshold_init)
- Best Switch config (num_experts, latent_per_expert, k)

**Use this for:**
- Identifying best hyperparameters per variant
- Informing downstream ablation analysis

---

### Phase 3: Cross-Variant Comparison (15 minutes)

#### Step 3: Compare All Variants
```bash
python compare_sae_variants.py
```

**What happens:**
- Aggregates metrics across all variants
- Generates comparison plots and report

**Outputs:**
```
outputs/
├── sae_variant_comparison.csv
├── variant_comparison_plots/
│   ├── pareto_frontier.png          (Reconstruction vs Sparsity)
│   ├── interpretability_comparison.png (Max |r_pb| by variant)
│   ├── convergence_comparison.png   (Training efficiency)
│   └── radar_chart.png              (Multi-metric overview)
└── variant_comparison_report.md     (Summary report)
```

**Interpret results:**
- Which variant has best reconstruction?
- Which variant has best interpretability?
- Which variant converges fastest?
- What are the trade-offs?

---

### Phase 4: Ablation Analysis (Latent Space)

#### Step 4: SAE Latent Ablations
```bash
# Single feature ablation
python run_ablation.py --variant topk --latent_dim 512 --k 8 --feature z1

# Batch ablation (all significant features)
python run_ablation.py --variant topk --latent_dim 512 --k 8 --all-features
```

**What happens:**
- For each feature, zeros out SAE latent representation
- Measures GNN performance degradation
- Generates ablation plots

**Outputs:**
```
ablations/
├── results/
│   ├── ablation_topk_feature*.csv
│   └── ...
└── plots/
    ├── ablation_topk_*.png
    └── ...
```

**Key metrics:**
- Δ MSE for each feature
- Feature importance ranking
- Motif-specific effects

**Typical findings:**
- Some features have large impact (important)
- Some features have small impact (noise)
- Impact varies by motif type

---

### Phase 5: Native GNN Ablations (Validation)

#### Step 5: Native Activation Patching
```bash
# Single feature, activation patching
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --feature 0

# All features
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --all-features
```

**What happens:**
- Identifies nodes most activated by SAE feature
- Zeros out their full 64-dim activations (no SAE reconstruction)
- Measures direct impact on GNN

**Outputs:**
```
outputs/native_gnn_ablations/
├── native_ablation_topk_feature*.csv
├── native_ablation_results.png
└── native_ablation_conditional_motif.csv
```

**Key metrics:**
- Δ Loss for each feature (native space)
- Conditional effects (with/without motif)
- Statistical significance (Wilcoxon test)

**Expected observation:**
- Strong correlation with SAE latent ablations validates SAE assumptions
- Weak correlation suggests nonlinear effects or reconstruction confounding

---

### Phase 6: Strategy Validation & Comparison

#### Step 6: Compare Ablation Strategies
```bash
# Compare SAE latent vs native for specific variant
python compare_ablation_strategies.py --variant topk --latent_dim 512 --k 8

# All variants
python compare_ablation_strategies.py --all-variants
```

**What happens:**
- Loads SAE latent ablation results from run_ablation.py
- Loads native GNN ablation results from native_gnn_ablation.py
- Merges by graph_id and computes agreement metrics

**Outputs:**
```
outputs/ablation_strategy_comparison/
├── strategy_comparison_topk_feature*.png  (4-panel plots)
├── ablation_strategy_comparison.md        (Interpretation guide)
└── agreement_matrix.csv                   (Correlation per feature)
```

**Agreement metrics:**
- Pearson correlation of Δ Loss values
- Spearman rank correlation
- Direction agreement (% same sign effect)
- Wilcoxon test (statistical significance)

**Interpretation:**
```
r > 0.8  ✓ Strong agreement      → SAE assumptions valid, reconstruction OK
0.5-0.8  ⚠ Moderate agreement   → Some confounding, report both methods
r < 0.5  🔍 Weak agreement      → Nonlinear effects, use native as primary
```

---

### Phase 7: Statistical Validation (Stanford Requirements)

#### Step 7: Comprehensive Statistical Analysis
```bash
# All analyses
python statistical_analysis_suite.py --variant all --redundancy --tradeoff

# Feature stability (requires multi-seed training)
python statistical_analysis_suite.py --seed-analysis

# Ablation conditional analysis
python statistical_analysis_suite.py --ablation-type both
```

**What happens:**
- Analyzes correlation distributions per architecture and motif
- Measures feature stability across seeds (if available)
- Conditional ablation analysis (with/without motif)
- Computes feature redundancy
- Analyzes sparsity-interpretability trade-offs

**Outputs:**
```
outputs/statistical_analysis/
├── correlation_distributions.png        (4×4: variant × motif)
├── feature_stability.png                (Multi-seed stability)
├── ablation_conditional_effects.png     (2×2: motif × direction)
├── feature_redundancy_heatmaps.png      (4: one per variant)
└── sparsity_tradeoff_analysis.png       (2-panel: trade-off curves)
```

**Key metrics:**
- Distribution of |r_pb| per variant-motif pair
- % stable features across seeds
- Wilcoxon p-values for conditional effects
- Cohen's d effect sizes
- Feature redundancy rates

---

### Phase 8: Multi-Seed Stability Analysis (Implemented)

#### Step 8a: Retrain Best Configs with Multiple Seeds
```bash
# Auto-detect best configs and retrain with 5 seeds each
python retrain_best_configs.py

# Or with custom seeds
python retrain_best_configs.py --seeds 42 123 456 789 1011

# Or specific variant only
python retrain_best_configs.py --variant topk
```

#### Step 8b: Analyze Feature Stability
```bash
python statistical_analysis_suite.py --seed-analysis
```

**What happens:**
- Loads decoder weights from all seeds
- Computes cosine similarity matrices between decoders
- Identifies stable features (consistent across seeds)

**Outputs:**
```
outputs/statistical_analysis/
├── feature_stability.png
├── stability_report.csv
└── seed_decoder_similarities.json
```

**Key metrics:**
- % stable features per variant
- Mean similarity (average agreement across seeds)
- Feature correspondence analysis

---

## Output Files Summary

### Training Outputs
```
checkpoints/                          # 30 SAE models
├── sae_topk_latent*_k*_seed42.pt
├── sae_gated_latent*_lambda*_seed42.pt
├── sae_jumprelu_latent*_thresh*_seed42.pt
└── sae_switch_experts*_latent*_seed42.pt

outputs/
├── sae_metrics_*.json                 # 30 training metrics files
├── sae_config_comparison.csv          # Within-variant comparison
├── sae_variant_comparison.csv         # Cross-variant metrics
│
├── variant_comparison_plots/
│   ├── pareto_frontier.png
│   ├── interpretability_comparison.png
│   ├── convergence_comparison.png
│   └── variant_comparison_report.md
│
├── native_gnn_ablations/
│   ├── native_ablation_*.csv
│   └── native_ablation_results.png
│
├── ablation_strategy_comparison/
│   ├── strategy_comparison_*.png
│   ├── ablation_strategy_comparison.md
│   └── agreement_matrix.csv
│
└── statistical_analysis/
    ├── correlation_distributions.png
    ├── feature_stability.png
    ├── ablation_conditional_effects.png
    ├── feature_redundancy_heatmaps.png
    └── sparsity_tradeoff_analysis.png

ablations/
├── results/
│   └── ablation_*.csv                 # SAE latent ablation results
└── plots/
    └── ablation_*.png                 # Ablation visualizations
```

---

## Interpretation Guide

### 1. Which Variant to Use?

**For best reconstruction:** Look at `pareto_frontier.png`
- Find variant with lowest test MSE
- Note: May have higher sparsity cost

**For best interpretability:** Look at `interpretability_comparison.png`
- Find variant with highest max |r_pb|
- Most selective feature-motif correlation

**For best trade-off:** Look at `sae_variant_comparison.csv`
- Sort by `composite_score` (balances all metrics)
- Best overall configuration

### 2. Are SAE Assumptions Valid?

**Check strategy agreement:** `ablation_strategy_comparison.md`
- If r > 0.8: ✓ SAE assumptions valid
- If 0.5 < r < 0.8: ⚠ Report both methods
- If r < 0.5: 🔍 Use native ablations as primary

### 3. What Features are Important?

**Check ablation rankings:**
- Load `ablation_topk_feature*.csv`
- Sort by |delta_mse| to find top features
- Cross-reference with correlation analysis
- Confirm selectivity (high when motif present)

### 4. Are Features Stable?

**Check feature stability** (if multi-seed available):
- Look at `feature_stability.png`
- % stable features > 80% = reproducible
- < 50% = high variance, questionable

### 5. How Much Redundancy?

**Check `feature_redundancy_heatmaps.png`:**
- High off-diagonal values = redundant features
- Redundancy rate > 30% = consider smaller latent dim
- Rate < 10% = efficient feature usage

---

## Troubleshooting

### Training stuck or slow?
```bash
# Check GPU usage
nvidia-smi

# Check data availability
ls -lh outputs/activations/layer2/*/
```

### Missing checkpoint files?
```bash
# Verify checkpoint path
ls -lh checkpoints/sae_*.pt | wc -l  # Should see 30 files

# Check specific variant
ls -lh checkpoints/sae_topk*.pt | wc -l  # Should see 11 files
```

### No results in comparison plots?
```bash
# Verify metrics files exist
ls -lh outputs/sae_metrics_*.json | wc -l  # Should see 30 files

# Check one metrics file
python -c "import json; print(json.load(open('outputs/sae_metrics_topk_latent512_k8_seed42.json')))"
```

### Ablation results not found?
```bash
# For SAE latent ablations
ls -lh ablations/results/ablation_*.csv

# For native ablations
ls -lh outputs/native_gnn_ablations/native_*.csv
```

---

## Generated Figures

After running the complete workflow, you have comprehensive analysis figures:

1. **Variant Comparison** → `pareto_frontier.png`
   - Shows reconstruction-sparsity trade-off
   - Clearly highlights variant differences

2. **Interpretability** → `interpretability_comparison.png`
   - Bar chart of max |r_pb| per variant
   - Demonstrates feature-motif selectivity

3. **Ablation Validation** → `ablation_conditional_effects.png`
   - Shows selective degradation by motif
   - Key evidence for causal mechanism

4. **Strategy Comparison** → `strategy_comparison_*.png`
   - Validates SAE assumptions

5. **Feature Redundancy** → `feature_redundancy_heatmaps.png`
   - Shows decoder feature relationships
   - Suggests optimal latent dimensionality

6. **Sparsity Trade-off** → `sparsity_tradeoff_analysis.png`
   - Visualization of key design choices
   - Justifies hyperparameter selection

---

## References

This analysis framework implements TopK, Gated, JumpReLU, and Switch SAE variants with comprehensive cross-variant comparison, native activation space ablations, and statistical validation following established best practices.

### References for Methods
- **Gated SAE:** Rajamanoharan et al. (2024)
- **JumpReLU SAE:** Rajamanoharan et al. (2024)
- **Switch SAE:** Wei et al. (2024)
- **Activation Patching:** Geiger et al. (2023)

---

## Contact

For questions or issues:
1. Check `IMPLEMENTATION_SUMMARY.md` for technical details
2. Review tool docstrings: `python compare_sae_variants.py --help`
3. Check output CSV files for raw data
4. Review generated markdown reports for interpretation

---

**Last Updated:** 2026-01-10
**Status:** Ready for Training & Analysis
