# Multi-Seed Training Architecture (CORRECTED)

## Overview

The project uses a multi-seed strategy for training and analyzing SAE variants. This document clarifies the flow from Phase 1 through Phase 4.

## Phase Breakdown

### Phase 1: Initial SAE Training (30 configs, seed=42)

**What happens:**
- Trains all 30 SAE configurations across 4 variants (TopK: 11 configs, Gated: 9 configs, JumpReLU: 6 configs, Switch: 4 configs)
- All training uses **seed=42 only**
- Saves checkpoints with seed in filename

**Checkpoint naming:**
```
TopK:       checkpoints/sae_topk_latent{latent_dim}_k{k}_seed42.pt
Gated:      checkpoints/sae_gated_latent{latent_dim}_lambda{sparsity_coef:.0e}_seed42.pt
JumpReLU:   checkpoints/sae_jumprelu_latent{latent_dim}_thresh{threshold_init:.0e}_bw{bandwidth:.0e}_seed42.pt
Switch:     checkpoints/sae_switch_experts{num_experts}_latent{total_latent}_k{k_per_expert}_seed42.pt
```

**Outputs:**
- 30 checkpoint files (one per config, all seed=42)
- Saved to: `checkpoints/`

---

### Phase 2: Evaluation & Ranking (All 30 seed=42 configs)

**What happens:**
- Evaluates all 30 seed=42 checkpoints
- Computes point-biserial correlation (max_rpb_abs) for each config
- Identifies **best config from EACH variant** (4 best configs total)
- Also computes feature-motif correlations for all configs

**Outputs:**
1. **`outputs/sae_config_comparison.csv`** - All 30 configs ranked
   - Columns: variant, latent_dim, k (TopK), sparsity_coef (Gated), threshold_init (JumpReLU), num_experts/latent_per_expert/k_per_expert (Switch), max_rpb_abs, composite_score, etc.
   - **Sorted by composite_score** (CURRENT - should change to max_rpb_abs per Phase 3 needs)
   - Contains all variant-specific parameters needed to reconstruct checkpoint filenames

2. **`outputs/latent_correlations.csv`** - Feature-motif correlations
   - Columns: feature, motif, rpb, rpb_abs, variant, latent_dim, k/sparsity_coef/threshold_init/num_experts, etc.
   - Used by Phase 3a to identify top features per motif

---

### Phase 2b: Multi-Seed Retraining (Best configs from each variant)

**What happens:**
- Identifies the 4 best configs from Phase 2 (one per variant)
- Retrains ONLY those 4 best configs with seeds [123, 456, 789, 1011]
- Does NOT retrain with seed=42 (already exists from Phase 1)

**Checkpoint naming:**
```
For best TopK config from Phase 2:
  checkpoints/sae_topk_latent512_k8_seed123.pt
  checkpoints/sae_topk_latent512_k8_seed456.pt
  checkpoints/sae_topk_latent512_k8_seed789.pt
  checkpoints/sae_topk_latent512_k8_seed1011.pt

(Same pattern for best Gated, JumpReLU, Switch)
```

**Outputs:**
- 4 configs × 4 new seeds = 16 new checkpoint files
- Saved to: `checkpoints/`

**Total checkpoints after Phase 2b:**
- 30 (Phase 1, seed=42) + 16 (Phase 2b, seeds 123/456/789/1011) = 46 total

---

### Phase 3: Ablation Studies (Single best config + 4 motifs)

**Which checkpoint does Phase 3 use?**
- Phase 3a selects the **best overall config from Phase 2** (highest max_rpb_abs in sae_config_comparison.csv)
- Uses the **seed=42 checkpoint** (from Phase 1)
- Example: If topk_latent512_k8 is best, uses `checkpoints/sae_topk_latent512_k8_seed42.pt`

**Why seed=42 not the retrained versions?**
- Phase 3a is designed for quick ablation analysis on the single best config
- Phase 3 evaluates the SAME latent space discovered in Phase 1 (seed=42)
- Multi-seed analysis happens in Phase 4, not Phase 3

**Phase 3 Phases:**

#### Phase 3a: SAE Latent Space Ablations
- **Input:** Best config's seed=42 checkpoint + feature-motif correlations from Phase 2
- **Process:** For each of 4 motifs, ablate top SAE features correlated with that motif
- **Output:** 4 CSV files (one per motif), saved with variant in filename

#### Phase 3b: Native GNN Validation
- **Input:** Best config's seed=42 checkpoint + top features per motif
- **Process:** Direct activation patching of same feature groups (4 motif-specific runs)
- **Output:** 4 CSV files (one per motif), saved with variant in filename

#### Phase 3c: SAE vs Native Comparison
- **Input:** Phase 3a + Phase 3b outputs (4 motifs each)
- **Process:** Compare ablation impacts between SAE and native approaches
- **Output:** Agreement metrics and plots

---

### Phase 4: Statistical Validation & Stability (All 5 seeds of 4 best configs)

**Which checkpoints does Phase 4 use?**
- All 5 seed versions of the 4 best configs identified in Phase 2
- Example: All 5 seeds of best TopK, all 5 seeds of best Gated, etc.

**Checkpoints needed per variant:**
```
For best TopK config:
  checkpoints/sae_topk_latent512_k8_seed42.pt    (from Phase 1)
  checkpoints/sae_topk_latent512_k8_seed123.pt   (from Phase 2b)
  checkpoints/sae_topk_latent512_k8_seed456.pt   (from Phase 2b)
  checkpoints/sae_topk_latent512_k8_seed789.pt   (from Phase 2b)
  checkpoints/sae_topk_latent512_k8_seed1011.pt  (from Phase 2b)

(Repeat for best Gated, JumpReLU, Switch)
```

**Phase 4 analyses:**
- Feature stability across seeds (which features consistently encode motifs)
- Redundancy analysis (correlation across seeds)
- Robustness metrics

---

## Summary: What Phase 3a Needs to Work

After Phase 1 + Phase 2 complete, Phase 3a has everything needed:

1. ✅ **Checkpoint files exist:**
   - All seed=42 checkpoints saved by Phase 1
   - Best config's seed=42 is available for Phase 3

2. ✅ **Metadata available in sae_config_comparison.csv:**
   - Which variant is best (column: `variant`)
   - All variant-specific parameters (k, sparsity_coef, threshold_init, num_experts, etc.)
   - Feature quality metrics (max_rpb_abs, composite_score, etc.)

3. ✅ **Feature selection available in latent_correlations.csv:**
   - Feature-motif correlations for all 30 configs
   - Per-feature significance scores

4. **No additional outputs needed from Phase 1/2**
   - Everything required by Phase 3a is already saved

---

## Critical Files Referenced

| Phase | Output File | Purpose | Used By |
|-------|------------|---------|---------|
| 1 | `checkpoints/sae_VARIANT_PARAMS_seed42.pt` | SAE models | Phase 2, 3, 4 |
| 2 | `outputs/sae_config_comparison.csv` | Config ranking (best per variant) | Phase 3a |
| 2 | `outputs/latent_correlations.csv` | Feature-motif correlations | Phase 3a |
| 3a | `ablations/results/MOTIF_VARIANT_PARAMS_results.csv` | SAE ablation results | Phase 3c |
| 3b | `outputs/native_gnn_ablations/native_ablation_VARIANT_rpb_MOTIF.csv` | Native ablation results | Phase 3c |
| 2b | `checkpoints/sae_VARIANT_PARAMS_seed{123,456,789,1011}.pt` | Multi-seed models | Phase 4 |

---

## Key Design Decisions

1. **Phase 3 uses single best config, not per-variant**: Focuses analysis on the overall best performing configuration
2. **Phase 3 uses seed=42 only**: Analyzes the specific latent space discovered in initial training
3. **Phase 4 uses all 5 seeds**: Validates stability and robustness across multiple random initializations
4. **Best config selection by max_rpb_abs**: Most directly relevant metric for feature-motif relationship strength
