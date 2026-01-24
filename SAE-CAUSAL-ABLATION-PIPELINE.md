# GNN-SAE Causal Ablation Pipeline: Complete Reference

**Status**: ✅ Production Ready
**Last Updated**: January 2026
**Audience**: Pipeline operators, researchers, reproducibility verification

---

## Table of Contents
1. [Critical Prerequisites](#critical-prerequisites)
2. [Pipeline Architecture Overview](#pipeline-architecture-overview)
3. [SAE Variants Architecture](#sae-variants-architecture)
4. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
5. [Multi-Seed Training Strategy](#multi-seed-training-strategy)
6. [Execution Sequences](#execution-sequences)
7. [File Responsibilities & Data Flow](#file-responsibilities--data-flow)
8. [Phase 3c: Ablation Strategy Comparison (Single vs All-Variants)](#phase-3c-ablation-strategy-comparison)
9. [Key Data Files](#key-data-files)
10. [Critical Checkpoints](#critical-checkpoints)
11. [Troubleshooting Guide](#troubleshooting-guide)

---

## Critical Prerequisites

**Before running the SAE training/ablation pipeline, complete these steps first:**

1. **Generate Virtual Graphs** (run locally)
   - Script: `graph_motif_generator.py`
   - Creates: `virtual_graphs/data/all_graphs/raw_graphs/*.pkl` (5000 graphs, 4000 single-motif + 1000 mixed-motif)

2. **Train GNN Model & Extract Activations** (run locally)
   - Script: `gnn_train.py`
   - Creates:
     - `checkpoints/gnn_model.pt` (trained GNN model)
     - `outputs/activations/layer2/{train,val,test,mixed}/` (activations)
       - train: 3000 graphs, graphs 0-2999
       - val: 500 graphs, graphs 3000-3499
       - test: 500 graphs, graphs 3500-3999
       - mixed: 1000 graphs, graphs 4000-4999
     - `outputs/test_graph_ids.json` (test set definition)
     - `virtual_graphs/data/all_graphs/graph_motif_metadata/*.csv` (motif labels: Feedback Loop, Cascade, Feedforward Loop, Single Input Module)

3. **Upload to Google Drive** (before running Colab notebook)
   - All Python scripts
   - Activation data (`outputs/activations/layer2/`)
   - Test graph IDs (`outputs/test_graph_ids.json`)
   - GNN model checkpoint
   - Motif metadata

**✅ The Colab notebook assumes this prerequisite data already exists.**

---

## Pipeline Architecture Overview

```
RAW DATA (Single-Motif & Mixed-Motif Graphs)
├─ Virtual Graphs (5000 total)
├─ GNN Activations (64-dim Layer 2)
└─ Motif Metadata (FFL, FBL, SIM, CASC)
        ↓
PHASE 1: SAE TRAINING (30 configs, seed=42)
├─ 4 variants: TopK (11), Gated (9), JumpReLU (6), Switch (4)
└─ Output: 30 checkpoints + metrics
        ↓
PHASE 2: CONFIGURATION COMPARISON (rank all 30)
├─ Compute feature-motif correlations (max_rpb_abs)
├─ Identify best config per variant (4 best total)
└─ Output: sae_config_comparison.csv, latent_correlations.csv
        ↓
PHASE 2.5a: FEATURE SIGNIFICANCE ANALYSIS (Optional, for publication)
├─ Statistical testing per variant
├─ Permutation testing + FDR correction
└─ Output: Statistical significance metrics
        ↓
PHASE 2.5b: CROSS-VARIANT COMPARISON (Pareto analysis)
├─ All 30 configs - reconstruction vs sparsity trade-offs
└─ Output: Variant comparison plots
        ↓
PHASE 2b: MULTI-SEED RETRAINING (REQUIRED for publication)
├─ Retrain 4 best configs with seeds [123, 456, 789, 1011]
└─ Output: 16 additional checkpoints
        ↓
PHASE 3: ABLATION STUDIES (3a/3b/3c/3d can run in parallel)
├─ 3a: SAE Latent Space Ablations (single-motif, grouped by motif)
├─ 3b: Native GNN Validation (single-motif, 4 motif-specific runs)
├─ 3c: Ablation Strategy Comparison (SAE vs native)
│   └─ Option A (Default): Single best variant
│   └─ Option B (Extended): All 4 variants
└─ 3d: Mixed-Motif Generalization (REQUIRED for publication)
        ↓
PHASE 4: STATISTICAL VALIDATION (multi-seed analysis)
├─ Feature stability across 5 seeds
├─ Redundancy analysis
├─ Permutation testing + FDR
└─ Output: Statistical tables + plots
        ↓
PHASE 5: VISUALIZATION & ANALYSIS
├─ Feature activation heatmaps
├─ Reconstruction fidelity (PCA)
└─ Output: Publication-ready visualizations
```

---

## SAE Variants Architecture

The pipeline trains 4 distinct Sparse Autoencoder variants, each with different sparsity mechanisms and design tradeoffs. All variants follow a common **Abstract Base Class** pattern for polymorphic training and consistent interfaces.

### Why These 4 Variants? (Research Grounding)

Rather than implementing baseline variants like L1-penalty SAEs, this pipeline uses state-of-the-art architectures validated by recent SAE research. **Key rationale**:

**Known Limitations of Standard Approaches:**
- **Linear encoders** (standard TopK, L1-penalty SAEs) suffer from **amortization gaps**: discrepancy between fast amortized encoding and inference-time optimization
- **Magnitude penalties** cause **feature shrinkage**: features trained with L1/L2 penalties on reconstructions tend toward weak signals
- **Single-architecture studies** cannot validate that mechanistic interpretability findings generalize across designs

**Research-Grounded Alternatives:**
Rather than L1SAE (a basic baseline), we implemented three state-of-the-art variants that address these known limitations:

1. **Gated SAE** (Rajamanoharan et al., 2024): Decouples feature detection from magnitude, eliminating shrinkage
2. **JumpReLU SAE** (Rajamanoharan et al., 2024): Direct L0 sparsity with straight-through estimators, no magnitude penalties
3. **Switch SAE** (Wei et al., 2024): Mixture-of-experts routing for sample-efficient scaling
4. **TopK SAE** (standard, included for baseline): Fixed-sparsity reference point

**Scientific Advantage**: By systematically comparing all four architectures, we:
- Contextualize limitations of any single design choice
- Strengthen claims about mechanistic interpretability (if effects persist across diverse configs)
- Avoid over-fitting analysis to a single architecture's quirks
- Ground results in validated research rather than ad-hoc baselines

See [encoder_limitation_justification.tex](encoder_limitation_justification.tex) for detailed encoder insufficiency analysis and references.

---

### Overview: 30 Total Configurations

| Variant | Count | Key Parameter | Hyperparameter Range | Best For |
|---------|-------|---|---|---|
| **TopK** | 11 | k (sparsity level) | latent_dim: [128, 256, 512], k: [4, 8, 16, 32] | Stable, interpretable sparsity |
| **Gated** | 9 | sparsity_coef | latent_dim: [128, 256, 512], λ: [1e-4, 5e-4, 1e-3] | Solves shrinkage, 2× feature efficiency |
| **JumpReLU** | 6 | threshold_init | latent_dim: [128, 256, 512], θ: [0.01, 0.1] | State-of-the-art reconstruction |
| **Switch** | 4 | num_experts | experts: 4/8, latent_per_expert: 64/128 | Scalable capacity with routing |

---

### Variant 1: TopK SAE (Standard Approach)

**Architecture**: Linear encoder → ReLU → Top-K selection

**Sparsity Mechanism**:
- For each input, keeps only top K neurons active by value
- Sets remaining neurons to zero (hard sparsity)
- **Formula**: `z_sparse[i] = z[i] if rank(z[i]) ≤ K else 0`

**Key Characteristics**:
- Predictable, fixed sparsity (exactly K active per sample)
- Fast encoding (single pass through encoder + topk)
- Works well with interpretability analysis (clear feature selection)

**Hyperparameter Details**:
```python
topk_configs = [
    {'latent_dim': 128, 'k': 4},   # Conservative
    {'latent_dim': 128, 'k': 8},   # Moderate
    {'latent_dim': 128, 'k': 16},  # Liberal
    {'latent_dim': 256, 'k': 4},
    {'latent_dim': 256, 'k': 8},
    {'latent_dim': 256, 'k': 16},
    {'latent_dim': 256, 'k': 32},  # High latent dim with more features
    {'latent_dim': 512, 'k': 4},   # Large latent space
    {'latent_dim': 512, 'k': 8},
    {'latent_dim': 512, 'k': 16},
    {'latent_dim': 512, 'k': 32},  # Maximum configuration
]
```

**Loss Function**: Reconstruction MSE only
- No sparsity penalty (sparsity is enforced structurally)
- Simple, interpretable training dynamics

---

### Variant 2: Gated SAE (Improved Sparsity Detection)

**Architecture**: Separate gate network + magnitude network

**Sparsity Mechanism**:
- Gate network: Learns which features to activate (binary mask via Heaviside)
- Magnitude network: Learns how strong each feature should be
- **Formula**: `z = Heaviside(gate) * magnitude`

**Key Innovation**: Decouples feature detection from strength estimation, solving **shrinkage problem** (where features trained with magnitude penalties tend to weak signals).

**Why This Variant**:
- Achieves 2× feature efficiency compared to TopK for same reconstruction
- More expressive sparsity patterns (varies across samples)
- Requires explicit L1 penalty on gate activations

**Hyperparameter Details**:
```python
gated_configs = [
    {'latent_dim': 128, 'sparsity_coef': 1e-4},   # Weak penalty
    {'latent_dim': 128, 'sparsity_coef': 5e-4},   # Medium
    {'latent_dim': 128, 'sparsity_coef': 1e-3},   # Strong penalty
    {'latent_dim': 256, 'sparsity_coef': 1e-4},
    {'latent_dim': 256, 'sparsity_coef': 5e-4},
    {'latent_dim': 256, 'sparsity_coef': 1e-3},
    {'latent_dim': 512, 'sparsity_coef': 1e-4},
    {'latent_dim': 512, 'sparsity_coef': 5e-4},
    {'latent_dim': 512, 'sparsity_coef': 1e-3},
]
```

**Loss Function**: Multi-component
- Reconstruction MSE + L1(gate activations) + Auxiliary decoder loss
- Auxiliary loss helps magnitude network learn meaningful representations

---

### Variant 3: JumpReLU SAE (Best Reconstruction Fidelity)

**Architecture**: Linear encoder + learnable per-feature thresholds + straight-through estimator

**Sparsity Mechanism**:
- Each feature has learnable threshold θ_i
- Activation: `z_i = x_i if x_i > θ_i else 0`
- Discontinuous (non-differentiable), trained via straight-through estimators (STE)

**Key Innovation**:
- Replaces soft sparsity penalties (L1) with direct L0 sparsity
- Uses Gaussian approximation for gradient computation through discontinuity
- Achieves state-of-the-art reconstruction while maintaining sparsity

**Why This Variant**:
- No shrinkage problem (feature values not penalized)
- Can learn heterogeneous sparsity thresholds per feature
- Higher reconstruction fidelity than TopK/Gated at comparable sparsity

**Hyperparameter Details**:
```python
jumprelu_configs = [
    {'latent_dim': 128, 'threshold_init': 0.01, 'bandwidth': 0.01},
    {'latent_dim': 128, 'threshold_init': 0.1, 'bandwidth': 0.01},
    {'latent_dim': 256, 'threshold_init': 0.01, 'bandwidth': 0.01},
    {'latent_dim': 256, 'threshold_init': 0.1, 'bandwidth': 0.01},
    {'latent_dim': 512, 'threshold_init': 0.01, 'bandwidth': 0.01},
    {'latent_dim': 512, 'threshold_init': 0.1, 'bandwidth': 0.01},
]
```
- `threshold_init`: Initial value for learnable thresholds (0.01 = sensitive, 0.1 = selective)
- `bandwidth`: Controls gradient smoothness in straight-through estimator

**Loss Function**: Direct L0 sparsity
- Reconstruction MSE + count of active features
- No proxy penalties, direct sparsity control

---

### Variant 4: Switch SAE (Scalable Routing)

**Architecture**: Router network + multiple expert SAEs

**Sparsity Mechanism**:
- Router network: Hard routing via `argmax(softmax(logits / temperature))`
- Multiple expert SAEs: Each expert specializes in different feature patterns
- Each sample routed to single expert (or soft routing with temperature scheduling)
- Experts can be TopK-based for per-expert sparsity

**Key Innovation**:
- Mixture-of-experts routing increases capacity without proportional parameter growth
- Each expert learns different sparsity patterns
- Enables 5× sample efficiency in training

**Why This Variant**:
- Scalable to larger latent dimensions
- Can model multi-modal feature distributions
- Different experts might specialize in different motif types

**Hyperparameter Details**:
```python
switch_configs = [
    {'num_experts': 4, 'latent_per_expert': 64, 'k_per_expert': 8},      # Small experts
    {'num_experts': 4, 'latent_per_expert': 128, 'k_per_expert': 16},     # Medium
    {'num_experts': 8, 'latent_per_expert': 64, 'k_per_expert': 8},       # Many small
    {'num_experts': 8, 'latent_per_expert': 128, 'k_per_expert': 16},     # Many medium
]
```
- Total latent_dim = num_experts × latent_per_expert
- Each expert applies TopK sparsity with k_per_expert

**Loss Function**: Reconstruction + load balancing
- Ensures even usage of all experts
- Prevents mode collapse where few experts dominate

---

### Design Patterns

#### 1. Abstract Base Class Pattern

All variants inherit from **`BaseSAE`** base class:

```python
class BaseSAE(nn.Module, ABC):
    """Base class enforcing common interface."""

    def __init__(self, input_dim: int, latent_dim: int):
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse latent."""
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        pass

    @abstractmethod
    def compute_loss(self, x, x_hat, z) -> Tuple[Tensor, Dict]:
        """Compute loss with all components."""
        pass

    @abstractmethod
    def get_config(self) -> Dict:
        """Return configuration dict."""
        pass
```

**Benefits**:
- Polymorphic training: `train_single_variant()` works for any variant
- Consistent checkpoint format across all variants
- Easy to add new variants without modifying training code

#### 2. Factory Pattern

**`compare_sae_configs.py`** uses factory functions for polymorphic loading:

```python
def load_sae_model(variant: str, config: Dict, device: str) -> BaseSAE:
    """Factory function for loading SAE by variant."""
    if variant == 'topk':
        return TopKSAE(input_dim=64, **config)
    elif variant == 'gated':
        return GatedSAE(input_dim=64, **config)
    elif variant == 'jumprelu':
        return JumpReLUSAE(input_dim=64, **config)
    elif variant == 'switch':
        return SwitchSAE(input_dim=64, **config)
    else:
        raise ValueError(f"Unknown variant: {variant}")
```

**Benefits**:
- Enables auto-detection of variant from checkpoint filenames
- Allows downstream scripts to work with any variant
- Simplifies integration with ablation/analysis tools

#### 3. Configuration Management

Each variant's **`get_config()`** method returns a dict with all hyperparameters:

```python
# TopKSAE example
def get_config(self) -> Dict[str, Any]:
    return {
        'input_dim': 64,
        'latent_dim': self.latent_dim,
        'k': self.k,
        'sparsity_method': 'topk',
        'target_sparsity': self.k / self.latent_dim
    }
```

**Benefits**:
- Config saved with checkpoint for reproducibility
- Enables reconstruction of model from filename
- Stored in JSON for analysis scripts

#### 4. Seed-Aware Checkpointing

All checkpoints include seed in filename:

```
checkpoints/sae_topk_latent512_k8_seed42.pt       # Initial training
checkpoints/sae_topk_latent512_k8_seed123.pt      # Multi-seed retrain
checkpoints/sae_topk_latent512_k8_seed456.pt
checkpoints/sae_topk_latent512_k8_seed789.pt
checkpoints/sae_topk_latent512_k8_seed1011.pt
```

**Benefits**:
- Enables feature stability analysis across random initializations
- Maintains reproducibility
- Supports Phase 4 multi-seed validation

---

### Summary: Why 4 Variants?

| Challenge | TopK | Gated | JumpReLU | Switch |
|-----------|------|-------|----------|--------|
| Reconstruction fidelity | ⚠️ Good | ✓ Better | ✓✓ Best | ✓ Better |
| Shrinkage problem | ✓ Immune | ✓ Solved | ✓✓ Eliminated | ✓ Solved |
| Sparsity interpretability | ✓✓ Excellent | ✓ Good | ✓ Good | ✓ Good |
| Scalability | ✓ Efficient | ✓ Efficient | ✓ Efficient | ✓✓ Excellent |
| Feature detectability | ✓ Fixed | ✓ Variable | ✓ Selective | ✓✓ Multi-modal |
| Training stability | ✓✓ Stable | ✓✓ Stable | ⚠️ Complex | ✓ Good |

**Pipeline strategy**: Train all 4 variants, use Phase 2 metrics (max_rpb_abs) to select overall best for Phases 3-5, then compare all 4 in Phase 4 robustness analysis.

---

## Phase-by-Phase Breakdown

### PHASE 1: SAE Training (30 Configurations, Seed=42)

**Input**: `outputs/activations/layer2/{train,val,test}/`
**Output**: 30 checkpoints + metrics
**Time**: ~5 hours (GPU required)

**Script**: `sparse_autoencoder.py`

**What happens**:
- Trains all 30 SAE configurations with seed=42
- Creates 30 checkpoint files in `checkpoints/`
- Saves training metrics in `outputs/sae_metrics_*.json`

**Checkpoint naming**:
```
TopK:       checkpoints/sae_topk_latent{latent_dim}_k{k}_seed42.pt
Gated:      checkpoints/sae_gated_latent{latent_dim}_lambda{sparsity_coef:.0e}_seed42.pt
JumpReLU:   checkpoints/sae_jumprelu_latent{latent_dim}_thresh{threshold_init:.0e}_bw{bandwidth:.0e}_seed42.pt
Switch:     checkpoints/sae_switch_experts{num_experts}_latent{total_latent}_k{k_per_expert}_seed42.pt
```

---

### PHASE 2: Configuration Comparison (Ranking)

**Input**: 30 seed=42 checkpoints + `layer2/test/`
**Output**: `sae_config_comparison.csv`, `latent_correlations.csv`
**Time**: ~30 minutes

**Script**: `compare_sae_configs.py`

**What happens**:
1. Evaluates all 30 seed=42 checkpoints
2. Computes point-biserial correlation (max_rpb_abs) for each config
3. Identifies best config from each variant (4 best total)
4. Saves feature-motif correlations for all configs
5. Creates test_graph_ids.json for downstream phases

**Output files**:
- `outputs/sae_config_comparison.csv` - All 30 configs ranked, includes variant-specific parameters
- `outputs/latent_correlations.csv` - Feature-motif correlations + significance for all configs
- `outputs/test_graph_ids.json` - Test set definition

**Key metric for Phase 3**: `max_rpb_abs` (used in Phase 3a to select single best config)

---

### PHASE 2.5a: Feature Significance Analysis (Optional)

**Input**: Feature-motif correlations from Phase 2
**Output**: Significance metrics per variant
**Time**: ~1 hour

**Script**: `analyze_feature_significance.py` (run once per variant)

**What happens**:
- Performs permutation testing (1000 iterations)
- FDR correction for multiple comparisons
- Computes effect sizes (Cohen's d, rank-biserial)

---

### PHASE 2.5b: Cross-Variant Comparison

**Input**: 30 seed=42 checkpoints + metrics
**Output**: Pareto frontier plots, variant comparison report
**Time**: ~5 minutes

**Script**: `compare_sae_variants.py`

**What happens**:
- Analyzes trade-offs: reconstruction quality vs sparsity
- Generates Pareto frontier plots per variant
- Compares efficiency: convergence speed, model size
- Justifies choice of best variant for subsequent analysis

---

### PHASE 2b: Multi-Seed Retraining (REQUIRED for Publication)

**Input**: Best config from each variant (from Phase 2)
**Output**: 16 additional checkpoints (4 configs × 4 new seeds)
**Time**: ~2.7 hours

**Script**: `retrain_best_configs.py`

**What happens**:
1. Identifies 4 best configs (one per variant) from Phase 2
2. Retrains ONLY those 4 configs with seeds [123, 456, 789, 1011]
3. Does NOT retrain with seed=42 (already exists from Phase 1)

**Checkpoint naming** (for best TopK as example):
```
checkpoints/sae_topk_latent512_k8_seed123.pt
checkpoints/sae_topk_latent512_k8_seed456.pt
checkpoints/sae_topk_latent512_k8_seed789.pt
checkpoints/sae_topk_latent512_k8_seed1011.pt
```

**Total checkpoints after Phase 2b**: 30 (Phase 1) + 16 (Phase 2b) = 46

**Why mandatory**: Enables feature stability analysis (Phase 4), provides confidence intervals, required for publication-ready reproducibility claims.

---

### PHASE 3a: SAE Latent Space Ablations

**Input**: Best overall config (by max_rpb_abs) + feature-motif correlations
**Output**: 4 CSV files (one per motif)
**Time**: ~2 hours

**Script**: `run_interpretability_experiments.py --variant {best_variant}`

**What happens**:
1. Phase 3a loads Phase 2 CSV, selects best config by **max_rpb_abs** (not composite_score)
2. Extracts variant-specific parameters (k, sparsity_coef, threshold_init, or num_experts)
3. **Saves metadata to** `ablations/phase_3a_config.json` for Phase 3b/3c to use
4. For each of 4 motif groups (FFL, Cascade, FFL, SIM):
   - Filters top features by: FDR significance + |rpb| ≥ min_rpb threshold
   - Groups filtered features by motif
   - Ablates each feature group via `run_ablation.py`
   - Runs random control trials (20 per feature count)
   - Computes z-scores, percentiles, p-values vs random

**Output files**:
```
ablations/phase_3a_config.json  ← CRITICAL: metadata for Phase 3b/3c
ablations/results/
├── feedback_loop_VARIANT_lLATENT_kK_results.csv
├── cascade_VARIANT_lLATENT_kK_results.csv
├── feedforward_loop_VARIANT_lLATENT_kK_results.csv
└── single_input_module_VARIANT_lLATENT_kK_results.csv

ablations/interpretability_VARIANT_lLATENT_kK_rpb*/
├── motif_specific_results.csv
├── statistical_tests.csv
└── feature_motif_mapping.json
```

**Key metrics**:
- `Loss(Original)` - GNN loss on native 64D activations
- `Loss(Full SAE)` - GNN loss on SAE reconstruction
- `Loss(Ablated)` - GNN loss with features zeroed
- `Ablation Impact` = Loss(Ablated) - Loss(Full SAE)

---

### PHASE 3b: Native GNN Activation Space Validation

**Input**: Best config from Phase 3a metadata + top features per motif
**Output**: 4 CSV files (one per motif)
**Time**: ~2 hours

**Script**: `native_gnn_ablation.py --variant {best_variant} --motif {motif}` (run 4 times)

**What happens**:
1. **CRITICAL**: Loads metadata from `ablations/phase_3a_config.json` (ensures same config as Phase 3a)
2. For each of 4 motifs (FFL, Cascade, FFL, SIM):
   - Gets top features correlated with that motif (from Phase 2)
   - Directly ablates those native GNN activations (no SAE encoder/decoder)
   - Measures GNN loss impact

**How to run** (4 separate calls):
```bash
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_feedback_loop
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_cascade
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_feedforward_loop
python native_gnn_ablation.py --variant topk --latent_dim 512 --k 8 --use-rpb --motif in_single_input_module
```

**Output files**:
```
outputs/native_gnn_ablations/
├── native_ablation_topk_rpb_in_feedback_loop.csv
├── native_ablation_topk_rpb_in_cascade.csv
├── native_ablation_topk_rpb_in_feedforward_loop.csv
└── native_ablation_topk_rpb_in_single_input_module.csv
```

**Purpose**: Validates that SAE ablations reflect true causal mechanisms (not artifacts of SAE encoder/decoder)

---

### PHASE 3c: Ablation Strategy Comparison

**Input**: Phase 3a + Phase 3b outputs
**Output**: Agreement metrics + comparison plots
**Time**: ~20 min (Option A) or ~80 min (Option B)

**Script**: `compare_ablation_strategies.py`

---

#### **OPTION A: Single Best Variant (DEFAULT - Current Pipeline)**

**What it does**:
- Loads best variant from Phase 3a metadata
- Compares SAE vs native ablation impacts for THAT VARIANT ONLY
- Measures per-motif agreement (Pearson/Spearman correlation)

**How to run**:
```bash
# Load variant from metadata, compare single variant
python compare_ablation_strategies.py --variant topk --latent_dim 512 --motif-mode
```

**Output**:
```
ablation_strategy_comparison/
├── motif_agreement_feedback_loop.csv
├── motif_agreement_cascade.csv
├── motif_agreement_feedforward_loop.csv
├── motif_agreement_single_input_module.csv
└── comparison_plots/
    └── strategy_comparison_*.png
```

**Interpretation**:
- High correlation between SAE and native ablations → SAE is mechanistically valid for this variant
- Low correlation → SAE may not capture true mechanisms (check for encoder/decoder limitations)

**Time**: ~20 minutes
**Computational cost**: Low
**Recommended**: Yes (default for publication pipeline)

---

#### **OPTION B: All 4 Variants (Extended Analysis)**

**What it does**:
- Compares SAE vs native ablations for ALL 4 variants
- Determines which variant is most mechanistically faithful
- Useful for: variant-specific mechanism exploration, robustness verification

**How to run**:
```bash
# Compare all 4 variants
python compare_ablation_strategies.py --all-variants --latent_dim 512 --motif-mode
```

**Output** (extended):
```
ablation_strategy_comparison/
├── variant_topk_agreement_*.csv
├── variant_gated_agreement_*.csv
├── variant_jumprelu_agreement_*.csv
├── variant_switch_agreement_*.csv
└── comparison_plots/
    ├── topk_vs_native_*.png
    ├── gated_vs_native_*.png
    ├── jumprelu_vs_native_*.png
    └── switch_vs_native_*.png
```

**Interpretation**:
- Identify which variant has highest SAE-native agreement
- If best config (selected by max_rpb_abs) also has highest agreement → validates selection
- If different variant has higher agreement → indicates mechanistic trade-off

**Time**: ~80 minutes (4× Option A)
**Computational cost**: High
**Recommended**: Only if exploring variant-specific mechanisms

---

### PHASE 3d: Mixed-Motif Generalization Test (REQUIRED for Publication)

**Input**: Best variant + top features from Phase 2
**Output**: Mixed-motif ablation results
**Time**: ~45 minutes

**Scripts**:
1. `generate_mixed_motif_activations.py` (preprocessing, one-time)
2. `run_ablation.py --use_mixed_motifs --feature [features]`
3. `native_gnn_ablation.py --use_mixed_motifs --feature [features]`

**What happens**:
1. Generates GNN activations for mixed-motif graphs (one-time preprocessing)
2. Tests if single-motif-discovered features generalize to mixed-motif (2-3 interacting motifs)
3. Compares ablation impacts: single-motif vs mixed-motif

**How to run**:
```bash
# Step 1: Generate mixed-motif activations (one-time)
python generate_mixed_motif_activations.py

# Step 2: Run SAE ablations on mixed-motif graphs
python run_ablation.py \
  --variant topk \
  --latent_dim 512 \
  --use_mixed_motifs \
  --feature z1,z2,z3,z10,z50,z100 \
  --experiment_name mixed_motifs_topk

# Step 3: Run native ablations on mixed-motif graphs
python native_gnn_ablation.py \
  --variant topk \
  --latent_dim 512 \
  --use_mixed_motifs \
  --feature z1,z2,z3,z10,z50,z100
```

**Output**:
```
ablations/results/ablation_mixed_motifs_topk.csv
outputs/native_gnn_ablations/native_ablation_topk_mixed_motifs.csv
```

**Purpose**:
- Validates that mechanistic interpretations don't depend on overly-simple single-motif graphs
- Demonstrates robustness of feature-motif associations
- Strengthens claims for publication

**Why mandatory**: Addresses reviewer concerns about generalization; shows features work in realistic multi-motif settings.

---

### PHASE 4: Statistical Validation (Multi-Seed Analysis)

**Input**: 5-seed checkpoints (from Phase 2b) + Phase 2-3 outputs
**Output**: Statistical tables, stability plots, redundancy analysis
**Time**: ~30 minutes

**Script**: `statistical_analysis_suite.py`

**What happens**:
1. **Feature Stability**: Computes decoder cosine similarity across 5 seeds
2. **Correlation Distributions**: Per variant, per motif
3. **Redundancy Analysis**: Which features are statistically independent
4. **Effect Sizes**: Cohen's d, rank-biserial for ablation impacts
5. **Sparsity-Interpretability Trade-off**: Analyzes K vs correlation strength

**Output**:
```
outputs/statistical_analysis/
├── feature_stability.png              (% features with >0.8 cosine sim)
├── correlation_distributions_*.png    (per variant, per motif)
├── feature_redundancy_heatmap.png
├── sparsity_vs_interpretability.png
├── ablation_effect_sizes.csv
└── statistical_summary.json
```

**Critical dependency**: Requires Phase 2b multi-seed checkpoints

---

### PHASE 5: Visualization & Reconstruction Analysis

**Input**: Best variant checkpoint + activations
**Output**: Feature visualization plots, reconstruction metrics
**Time**: ~20 minutes

**Scripts**:
- `visualize_feature_activations.py` - Feature selectivity heatmaps, decoder weights
- `analyze_sae_reconstruction_fidelity.py` - PCA histogram comparison (native vs SAE)

**Output**:
```
outputs/feature_activation_visualizations/
├── feature_activation_heatmap.png
├── feature_selectivity_scores.csv
└── decoder_weight_distributions.png

outputs/sae_reconstruction_fidelity/
├── pca_histograms_component_1.png
├── pca_histograms_component_2.png
└── reconstruction_fidelity_report.json
```

**Purpose**: Publication-ready visualizations demonstrating feature interpretability and reconstruction quality.

---

## Multi-Seed Training Strategy

### Overview

The project uses a coordinated multi-seed strategy across phases:

| Phase | Configs | Seeds | Purpose | Output |
|-------|---------|-------|---------|--------|
| **1** | All 30 | seed=42 only | Initial exploration | 30 checkpoints |
| **2** | All 30 | seed=42 | Ranking | Best config per variant identified |
| **2b** | Best 4 | seeds [123, 456, 789, 1011] | Multi-seed retraining | 16 additional checkpoints |
| **3** | Best 1 | seed=42 | Ablation analysis | Mechanistic interpretability |
| **4** | Best 4 | all 5 seeds | Stability analysis | Feature stability metrics |

### Why This Strategy?

1. **Phase 1**: Single seed (42) for efficiency - explores 30 configs
2. **Phase 2**: Ranks by max_rpb_abs - identifies 4 best (one per variant)
3. **Phase 2b**: Retrains 4 best with 4 new seeds - validates stability
4. **Phase 3**: Uses seed=42 of best - analyzes specific latent space discovered initially
5. **Phase 4**: All 5 seeds of 4 best - statistical validation across initializations

### Checkpoint Directory After All Phases

```
checkpoints/
├── (Phase 1) 30 files: sae_VARIANT_PARAMS_seed42.pt
├── (Phase 2b) 16 files: sae_VARIANT_PARAMS_seed{123,456,789,1011}.pt
└── gnn_model.pt
Total: 47 checkpoint files
```

---

## Execution Sequences

### Standard Pipeline with Multi-Seed (RECOMMENDED for Publication)

**Phases**: 1→2→2.5a→2.5b→2b→3a→3b→3c(A)→3d→4→5
**Total time**: ~17 hours
**Computational**: GPU required throughout

```bash
# PHASE 1: Train all 30 SAE configs (~5 hours)
python sparse_autoencoder.py

# PHASE 2: Rank configs + extract features (~30 min)
python compare_sae_configs.py

# PHASE 2.5a: Feature significance testing (optional, ~1 hour)
python analyze_feature_significance.py --variant topk
python analyze_feature_significance.py --variant gated
python analyze_feature_significance.py --variant jumprelu
python analyze_feature_significance.py --variant switch

# PHASE 2.5b: Cross-variant comparison (~5 min)
python compare_sae_variants.py

# PHASE 2b: Multi-seed retraining (~2.7 hours)
python retrain_best_configs.py

# PHASE 3a: SAE latent ablations (~2 hours)
# (Notebook cell handles this with metadata creation)

# PHASE 3b: Native GNN validation (~2 hours, 4 runs)
# (Notebook cell loops through 4 motifs)

# PHASE 3c-OPTION-A: Single-variant comparison (default, ~20 min)
python compare_ablation_strategies.py --variant topk --latent_dim 512 --motif-mode

# PHASE 3d: Mixed-motif validation (~45 min)
python generate_mixed_motif_activations.py
python run_ablation.py --variant topk --latent_dim 512 --use_mixed_motifs --feature z1,z2,z3
python native_gnn_ablation.py --variant topk --latent_dim 512 --use_mixed_motifs --feature z1,z2,z3

# PHASE 4: Statistical analysis (~30 min)
python statistical_analysis_suite.py --seed-analysis

# PHASE 5: Visualization (~20 min)
python visualize_feature_activations.py --variant topk --latent_dim 512 --features 20
python analyze_sae_reconstruction_fidelity.py --variant topk --latent-dim 512 --k 8
```

### Quick Pipeline - Single-Seed Only (~8 hours, NOT RECOMMENDED for publication)

```bash
python sparse_autoencoder.py              # 5 hours
python compare_sae_configs.py             # 30 min
python compare_sae_variants.py            # 5 min
python run_interpretability_experiments.py  # 2 hours
# Skip Phase 2b, Phase 4 --seed-analysis
```

### Extended Analysis - All-Variant Phase 3c (~20 hours)

Same as Standard Pipeline, but replace Phase 3c:

```bash
# PHASE 3c-OPTION-B: All-variant comparison (~80 min)
python compare_ablation_strategies.py --all-variants --latent_dim 512 --motif-mode
```

---

## File Responsibilities & Data Flow

### Input Requirements

| File | Purpose | Generated By |
|------|---------|--------------|
| `outputs/activations/layer2/train/*.pt` | Training activations | gnn_train.py |
| `outputs/activations/layer2/val/*.pt` | Validation activations | gnn_train.py |
| `outputs/activations/layer2/test/*.pt` | Test activations | gnn_train.py |
| `outputs/activations/layer2/mixed/*.pt` | Mixed-motif activations | gnn_train.py |
| `checkpoints/gnn_model.pt` | GNN model | gnn_train.py |
| `virtual_graphs/data/all_graphs/raw_graphs/*.pkl` | Graph structures | graph_motif_generator.py |
| `virtual_graphs/data/all_graphs/graph_motif_metadata/*.csv` | Motif labels | graph_motif_generator.py |

### Phase 1 Outputs

| File | Used By | Purpose |
|------|---------|---------|
| `checkpoints/sae_*.pt` (30 files) | All downstream phases | Trained SAE models |
| `outputs/sae_metrics_*.json` (30 files) | Phase 2, 2.5 | Training curves, final metrics |

### Phase 2 Outputs

| File | Used By | Purpose |
|------|---------|---------|
| `outputs/sae_config_comparison.csv` | Phase 3a, Phase 2b | Config ranking, best per variant |
| `outputs/latent_correlations.csv` | Phase 3a, 3b, 3d | Feature-motif correlations |
| `outputs/test_graph_ids.json` | Phase 3, 4, 5 | Test set definition |
| `outputs/latent_cache/*.pkl` | Phase 2 | Cached latents (optimization) |

### Phase 2b Outputs

| File | Used By | Purpose |
|------|---------|---------|
| `checkpoints/sae_*_seed{123,456,789,1011}.pt` (16 files) | Phase 4 | Multi-seed checkpoints |
| `outputs/retrain_summary.json` | Phase 4 | Retraining metrics |

### Phase 3 Outputs

| File | Used By | Purpose |
|------|---------|---------|
| `ablations/phase_3a_config.json` | Phase 3b, 3c, 3d | Selected config metadata |
| `ablations/results/*.csv` | Phase 3c, 4 | SAE ablation results |
| `outputs/native_gnn_ablations/*.csv` | Phase 3c, 4 | Native ablation results |
| `ablation_strategy_comparison/*.csv` | Phase 4 | SAE vs native comparison |

### Phase 4 Outputs

| File | Purpose |
|------|---------|
| `outputs/statistical_analysis/feature_stability.png` | Reproducibility metrics |
| `outputs/statistical_analysis/correlation_distributions_*.png` | Feature-motif associations |
| `outputs/statistical_analysis/redundancy_heatmap.png` | Model efficiency |

---

## Phase 3c: Ablation Strategy Comparison

### Key Clarification: Single Best Variant vs All 4 Variants

The script `compare_ablation_strategies.py` has two distinct operating modes:

**Both approaches measure the same thing**: Do SAE and native ablations produce similar results?
**Difference**: Scope of comparison (single variant vs all variants)

#### Option A: Single Best Variant (RECOMMENDED - Current Pipeline)

- **Compares**: SAE vs native for 1 variant only
- **Which variant**: The one selected by Phase 3a (highest max_rpb_abs)
- **Time**: ~20 minutes
- **Output**: Agreement metrics for 4 motifs × 1 variant
- **Recommended for**: Publication pipelines (time efficient, sufficient evidence)

**Example**:
```bash
python compare_ablation_strategies.py --variant topk --latent_dim 512 --motif-mode
```

**Interpretation**:
- Validates that the BEST variant (selected by max_rpb_abs) is mechanistically sound
- High SAE-native agreement → best config is truly causal (not just correlative)
- Sufficient for publication: demonstrates mechanistic validity of the chosen variant

#### Option B: All 4 Variants (ADVANCED - Optional Extended Analysis)

- **Compares**: SAE vs native for all 4 variants
- **Time**: ~80 minutes (4× Option A)
- **Output**: Agreement metrics for 4 motifs × 4 variants = 16 comparisons
- **Recommended for**: Variant-specific mechanism exploration, robustness studies

**Example**:
```bash
python compare_ablation_strategies.py --all-variants --latent_dim 512 --motif-mode
```

**Interpretation**:
- Shows which variant(s) are most mechanistically faithful
- Identifies mechanistic trade-offs between variants
- Useful for: understanding why one variant was selected, exploring alternatives
- NOT necessary for publication (Option A sufficient)

### Implementation in Different Scenarios

**Google Colab Notebook** (sae_colab_pipeline.ipynb):
- Uses Option A (single best variant)
- Time and memory efficient for cloud execution
- Focuses on reproducing main results

**Research Exploration** (command line):
- Can use Option B to compare all 4 variants
- Investigate variant-specific mechanisms
- Longer execution, use local GPU

**Quick Verification** (debugging):
- Use Option A for fast feedback
- Verify SAE assumptions hold for best config
- ~20 min total for both ablations + comparison

---

## Key Data Files

### Checkpoint Naming Convention

**Phase 1 & 3 checkpoints** (single seed=42):
```
TopK:       sae_topk_latent{latent_dim}_k{k}_seed42.pt
Gated:      sae_gated_latent{latent_dim}_lambda{sparsity_coef:.0e}_seed42.pt
JumpReLU:   sae_jumprelu_latent{latent_dim}_thresh{threshold_init:.0e}_bw{bandwidth:.0e}_seed42.pt
Switch:     sae_switch_experts{num_experts}_latent{total_latent}_k{k_per_expert}_seed42.pt
```

**Phase 2b checkpoints** (multi-seed):
```
sae_VARIANT_PARAMS_seed{123,456,789,1011}.pt
```

### Directory Structure After Full Pipeline

```
checkpoints/
├── sae_topk_latent512_k8_seed42.pt          (Phase 1)
├── sae_topk_latent512_k8_seed123.pt         (Phase 2b)
├── sae_topk_latent512_k8_seed456.pt         (Phase 2b)
├── sae_topk_latent512_k8_seed789.pt         (Phase 2b)
├── sae_topk_latent512_k8_seed1011.pt        (Phase 2b)
├── [30 files total from Phase 1 + 16 files from Phase 2b]
└── gnn_model.pt

outputs/
├── sae_config_comparison.csv                (Phase 2)
├── sae_variant_comparison.csv               (Phase 2.5b)
├── latent_correlations.csv                  (Phase 2)
├── test_graph_ids.json                      (Phase 2)
├── retrain_summary.json                     (Phase 2b)
├── native_gnn_ablations/                    (Phase 3b)
│   ├── native_ablation_topk_rpb_in_feedback_loop.csv
│   ├── native_ablation_topk_rpb_in_cascade.csv
│   ├── native_ablation_topk_rpb_in_feedforward_loop.csv
│   └── native_ablation_topk_rpb_in_single_input_module.csv
├── ablation_strategy_comparison/            (Phase 3c)
│   ├── motif_agreement_*.csv
│   └── comparison_plots/
├── statistical_analysis/                    (Phase 4)
│   ├── feature_stability.png
│   ├── correlation_distributions_*.png
│   ├── feature_redundancy_heatmap.png
│   ├── sparsity_vs_interpretability.png
│   └── statistical_summary.json
├── feature_activation_visualizations/       (Phase 5)
│   ├── feature_activation_heatmap.png
│   └── feature_selectivity_scores.csv
└── sae_reconstruction_fidelity/             (Phase 5)
    ├── pca_histograms_*.png
    └── reconstruction_fidelity_report.json

ablations/
├── phase_3a_config.json                     (Phase 3a)
├── results/                                 (Phase 3a)
│   ├── feedback_loop_topk_l512_k8_results.csv
│   ├── cascade_topk_l512_k8_results.csv
│   ├── feedforward_loop_topk_l512_k8_results.csv
│   └── single_input_module_topk_l512_k8_results.csv
├── interpretability_topk_l512_k8_rpb0.05_results/  (Phase 3a)
│   ├── motif_specific_results.csv
│   ├── statistical_tests.csv
│   └── feature_motif_mapping.json
└── interpretability_topk_l512_k8_rpb0.05_plots/    (Phase 3a)
    └── interpretability_vs_random_controls.png
```

---

## Critical Checkpoints

Before proceeding to next phase, verify:

| Phase | Checkpoint | Verification |
|-------|-----------|--------------|
| **1** | SAE training completed | `ls checkpoints/sae_*.pt` → 30 files with seed42 |
| **2** | Config ranking saved | `ls outputs/sae_config_comparison.csv` exists |
| **2** | Feature correlations saved | `ls outputs/latent_correlations.csv` exists |
| **2b** | Multi-seed checkpoints | `ls checkpoints/sae_*_seed{123,456,789,1011}.pt` → 16 files |
| **3a** | Metadata created | `ls ablations/phase_3a_config.json` exists |
| **3a** | Motif results created | `ls ablations/results/*.csv` → 4 files |
| **3b** | Native ablations done | `ls outputs/native_gnn_ablations/*.csv` → 4 files |
| **3c** | Comparison complete | `ls ablation_strategy_comparison/*.csv` → 4 files |
| **3d** | Mixed-motif results | `ls ablations/results/*_mixed_motifs.csv` exists |
| **4** | Statistics computed | `ls outputs/statistical_analysis/feature_stability.png` exists |

---

## Troubleshooting Guide

### Problem: "Checkpoint not found"

**Symptom**: Phase 2 fails with "sae_*.pt not found"
**Solution**: Verify Phase 1 completed successfully
```bash
ls -la checkpoints/sae_*.pt  # Should have 30 files
# If not enough: re-run Phase 1
python sparse_autoencoder.py
```

### Problem: "Feature correlations missing"

**Symptom**: Phase 3a fails with "latent_correlations.csv not found"
**Solution**: Re-run Phase 2
```bash
python compare_sae_configs.py  # Creates required CSV
```

### Problem: "Phase 3b says metadata not found"

**Symptom**: Phase 3b fails with "ablations/phase_3a_config.json not found"
**Solution**: Run Phase 3a first
```bash
# Phase 3a creates the metadata JSON
python run_interpretability_experiments.py --variant topk ...
```

### Problem: "Wrong variant used in Phase 3b"

**Symptom**: Phase 3b runs with different variant than Phase 3a
**Solution**: Check metadata file content
```bash
cat ablations/phase_3a_config.json  # Verify variant matches
```

### Problem: "All-features flag not recognized"

**Symptom**: `native_gnn_ablation.py --all-features` throws error
**Solution**: Use `--feature` with comma-separated list instead
```bash
# WRONG:
python native_gnn_ablation.py --variant topk --all-features

# CORRECT:
python native_gnn_ablation.py --variant topk --feature z1,z2,z3,z10,z50
```

### Problem: "Multi-seed checkpoints not found"

**Symptom**: Phase 4 fails - can't find seed123, seed456, etc.
**Solution**: Run Phase 2b first
```bash
python retrain_best_configs.py  # Creates multi-seed checkpoints
```

### Problem: "Mixed-motif activations not found"

**Symptom**: Phase 3d fails - "outputs/activations/layer2/mixed/*.pt not found"
**Solution**: Run preprocessing step
```bash
python generate_mixed_motif_activations.py  # One-time preprocessing
```

---

## Publication Outputs

After completing full pipeline with Phase 2b and Phase 3d:

**For Methods Section**:
- `outputs/sae_config_comparison.csv` → SAE hyperparameter selection & ranking
- `outputs/native_gnn_ablations/*.csv` → Native validation approach

**For Results Section**:
- `outputs/statistical_analysis/correlation_distributions_*.png` → Feature-motif associations
- `ablations/results/*.csv` → Feature causality (ablation impacts)
- `outputs/statistical_analysis/feature_stability.png` → Reproducibility (multi-seed)
- `ablation_strategy_comparison/*.csv` → SAE assumption validation
- `ablations/results/*_mixed_motifs.csv` → Generalization to realistic multi-motif graphs

**For Appendix**:
- `outputs/sae_reconstruction_fidelity/*.png` → Reconstruction quality
- `outputs/feature_activation_visualizations/*.png` → Feature selectivity
- `outputs/statistical_analysis/redundancy_heatmap.png` → Model efficiency
- `outputs/statistical_analysis/sparsity_vs_interpretability.png` → Trade-off analysis

---

## Key Research Questions Answered

| Question | Phase | Answer Location |
|----------|-------|-----------------|
| Which features encode motifs? | 2 | latent_correlations.csv (r_pb column) |
| Are features causal? | 3a | ablations/results/*.csv (ablation_impact) |
| Is SAE valid? | 3b, 3c | agree_metrics.csv (SAE vs native correlation) |
| Do features generalize? | 3d | ablations/results/*_mixed_motifs.csv |
| Are findings reproducible? | 4 | feature_stability.png (% stable across seeds) |
| How efficient is model? | 4 | redundancy_heatmap.png (% independent) |
| Best sparsity trade-off? | 4 | sparsity_vs_interpretability.png |
| How good is reconstruction? | 5 | pca_histograms.png (distribution match) |
