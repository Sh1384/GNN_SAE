# Comprehensive Benchmarking Workflow Documentation

**File**: `benchmarking.py` (~1554 lines)
**Purpose**: Evaluate GNN models (GCN, GAT) vs baselines (MLP, MeanMedian) on gene expression prediction with statistical rigor
**Framework**: Multi-seed training, pairwise statistical comparisons, motif-specific analysis, sensitivity analysis

---
## Overview

The benchmarking system provides comprehensive evaluation:

1. **Baseline Models** - Demonstrates GNN's advantage over simple baselines
2. **Data Sensitivity Analysis** - Validates hyperparameter choices (timesteps, noise)
3. **Multi-Seed Statistical Analysis** - Ensures statistical rigor with comprehensive metrics

## Table of Contents

1. [Overview & Architecture](#overview--architecture)
2. [Phase 1: Multi-Seed Training](#phase-1-multi-seed-training)
3. [Phase 2: Pairwise Statistical Comparisons](#phase-2-pairwise-statistical-comparisons)
4. [Phase 3: Motif-Specific Analysis](#phase-3-motif-specific-analysis)
5. [Phase 4: Sensitivity Analysis](#phase-4-sensitivity-analysis-optional)
6. [Phase 5: Data Storage](#phase-5-data-storage)
7. [Complete Workflow Summary](#complete-workflow-summary)
8. [Command Usage Examples](#command-usage-examples)

---

## Overview & Architecture

### High-Level Workflow

```
Input: Synthetic single-motif graphs (4000 total, 1000 per motif)
  ↓
PHASE 1: Multi-Seed Training (4 models × N seeds)
  ├─ Task A: train_multi_seed_training()
  ├─ Task B: generate_detailed_statistics()
  ├─ Task C: plot_baseline_comparison()
  ├─ Task D: compute_motif_metrics()
  ├─ Task E: plot_motif_comparison()
  ├─ Task F: plot_motif_heatmap()
  ├─ Task I: plot_seed_variance()
  ├─ Task J: plot_statistical_summary_table()
  └─ Task K: plot_train_val_test_progression()
  ↓
PHASE 2: Pairwise Statistical Comparisons (4 pairs)
  ├─ Task G: compute_pairwise_comparisons()
  └─ Task H: plot_pairwise_comparisons()
  ↓
PHASE 4: Sensitivity Analysis [OPTIONAL] (GCN, GAT only)
  ├─ Task L: run_sensitivity_analysis()
  ├─ Task M: plot_sensitivity_analysis()
  └─ Task N: plot_individual_sensitivity_analysis()
  ↓
Output: 9 PNG visualizations + 9 JSON data files
```

### Key Design Principles

- **Reproducibility**: All random seeds explicitly controlled (torch.manual_seed, np.random.seed)
- **Statistical Rigor**: Wilcoxon test + bootstrap confidence intervals + multi-seed analysis
- **Practical Significance**: Effect sizes (rank-biserial) not just p-values
- **Modularity**: Each task is independently callable and generates outputs
- **Visualization**: Publication-ready PNG outputs for all metrics

---

## Phase 1: Multi-Seed Training

The core benchmarking task that trains all models with proper stochasticity control.

### Task A: `run_multi_seed_training()` (lines 267-387)

#### Purpose
Train 4 models across N seeds with seed control and motif-specific metric computation

#### Configuration
| Parameter | Default | Configurable |
|-----------|---------|--------------|
| Seeds | 5 | `--seeds N` |
| Max epochs | 100 | `--epochs N` |
| Early stopping patience | 10 | ✗ |
| Batch size | 32 | `--batch-size N` |
| Learning rate | 1e-3 | ✗ |
| Models tested | GCN, GAT, MLP, MeanMedian | ✗ |

#### Data Configuration
- **Data source**: Single-motif graphs only (4000 total)
- **Motif types**: cascade, feedback_loop, feedforward_loop, single_input_module (250 each)
- **Train/Val/Test split**: 80/10/10 (stratified by motif)
- **Mask probability**: 30% of nodes masked during training
- **Expression dynamics**: 50 timesteps (configurable in GraphDataset)
- **Noise level**: 0.01 standard deviation (configurable in GraphDataset)

#### Seed Control (Critical for Reproducibility)

```python
# For each seed (0 to n_seeds-1):
torch.manual_seed(seed)        # Controls weight initialization
np.random.seed(seed)           # Controls data split and masking
```

This ensures:
- ✓ Different weight initializations per seed
- ✓ Different train/val/test splits per seed
- ✓ Different data masking patterns per seed
- ✓ Different batch sampling orders per seed

#### Model Architectures

**GCN** (Graph Convolutional Network):
```
Input (2) → GCNConv(2→128, ReLU) → Dropout(0.2)
         → GCNConv(128→64, ReLU) → Dropout(0.2)
         → GCNConv(64→1)
         → Output (1)
```

**GAT** (Graph Attention Network):
```
Input (2) → GATConv(2→128, 4 heads, ELU) → Dropout(0.2)
         → GATConv(128→64, 4 heads, ELU) → Dropout(0.2)
         → GATConv(64→1, 1 head)
         → Output (1)
```

**MLP** (Baseline - no graph structure):
```
Input (2) → Linear(2→64, ReLU) → Dropout(0.2)
         → Linear(64→32, ReLU) → Dropout(0.2)
         → Linear(32→1)
         → Output (1)
```

**MeanMedian** (Baseline - non-learning):
```
Fit: Compute mean of all observed node values in training data
Predict: Return same mean value for all masked nodes
```

#### Training Process (Per Seed)

1. **Data loading**:
   - Load all graph paths
   - Split with seed-specific randomness
   - Create GraphDataset with mask_prob=0.3
   - Create DataLoaders with batch_size=32

2. **Model initialization**:
   - For GCN/GAT: Random weight initialization (controlled by seed)
   - For MLP: Random weight initialization (controlled by seed)
   - For MeanMedian: Fit to training data mean

3. **Training loop** (max 100 epochs):
   ```python
   for epoch in range(num_epochs):
       train_loss = trainer.train_epoch(train_loader)
       val_loss = trainer.validate(val_loader)

       if val_loss < best_val_loss:
           best_val_loss = val_loss
           patience_counter = 0
       else:
           patience_counter += 1
           if patience_counter >= 10:
               break  # Early stopping
   ```

4. **Final evaluation**:
   - Compute test_loss on held-out test set
   - Record: train_loss, val_loss, test_loss, best_epoch

5. **Motif metrics computation**:
   - Compute per-motif MSE and MAE (see Task D)

#### Output Format

**Per seed**: Single dictionary with lists of N values:
```python
seed_results = {
    'train_loss': [loss0, loss1, ..., loss_n],
    'val_loss': [loss0, loss1, ..., loss_n],
    'test_loss': [loss0, loss1, ..., loss_n],
    'best_epoch': [epoch0, epoch1, ..., epoch_n],
    'motif_metrics': [motif_dict0, motif_dict1, ..., motif_dict_n]
}
```

**Aggregated across all models**:
```python
all_results = {
    'GCN': seed_results,
    'GAT': seed_results,
    'MLP': seed_results,
    'MeanMedian': seed_results
}
```

**Saved to**: `{model}_results.json` (4 files)

#### Training Runs Count

| Model | Seeds | Runs |
|-------|-------|------|
| GCN | 5 | 5 |
| GAT | 5 | 5 |
| MLP | 5 | 5 |
| MeanMedian | 5 | 0 (non-learning) |
| **Total** | | **15** |

---

### Task B: `generate_detailed_statistics()` (lines 573-633)

#### Purpose
Compute comprehensive statistics from N seed results for each model

#### Input
Single model's seed results:
```python
test_losses = [0.45, 0.42, 0.48, 0.41, 0.44]  # 5 seed values
val_losses = [0.48, 0.44, 0.51, 0.43, 0.46]
train_losses = [0.35, 0.32, 0.38, 0.31, 0.34]
best_epochs = [45, 52, 38, 61, 49]
```

#### Computations

**Test Loss Statistics**:
- **Mean**: `np.mean(test_losses)` = 0.442
- **Median**: `np.median(test_losses)` = 0.442
- **Std Dev**: `np.std(test_losses)` = 0.0289
- **Min/Max**: 0.41 / 0.48
- **P25/P75**: 0.417 / 0.461
- **IQR**: 0.461 - 0.417 = 0.044
- **95% CI**: ±1.96 × SE where SE = std / √n = 0.0289 / √5 = ±0.0253

**Validation Loss Statistics**:
- Same as above

**Training Loss Statistics**:
- Mean, median, std only (no CI needed)

**Best Epoch Statistics**:
- Mean, median, values only

#### Output JSON Structure

```json
{
  "model": "GCN",
  "n_seeds": 5,
  "test_loss": {
    "mean": 0.442,
    "median": 0.442,
    "std": 0.0289,
    "p25": 0.417,
    "p75": 0.461,
    "iqr": 0.044,
    "min": 0.41,
    "max": 0.48,
    "ci_95": 0.0253,
    "se": 0.0129,
    "values": [0.45, 0.42, 0.48, 0.41, 0.44]
  },
  "val_loss": {
    "mean": 0.464,
    "median": 0.46,
    "std": 0.0334,
    "p25": 0.43,
    "p75": 0.49,
    "iqr": 0.06,
    "min": 0.43,
    "max": 0.51,
    "values": [0.48, 0.44, 0.51, 0.43, 0.46]
  },
  "train_loss": {
    "mean": 0.342,
    "median": 0.34,
    "std": 0.0289,
    "values": [0.35, 0.32, 0.38, 0.31, 0.34]
  },
  "best_epoch": {
    "mean": 49.0,
    "median": 49.0,
    "values": [45, 52, 38, 61, 49]
  }
}
```

**Stored**: `all_detailed_stats[model_name] = stats_dict`

**Saved to**: `detailed_statistics.json`

#### Interpretation

- **Tight CI** (±0.02): Estimates are precise, model is stable across seeds
- **Wide CI** (±0.10): Estimates are uncertain, model has high variance
- **High std**: Model performance varies significantly with initialization
- **Low std**: Model is robust to random seed variations

---

### Task C: `plot_baseline_comparison()` (lines 758-821)

#### Purpose
Visually compare all 4 models showing GNN advantage over baselines

#### Figure Specification
- **Size**: 16 × 6 inches
- **Format**: 2-panel plot
- **DPI**: 300

#### Panel 1: Mean vs Median Test Loss
- **Type**: Side-by-side bar chart
- **X-axis**: Model names (GCN, GAT, MLP, MeanMedian)
- **Y-axis**: Test loss value
- **Bars**:
  - Blue bars: Mean test loss
  - Coral bars: Median test loss
- **Purpose**: Show that mean and median are similar (robust measure)

#### Panel 2: Distribution with Quartiles
- **Type**: Statistical range plot
- **For each model**:
  - Red horizontal line: [Min, Max] range
  - Blue box: IQR (P25 to P75)
  - Green diamond: Mean
  - Purple circle: Median

**Example visualization**:
```
GCN:       ◆─────[▮▮▮▮▮]─────◯      ← Mean and IQR shown
GAT:       ◆──────[▮▮▮▮]──────◯
MLP:            ◆────[▮▮▮▮▮▮]──◯     ← Larger range, more spread
MeanMedian:        ◆──[▮▮▮▮▮▮▮]──◯   ← Worst performance
```

#### Output
**File**: `baseline_comparison.png`

#### Key Insights
- **GNN models (GCN, GAT)** should have lower loss than baselines
- **Tight IQR** = consistent performance across seeds
- **Wide IQR** = high variance (model unstable)
- **Gap between GCN/GAT and baselines** = extent of GNN advantage

#### Interpretation Example
- GCN mean = 0.442, MeanMedian mean = 0.612
- **GNN advantage**: 0.170 loss reduction (27% better)
- **Statistical significance**: Confirmed by Task G (Wilcoxon test)

---

### Task D: `compute_motif_metrics()` (lines 745-816)

#### Purpose
Evaluate model performance on each network motif structure separately

#### Network Motif Types
| Motif | Description | Count |
|-------|-------------|-------|
| **cascade** | Linear chain: A→B→C→D | 1000 graphs |
| **feedback_loop** | Cyclic: A→B→C→A | 1000 graphs |
| **feedforward_loop** | Diamond: A→B, A→C, B→D, C→D | 1000 graphs |
| **single_input_module** | Hub: A→B,C,D,E | 1000 graphs |

#### Computation Process (Per Seed)

For each trained model:
1. **Inference on test set**:
   - Forward pass through model
   - Get predictions for all nodes

2. **Per-motif evaluation**:
   ```python
   for each graph in test_set:
       motif_type = graph.motif_id
       for masked_node in graph:
           mse = (prediction - true_value)^2
           mae = |prediction - true_value|
       aggregate by motif_type
   ```

3. **Metrics computed per motif**:
   - Mean MSE: Average squared error across all graphs of that motif
   - Std MSE: Variation in MSE across graphs
   - Mean MAE: Average absolute error
   - Std MAE: Variation in MAE across graphs
   - Num graphs: Count of graphs for that motif

#### Output Per Seed

```json
{
  "cascade": {
    "mean_mse": 0.0245,
    "std_mse": 0.0034,
    "mean_mae": 0.1123,
    "std_mae": 0.0089,
    "num_graphs": 250
  },
  "feedback_loop": {
    "mean_mse": 0.0312,
    "std_mse": 0.0045,
    "mean_mae": 0.1456,
    "std_mae": 0.0102,
    "num_graphs": 250
  },
  "feedforward_loop": {...},
  "single_input_module": {...}
}
```

**Stored**: One per seed, aggregated in Task E

#### Aggregation Across Seeds

For each motif and model:
```python
mses_across_seeds = [0.0245, 0.0251, 0.0238, 0.0248, 0.0242]
mean_mse = np.mean(mses_across_seeds) = 0.0245
std_mse = np.std(mses_across_seeds) = 0.0050
```

#### Final Output

```json
{
  "GCN": {
    "cascade": {
      "num_graphs": 250,
      "mean_mse": 0.0245,
      "std_mse": 0.0050,
      "mean_mae": 0.1123,
      "std_mae": 0.0089
    },
    "feedback_loop": {...},
    ...
  },
  "GAT": {...},
  "MLP": {...},
  "MeanMedian": {...}
}
```

**Saved to**: `motif_metrics_summary.json`

#### Interpretation

- **GCN/GAT should have lower MSE/MAE** than MLP/MeanMedian on all motifs
- **Motif-specific differences** indicate structure affects learning:
  - Some models better on cascades
  - Others better on feedback loops
- **Large std** within a motif = inconsistent performance
- **Comparing across motifs** shows if some structures are harder to predict

---

### Task E: `plot_motif_comparison()` (lines 1218-1277)

#### Purpose
Bar chart comparison of per-motif performance across models

#### Figure Specification
- **Size**: 5n_motifs × 12 inches (depends on motif count, typically 20 × 12)
- **Format**: 2 × 4 grid (2 rows × 4 columns)
- **Rows**: MSE (top), MAE (bottom)
- **Columns**: One per motif type

#### Layout

```
┌─────────────────────────────────────────────────────┐
│           Motif-Specific Performance                │
├─────────────────┬─────────────────┬─────────────────┤
│  CASCADE MSE    │  FEEDBACK MSE   │ FEEDFORWARD MSE │
│                 │                 │                 │
│  ████ GCN       │  ████ GCN       │  ████ GCN       │
│  ███  GAT       │  ███  GAT       │  ███  GAT       │
│  ██   MLP       │  ██   MLP       │  ██   MLP       │
│  ██   MM        │  ██   MM        │  ██   MM        │
├─────────────────┼─────────────────┼─────────────────┤
│ CASCADE MAE     │ FEEDBACK MAE    │ FEEDFORWARD MAE │
│                 │                 │                 │
│  ████ GCN       │  ████ GCN       │  ████ GCN       │
│  ███  GAT       │  ███  GAT       │  ███  GAT       │
│  ██   MLP       │  ██   MLP       │  ██   MLP       │
│  ██   MM        │  ██   MM        │  ██   MM        │
└─────────────────┴─────────────────┴─────────────────┘
```

#### Per-Subplot Details

**Top row (MSE)**:
- X-axis: Model names (GCN, GAT, MLP, MeanMedian)
- Y-axis: Mean Squared Error value
- Bars: One per model with error bars (±std)
- Color: Default matplotlib palette

**Bottom row (MAE)**:
- Same structure as top row but for Mean Absolute Error

#### Output
**File**: `motif_comparison.png`

#### Interpretation

- **GCN/GAT bars should be shorter** (lower error) than MLP/MeanMedian
- **Consistent across motifs** = model generalizes well
- **Variable across motifs** = some structures harder to learn
- **Error bars** show uncertainty in metrics across seeds
- **Comparing within motif** shows relative model performance
- **Comparing across motifs** identifies challenging structures

---

### Task F: `plot_motif_heatmap()` (lines 1279-1338)

#### Purpose
Heatmap view of motif-specific performance for quick visual comparison

#### Figure Specification
- **Size**: 20 × 8 inches
- **Format**: Side-by-side heatmaps
- **DPI**: 300

#### Heatmap Structure

**Left Heatmap - MSE**:
```
           Cascade  Feedback  Feedforward  SingleInput
GCN        0.0245   0.0312    0.0198       0.0267
GAT        0.0268   0.0334    0.0215       0.0289
MLP        0.0456   0.0512    0.0401       0.0498
MeanMedian 0.0892   0.0934    0.0876       0.0911

Color scale (YlOrRd - Yellow to Red):
Light yellow = low MSE (good)
Dark red = high MSE (bad)
```

**Right Heatmap - MAE**:
```
           Cascade  Feedback  Feedforward  SingleInput
GCN        0.1123   0.1456    0.0987       0.1234
GAT        0.1245   0.1567    0.1098       0.1345
MLP        0.1876   0.2012    0.1654       0.1945
MeanMedian 0.2456   0.2534    0.2378       0.2489

Color scale (YlGnBu - Yellow to Green to Blue):
Light yellow = low MAE (good)
Dark blue = high MAE (bad)
```

#### Key Visual Patterns

1. **GCN/GAT rows** should be light colored (low error)
2. **MLP/MeanMedian rows** should be darker (higher error)
3. **Consistent coloring within rows** = model generalizes across structures
4. **Variable coloring across columns** = structures have different difficulty

#### Output
**File**: `motif_heatmap.png`

#### Interpretation
- Quick visual identification of best model (lightest row)
- Easy comparison across motif types
- Color intensity shows relative model quality
- Useful for identifying motif-specific challenges

---

### Task I: `plot_seed_variance()` (lines 821-866)

#### Purpose
Show distribution of test losses across seeds, indicating model stability

#### Figure Specification
- **Size**: 14 × 7 inches
- **Type**: Box plot with overlaid individual points
- **DPI**: 300

#### Box Plot Elements

For each model (one box):
```
                    max
                     |
                    [o]  ← whisker
                     |
                  ┌──┴──┐
                  │ ╱▮▮ │  ← P75
                  │▮▮╱◆ │  ← mean (green diamond)
                  │▮▮ ◯ │  ← median (purple circle)
                  │ ╱▮▮ │  ← P25
                  └──┴──┘
                     |
                    [o]  ← whisker
                     |
                    min

      ● ● ●          ← individual seed points (jittered)
      ●   ●
        ●
```

#### Point Overlay

- **Jittered x-position**: Add small random offset so points don't overlap
- **Blue dots**: Individual seed test losses
- **Alpha 0.6**: Semi-transparent to show overlapping points

#### Output
**File**: `seed_variance.png`

#### Interpretation

**Tight box + clustered points**:
- Model is stable across seeds
- Similar initialization/data split effects are small
- Good reproducibility

**Wide box + spread points**:
- High variance across seeds
- Initialization or data split significantly affects results
- May need more seeds or architectural tuning

**Outlier points**:
- Single seed performed much worse
- Investigate that training run (possible failure mode)

#### Example Analysis

```
GCN:   • • •           ← Tight cluster, good reproducibility
       •  ◆ •
         ◯

MeanMedian:  ●         ← Tight but higher overall (baseline)
          ● ◆ ●
            ◯
```

---

### Task J: `plot_statistical_summary_table()` (lines 980-1028)

#### Purpose
Comprehensive statistics table in comprehensive format

#### Figure Specification
- **Size**: 18 × 8 inches
- **Type**: Formatted table visualization
- **DPI**: 300

#### Table Columns

| Column | Content | Example |
|--------|---------|---------|
| Model | Model name | GCN |
| Mean | Mean test loss | 0.4420 |
| Median | Median test loss | 0.4420 |
| P25 | 25th percentile | 0.4170 |
| P75 | 75th percentile | 0.4610 |
| IQR | Interquartile range | 0.0440 |
| Std Dev | Standard deviation | 0.0289 |
| 95% CI | ±Confidence interval | ±0.0253 |
| Min | Minimum value | 0.4100 |
| Max | Maximum value | 0.4800 |
| N Seeds | Number of seeds | 5 |

#### Example Table

```
┌───────────┬────────┬────────┬────────┬────────┬────────┬────────┬──────────┬────────┬────────┬─────────┐
│ Model     │ Mean   │ Median │  P25   │  P75   │  IQR   │ Std    │  95% CI  │  Min   │  Max   │ N Seeds │
├───────────┼────────┼────────┼────────┼────────┼────────┼────────┼──────────┼────────┼────────┼─────────┤
│ GCN       │ 0.4420 │ 0.4420 │ 0.4170 │ 0.4610 │ 0.0440 │ 0.0289 │ ±0.0253  │ 0.4100 │ 0.4800 │    5    │
│ GAT       │ 0.4650 │ 0.4620 │ 0.4380 │ 0.4890 │ 0.0510 │ 0.0321 │ ±0.0282  │ 0.4350 │ 0.5120 │    5    │
│ MLP       │ 0.5340 │ 0.5280 │ 0.5120 │ 0.5560 │ 0.0440 │ 0.0178 │ ±0.0156  │ 0.5200 │ 0.5680 │    5    │
│ MeanMedian│ 0.6120 │ 0.6150 │ 0.5890 │ 0.6340 │ 0.0450 │ 0.0192 │ ±0.0168  │ 0.5850 │ 0.6450 │    5    │
└───────────┴────────┴────────┴────────┴────────┴────────┴────────┴──────────┴────────┴────────┴─────────┘
```

#### Formatting

- **Header row**: Dark background (#40466e), white bold text
- **Data rows**: Alternating white and light gray background
- **Font size**: 10pt (readable in PNG)
- **Cell borders**: Visible grid lines

#### Output
**File**: `statistical_summary_table.png`

#### Use Cases

- **Quick comparison**: Scan all models' statistics at once
- **Publication**: Include directly in papers/reports
- **Reproducibility**: Complete statistical summary in one image
- **Decision-making**: Identify best model and its variability

---

### Task K: `plot_train_val_test_progression()` (lines 1030-1150)

#### Purpose
Show how different loss metrics evolve across seeds, indicating training dynamics

#### Figure Specification
- **Size**: 5*n_models × 6 inches (typically 20 × 6)
- **Format**: Multi-panel plot (one subplot per model)
- **DPI**: 300

#### Per-Model Subplot

For each model (one subplot):

```
Loss
 0.7 ┤
     │
 0.6 ┤
     │                     ▲
 0.5 ┤        ▲    ▲  ▲    ▲  ▲
     │      ▲     ▲  ▲  ▲ ▲
 0.4 ┤    ●   ●  ●    ●
     │  ●
 0.3 ┤ ●
     │
 0.2 ┤
     │______________▬▬▬▬▬▬▬▬___
     └─────────────────────────
       1   2   3   4   5   Seed

Legend:
● = Train loss (green circles)
■ = Val loss (orange squares)
▲ = Test loss (red triangles)
─── = Mean line for each loss type
```

#### Elements Per Subplot

- **X-axis**: Seed number (1 to N)
- **Y-axis**: Loss value
- **Three scatter series**:
  - **Green circles**: Train loss per seed
  - **Orange squares**: Val loss per seed
  - **Red triangles**: Test loss per seed
- **Three dashed lines**: Mean for each loss type
- **Grid**: Background grid for readability

#### Output
**File**: `train_val_test_progression.png`

#### Interpretation

**Train loss (green)**:
- Should be lowest (model fits training data)
- Stable across seeds = reproducible training
- Increasing trend = not enough training capacity

**Val loss (orange)**:
- Should be between train and test
- Sharp increase from early to late seed = sensitivity to initialization

**Test loss (red)**:
- Most important metric
- Should be stable across seeds (tight cluster)
- Variance shows generalization reliability

**Relative spacing**:
- Small train-test gap = good generalization
- Large train-test gap = overfitting

#### Example Patterns

```
Good model:        Overfitting:        High variance:
●●●●●              ●●●●●               ●  ●
 ■■■■■              ■  ■  ■             ■ ■ ■ ■
  ▲▲▲▲               ▲    ▲▲▲           ▲   ▲
  stable            diverges             unstable
```

---

## Phase 2: Pairwise Statistical Comparisons

Statistical hypothesis testing with effect sizes and confidence intervals.

### Task G: `compute_pairwise_comparisons()` (lines 635-743)

#### Purpose
Compare pairs of models using non-parametric Wilcoxon signed-rank test with bootstrap CIs

#### Comparison Pairs

GNN-based vs non-GNN-based (4 pairs total):
1. **GCN vs MeanMedian**: GNN vs non-learning baseline
2. **GAT vs MeanMedian**: Another GNN vs non-learning baseline
3. **GCN vs MLP**: GNN vs learning-based baseline
4. **GAT vs MLP**: Another GNN vs learning-based baseline

**Rationale**: Compare GNN advantage over baselines, not GNN vs GNN

#### Step 1: Wilcoxon Signed-Rank Test (line 676)

**Input**: Two paired arrays of test losses
```python
losses_gcn = [0.45, 0.42, 0.48, 0.41, 0.44]  (5 seeds)
losses_mm  = [0.61, 0.59, 0.64, 0.58, 0.62]  (5 seeds)
```

**Process**:
1. Compute differences: `diff = losses_gcn - losses_mm = [-0.16, -0.17, -0.16, -0.17, -0.18]`
2. Rank absolute differences: `|diff| = [0.16, 0.17, 0.16, 0.17, 0.18]` → ranks: [2, 3.5, 2, 3.5, 5]
3. Sum ranks for positive differences (none here)
4. Compute Wilcoxon statistic
5. Calculate two-tailed p-value

**Output**:
- **statistic**: Wilcoxon test statistic (small = significant difference)
- **p_value**: Two-tailed significance level (0-1 scale)
- **Interpretation**: p < 0.05 means significantly different at α=0.05

#### Step 2: Rank-Biserial Effect Size (lines 678-693)

**Formula**:
```
r = 1 - (2R / n(n+1))

where:
  R = sum of ranks for positive differences (model_a - model_b > 0)
  n = number of seed pairs
```

**Range**: -1 to +1
- **r = 1**: Model A always better
- **r = 0**: No consistent difference
- **r = -1**: Model B always better

**Interpretation**:
- |r| < 0.1: Negligible effect
- 0.1 ≤ |r| < 0.3: Small effect
- 0.3 ≤ |r| < 0.5: Medium effect
- |r| ≥ 0.5: Large effect

#### Step 3: Bootstrap 95% Confidence Interval (lines 695-715)

**Purpose**: Quantify uncertainty in the rank-biserial effect size

**Process**:
```
for i = 1 to 10,000:
    1. Resample differences with replacement
       (e.g., pick seeds [0, 2, 2, 4, 1] from original 5 seeds)
    2. Compute rank-biserial for this bootstrap sample
    3. Store bootstrap_r[i]

95% CI = [percentile(bootstrap_r, 2.5), percentile(bootstrap_r, 97.5)]
```

**Example**:
- Bootstrap samples yield: r values from -0.1 to 0.9
- 2.5th percentile = 0.42
- 97.5th percentile = 0.82
- **Result**: r = 0.65 [95% CI: 0.42 to 0.82]

**Interpretation**:
- **Narrow CI** (0.42 to 0.82): Precise effect estimate
- **Wide CI** (-0.10 to 0.95): Uncertain effect (possibly due to small n)
- **CI includes 0**: Effect could be negligible

#### Output JSON

```json
{
  "GCN vs MeanMedian": {
    "wilcoxon_statistic": 0.0,
    "p_value": 0.0625,
    "rank_biserial": 0.8000,
    "rank_biserial_ci_lower": 0.6234,
    "rank_biserial_ci_upper": 0.9345,
    "rank_biserial_ci_str": "0.8000 [95% CI: 0.6234 to 0.9345]",
    "mean_loss_a": 0.4420,
    "mean_loss_b": 0.6120,
    "mean_diff": 0.1700,
    "better_model": "GCN",
    "is_significant": false,
    "interpretation": "No significant difference (p=0.0625)",
    "n_seeds": 5
  },
  "GAT vs MeanMedian": {...},
  "GCN vs MLP": {...},
  "GAT vs MLP": {...}
}
```

**Saved to**: `pairwise_comparisons.json`

#### Statistical Interpretation

**Example 1: Significant difference**
```
p_value = 0.0234 (< 0.05) ✓ Significant
rank_biserial = 0.65 [95% CI: 0.42 to 0.82]
→ GCN significantly outperforms MLP with medium-large effect
```

**Example 2: Non-significant difference**
```
p_value = 0.156 (> 0.05) ✗ Not significant
rank_biserial = 0.35 [95% CI: -0.05 to 0.68]
→ No significant difference, though CI overlaps 0
→ Small sample size (5 seeds) lacks power to detect medium effect
```

**Example 3: Small but significant effect**
```
p_value = 0.0423 (< 0.05) ✓ Significant
rank_biserial = 0.18 [95% CI: 0.02 to 0.35]
→ Statistically significant but practically negligible (|r| < 0.3)
→ May need domain expertise to judge practical importance
```

---

### Task H: `plot_pairwise_comparisons()` (lines 1152-1216)

#### Purpose
Visualize pairwise comparison results in comprehensive table format

#### Figure Specification
- **Size**: 20 × 8 inches
- **Type**: Formatted table
- **DPI**: 300

#### Table Structure

| Column | Content | Example |
|--------|---------|---------|
| Comparison | Pair label | "GCN vs MeanMedian" |
| Model A | First model | "GCN" |
| Model B | Second model | "MeanMedian" |
| Mean Loss A | Average loss for model A | "0.4420" |
| Mean Loss B | Average loss for model B | "0.6120" |
| Mean Diff | Absolute difference | "0.1700" |
| Wilcoxon p-value | Two-tailed p-value | "0.0625" |
| Effect Size (r) [95% CI] | Rank-biserial with CI | "0.8000 [95% CI: 0.6234 to 0.9345]" |
| Significant? | Better model if p<0.05 | "GCN *" or "No" |

#### Example Table

```
┌──────────────────────┬─────────┬────────────┬──────────────┬──────────────┬──────────────┬─────────────┬─────────────────────────────────┬────────────────┐
│ Comparison           │ Model A │ Model B    │ Mean Loss A  │ Mean Loss B  │ Mean Diff    │ Wilcoxon p  │ Effect Size (r) [95% CI]        │ Significant?   │
├──────────────────────┼─────────┼────────────┼──────────────┼──────────────┼──────────────┼─────────────┼─────────────────────────────────┼────────────────┤
│ GCN vs MeanMedian    │ GCN     │ MeanMedian │    0.4420    │    0.6120    │    0.1700    │   0.0625    │ 0.8000 [95% CI: 0.6234-0.9345] │ GCN *          │
│ GAT vs MeanMedian    │ GAT     │ MeanMedian │    0.4650    │    0.6120    │    0.1470    │   0.0938    │ 0.7500 [95% CI: 0.5123-0.9012] │ No             │
│ GCN vs MLP           │ GCN     │ MLP        │    0.4420    │    0.5340    │    0.0920    │   0.0313    │ 0.6234 [95% CI: 0.3456-0.8123] │ GCN *          │
│ GAT vs MLP           │ GAT     │ MLP        │    0.4650    │    0.5340    │    0.0690    │   0.1250    │ 0.5234 [95% CI: 0.2341-0.7654] │ No             │
└──────────────────────┴─────────┴────────────┴──────────────┴──────────────┴──────────────┴─────────────┴─────────────────────────────────┴────────────────┘
```

#### Formatting

- **Header row**: Dark background (#40466e), white bold text
- **Data rows**:
  - White background: 1st, 3rd row
  - Light gray (#f0f0f0): 2nd, 4th row
  - Light yellow (#ffffcc): Rows with significant results (p < 0.05)
- **Font**: 9pt (compact for wide table)
- **Cell width**: Effect size column widest to show full CI

#### Output
**File**: `pairwise_comparisons.png`

#### Interpretation Guide

**Reading the table**:

1. **Find your comparison row**: e.g., "GCN vs MeanMedian"
2. **Check Mean Diff**: How much better is Model A? (0.1700 = 17% loss reduction)
3. **Check Wilcoxon p-value**: Is difference statistically significant? (0.0625 > 0.05 = NO)
4. **Check Effect Size**: How consistent is the advantage?
   - 0.80 with CI [0.62-0.93] = Large, consistent effect
   - 0.35 with CI [-0.05-0.68] = Small, uncertain effect
5. **Check Significant column**: Quick visual indicator (yellow highlight)

**Decision logic**:
```
If p < 0.05 AND |r| > 0.3:
  → Significant difference with meaningful effect size
  → Model A is genuinely better

If p < 0.05 AND |r| < 0.1:
  → Significant by chance/power, but trivial effect
  → Practical difference negligible

If p >= 0.05 AND CI overlaps 0:
  → No significant difference
  → Insufficient evidence of difference

If p >= 0.05 BUT CI > 0:
  → Non-significant but suggests trend
  → May benefit from more seeds
```

---

## Phase 3: Motif-Specific Analysis

**Summary**: Task D computes metrics, Tasks E-F visualize them. Already covered above in Phase 1.

---

## Phase 4: Sensitivity Analysis (Optional)

Run with `--sensitivity` flag. Tests only GNN-based methods (GCN, GAT).

### Task L: `run_sensitivity_analysis()` (lines 390-529)

#### Purpose
Test how model performance varies with data generation parameters

#### Configuration
| Parameter | Range | Default | Purpose |
|-----------|-------|---------|---------|
| Timesteps | [25, 50, 75] | 50 | Expression dynamics length |
| Noise | [0.005, 0.01, 0.05] | 0.01 | Gaussian noise std dev |
| Epochs per config | 50 (fixed) | - | Shorter than main training |
| Early stopping | patience=5 | - | Faster than main (patience=10) |

#### Timestep Sensitivity (lines 409-468)

**Task**: Train model at different expression simulation lengths

**Fixed parameters**:
- noise_std = 0.01 (default value)
- Data split seed = 42 (constant for fair comparison)
- Model weights initialized with seed 42 (constant)

**Variable parameter**:
- timesteps in [25, 50, 75]

**Process**:
```python
for timestep_value in [25, 50, 75]:
    # Create data with specified timesteps
    train_dataset = DataVariationDataset(
        train_paths,
        steps=timestep_value,  # ← Variable
        noise_std=0.01,         # ← Fixed
        seed=42
    )
    # Train model
    # Record best val_loss, test_loss
```

**Output**:
```json
{
  "timesteps": [
    {"timesteps": 25, "val_loss": 0.445, "test_loss": 0.451},
    {"timesteps": 50, "val_loss": 0.442, "test_loss": 0.449},
    {"timesteps": 75, "val_loss": 0.440, "test_loss": 0.448}
  ]
}
```

**Interpretation**:
- **Decreasing loss**: Model benefits from longer dynamics (more signal)
- **Increasing loss**: Longer dynamics introduce noise/complexity
- **Flat**: Model robust to expression length changes
- **Optimum**: Lowest loss value = best timestep choice

#### Noise Sensitivity (lines 471-528)

**Task**: Train model at different noise levels

**Fixed parameters**:
- timesteps = 50 (default value)
- Data split seed = 42 (constant)
- Model weights seed = 42 (constant)

**Variable parameter**:
- noise_std in [0.005, 0.01, 0.05]

**Process**:
```python
for noise_value in [0.005, 0.01, 0.05]:
    # Create data with specified noise
    train_dataset = DataVariationDataset(
        train_paths,
        steps=50,              # ← Fixed
        noise_std=noise_value, # ← Variable
        seed=42
    )
    # Train model
    # Record best val_loss, test_loss
```

**Output**:
```json
{
  "noise": [
    {"noise_std": 0.005, "val_loss": 0.438, "test_loss": 0.445},
    {"noise_std": 0.01, "val_loss": 0.442, "test_loss": 0.449},
    {"noise_std": 0.05, "val_loss": 0.450, "test_loss": 0.456}
  ]
}
```

**Interpretation**:
- **Increasing loss with noise**: Model sensitive to noise (robustness issue)
- **Stable loss**: Model noise-robust
- **Best at 0.01**: Original choice is optimal
- **Bi-modal curve**: Suggests different regimes (low/high noise)

#### Output JSON Structure

**Per model** (GCN, GAT):
```json
{
  "GCN": {
    "timesteps": [
      {"timesteps": 25, "val_loss": 0.445, "test_loss": 0.451},
      {"timesteps": 50, "val_loss": 0.442, "test_loss": 0.449},
      {"timesteps": 75, "val_loss": 0.440, "test_loss": 0.448}
    ],
    "noise": [
      {"noise_std": 0.005, "val_loss": 0.438, "test_loss": 0.445},
      {"noise_std": 0.01, "val_loss": 0.442, "test_loss": 0.449},
      {"noise_std": 0.05, "val_loss": 0.450, "test_loss": 0.456}
    ]
  }
}
```

**Saved to**: `gcn_sensitivity.json`, `gat_sensitivity.json`

---

### Task M: `plot_sensitivity_analysis()` (lines 868-922)

#### Purpose
Combined visualization of timestep and noise sensitivity

#### Figure Specification
- **Size**: 12 × 12 inches
- **Format**: 2 × 2 grid
- **DPI**: 300

#### Layout

```
┌────────────────────────────────────────────────────────┐
│     Data Sensitivity Analysis: Timesteps and Noise     │
├──────────────────────────┬──────────────────────────────┤
│  GCN: Timestep Sens      │  GAT: Timestep Sens          │
│  (2 lines: val, test)    │  (2 lines: val, test)        │
│                          │                              │
├──────────────────────────┼──────────────────────────────┤
│  GCN: Noise Sens         │  GAT: Noise Sens             │
│  (2 lines: val, test)    │  (2 lines: val, test)        │
│                          │                              │
└──────────────────────────┴──────────────────────────────┘
```

#### Top Row: Timestep Sensitivity

**Left subplot (GCN)**:
- X-axis: Timesteps [25, 50, 75]
- Y-axis: Loss value
- **Blue line (circles)**: Test loss
- **Orange line (squares)**: Val loss
- **Red dashed vertical line**: Original timesteps = 50
- **Grid**: Background grid for readability

**Right subplot (GAT)**: Same structure

#### Bottom Row: Noise Sensitivity

**Left subplot (GCN)**:
- X-axis: Noise std [0.005, 0.01, 0.05]
- Y-axis: Loss value
- **Blue line**: Test loss
- **Orange line**: Val loss
- **Red dashed vertical line**: Original noise = 0.01

**Right subplot (GAT)**: Same structure

#### Output
**File**: `sensitivity_analysis.png`

#### Interpretation Patterns

**Pattern 1: Flat lines**
```
Loss
  0.45 ─────────────────
       Robust parameter
```
→ Model insensitive to parameter variations (good)

**Pattern 2: Declining loss**
```
Loss
  0.46 ──────
       ────────  → Improves with parameter
  0.44 ──────────────
```
→ More is better (longer timesteps, less noise)

**Pattern 3: Increasing loss**
```
Loss
  0.42 ──────────────
       ────────
  0.44 ──────
       ↑ Degrades with parameter increase
```
→ Sensitive to parameter (too much causes issues)

**Pattern 4: U-shaped curve**
```
Loss
  0.44 ──────
       \    /
  0.42  \──/  ← Optimal around middle
```
→ Sweet spot exists at specific parameter value

#### Decision Making

**For timesteps**:
- If flat: 50 is arbitrary, any value works
- If decreasing: Use higher timesteps (more signal)
- If increasing after 50: Use lower timesteps (avoid noise)
- If U-shaped: 50 is already near optimal

**For noise**:
- If flat: Model is noise-robust
- If increasing: Model sensitive to noise (be careful with real noisy data)
- If decreasing: Noise acts as regularization (unusual)

---

### Task N: `plot_individual_sensitivity_analysis()` (lines 924-1026)

#### Purpose
High-resolution individual sensitivity plots per model
#### Figure Specification Per Model
- **Size**: 16 × 10 inches (per model)
- **Format**: 2 subplots (timestep, noise)
- **DPI**: 300

#### Timestep Subplot

```
Loss
 0.46 ┤ ╱────── test loss
      │╱╱
 0.45 ┤     ├─── original (50)
      │\╱╱  │
 0.44 ┤ \╲╱
      │  ╲╱──────── val loss
 0.43 ┤    ╲
      │     └─────── shaded region between curves
      └─────────────────────────
        25    50    75
        Timesteps
```

**Elements**:
- X-axis: Timestep values [25, 50, 75]
- Y-axis: Loss
- **Blue line with circles**: Test loss
- **Orange line with squares**: Val loss
- **Shaded area**: Region between test and val (generalization gap)
- **Vertical red dashed line**: Original timesteps = 50
- **Legend**: Clear labeling

#### Noise Subplot

```
Loss
 0.46 ┤        ╱────── test loss
      │       ╱
 0.45 ┤      ├─── original (0.01)
      │ ─────┤
 0.44 ┤╱╱    │    ╲
      │╱╱╱   │     ╲──── val loss
 0.43 ┤      │      ╲╲╲
      │      └──────── shaded region
      └───────────────────────────
      0.005  0.01  0.05
      Noise Std Dev
```

**Elements**:
- X-axis: Noise values [0.005, 0.01, 0.05]
- Y-axis: Loss
- **Blue line**: Test loss
- **Orange line**: Val loss
- **Shaded area**: Generalization gap
- **Vertical red dashed line**: Original noise = 0.01

#### Output Files
**Files**:
- `gcn_sensitivity_detailed.png`
- `gat_sensitivity_detailed.png`

#### Interpretation

**Shaded region size** indicates generalization:
- **Narrow shading**: Val and test loss close (good generalization)
- **Wide shading**: Large val-test gap (overfitting)

**Line slopes**:
- **Steep slope**: Sensitive to parameter
- **Shallow slope**: Robust to parameter

**Distance from original (50 timesteps, 0.01 noise)**:
- **Losses increase when moving away**: Original choice is good
- **Losses decrease when moving away**: Original choice could be improved

---

## Phase 5: Data Storage

### Output Directory Structure

```
outputs/benchmark/
├── statistics/
│   ├── gcn_results.json
│   │   └─ Individual seed losses for GCN (5 rows)
│   ├── gat_results.json
│   │   └─ Individual seed losses for GAT (5 rows)
│   ├── mlp_results.json
│   │   └─ Individual seed losses for MLP (5 rows)
│   ├── meanmedian_results.json
│   │   └─ Individual seed losses for MeanMedian (5 rows)
│   ├── detailed_statistics.json
│   │   └─ Comprehensive stats for all 4 models (mean, median, CI, std, etc.)
│   ├── pairwise_comparisons.json
│   │   └─ Wilcoxon test results, rank-biserial, bootstrap CI for 4 pairs
│   ├── motif_metrics_summary.json
│   │   └─ Per-motif MSE/MAE aggregated across seeds for all models
│   ├── multi_seed_summary.json
│   │   └─ Summary of all results
│   ├── gcn_sensitivity.json [if --sensitivity]
│   │   └─ GCN losses at different timesteps and noise levels
│   └── gat_sensitivity.json [if --sensitivity]
│       └─ GAT losses at different timesteps and noise levels
│
└── visualizations/
    ├── baseline_comparison.png
    │   └─ Mean vs median, IQR comparison across 4 models
    ├── seed_variance.png
    │   └─ Box plot with individual seed points
    ├── statistical_summary_table.png
    │   └─ Table of all statistics (mean, median, CI, etc.)
    ├── train_val_test_progression.png
    │   └─ Per-seed loss progression for each model
    ├── pairwise_comparisons.png
    │   └─ Table of Wilcoxon tests and effect sizes
    ├── motif_comparison.png
    │   └─ 2×4 grid of per-motif MSE/MAE bars
    ├── motif_heatmap.png
    │   └─ Side-by-side MSE and MAE heatmaps
    ├── sensitivity_analysis.png [if --sensitivity]
    │   └─ 2×2 grid of timestep and noise sensitivity
    ├── gcn_sensitivity_detailed.png [if --sensitivity]
    │   └─ High-res timestep and noise sensitivity for GCN
    └── gat_sensitivity_detailed.png [if --sensitivity]
        └─ High-res timestep and noise sensitivity for GAT
```

### JSON File Contents

#### `gcn_results.json` (example)

```json
{
  "train_loss": [0.35, 0.32, 0.38, 0.31, 0.34],
  "val_loss": [0.48, 0.44, 0.51, 0.43, 0.46],
  "test_loss": [0.45, 0.42, 0.48, 0.41, 0.44],
  "best_epoch": [45, 52, 38, 61, 49],
  "motif_metrics": [
    {
      "cascade": {"mean_mse": 0.0245, "std_mse": 0.0034, ...},
      "feedback_loop": {...},
      ...
    },
    ...  # 5 entries (one per seed)
  ]
}
```

#### `detailed_statistics.json` (example)

```json
{
  "GCN": {
    "model": "GCN",
    "n_seeds": 5,
    "test_loss": {
      "mean": 0.442,
      "median": 0.442,
      "std": 0.0289,
      "p25": 0.417,
      "p75": 0.461,
      "iqr": 0.044,
      "min": 0.41,
      "max": 0.48,
      "ci_95": 0.0253,
      "se": 0.0129,
      "values": [0.45, 0.42, 0.48, 0.41, 0.44]
    },
    ...
  },
  "GAT": {...},
  "MLP": {...},
  "MeanMedian": {...}
}
```

#### `pairwise_comparisons.json` (example)

```json
{
  "GCN vs MeanMedian": {
    "wilcoxon_statistic": 0.0,
    "p_value": 0.0625,
    "rank_biserial": 0.8000,
    "rank_biserial_ci_lower": 0.6234,
    "rank_biserial_ci_upper": 0.9345,
    "rank_biserial_ci_str": "0.8000 [95% CI: 0.6234 to 0.9345]",
    "mean_loss_a": 0.4420,
    "mean_loss_b": 0.6120,
    "mean_diff": 0.1700,
    "better_model": "GCN",
    "is_significant": false,
    "interpretation": "No significant difference (p=0.0625)",
    "n_seeds": 5
  },
  "GAT vs MeanMedian": {...},
  "GCN vs MLP": {...},
  "GAT vs MLP": {...}
}
```

---

## Complete Workflow Summary

### All Tasks at a Glance

| Phase | Task | Calculation | Visualization | Output |
|-------|------|-------------|----------------|--------|
| **1** | A | run_multi_seed_training | - | {model}_results.json |
| **1** | B | generate_detailed_statistics | - | detailed_statistics.json |
| **1** | C | - | plot_baseline_comparison | baseline_comparison.png |
| **1** | D | compute_motif_metrics | - | (embedded in A) |
| **1** | E | - | plot_motif_comparison | motif_comparison.png |
| **1** | F | - | plot_motif_heatmap | motif_heatmap.png |
| **1** | I | - | plot_seed_variance | seed_variance.png |
| **1** | J | - | plot_statistical_summary_table | statistical_summary_table.png |
| **1** | K | - | plot_train_val_test_progression | train_val_test_progression.png |
| **2** | G | compute_pairwise_comparisons | - | pairwise_comparisons.json |
| **2** | H | - | plot_pairwise_comparisons | pairwise_comparisons.png |
| **4** | L | run_sensitivity_analysis | - | {model}_sensitivity.json |
| **4** | M | - | plot_sensitivity_analysis | sensitivity_analysis.png |
| **4** | N | - | plot_individual_sensitivity_analysis | {model}_sensitivity_detailed.png |

### Execution Flow

```
main() called
  ├─ Parse arguments (--seeds, --epochs, --batch-size, --sensitivity)
  ├─ Initialize BenchmarkExperiment
  │
  ├─ PHASE 1: Multi-Seed Training
  │  └─ For each model in [GCN, GAT, MLP, MeanMedian]:
  │     ├─ Task A: run_multi_seed_training()
  │     │  └─ For each seed 0 to n_seeds-1:
  │     │     ├─ Set torch/np random seeds
  │     │     ├─ Split data, create loaders
  │     │     ├─ Create model, train
  │     │     └─ Task D: compute_motif_metrics()
  │     ├─ Task B: generate_detailed_statistics()
  │     └─ Save {model}_results.json
  │
  ├─ PHASE 1: Aggregate Motif Metrics
  │  └─ For each model:
  │     └─ Aggregate motif metrics across seeds
  │     └─ Save to motif_metrics_summary.json
  │
  ├─ PHASE 2: Pairwise Comparisons (always)
  │  └─ Task G: compute_pairwise_comparisons()
  │     ├─ GCN vs MeanMedian: Wilcoxon + bootstrap CI
  │     ├─ GAT vs MeanMedian: Wilcoxon + bootstrap CI
  │     ├─ GCN vs MLP: Wilcoxon + bootstrap CI
  │     └─ GAT vs MLP: Wilcoxon + bootstrap CI
  │     └─ Save to pairwise_comparisons.json
  │
  ├─ PHASE 1: Visualizations (always)
  │  ├─ Task C: plot_baseline_comparison()
  │  ├─ Task I: plot_seed_variance()
  │  ├─ Task J: plot_statistical_summary_table()
  │  ├─ Task K: plot_train_val_test_progression()
  │  └─ Task H: plot_pairwise_comparisons()
  │
  ├─ PHASE 3: Motif Visualizations (if metrics exist)
  │  ├─ Task E: plot_motif_comparison()
  │  └─ Task F: plot_motif_heatmap()
  │
  ├─ PHASE 4: Sensitivity Analysis (if --sensitivity)
  │  └─ For each model in [GCN, GAT]:
  │     ├─ Task L: run_sensitivity_analysis()
  │     │  ├─ Timestep sensitivity [25, 50, 75]
  │     │  └─ Noise sensitivity [0.005, 0.01, 0.05]
  │     └─ Save {model}_sensitivity.json
  │
  ├─ PHASE 4: Sensitivity Visualizations (if --sensitivity)
  │  ├─ Task M: plot_sensitivity_analysis()
  │  └─ Task N: plot_individual_sensitivity_analysis()
  │
  └─ Print summary and file locations
```

---

## Command Usage Examples

### Example 1: Quick Validation (5 seeds, no sensitivity)

```bash
python benchmarking.py --seeds 5 --epochs 100
```

**What runs**:
- Task A: Train 15 models (5 seeds × 3 trainable models)
- Tasks B-K: Statistics and visualizations
- Task G-H: Pairwise comparisons
- **No sensitivity analysis**

**Output**:
- 4 JSON files ({model}_results.json)
- 9 JSON files (statistics, detailed_stats, pairwise, motif_summary, multi_seed_summary)
- 9 PNG files (all visualizations except sensitivity)
- **Time**: ~30-45 minutes

### Example 2: Standard Evaluation (5 seeds, with sensitivity)

```bash
python benchmarking.py --seeds 5 --epochs 100 --sensitivity
```

**What runs**:
- Phase 1: Multi-seed training (15 training runs)
- Phase 2: Pairwise comparisons
- Phase 3: Motif metrics
- Phase 4: Sensitivity analysis (9 + 9 = 18 training runs for GCN and GAT)

**Output**:
- All 13 JSON files (including sensitivity files)
- All 12 PNG files (including sensitivity plots)
- **Time**: ~2-3 hours

### Example 3: Publication-Quality (10 seeds, with sensitivity)

```bash
python benchmarking.py --seeds 10 --epochs 150 --sensitivity
```

**What runs**:
- Phase 1: Multi-seed training (30 training runs, 150 epochs each)
- Phase 2: Pairwise comparisons (bootstrap CIs from 10 seed pairs)
- Phases 3-4: All metrics and visualizations

**Output**:
- Complete statistical results with tight CIs
- High-power statistical tests
- Production-ready visualizations
- **Time**: ~3-4 hours

### Example 4: Custom Configuration

```bash
python benchmarking.py --seeds 7 --epochs 120 --batch-size 64 --output-dir results/custom_run --sensitivity
```

**Custom settings**:
- 7 random seeds (better than 5, less than 10)
- 120 epochs (careful training)
- Batch size 64 (if GPU has memory)
- Custom output directory
- Sensitivity analysis enabled

---

## Statistical Concepts Used

### Wilcoxon Signed-Rank Test
- **Type**: Non-parametric paired comparison test
- **Assumption**: No normality assumption (robust)
- **Null hypothesis**: Two models produce same distribution of losses
- **Test statistic**: Sum of ranks where first model < second
- **Output**: p-value (two-tailed)

### Rank-Biserial Correlation
- **Type**: Non-parametric effect size
- **Formula**: r = 1 - (2R / n(n+1))
- **Range**: -1 to +1
- **Interpretation**: Probability that random sample from model A < model B
- **Advantage**: Independent of sample size

### Bootstrap Confidence Intervals
- **Method**: Non-parametric resampling
- **Process**: Resample with replacement 10,000 times, compute statistic
- **CI**: [2.5th percentile, 97.5th percentile] of bootstrap distribution
- **Advantage**: No assumption of parametric distribution

### Multi-Seed Analysis
- **Purpose**: Account for stochasticity in initialization and data splits
- **Typical n_seeds**: 5-20 (trade-off between rigor and computation)
- **Benefit**: Confidence that results not due to lucky seed
- **SE formula**: std / √n (decreases with more seeds)

---

## Key Metrics Explained

### Test Loss (Primary Metric)
- **Definition**: MSE on held-out test set, computed only on masked nodes
- **Why important**: Measures generalization to unseen nodes
- **Range**: 0 to ∞ (lower is better)
- **Reported as**: Mean ± 95% CI across N seeds

### Wilcoxon p-value
- **Definition**: Probability of observing given or more extreme difference under null hypothesis
- **Significance threshold**: p < 0.05 (α = 0.05)
- **Interpretation**: p = 0.03 means 3% chance of difference if models identical

### Rank-Biserial Effect Size
- **Definition**: How consistently model A outperforms model B
- **Range**: -1 (always B better) to +1 (always A better)
- **Thresholds**:
  - |r| < 0.1: Negligible
  - 0.1-0.3: Small
  - 0.3-0.5: Medium
  - > 0.5: Large

### 95% Confidence Interval
- **Definition**: Range where true population parameter likely resides
- **Interpretation**: 95% of such intervals contain true value (if repeated)
- **Width**: Narrow = precise, wide = uncertain
- **On rank-biserial**: Bootstrap CI shows effect size uncertainty

---

## Troubleshooting & Tips

### Common Issues

**Out of Memory**
```bash
# Reduce batch size
python benchmarking.py --batch-size 16
```

**Training Too Slow**
```bash
# Reduce seeds and epochs
python benchmarking.py --seeds 3 --epochs 50
```

**NaN or Inf Values**
- Check that graphs are loaded correctly
- Verify GraphDataset normalizes features properly
- Check data file paths

**Missing Visualizations**
- Ensure `--sensitivity` flag if expecting sensitivity plots
- Check output directory permissions
- Verify matplotlib can write PNG files

### Performance Optimization

1. **Use GPU**: Script auto-detects CUDA availability
2. **Reduce batch-size trades memory for speed**: --batch-size 16 (slower but lower memory)
3. **Run sensitivity separately**: Could parallelize across models
4. **Archive old runs**: Remove large benchmark directories to save space

---

## References & Further Reading

- **Wilcoxon test**: https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
- **Rank-biserial correlation**: https://en.wikipedia.org/wiki/Rank_biserial_correlation
- **Bootstrap CIs**: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
- **Multi-seed evaluation**: https://openml.org/articles/machine-learning-evaluation/

---

**Document Version**: 1.0
**Last Updated**: 2025-01-28
**Benchmarking.py Version**: ~1554 lines
