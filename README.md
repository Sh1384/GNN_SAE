# GNN-SAE: Graph Neural Network Sparse Autoencoder Analysis

Interpretable analysis of Graph Neural Networks using Sparse Autoencoders (SAEs) to discover motif-specific features in synthetic graph data.

## Overview

This project trains Graph Neural Networks (GNNs) on synthetic graphs containing canonical network motifs (feedforward loops, feedback loops, single-input modules, and cascades), then applies Sparse Autoencoders to GNN hidden layer activations to discover interpretable features that correspond to these motifs.

**Key Innovation**: Multi-variant SAE analysis with comprehensive ablation studies to validate that discovered features are genuinely motif-specific rather than artifacts of degree or other confounds.

## Project Structure

```
182-GNN_SAE/
├── gnn_train.py                          # Train GNN and extract layer activations
├── hyperparameter_sweep_multi_gpu.py     # GNN hyperparameter optimization
├── run_sae_pipeline_multi_gpu.py         # Main pipeline - trains all 30 SAE variants
│
├── sparse_autoencoder.py                 # SAE model definitions (TopK, Gated, JumpReLU, Switch)
│
├── SAE Training & Comparison
│   ├── compare_sae_configs.py            # Compare configurations within variants
│   ├── compare_sae_variants.py           # Compare across SAE variants
│   ├── retrain_best_configs.py           # Multi-seed validation
│
├── Feature Analysis & Ablation
│   ├── analyze_feature_significance.py   # Feature-motif correlation analysis
│   ├── run_ablation.py                   # SAE latent space ablations
│   ├── native_gnn_ablation.py            # Direct GNN activation ablations
│   ├── compare_ablation_strategies.py    # Compare ablation methods
│   ├── compare_sae_vs_gnnexplainer.py    # Compare with GNNExplainer baseline
│
├── Statistical Analysis & Visualization
│   ├── statistical_analysis_suite.py     # Permutation tests, FDR correction
│   ├── aggregate_validation_report.py    # Generate comprehensive reports
│   ├── visualize_feature_activations.py  # Feature heatmaps
│   ├── analyze_sae_reconstruction_fidelity.py
│   ├── plot_random_control_distributions.py
│   ├── visualize_sweep_results.py
│   ├── visualize_training_progress.py
│   ├── analyze_comparison_results.py
│
├── Utilities
│   ├── generate_mixed_motif_activations.py  # Generate mixed-motif test graphs
│   ├── identify_top_sae_features.py
│   ├── print_phase2_summary.py
│
├── Shell Scripts
│   ├── RUN_PIPELINE.sh                   # Quick pipeline launcher
│   ├── run_multi_gpu_sweep.sh            # Hyperparameter sweep launcher
│   ├── monitor_gpu_training.sh           # Real-time GPU monitoring
│   ├── check_prerequisites.sh            # Verify setup
│
├── Graph Generation (virtual_graphs/)
│   ├── graph_motif_generator.py          # Generate synthetic motif graphs
│   ├── compute_node_features.py          # Compute degree features
│   ├── compute_node_features_with_centrality.py  # Add centrality controls
│   ├── detect_actual_motifs.py           # Validate motif presence
│   ├── validate_motif_detection.py
│   └── data/                             # Generated graphs and metadata
│
├── Outputs
│   ├── activations/                      # GNN layer activations
│   ├── checkpoints/                      # Trained model checkpoints
│   ├── ablations/                        # Ablation study results
│   ├── pipeline_logs/                    # Execution logs
│   └── [various analysis outputs]
│
└── Documentation
    ├── README.md                         # This file
    ├── MULTI_GPU_PIPELINE_README.md      # Detailed pipeline documentation
    ├── SAE_PIPELINE_README.md            # SAE variant documentation
    ├── RUN_WORKFLOW.md                   # Step-by-step workflow
    └── graph_generation.md               # Graph generation details
```

## Quick Start

### Prerequisites

**Hardware**:
- 4x NVIDIA GPUs (e.g., TITAN V with 12GB VRAM each)
- ~16GB system RAM
- ~20GB storage

**Software**:
- Python 3.8+
- PyTorch with CUDA support
- PyTorch Geometric
- Standard scientific Python stack (numpy, pandas, matplotlib, scikit-learn)

### Installation

```bash
# Clone or navigate to the repository
cd /path/to/182-GNN_SAE

# Install dependencies (adjust for your environment)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install numpy pandas matplotlib scikit-learn tqdm networkx
```

### Complete Pipeline (All Phases)

```bash
# Run the complete SAE variant analysis pipeline
python run_sae_pipeline_multi_gpu.py --all
```

This executes all 5 phases (~12-15 hours total):
1. **Phase 1**: Train 30 SAE configurations in parallel (~5 hours)
2. **Phase 2**: Compare configurations and variants (~20 min)
3. **Phase 3**: Ablation studies (~4 hours)
4. **Phase 4**: Statistical validation (~30 min)
5. **Phase 5**: Visualization (~15 min)

### Individual Phases

```bash
# Train GNN and extract activations (if not already done)
python gnn_train.py

# Train all 30 SAE variants in parallel
python run_sae_pipeline_multi_gpu.py --phase 1

# Compare configurations
python run_sae_pipeline_multi_gpu.py --phase 2

# Run ablation studies
python run_sae_pipeline_multi_gpu.py --phase 3

# Statistical validation
python run_sae_pipeline_multi_gpu.py --phase 4

# Generate visualizations
python run_sae_pipeline_multi_gpu.py --phase 5
```

### Monitor Progress

```bash
# Real-time GPU monitoring (in separate terminal)
./monitor_gpu_training.sh

# Watch pipeline logs
tail -f outputs/pipeline_logs/pipeline_*.log

# Check GPU utilization
watch -n 1 nvidia-smi

# Count completed checkpoints
ls checkpoints/sae_*.pt | wc -l
```

## Pipeline Details

### Phase 1: SAE Training

Trains **30 different SAE configurations** across **4 architectural variants**:

**TopK SAE** (11 configs):
- Latent dimensions: 128, 256, 512
- Top-K sparsity: k ∈ {4, 8, 16, 32}
- Standard sparse autoencoder with hard Top-K activation

**Gated SAE** (9 configs):
- Latent dimensions: 128, 256, 512
- Sparsity coefficients: {1e-4, 5e-4, 1e-3}
- Separates feature detection from magnitude

**JumpReLU SAE** (6 configs):
- Latent dimensions: 128, 256, 512
- Threshold initialization: {0.01, 0.1}
- Discontinuous activation with straight-through estimator

**Switch SAE** (4 configs):
- Number of experts: {4, 8}
- Latent per expert: {64, 128}
- Mixture-of-experts routing

**Input**: Layer 2 GNN activations (80-dim)
**Duration**: ~5 hours on 4 GPUs
**Outputs**: `checkpoints/sae_*.pt`, `outputs/sae_metrics_*.json`

### Phase 2: Configuration Comparison

**Phase 2.0** - Within-variant comparison:
- Loads all trained SAE checkpoints
- Computes feature-motif correlations with degree controls
- Ranks configurations by composite score
- Output: `outputs/sae_config_comparison.csv`

**Phase 2.5** - Cross-variant comparison:
- Compares TopK vs Gated vs JumpReLU vs Switch
- Analyzes reconstruction vs sparsity trade-offs
- Output: `outputs/sae_variant_comparison.csv`

**Phase 2b** - Multi-seed retraining:
- Retrains best 4 configs with seeds [42, 123, 456, 789, 1011]
- Validates reproducibility and stability
- Duration: ~2.7 hours

**Key Metrics**:
- `max_rpb_abs`: Maximum bivariate correlation
- `max_rpb_partial_abs`: Maximum partial correlation (degree-controlled)
- `composite_score`: Weighted combination (50% correlation + 35% F1 + 15% capacity)
- `test_reconstruction`: MSE reconstruction loss

### Phase 3: Ablation Studies

**Phase 3a** - SAE latent space ablations:
- Identifies top features correlated with each motif
- Systematically zeros out features
- Measures impact on GNN predictions
- Script: `run_interpretability_experiments.py`

**Phase 3b** - Native GNN ablations:
- Directly ablates GNN activations (no SAE)
- Uses same methodology for fair comparison
- Script: `native_gnn_ablation.py`

**Phase 3c** - Ablation strategy comparison:
- Compares SAE-based vs direct ablation
- Validates that SAE features are genuinely interpretable
- Script: `compare_ablation_strategies.py`

**Phase 3d** - Mixed-motif generalization:
- Tests on graphs with multiple concurrent motifs
- Validates feature specificity
- Script: `generate_mixed_motif_activations.py`

### Phase 4: Statistical Validation

- Permutation testing (1000 permutations)
- False Discovery Rate (FDR) correction
- Feature stability across random seeds
- Confidence intervals for all metrics
- Script: `statistical_analysis_suite.py`

### Phase 5: Visualization

- Feature activation heatmaps
- Reconstruction fidelity analysis
- Motif-specific feature distributions
- Ablation impact visualizations
- Scripts: `visualize_feature_activations.py`, `analyze_sae_reconstruction_fidelity.py`

## Graph Generation

### Synthetic Motif Graphs

The `virtual_graphs/` directory contains code to generate synthetic directed graphs with canonical network motifs:

**Motif Types**:
1. **Feedforward Loop**: A→B→C, A→C (coherent/incoherent)
2. **Feedback Loop**: A↔B (mutual regulation)
3. **Single-Input Module**: A→{B, C, D, ...} (one regulator, multiple targets)
4. **Cascade**: A→B→C→D (linear chain)

**Dataset Composition**:
- 4,000 single-motif graphs (1,000 per motif type)
- 1,000 mixed-motif graphs (2-3 concurrent motifs)
- Each graph has 10 nodes, continuous edge weights [0,1]

**Generation**:
```bash
cd virtual_graphs
python graph_motif_generator.py
```

**Feature Computation**:
```bash
# Compute degree features (required for controls)
python virtual_graphs/compute_node_features.py

# Add centrality features (optional, for deeper controls)
python virtual_graphs/compute_node_features_with_centrality.py
```

## GNN Training

The GNN architecture is a 4-layer Graph Convolutional Network (GCN):

**Architecture**:
- Input: 2D node features (random initialization)
- Layer 1: GCNConv(2 → 80) + ReLU + Dropout(0.2)
- Layer 2: GCNConv(80 → 80) + ReLU + Dropout(0.2)  ← **SAE trained on this**
- Layer 3: GCNConv(80 → 64) + ReLU + Dropout(0.2)
- Layer 4: GCNConv(64 → 1) (graph-level binary prediction)

**Task**: Classify whether graph contains specific motif type

**Training**:
```bash
python gnn_train.py
```

**Outputs**:
- `checkpoints/gnn_model.pt`: Trained GNN weights
- `outputs/activations/layer2_new/{train,val,test}/`: Layer 2 activations
- `outputs/training_metrics.json`: Training curves
- `outputs/motif_metrics.json`: Per-motif performance

## Advanced Usage

### Custom GPU Selection

```bash
# Use only GPUs 0 and 1
python run_sae_pipeline_multi_gpu.py --phase 1 --gpus 0,1

# Use all 4 GPUs (default)
python run_sae_pipeline_multi_gpu.py --phase 1 --gpus 0,1,2,3
```

### Force Retrain

```bash
# Retrain all configurations even if checkpoints exist
python run_sae_pipeline_multi_gpu.py --phase 1 --force-retrain
```

### Run Specific Analyses

```bash
# Compare only specific SAE variant
python compare_sae_configs.py --variant topk

# Run ablation for specific motif
python run_ablation.py --variant topk --latent_dim 256 --k 16 --motif in_feedforward_loop

# Compare with GNNExplainer
python compare_sae_vs_gnnexplainer.py --variant topk

# Generate validation report
python aggregate_validation_report.py
```

### Hyperparameter Tuning

```bash
# Grid search over GNN hyperparameters
python hyperparameter_sweep_multi_gpu.py

# Visualize sweep results
python visualize_sweep_results.py
```

## Key Results Files

After running the pipeline, check these outputs:

**Configuration Comparison**:
- `outputs/sae_config_comparison.csv`: All 30 configs ranked
- `outputs/sae_variant_comparison.csv`: Variant comparison summary

**Ablation Results**:
- `ablations/interpretability_*/`: Per-config ablation results
- `ablations/phase_3a_config.json`: Best configuration metadata
- `outputs/native_gnn_ablations/`: GNN ablation baselines

**Statistical Validation**:
- `outputs/statistical_analysis/`: Permutation test results, FDR corrections

**Visualizations**:
- `outputs/feature_activation_visualizations/`: Feature heatmaps
- `outputs/comparison_plots/`: Variant comparison plots
- `outputs/paper_figures/`: Publication-ready figures

**Logs**:
- `outputs/pipeline_logs/pipeline_*.log`: Complete execution logs

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size (edit run_sae_pipeline_multi_gpu.py line 56)
BATCH_SIZE = 512  # Default: 1024

# Use fewer GPUs
python run_sae_pipeline_multi_gpu.py --gpus 0,1

# Train configs sequentially instead of in parallel
python sparse_autoencoder.py
```

### Missing Activations

```bash
# Verify activations exist
ls outputs/activations/layer2_new/train/ | wc -l  # Should show ~3200

# If missing, retrain GNN
python gnn_train.py
```

### Import Errors

All SAE analysis scripts now import directly from the project root. Ensure you're running from the `182-GNN_SAE/` directory:

```bash
cd /path/to/182-GNN_SAE
python run_sae_pipeline_multi_gpu.py --phase 1
```

### Process Hangs

```bash
# Check if process is running
ps aux | grep run_sae_pipeline

# Check GPU utilization
nvidia-smi

# Kill stuck processes
pkill -9 -f run_sae_pipeline

# Restart with force-retrain
python run_sae_pipeline_multi_gpu.py --phase 1 --force-retrain
```

## Performance Optimization

**Maximize GPU Utilization**:
```python
# Edit run_sae_pipeline_multi_gpu.py
BATCH_SIZE = 2048  # Default: 1024 (if you have more VRAM)
num_workers = 4    # Default: 2 (for data loading)
```

**Reduce Training Time** (may affect quality):
```python
NUM_EPOCHS = 100   # Default: 200
patience = 10      # Default: 15 (early stopping)
```

## Citation

If you use this code or methodology, please cite:

```bibtex
@software{gnn_sae_2025,
  title={GNN-SAE: Interpretable Graph Neural Network Analysis via Sparse Autoencoders},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/182-GNN_SAE}
}
```

## Additional Documentation

- **[MULTI_GPU_PIPELINE_README.md](MULTI_GPU_PIPELINE_README.md)**: Detailed multi-GPU pipeline documentation
- **[SAE_PIPELINE_README.md](SAE_PIPELINE_README.md)**: SAE variants and configurations
- **[RUN_WORKFLOW.md](RUN_WORKFLOW.md)**: Step-by-step workflow guide
- **[graph_generation.md](graph_generation.md)**: Graph generation technical details

## License

[Specify your license here]

## Contact

[Your contact information]
