# Multi-GPU SAE Training Pipeline

Complete pipeline for training and analyzing Sparse Autoencoders (SAEs) on GNN layer 3 activations using 4 NVIDIA TITAN V GPUs in parallel.

## Overview

This pipeline automates the complete SAE workflow:
1. **Phase 1**: Parallel training of multiple SAE configurations across 4 GPUs
2. **Phase 2**: Configuration comparison and ranking
3. **Phase 3**: Interpretability analysis on best configuration

## Hardware Requirements

- **GPUs**: 4x NVIDIA TITAN V (12 GB each)
- **RAM**: ~16 GB system memory
- **Storage**: ~10 GB for checkpoints and activations

## Prerequisites

### 1. GNN Training Complete
Ensure layer 3 activations are available:
```bash
ls outputs/activations/layer3_new/train/  # Should have ~3000 files
ls outputs/activations/layer3_new/val/    # Should have ~500 files
ls outputs/activations/layer3_new/test/   # Should have ~400 files
```

If not, run GNN training first:
```bash
python gnn_train_copy.py --model_type GCN
```

### 2. Centrality Features Computed
Ensure topological features are available for controls:
```bash
ls virtual_graphs/data/all_graphs/graph_motif_metadata/
```

If metadata doesn't include centrality features, run:
```bash
python virtual_graphs/compute_node_features_with_centrality.py
```

## Quick Start

### 1. View Configurations
```bash
python run_sae_pipeline_multi_gpu.py --configs-only
```

### 2. Run Complete Pipeline
```bash
# Run all phases (training, comparison, analysis)
python run_sae_pipeline_multi_gpu.py
```

### 3. Monitor Training (in separate terminal)
```bash
./monitor_gpu_training.sh
```

## SAE Configurations

The pipeline trains **11 different configurations**:

| Latent Dim | K  | Sparsity | Description                      |
|------------|----|----------|----------------------------------|
| 128        | 4  | 3.1%     | Low capacity, very low sparsity  |
| 128        | 8  | 6.2%     | Low capacity, low sparsity       |
| 128        | 16 | 12.5%    | Low capacity, moderate sparsity  |
| 256        | 4  | 1.6%     | Medium capacity, very low        |
| 256        | 8  | 3.1%     | Medium capacity, low             |
| 256        | 16 | 6.2%     | Medium capacity, moderate        |
| 256        | 32 | 12.5%    | Medium capacity, high            |
| 512        | 4  | 0.8%     | High capacity, very low          |
| 512        | 8  | 1.6%     | High capacity, low               |
| 512        | 16 | 3.1%     | High capacity, moderate          |
| 512        | 32 | 6.2%     | High capacity, high              |

## Usage Examples

### Run Only Training Phase
```bash
# Train all configs in parallel
python run_sae_pipeline_multi_gpu.py --phase train

# Force retrain even if checkpoints exist
python run_sae_pipeline_multi_gpu.py --phase train --force-retrain
```

### Run Only Comparison Phase
```bash
# Compare existing checkpoints
python run_sae_pipeline_multi_gpu.py --phase compare
```

### Use Specific GPUs
```bash
# Use only GPUs 0 and 1
python run_sae_pipeline_multi_gpu.py --gpus 0,1

# Use all 4 GPUs (default)
python run_sae_pipeline_multi_gpu.py --gpus 0,1,2,3
```

### Advanced Options
```bash
# Skip comparison after training
python run_sae_pipeline_multi_gpu.py --skip-comparison

# Force retrain all configurations
python run_sae_pipeline_multi_gpu.py --force-retrain
```

## Pipeline Phases

### Phase 1: Parallel Training

**What it does:**
- Trains 11 SAE configurations in parallel across 4 GPUs
- Automatically assigns configs to available GPUs
- Implements early stopping (patience=15)
- Saves checkpoints and metrics

**Duration:** ~2-3 hours (depending on convergence)

**GPU Assignment:**
- GPU 0: Config 1, then Config 5, then Config 9...
- GPU 1: Config 2, then Config 6, then Config 10...
- GPU 2: Config 3, then Config 7, then Config 11...
- GPU 3: Config 4, then Config 8...

**Outputs:**
```
checkpoints/
├── sae_latent128_k4.pt
├── sae_latent128_k8.pt
├── ... (11 checkpoints total)

outputs/
├── sae_metrics_latent128_k4.json
├── sae_metrics_latent128_k8.json
├── ... (11 metric files)
```

### Phase 2: Configuration Comparison

**What it does:**
- Loads all trained checkpoints
- Computes feature-motif correlations (with degree+centrality controls)
- Ranks configs by composite score
- Identifies best configuration

**Duration:** ~30 minutes

**Outputs:**
```
outputs/
├── sae_config_comparison.csv      # All configs ranked
├── latent_correlations.csv        # Feature-motif correlations
└── feature_motif_correlations_partial.csv
```

**Ranking Metrics:**
- **max_rpb_abs**: Maximum absolute correlation (default)
- **composite_score**: Weighted combination of correlation, F1, and capacity

### Phase 3: Interpretability Analysis

**What it does:**
- Identifies best configuration from Phase 2
- Provides instructions for running interpretability notebook
- Analyzes feature-motif correspondence with topology controls

**Manual Step Required:**
```bash
jupyter notebook sae_activations_motif_with_controls.ipynb
```

Then update the config cell with best parameters:
```python
LATENT_DIM = 256  # From Phase 2 results
K = 16            # From Phase 2 results
```

## Monitoring

### Real-Time GPU Monitor
```bash
./monitor_gpu_training.sh
```

Shows:
- GPU utilization (GPU%, Memory%)
- Temperature and power draw
- Running processes
- Training progress
- Recent log entries

### Manual Monitoring
```bash
# Check GPU usage
watch -n 1 nvidia-smi

# Check running processes
ps aux | grep sparse_autoencoder

# Check latest log
tail -f outputs/pipeline_logs/pipeline_*.log
```

### Check Progress
```bash
# Count completed checkpoints
ls checkpoints/sae_latent*.pt | wc -l

# View training metrics
cat outputs/sae_metrics_latent256_k16.json | jq '.test_reconstruction'

# Check comparison results
head -20 outputs/sae_config_comparison.csv
```

## Troubleshooting

### CUDA Out of Memory
**Problem:** Training fails with CUDA OOM error

**Solutions:**
```bash
# Reduce batch size (edit script, line 58)
BATCH_SIZE = 512  # Default: 1024

# Use fewer GPUs
python run_sae_pipeline_multi_gpu.py --gpus 0,1

# Train configs sequentially
python sparse_autoencoder.py  # Original single-GPU script
```

### Process Hangs
**Problem:** Training appears stuck

**Solutions:**
```bash
# Check if process is actually running
ps aux | grep sparse_autoencoder

# Check GPU utilization
nvidia-smi

# Kill stuck processes
pkill -9 -f sparse_autoencoder

# Restart with force-retrain
python run_sae_pipeline_multi_gpu.py --force-retrain
```

### Missing Prerequisites
**Problem:** Script errors about missing activations

**Solutions:**
```bash
# Check activations exist
ls outputs/activations/layer3_new/train/

# Re-run GNN training if needed
python gnn_train_copy.py --model_type GCN

# Verify centrality features
head -1 virtual_graphs/data/all_graphs/graph_motif_metadata/graph_0_metadata.csv
# Should show: betweenness_centrality, in_closeness_centrality, etc.
```

### Comparison Fails
**Problem:** Phase 2 comparison crashes

**Solutions:**
```bash
# Run comparison manually
python compare_sae_configs_with_centrality.py

# Check if all checkpoints exist
ls checkpoints/sae_latent*.pt

# Verify test graph IDs exist
ls outputs/test_graph_ids.json
```

## Performance Optimization

### Maximize GPU Utilization
```python
# Edit run_sae_pipeline_multi_gpu.py

# Increase batch size if memory allows
BATCH_SIZE = 2048  # Default: 1024

# Adjust num_workers for data loading
num_workers=4  # Default: 2
```

### Reduce Training Time
```python
# Reduce max epochs (may affect convergence)
NUM_EPOCHS = 100  # Default: 200

# Reduce early stopping patience
patience = 10  # Default: 15
```

### Memory Optimization
```python
# Use smaller configs first
SAE_CONFIGS = [
    (128, 4, "..."),
    (128, 8, "..."),
    # Remove 512-dim configs if OOM
]
```

## Output Files

### Checkpoints
```
checkpoints/
└── sae_latent{latent_dim}_k{k}.pt
    ├── model_state_dict: Model weights
    ├── optimizer_state_dict: Optimizer state
    └── history: Training curves
```

### Metrics
```json
{
  "best_val_loss": 1.5e-8,
  "test_reconstruction": 1.2e-8,
  "test_l0_sparsity": 0.0625,
  "config": {
    "latent_dim": 256,
    "k": 16,
    "sparsity_method": "topk"
  }
}
```

### Logs
```
outputs/pipeline_logs/
└── pipeline_20260125_143022.log
```

## Comparison with Original Pipeline

| Feature | Original | Multi-GPU Pipeline |
|---------|----------|-------------------|
| Training | Sequential | Parallel (4x GPUs) |
| Duration | ~8 hours | ~2-3 hours |
| Monitoring | Manual | Automated |
| Resume | Manual | Automatic |
| GPU Assignment | Manual | Automatic |
| Logging | Scattered | Centralized |

## Next Steps After Pipeline

### 1. Review Best Configuration
```bash
# View comparison results
less outputs/sae_config_comparison.csv

# Check best config metrics
cat outputs/sae_metrics_latent{BEST}_k{BEST}.json | jq '.'
```

### 2. Run Interpretability Analysis
```bash
jupyter notebook sae_activations_motif_with_controls.ipynb
```

### 3. Run Ablation Studies
```bash
# See SAE-CAUSAL-ABLATION-PIPELINE.md for full ablation workflow
python run_ablation.py --variant topk --latent_dim {BEST} --k {BEST}
```

### 4. Multi-Seed Validation
```bash
# Retrain best config with different seeds
python retrain_best_configs.py
```

## Support

For issues or questions:
1. Check the main documentation: `SAE-CAUSAL-ABLATION-PIPELINE.md`
2. Review logs in `outputs/pipeline_logs/`
3. Check GPU status with `nvidia-smi`

## Citation

If you use this pipeline, please cite:
- Original SAE architecture: [Your paper]
- Multi-GPU training framework: [This repository]
