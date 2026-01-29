# SAE Multi-GPU Pipeline - Quick Start

## Prerequisites Check
```bash
# 1. Check activations exist
ls outputs/activations/layer3_new/{train,val,test}/ | wc -l
# Should see: 3000+ (train), 500+ (val), 400+ (test)

# 2. Check GPU availability
nvidia-smi
# Should see 4 NVIDIA TITAN V GPUs
```

## Basic Usage

### 1. View configurations to train
```bash
python run_sae_pipeline_multi_gpu.py --configs-only
```

### 2. Run complete pipeline (recommended)
```bash
# Terminal 1: Start training
python run_sae_pipeline_multi_gpu.py

# Terminal 2: Monitor GPUs
./monitor_gpu_training.sh
```

### 3. After training completes
```bash
# Visualize training progress
python visualize_training_progress.py

# View comparison results
less outputs/sae_config_comparison.csv

# Run interpretability analysis
jupyter notebook sae_activations_motif_with_controls.ipynb
```

## Common Commands

### Training
```bash
# Train only (skip comparison)
python run_sae_pipeline_multi_gpu.py --phase train

# Use specific GPUs
python run_sae_pipeline_multi_gpu.py --gpus 0,1,2

# Force retrain all
python run_sae_pipeline_multi_gpu.py --force-retrain
```

### Monitoring
```bash
# GPU usage
./monitor_gpu_training.sh

# Training logs
tail -f outputs/pipeline_logs/pipeline_*.log

# Check progress
ls checkpoints/sae_*.pt | wc -l
```

### Analysis
```bash
# Compare configs
python run_sae_pipeline_multi_gpu.py --phase compare

# Visualize progress
python visualize_training_progress.py

# View plots
eog outputs/training_plots/*.png
```

## File Structure

```
outputs/
├── activations/layer3_new/    # Input activations (64-dim)
│   ├── train/                 # ~3000 graphs
│   ├── val/                   # ~500 graphs
│   └── test/                  # ~400 graphs
├── sae_metrics_*.json         # Training metrics (11 files)
├── sae_config_comparison.csv  # Config ranking
├── pipeline_logs/             # Execution logs
└── training_plots/            # Visualization plots

checkpoints/
└── sae_latent*_k*.pt          # Trained models (11 files)
```

## Expected Timeline

- **Phase 1 (Training)**: 2-3 hours
- **Phase 2 (Comparison)**: 30 minutes
- **Phase 3 (Interpretability)**: Manual, ~1 hour
- **Total**: ~3-4 hours

## Troubleshooting

### GPU Out of Memory
```bash
# Use fewer GPUs
python run_sae_pipeline_multi_gpu.py --gpus 0,1
```

### Process Stuck
```bash
# Kill and restart
pkill -9 -f sparse_autoencoder
python run_sae_pipeline_multi_gpu.py --force-retrain
```

### Missing Files
```bash
# Check activations
python gnn_train_copy.py --model_type GCN

# Check centrality features
python virtual_graphs/compute_node_features_with_centrality.py
```

## Output Interpretation

### Best Configuration
From `outputs/sae_config_comparison.csv`:
- **max_rpb_abs > 0.5**: Strong feature-motif correlation
- **composite_score > 0.6**: Excellent overall performance
- **dead_feature_rate < 0.3**: Good capacity utilization

### Visualization
From `outputs/training_plots/`:
- **Training curves**: Check convergence
- **Final metrics**: Compare reconstruction quality
- **Sparsity plot**: Verify target vs achieved sparsity

## Next Steps

1. ✓ View training results: `python visualize_training_progress.py`
2. ✓ Check best config: `less outputs/sae_config_comparison.csv`
3. ✓ Run interpretability: `jupyter notebook sae_activations_motif_with_controls.ipynb`
4. → Run ablations: See `SAE-CAUSAL-ABLATION-PIPELINE.md`

## Full Documentation

- Complete guide: `MULTI_GPU_PIPELINE_README.md`
- Original pipeline: `SAE-CAUSAL-ABLATION-PIPELINE.md`
- Code reference: `run_sae_pipeline_multi_gpu.py`
