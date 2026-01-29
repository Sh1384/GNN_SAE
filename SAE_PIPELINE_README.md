# Multi-GPU SAE Pipeline - Quick Start Guide

## Overview

This script runs the complete SAE variants analysis pipeline from `sae/sae_colab_pipeline.ipynb`, optimized for your 4x NVIDIA TITAN V GPUs.

## What It Does

Trains and analyzes **30 different SAE configurations** across **4 variants**:
- **TopK SAE**: 11 configs (standard sparse autoencoder)
- **Gated SAE**: 9 configs (separates feature detection from magnitude)
- **JumpReLU SAE**: 6 configs (discontinuous activation with STE)
- **Switch SAE**: 4 configs (mixture of experts routing)

Then runs comprehensive analysis across 5 phases.

## Prerequisites

Ensure you have layer 3 activations from GNN training:
```bash
ls outputs/activations/layer3_new/train/*.pt | wc -l   # Should show ~3200
ls outputs/activations/layer3_new/val/*.pt | wc -l     # Should show ~400
ls outputs/activations/layer3_new/test/*.pt | wc -l    # Should show ~400
```

If not, run GNN training first:
```bash
python gnn_train.py
```

## Quick Start

### Run Complete Pipeline (All Phases)
```bash
python run_sae_pipeline_multi_gpu.py --all
```

This will run:
- **Phase 1**: Train all 30 SAE configs in parallel (~5 hours)
- **Phase 2**: Configuration comparison (~15 min)
- **Phase 2.5**: Cross-variant comparison (~3 min)
- **Phase 2b**: Multi-seed retraining (~2.7 hours)
- **Phase 3**: Ablation studies (~4 hours)
- **Phase 4**: Statistical validation (~30 min)
- **Phase 5**: Visualization (~15 min)

**Total time**: ~12-15 hours

### Run Individual Phases

```bash
# Phase 1: Train all 30 configs in parallel
python run_sae_pipeline_multi_gpu.py --phase 1

# Phase 2: Compare configs
python run_sae_pipeline_multi_gpu.py --phase 2

# Phase 3: Ablation studies
python run_sae_pipeline_multi_gpu.py --phase 3
```

### Custom GPU Selection

```bash
# Use only GPUs 0 and 1
python run_sae_pipeline_multi_gpu.py --phase 1 --gpus 0,1

# Use all 4 GPUs (default)
python run_sae_pipeline_multi_gpu.py --phase 1 --gpus 0,1,2,3
```

### Force Retrain

```bash
# Force retrain even if checkpoints exist
python run_sae_pipeline_multi_gpu.py --phase 1 --force-retrain
```

## Pipeline Phases Explained

### Phase 1: Parallel SAE Training (~5 hours)
- Trains 30 SAE configurations in parallel across 4 GPUs
- Automatically assigns configs to available GPUs
- Skips already trained configs (unless `--force-retrain`)
- Saves checkpoints to `checkpoints/sae_*.pt`
- Saves metrics to `outputs/sae_metrics_*.json`

**GPU Assignment Example:**
- GPU 0: TopK configs 1, 5, 9...
- GPU 1: TopK configs 2, 6, 10...
- GPU 2: TopK configs 3, 7, 11...
- GPU 3: TopK configs 4, 8...
- Then moves to Gated, JumpReLU, Switch variants

### Phase 2: Configuration Comparison (~15 min)
- Analyzes all 30 trained configs
- Computes feature-motif correlations
- Ranks configs by composite score
- Identifies best config per variant
- Output: `outputs/sae_config_comparison.csv`

### Phase 2.5: Cross-Variant Comparison (~3 min)
- Compares all 4 variants
- Analyzes reconstruction vs sparsity trade-offs
- Output: `outputs/sae_variant_comparison.csv`

### Phase 2b: Multi-Seed Retraining (~2.7 hours)
- Retrains best 4 configs with seeds [123, 456, 789, 1011]
- Required for reproducibility analysis
- Output: 16 additional checkpoints

### Phase 3: Ablation Studies (~4 hours)
- **3a**: SAE latent space ablations
- **3b**: Native GNN ablations
- **3c**: Ablation strategy comparison
- **3d**: Mixed-motif generalization
- Outputs: `ablations/results/*.csv`

### Phase 4: Statistical Validation (~30 min)
- Feature stability across seeds
- Permutation testing
- FDR correction
- Output: `outputs/statistical_analysis/*.png`

### Phase 5: Visualization (~15 min)
- Feature activation heatmaps
- Reconstruction fidelity analysis
- Output: `outputs/feature_activation_visualizations/*.png`

## Monitoring Progress

### Real-Time Logs
```bash
# Watch the latest log file
tail -f outputs/pipeline_logs/pipeline_*.log

# Or get the most recent
tail -f $(ls -t outputs/pipeline_logs/*.log | head -1)
```

### GPU Usage
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

### Check Completed Configs
```bash
# Count completed checkpoints
ls checkpoints/sae_*_seed42.pt | wc -l

# View metrics for a specific config
cat outputs/sae_metrics_topk_latent256_k16_seed42.json | python -m json.tool
```

## Output Files

After completion, you'll have:

```
checkpoints/
├── sae_topk_latent128_k4_seed42.pt
├── sae_topk_latent128_k8_seed42.pt
├── ... (30 files total from Phase 1)
├── sae_topk_latent256_k16_seed123.pt
├── ... (16 files from Phase 2b)

outputs/
├── sae_config_comparison.csv          # All configs ranked
├── sae_variant_comparison.csv         # Variant comparison
├── test_graph_ids.json                # Test set definition
├── sae_metrics_*.json                 # Training metrics (30 files)
├── pipeline_logs/
│   └── pipeline_20260126_*.log        # Execution logs
├── native_gnn_ablations/              # Phase 3b results
├── statistical_analysis/              # Phase 4 results
└── feature_activation_visualizations/ # Phase 5 results

ablations/
├── phase_3a_config.json               # Best config metadata
├── results/                           # Ablation results
└── interpretability_*/                # Detailed analysis
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Use fewer GPUs
python run_sae_pipeline_multi_gpu.py --phase 1 --gpus 0,1
```

### Process Hangs
```bash
# Check running processes
ps aux | grep run_sae_pipeline

# Kill if needed
pkill -9 -f run_sae_pipeline

# Restart with force-retrain
python run_sae_pipeline_multi_gpu.py --phase 1 --force-retrain
```

### Missing Activations
```bash
# Verify activations exist
ls outputs/activations/layer3_new/train/ | head

# If missing, run GNN training
python gnn_train_copy.py --model_type GCN
```

### Import Errors
The script automatically adds `sae/` to the Python path. If you get import errors:
```bash
# Verify sae folder exists
ls sae/sparse_autoencoder.py

# Check all required scripts are present
ls sae/*.py
```

## Next Steps After Pipeline Completes

1. **Review Results**:
   ```bash
   # View ranked configs
   cat outputs/sae_config_comparison.csv | head -20

   # Check best config per variant
   grep -E "^topk|^gated|^jumprelu|^switch" outputs/sae_config_comparison.csv | head -4
   ```

2. **Analyze Visualizations**:
   ```bash
   # View plots
   eog outputs/statistical_analysis/*.png
   eog outputs/feature_activation_visualizations/*.png
   ```

3. **Run Interpretability Analysis**:
   ```bash
   # Open the interpretability notebook
   jupyter notebook sae/sae_activations_motif_new.ipynb
   ```

## Performance Tips

### Maximize GPU Utilization
- Default batch size (1024) is optimized for TITAN V (12 GB)
- All 4 GPUs will be utilized in Phase 1
- Later phases run sequentially but are much faster

### Reduce Training Time
If you want faster results (not recommended for publication):
- Edit line 56 in the script: `NUM_EPOCHS = 100` (default: 200)
- Edit line 253: `patience = 10` (default: 15)

### Skip Phases
Run only what you need:
```bash
# Train only (skip analysis)
python run_sae_pipeline_multi_gpu.py --phase 1

# Analysis only (assumes Phase 1 done)
python run_sae_pipeline_multi_gpu.py --phase 2-5
```

## Differences from Colab Notebook

| Feature | Colab Notebook | This Script |
|---------|---------------|-------------|
| Training | Sequential | Parallel (4 GPUs) |
| Duration | ~8 hours | ~5 hours (Phase 1) |
| GPU Assignment | Manual | Automatic |
| Monitoring | Manual cells | Continuous logging |
| Resume | Manual | Automatic |
| Drive Mounting | Required | Not needed |

## Technical Details

- **Multiprocessing**: Uses `spawn` context for CUDA compatibility
- **Auto-Resume**: Skips already trained configs automatically
- **Error Handling**: Each config trains independently; failures don't stop the pipeline
- **Path Management**: Automatically imports from `sae/` folder
- **Logging**: Centralized logs in `outputs/pipeline_logs/`

## Support

For issues:
1. Check the log file: `outputs/pipeline_logs/pipeline_*.log`
2. Verify GPU status: `nvidia-smi`
3. Check activations exist: `ls outputs/activations/layer3_new/*/`
4. Review the full pipeline docs: `sae/SAE-CAUSAL-ABLATION-PIPELINE.md`

## Citation

Based on the SAE variants analysis pipeline. See `sae/SAE-CAUSAL-ABLATION-PIPELINE.md` for full methodology.
