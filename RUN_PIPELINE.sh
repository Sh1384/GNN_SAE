#!/bin/bash
# Quick launcher script for SAE pipeline

# Activate conda environment (adjust if needed)
# source ~/miniconda3/bin/activate your_env_name

# Run the pipeline
python3 run_sae_pipeline_multi_gpu.py "$@"
