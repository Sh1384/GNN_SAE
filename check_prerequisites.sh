#!/bin/bash
# Pre-flight checklist for SAE pipeline

echo "═══════════════════════════════════════════════════════════════"
echo "          SAE PIPELINE - PRE-FLIGHT CHECKLIST"
echo "═══════════════════════════════════════════════════════════════"
echo

# Check 1: Key Python scripts
echo "✓ Checking required scripts..."
for script in sparse_autoencoder.py compare_sae_configs.py retrain_best_configs.py run_ablation.py; do
    if [ -f "$script" ]; then
        echo "  ✓ $script"
    else
        echo "  ✗ MISSING: $script"
    fi
done
echo

# Check 2: Layer 3 activations
echo "✓ Checking activations..."
for split in train val test; do
    dir="outputs/activations/layer2_new/$split"
    if [ -d "$dir" ]; then
        count=$(ls $dir/graph_*.pt 2>/dev/null | wc -l)
        echo "  $split: $count files"

        if [ "$split" = "train" ] && [ "$count" -lt 1000 ]; then
            echo "    ⚠ WARNING: Expected ~3200 train files, got $count"
        elif [ "$split" = "val" ] && [ "$count" -lt 100 ]; then
            echo "    ⚠ WARNING: Expected ~400 val files, got $count"
        elif [ "$split" = "test" ] && [ "$count" -lt 100 ]; then
            echo "    ⚠ WARNING: Expected ~400 test files, got $count"
        fi
    else
        echo "  ✗ MISSING: $dir/"
        echo "    Run: python gnn_train.py first"
    fi
done
echo

# Check 3: GPU availability
echo "✓ Checking GPUs..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
    echo "  Found $gpu_count CUDA devices"
    nvidia-smi -L | head -4
else
    echo "  ✗ nvidia-smi not found"
fi
echo

# Check 4: Python environment
echo "✓ Checking Python environment..."
if python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null; then
    python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
    python3 -c "import torch; print(f'  CUDA devices: {torch.cuda.device_count()}')" 2>/dev/null
else
    echo "  ✗ PyTorch not installed in current environment"
    echo "    Activate your conda environment first"
fi
echo

# Check 5: Pipeline script
echo "✓ Checking pipeline script..."
if [ -f "run_sae_pipeline_multi_gpu.py" ]; then
    lines=$(wc -l < run_sae_pipeline_multi_gpu.py)
    echo "  ✓ run_sae_pipeline_multi_gpu.py ($lines lines)"
else
    echo "  ✗ run_sae_pipeline_multi_gpu.py not found!"
    exit 1
fi
echo

# Check 6: Output directories
echo "✓ Checking/creating output directories..."
for dir in checkpoints outputs outputs/pipeline_logs ablations; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  Created: $dir/"
    else
        echo "  ✓ $dir/"
    fi
done
echo

# Summary
echo "═══════════════════════════════════════════════════════════════"
echo "                      SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo
echo "If all checks passed, you're ready to run:"
echo
echo "  python3 run_sae_pipeline_multi_gpu.py --phase 1"
echo
echo "Or run the complete pipeline:"
echo
echo "  python3 run_sae_pipeline_multi_gpu.py --all"
echo
echo "═══════════════════════════════════════════════════════════════"
