#!/bin/bash
# Complete GNN-SAE Analysis Pipeline with Degree Confound Controls
# Run these commands in sequence

set -e  # Exit on error

echo "=================================="
echo "GNN-SAE Analysis Pipeline"
echo "With Degree Confound Controls"
echo "=================================="
echo ""

# Change to project directory
cd /data/users/goodarzilab/shervin/182-GNN_SAE

# ============================================================================
# STEP 0: Compute Node Degree Features (ONE-TIME SETUP)
# ============================================================================
echo "STEP 0: Computing node degree features for all graphs..."
echo "  (Only needs to be run once, or when graphs are regenerated)"
echo ""

python virtual_graphs/compute_node_features.py

echo ""
echo "✓ Degree features computed and added to metadata CSVs"
echo ""
read -p "Press Enter to continue to Step 1 (or Ctrl+C to exit)..."

# ============================================================================
# STEP 1: (Optional) Hyperparameter Tuning
# ============================================================================
echo ""
echo "STEP 1 (OPTIONAL): Hyperparameter tuning for GNN..."
echo "  Skip this if you already have good hyperparameters"
echo ""
read -p "Run hyperparameter sweep? (y/N): " run_sweep

if [[ "$run_sweep" == "y" || "$run_sweep" == "Y" ]]; then
    python hyperparameter_sweep_multi_gpu.py
    echo ""
    echo "✓ Hyperparameter sweep complete"
else
    echo "  Skipping hyperparameter sweep"
fi

echo ""
read -p "Press Enter to continue to Step 2 (or Ctrl+C to exit)..."

# ============================================================================
# STEP 2: Train GNN and Extract Activations
# ============================================================================
echo ""
echo "STEP 2: Training GNN and extracting layer 1, 2, 3 activations..."
echo "  This will create activations for ~4000 single-motif graphs"
echo "  Duration: ~10-30 minutes"
echo ""

python gnn_train_copy.py

echo ""
echo "✓ GNN trained and activations extracted"
echo ""
echo "Verification:"
echo "  Layer 1 train files: $(ls outputs/activations/layer1_new/train/*.pt 2>/dev/null | wc -l)"
echo "  Layer 2 train files: $(ls outputs/activations/layer2_new/train/*.pt 2>/dev/null | wc -l)"
echo "  Layer 3 train files: $(ls outputs/activations/layer3_new/train/*.pt 2>/dev/null | wc -l)"
echo ""
read -p "Press Enter to continue to Step 3 (or Ctrl+C to exit)..."

# ============================================================================
# STEP 3: Train Sparse Autoencoders
# ============================================================================
echo ""
echo "STEP 3: Training SAE on layer 3 activations..."
echo "  Testing 11 configurations (latent_dim × k combinations)"
echo "  Duration: ~5-20 minutes per config"
echo ""

python sparse_autoencoder.py

echo ""
echo "✓ SAE training complete"
echo ""
echo "Verification:"
echo "  SAE checkpoints: $(ls checkpoints/sae_*.pt 2>/dev/null | wc -l)"
echo ""
read -p "Press Enter to continue to Step 4 (or Ctrl+C to exit)..."

# ============================================================================
# STEP 4: Compare SAE Configurations (WITH DEGREE CONTROLS)
# ============================================================================
echo ""
echo "STEP 4: Comparing SAE configurations with degree controls..."
echo "  Stage 1: Fast screening of all configs"
echo "  Stage 2: Full permutation tests on top 3"
echo "  Includes: Sanity check for degree-motif confounds"
echo "  Duration: ~10-30 minutes"
echo ""

python compare_sae_configs.py

echo ""
echo "✓ SAE configuration comparison complete"
echo ""
echo "Results saved to: outputs/sae_config_comparison.csv"
echo ""
echo "Quick view of top configurations:"
echo "=================================="
head -5 outputs/sae_config_comparison.csv | column -t -s,
echo ""

echo ""
echo "=================================="
echo "PIPELINE COMPLETE!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Review outputs/sae_config_comparison.csv"
echo "  2. Check the recommended SAE configuration"
echo "  3. Open analysis_notebook.ipynb for detailed analysis"
echo "  4. Compare bivariate vs partial correlations"
echo ""
echo "Key files:"
echo "  - outputs/sae_config_comparison.csv      (comparison results)"
echo "  - checkpoints/sae_latent*_k*.pt          (trained SAE models)"
echo "  - outputs/activations/layer3_new/        (GNN activations)"
echo "  - virtual_graphs/data/.../metadata/      (with degree features)"
echo ""
echo "Documentation:"
echo "  - RUN_WORKFLOW.md                        (detailed workflow guide)"
echo "  - ~/.claude/plans/groovy-napping-engelbart.md  (deep dive analysis)"
echo ""
