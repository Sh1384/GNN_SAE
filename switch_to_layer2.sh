#!/bin/bash
# Script to switch SAE pipeline from layer 3 (64-dim) to layer 2 (80-dim) activations

echo "Switching SAE pipeline from layer 3 (64-dim) to layer 2 (80-dim) activations..."
echo ""

# Backup original files
echo "Creating backups..."
mkdir -p backups
cp run_sae_pipeline_multi_gpu.py backups/
cp sae/sparse_autoencoder.py backups/
cp sae/analyze_feature_significance.py backups/
cp sae/native_gnn_ablation.py backups/
cp sae/analyze_sae_reconstruction_fidelity.py backups/
cp sae/retrain_best_configs.py backups/
cp sae/compare_sae_vs_gnnexplainer.py backups/
cp sae/run_ablation.py backups/
cp sae/generate_mixed_motif_activations.py backups/
cp sae/compare_sae_configs.py backups/

echo "Applying changes..."

# Main pipeline file
sed -i 's/Input: Layer 3 GNN activations (64-dim)/Input: Layer 2 GNN activations (80-dim)/g' run_sae_pipeline_multi_gpu.py
sed -i 's/INPUT_DIM = 64/INPUT_DIM = 80/g' run_sae_pipeline_multi_gpu.py
sed -i 's/layer3_new/layer2_new/g' run_sae_pipeline_multi_gpu.py

# Sparse autoencoder module
sed -i 's/GNN layer3 activations (64-dim/GNN layer2 activations (80-dim/g' sae/sparse_autoencoder.py
sed -i 's/input_dim: int = 64/input_dim: int = 80/g' sae/sparse_autoencoder.py
sed -i 's/INPUT_DIM = 64/INPUT_DIM = 80/g' sae/sparse_autoencoder.py
sed -i 's/layer3_new/layer2_new/g' sae/sparse_autoencoder.py
sed -i 's/Dataset for loading GNN layer3 activations/Dataset for loading GNN layer2 activations/g' sae/sparse_autoencoder.py

# Feature significance analysis
sed -i 's/INPUT_DIM = 64/INPUT_DIM = 80/g' sae/analyze_feature_significance.py
sed -i 's/layer3_new/layer2_new/g' sae/analyze_feature_significance.py

# Native GNN ablation
sed -i 's/INPUT_DIM = 64/INPUT_DIM = 80/g' sae/native_gnn_ablation.py
sed -i 's/layer3_new/layer2_new/g' sae/native_gnn_ablation.py
sed -i 's/layer3 activations/layer2 activations/g' sae/native_gnn_ablation.py

# SAE reconstruction fidelity
sed -i 's/input_dim=64/input_dim=80/g' sae/analyze_sae_reconstruction_fidelity.py
sed -i 's/layer3_new/layer2_new/g' sae/analyze_sae_reconstruction_fidelity.py
sed -i 's/Load native GNN layer3 activations/Load native GNN layer2 activations/g' sae/analyze_sae_reconstruction_fidelity.py

# Retrain best configs
sed -i 's/input_dim: int = 64/input_dim: int = 80/g' sae/retrain_best_configs.py
sed -i 's/Input dimension (default: 64 for GNN layer3)/Input dimension (default: 80 for GNN layer2)/g' sae/retrain_best_configs.py
sed -i 's/INPUT_DIM = 64/INPUT_DIM = 80/g' sae/retrain_best_configs.py
sed -i 's/layer3_new/layer2_new/g' sae/retrain_best_configs.py

# GNN explainer comparison
sed -i 's/input_dim = 64/input_dim = 80/g' sae/compare_sae_vs_gnnexplainer.py
sed -i 's/layer3_new/layer2_new/g' sae/compare_sae_vs_gnnexplainer.py

# Run ablation script
sed -i 's/input_dim=64/input_dim=80/g' sae/run_ablation.py
sed -i 's/INPUT_DIM = 64/INPUT_DIM = 80/g' sae/run_ablation.py
sed -i 's/layer3_new/layer2_new/g' sae/run_ablation.py

# Generate mixed motif activations
sed -i 's/layer3_activations/layer2_activations/g' sae/generate_mixed_motif_activations.py
sed -i 's/layer3_new/layer2_new/g' sae/generate_mixed_motif_activations.py
sed -i 's/Layer 3 activation/Layer 2 activation/g' sae/generate_mixed_motif_activations.py

# Compare SAE configs
sed -i 's/INPUT_DIM = 64/INPUT_DIM = 80/g' sae/compare_sae_configs.py
sed -i 's/layer3_new/layer2_new/g' sae/compare_sae_configs.py

echo ""
echo "✓ All changes applied successfully!"
echo ""
echo "Summary of changes:"
echo "  - Changed INPUT_DIM from 64 to 80"
echo "  - Changed activation path from layer3_new to layer2_new"
echo "  - Updated all model instantiations to use 80-dim input"
echo "  - Updated all documentation strings"
echo ""
echo "Backups saved to: backups/"
echo ""
echo "Next steps:"
echo "  1. Verify layer 2 activations exist:"
echo "     ls outputs/activations/layer2_new/train/ | wc -l"
echo ""
echo "  2. Run Phase 1 to train SAEs on layer 2 activations:"
echo "     python3 run_sae_pipeline_multi_gpu.py --phase 1"
echo ""
echo "Note: Old SAE checkpoints (trained on layer 3) will be incompatible."
echo "      Phase 1 will create new checkpoints with 80-dim input."
