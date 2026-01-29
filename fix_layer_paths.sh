#!/bin/bash
# Fix layer2 -> layer3_new path references in SAE folder

echo "Fixing activation paths from layer2 to layer3_new..."

# List of Python files to update
files=(
    "sae/sparse_autoencoder.py"
    "sae/run_ablation.py"
    "sae/analyze_feature_significance.py"
    "sae/analyze_sae_reconstruction_fidelity.py"
    "sae/compare_sae_variants.py"
    "sae/generate_mixed_motif_activations.py"
    "sae/native_gnn_ablation.py"
    "sae/retrain_best_configs.py"
)

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  Updating $file..."
        # Replace layer2 with layer3_new in activation paths
        sed -i 's|outputs/activations/layer2/|outputs/activations/layer3_new/|g' "$file"
        # Also update any references to "GNN layer2" in comments/docstrings
        sed -i 's/GNN layer2/GNN layer3/g' "$file"
        sed -i 's/layer2 activations/layer3 activations/g' "$file"
    else
        echo "  ⚠ File not found: $file"
    fi
done

echo "✓ Done! All activation paths updated."
echo ""
echo "Files updated:"
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        count=$(grep -c "layer3_new" "$file" 2>/dev/null || echo 0)
        echo "  $file: $count occurrences of layer3_new"
    fi
done
