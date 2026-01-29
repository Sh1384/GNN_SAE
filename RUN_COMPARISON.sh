#!/bin/bash
# Quick Start: SAE vs GNNExplainer Comparison Analysis

set -e  # Exit on error

echo "========================================================================"
echo "SAE vs GNNExplainer Comparison Analysis"
echo "========================================================================"
echo ""

cd /data/users/goodarzilab/shervin/182-GNN_SAE

# ============================================================================
# Step 1: Identify Top SAE Features for Each Motif
# ============================================================================
echo "STEP 1: Identifying top SAE features for each motif type..."
echo "  This analyzes correlations to find which features detect which motifs"
echo ""

python identify_top_sae_features.py

echo ""
echo "✓ Feature identification complete"
echo ""
echo "⚠ IMPORTANT: Check the output above and update compare_sae_vs_gnnexplainer.py"
echo "  with the identified feature indices in the 'targets' dictionary (around line 248)"
echo ""
read -p "Press Enter after updating the script (or Ctrl+C to exit)..."

# ============================================================================
# Step 2: Run Comparison Analysis
# ============================================================================
echo ""
echo "STEP 2: Running comparison analysis..."
echo "  This compares SAE Gradient Saliency vs GNNExplainer"
echo "  Duration: ~15-30 minutes"
echo ""

python compare_sae_vs_gnnexplainer.py

echo ""
echo "✓ Comparison analysis complete!"
echo ""
echo "========================================================================"
echo "RESULTS SUMMARY"
echo "========================================================================"
echo ""
echo "Quantitative results:"
echo "  - outputs/comparison_results/comparison_summary.csv"
echo "  - outputs/comparison_results/comparison_detailed.csv"
echo ""
echo "Visualizations:"
echo "  - outputs/comparison_plots/curves_*.png (ROC and PR curves)"
echo "  - outputs/comparison_plots/visualization_*_best.png"
echo "  - outputs/comparison_plots/visualization_*_median.png"
echo "  - outputs/comparison_plots/visualization_*_worst.png"
echo ""
echo "Next steps:"
echo "  1. Review quantitative metrics in comparison_summary.csv"
echo "  2. Check statistical significance tests in the console output"
echo "  3. Examine visualizations to assess qualitative differences"
echo "  4. Read COMPARISON_ANALYSIS_README.md for interpretation guide"
echo ""
echo "========================================================================"
