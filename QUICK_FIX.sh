#!/bin/bash
# Quick fix workflow for zero SAE activations

set -e

cd /data/users/goodarzilab/shervin/182-GNN_SAE

echo "="================================================================
echo "QUICK FIX: Finding Active SAE Features"
echo "="================================================================
echo ""
echo "Current issue: SAE feature z112 has ZERO activation"
echo "Solution: Find which features are actually active"
echo ""

# Step 1: Check which features are active
echo "Step 1: Checking which SAE features are active..."
echo "------------------------------------------------------------"
python check_sae_features_quick.py

echo ""
echo "------------------------------------------------------------"
echo ""
read -p "Did you see features with non-zero activation above? (y/n): " saw_features

if [[ "$saw_features" != "y" ]]; then
    echo ""
    echo "ERROR: No active features found!"
    echo "This suggests a deeper issue with the SAE training."
    echo "You may need to retrain the SAE."
    exit 1
fi

# Step 2: Identify top features for each motif
echo ""
echo "Step 2: Identifying top features for each motif type..."
echo "------------------------------------------------------------"
python identify_top_sae_features.py

echo ""
echo "------------------------------------------------------------"
echo ""
echo "IMPORTANT: Copy the feature indices from the output above"
echo ""
echo "Then update compare_sae_vs_gnnexplainer.py around line 324:"
echo ""
echo "targets = {"
echo "    'feedforward_loop': {"
echo "        'feature_idx': XXX,  # UPDATE THIS"
echo "        ..."
echo "    },"
echo "    ..."
echo "}"
echo ""
read -p "Have you updated the feature indices? (y/n): " updated_indices

if [[ "$updated_indices" != "y" ]]; then
    echo ""
    echo "Please update the indices in compare_sae_vs_gnnexplainer.py"
    echo "Then run: python compare_sae_vs_gnnexplainer.py"
    exit 0
fi

# Step 3: Re-run comparison
echo ""
echo "Step 3: Re-running comparison with correct feature indices..."
echo "------------------------------------------------------------"
python compare_sae_vs_gnnexplainer.py

echo ""
echo "="================================================================
echo "COMPLETE"
echo "="================================================================
echo ""
echo "Check the output above for:"
echo "  ✓ Non-zero SAE activations"
echo "  ✓ Non-constant gradients"
echo "  ✓ Valid AUROC values (not NaN)"
echo ""
echo "If still seeing NaN, check DIAGNOSIS_AND_NEXT_STEPS.md"
