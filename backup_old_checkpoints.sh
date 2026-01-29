#!/bin/bash
# Backup old SAE checkpoints (trained on 64-dim layer 3)
# before retraining on 80-dim layer 2

echo "Backing up old SAE checkpoints (64-dim layer 3)..."

# Create backup directory with timestamp
BACKUP_DIR="checkpoints_layer3_64dim_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Count SAE checkpoints
NUM_SAE=$(ls checkpoints/sae_*.pt 2>/dev/null | wc -l)

if [ "$NUM_SAE" -eq 0 ]; then
    echo "No SAE checkpoints found to backup."
    echo "Checkpoints directory may already be clean."
    exit 0
fi

echo "Found $NUM_SAE SAE checkpoint files"
echo "Moving to: $BACKUP_DIR/"

# Move all SAE checkpoints (keep GNN checkpoint!)
mv checkpoints/sae_*.pt "$BACKUP_DIR/" 2>/dev/null

# Verify move
NUM_REMAINING=$(ls checkpoints/sae_*.pt 2>/dev/null | wc -l)
NUM_BACKED_UP=$(ls "$BACKUP_DIR"/sae_*.pt 2>/dev/null | wc -l)

echo ""
echo "✓ Backup complete!"
echo "  - Moved: $NUM_BACKED_UP SAE checkpoint files"
echo "  - Remaining in checkpoints/: $NUM_REMAINING SAE files"
echo "  - Backup location: $BACKUP_DIR/"
echo ""
echo "GNN checkpoint (checkpoints/gnn_model.pt) was NOT moved."
echo ""
echo "Next step: Run Phase 1 to train new 80-dim SAEs"
echo "  python3 run_sae_pipeline_multi_gpu.py --phase 1"
