#!/bin/bash
# Check activation dimensions in saved files

cd /data/users/goodarzilab/shervin/182-GNN_SAE

echo "Checking activation dimensions..."
echo ""

for layer in layer1_new layer2_new layer3_new; do
    test_file=$(find outputs/activations/$layer/test -name "*.pt" -type f | head -1)
    if [ -n "$test_file" ]; then
        echo "Layer: $layer"
        echo "Sample file: $test_file"
        /usr/bin/python3 << EOF
import torch
import sys
try:
    data = torch.load("$test_file", weights_only=True)
    print(f"  Shape: {data.shape}")
    print(f"  Dimension: {data.shape[1]}")
except Exception as e:
    print(f"  Error: {e}")
EOF
        echo ""
    fi
done

echo "Checking GNN checkpoint architecture..."
/usr/bin/python3 << 'PYEOF'
import torch
try:
    ckpt = torch.load("checkpoints/gnn_model.pt", weights_only=True, map_location='cpu')
    print("GNN layer dimensions:")
    print(f"  Conv1 out: {ckpt['conv1.bias'].shape[0]}")
    print(f"  Conv2 out: {ckpt['conv2.bias'].shape[0]}")
    print(f"  Conv3 out: {ckpt['conv3.bias'].shape[0]}")
    print(f"  Conv4 out: {ckpt['conv4.bias'].shape[0]}")
except Exception as e:
    print(f"Error: {e}")
PYEOF
