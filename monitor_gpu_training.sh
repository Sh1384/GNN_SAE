#!/bin/bash
# GPU Training Monitor
# Monitors GPU usage, temperature, and training progress in real-time

echo "==================================================================="
echo "GPU Training Monitor - Press Ctrl+C to exit"
echo "==================================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

while true; do
    clear
    echo "==================================================================="
    echo "GPU Status - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "==================================================================="
    echo ""

    # Show GPU utilization
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | while IFS=, read -r idx name gpu_util mem_util mem_used mem_total temp power; do
        # Color coding based on utilization
        if [ "$gpu_util" -gt 80 ]; then
            color=$GREEN
        elif [ "$gpu_util" -gt 50 ]; then
            color=$YELLOW
        else
            color=$RED
        fi

        printf "${color}GPU %s: %s${NC}\n" "$idx" "$name"
        printf "  GPU Util: %3d%% | Mem: %5d/%5d MB (%2d%%) | Temp: %2d°C | Power: %5.1f W\n" \
            "$gpu_util" "$mem_used" "$mem_total" "$mem_util" "$temp" "$power"
        echo ""
    done

    echo "-------------------------------------------------------------------"
    echo "Training Progress"
    echo "-------------------------------------------------------------------"

    # Check for running SAE training processes
    if pgrep -f "sparse_autoencoder" > /dev/null; then
        echo -e "${GREEN}✓ SAE training processes running${NC}"
        ps aux | grep -E "sparse_autoencoder|run_sae_pipeline" | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "MEM:", $4"%"}'
    else
        echo -e "${YELLOW}⚠ No SAE training processes detected${NC}"
    fi

    echo ""

    # Check for checkpoints
    checkpoint_count=$(ls checkpoints/sae_latent*.pt 2>/dev/null | wc -l)
    echo "Checkpoints saved: $checkpoint_count"

    # Show latest log entries if log exists
    latest_log=$(ls -t outputs/pipeline_logs/pipeline_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo ""
        echo "Recent log entries (last 5):"
        tail -5 "$latest_log" | sed 's/^/  /'
    fi

    echo ""
    echo "==================================================================="
    echo "Press Ctrl+C to exit | Refreshing every 5 seconds..."
    echo "==================================================================="

    sleep 5
done
