#!/bin/bash
set -e

echo "=== Hypervision VPS Pipeline ==="
echo "Worker: ${WORKER_INDEX}/${NUM_WORKERS}"
echo "City: ${CITY_NAME}"
echo "Instance: ${INSTANCE_ID}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "================================"

# Run the pipeline
python pipeline.py

# If pipeline exits with error, keep container alive for debugging
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Pipeline exited with code $EXIT_CODE"
    echo "[INFO] Container kept alive for debugging. SSH in to investigate."
    tail -f /dev/null
fi
