#!/bin/bash
set -euo pipefail

# Floorplan Diffusion — Training Launch Script for GCP (4× A100 80GB)
#
# Usage:
#   # With Docker:
#   docker run --gpus all -e WANDB_API_KEY=$WANDB_API_KEY -v /data:/app/data floorplan-diffusion
#
#   # Without Docker:
#   export WANDB_API_KEY=your_key_here
#   bash scripts/run_training.sh
#
#   # Resume from checkpoint:
#   RESUME=checkpoints/checkpoint_epoch_0050.pt bash scripts/run_training.sh

echo "=========================================="
echo "Floorplan Diffusion — Training"
echo "=========================================="

# Check GPUs
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} GPUs"

# wandb setup
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY not set. wandb will run in offline mode."
    echo "Set it with: export WANDB_API_KEY=your_key"
fi

# Config
CONFIG=${CONFIG:-configs/train_config.yaml}
MASTER_PORT=${MASTER_PORT:-29500}

# Build launch command
CMD="torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} train.py --config ${CONFIG}"

# Resume support
if [ -n "${RESUME:-}" ]; then
    echo "Resuming from: ${RESUME}"
    CMD="${CMD} --resume ${RESUME}"
fi

echo "Launch command: ${CMD}"
echo "=========================================="

exec ${CMD}
