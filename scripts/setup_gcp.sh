#!/bin/bash
set -euo pipefail

# GCP Instance Setup Script
# Run this on a fresh GCP VM with A100 GPUs
#
# Usage: bash scripts/setup_gcp.sh

echo "=== Setting up GCP instance for floorplan diffusion training ==="

# 1. Install system dependencies
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv unzip wget

# 2. Clone/copy the project
# (Assumes you've already copied the project to the instance)

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Setup wandb
echo "Please set your wandb API key:"
echo "  export WANDB_API_KEY=your_key_here"
echo "  wandb login"

# 6. Download and preprocess dataset
echo "=== Downloading MSD dataset ==="
echo "Option A: Kaggle CLI"
echo "  pip install kaggle"
echo "  export KAGGLE_USERNAME=your_username"
echo "  export KAGGLE_KEY=your_key"
echo "  python data/download_msd.py"
echo ""
echo "Option B: Manual download from https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings"
echo "  Place the zip in data/ and run: python data/download_msd.py --zip_path data/modified-swiss-dwellings.zip"
echo ""
echo "Then preprocess:"
echo "  python data/preprocess.py --data_dir data/msd_raw --output_dir data/msd_processed"

# 7. Verify GPU setup
echo ""
echo "=== GPU Status ==="
nvidia-smi

echo ""
echo "=== Setup complete ==="
echo "To start training:"
echo "  source venv/bin/activate"
echo "  export WANDB_API_KEY=your_key_here"
echo "  bash scripts/run_training.sh"
