# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reproduction of "Generating accessible multi-occupancy floor plans with fine-grained control using a diffusion model" (Zhang & Zhang, Automation in Construction 2025). A constrained latent transformer-based diffusion model generating 512x512 multi-occupancy floor plans conditioned on design constraints.

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Preprocess MSD dataset (CSV polygons → image pairs)
python data/preprocess.py --csv_path data/msd_sample/mds_V2_5.372k.csv --output_dir data/msd_processed
# Quick test with few plans:
python data/preprocess.py --csv_path data/msd_sample/mds_V2_5.372k.csv --output_dir data/msd_processed --max_plans 10

# Train (debug mode, single GPU)
python train.py --config configs/train_config.yaml --debug

# Train (multi-GPU with DDP)
torchrun --nproc_per_node=4 --master_port=29500 train.py --config configs/train_config.yaml

# Resume training
torchrun --nproc_per_node=4 train.py --config configs/train_config.yaml --resume checkpoints/checkpoint_epoch_0050.pt

# Generate (single image)
python generate.py --checkpoint checkpoints/best_model.pt --condition_image input.png --output output.png

# Generate (directory, DDIM)
python generate.py --checkpoint checkpoints/best_model.pt --condition_dir data/msd_processed/test/conditions/ --output_dir outputs/ --steps 25

# Evaluate
python evaluate.py --generated_dir outputs/ --ground_truth_dir data/msd_processed/test/floor_plans/ --metrics fid miou
```

## Architecture

The model has three components, two trainable:

1. **Frozen VAE** (`src/model.py`): Stable Diffusion 2.1 VAE (`stabilityai/sd-vae-ft-mse`). Encodes 512x512 RGB → 64x64x4 latent, decodes back. Not trained.

2. **Condition Encoder** (`src/condition_encoder.py`): Trainable conv network. 3 strided conv layers (512→256→128→64) with GroupNorm + SiLU, then 1x1 conv to 4 channels. Maps 512x512x3 condition image → 64x64x4 latent.

3. **ViT Denoiser** (`src/vit_denoiser.py`): ~200M param Vision Transformer. Takes concatenated [noisy_latent, condition_latent] (8ch, 64x64), predicts noise (4ch, 64x64). 28 blocks, 16 heads, 768 embed dim, patch size 2. Timestep conditioning via sinusoidal embedding added to all tokens.

**Data flow**: condition_image → ConditionEncoder → cond_latent; floor_plan → VAE.encode → x0; diffusion adds noise to x0; ViT predicts noise from [noisy_x0 ∥ cond_latent, t]; loss = MSE(predicted_noise, actual_noise).

**Diffusion** (`src/diffusion.py`): DDPM with 1000 timesteps, linear β schedule. Supports both full DDPM and DDIM sampling. `GaussianDiffusion` is a plain class (not nn.Module), holds schedule tensors on CPU and moves them in `_extract`.

## Data Pipeline

- **Raw data**: MSD (Modified Swiss Dwellings) CSV with polygon geometries per room per floor plan
- **Preprocessing** (`data/preprocess.py`): Renders polygons to 512x512 images using OpenCV/Shapely. Each floor plan produces 2 condition variants (with/without structural elements). Training data augmented with 90°/180° rotations (6x multiplier). Room conditioning types (mask, bbox, circle, unconditioned) randomly assigned per room.
- **Dataset** (`src/dataset.py`): Loads paired PNGs from `{split}/conditions/` and `{split}/floor_plans/`. Images normalized to [-1, 1].
- **Room colors**: Defined as RGB dicts in both `data/preprocess.py` (`ROOM_COLORS`) and `evaluate.py` (`ROOM_COLORS`). 13 room types total (9 conditionable + 4 structural).

## Key Config

All hyperparameters in `configs/train_config.yaml`. Training uses FP16 mixed precision, gradient clipping (max_norm=1.0), AdamW optimizer. Only `condition_encoder` and `denoiser` parameters are trained (`model.get_trainable_parameters()`).

## Training Infrastructure

- DDP via `torchrun`, NCCL backend
- wandb logging (rank 0 only): loss every 10 steps, samples every `sample_every` epochs
- Checkpoint management: keeps last 3 + best model, saves every `save_every` epochs
- `--debug` flag: 1 GPU, 2 epochs, batch_size=1, fast iteration
