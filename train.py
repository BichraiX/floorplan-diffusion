import os
import sys
import argparse
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from torchvision.utils import make_grid, save_image
import wandb
from tqdm import tqdm

from src.model import FloorPlanDiffusionModel
from src.diffusion import GaussianDiffusion
from src.dataset import FloorPlanDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train floorplan diffusion model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Single GPU debug mode with small subset')
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_distributed():
    """Initialize DDP. Returns (local_rank, world_size, is_distributed)."""
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True
    else:
        return 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, scaler, epoch, val_loss, config, is_best=False):
    """Save training checkpoint."""
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # Get the underlying model if wrapped in DDP
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
        'config': config,
    }

    path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch:04d}.pt')
    torch.save(checkpoint, path)

    if is_best:
        best_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
        torch.save(checkpoint, best_path)

    # Keep only last 3 checkpoints + best to save disk
    checkpoints = sorted([
        f for f in os.listdir(config['checkpoint_dir'])
        if f.startswith('checkpoint_epoch_')
    ])
    for old_ckpt in checkpoints[:-3]:
        os.remove(os.path.join(config['checkpoint_dir'], old_ckpt))


def load_checkpoint(path, model, optimizer=None, scaler=None):
    """Load checkpoint. Returns start_epoch."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint['epoch'] + 1


@torch.no_grad()
def generate_samples(model, diffusion, val_dataset, config, device, epoch):
    """Generate sample images and return as a grid for wandb logging."""
    model_raw = model.module if hasattr(model, 'module') else model
    model_raw.eval()

    num_samples = min(config.get('num_samples', 8), len(val_dataset))
    conditions = []
    ground_truths = []

    for i in range(num_samples):
        cond, gt = val_dataset[i]
        conditions.append(cond)
        ground_truths.append(gt)

    conditions = torch.stack(conditions).to(device)
    ground_truths = torch.stack(ground_truths).to(device)

    # Encode conditions
    cond_latent = model_raw.encode_condition(conditions)

    # Generate via DDIM (faster than full DDPM for monitoring)
    shape = (num_samples, 4, config.get('latent_size', 64), config.get('latent_size', 64))
    generated_latent = diffusion.ddim_sample(
        model_raw.denoiser, cond_latent, shape, device=device, num_inference_steps=50
    )

    # Decode
    generated_images = model_raw.decode_latent(generated_latent)
    generated_images = generated_images.clamp(-1, 1)

    # Denormalize to [0, 1] for visualization
    conditions_vis = (conditions + 1) / 2
    ground_truths_vis = (ground_truths + 1) / 2
    generated_vis = (generated_images + 1) / 2

    # Create comparison grid: condition | generated | ground truth
    rows = []
    for i in range(num_samples):
        rows.extend([conditions_vis[i], generated_vis[i], ground_truths_vis[i]])

    grid = make_grid(rows, nrow=3, padding=4, pad_value=1.0)

    # Save locally
    os.makedirs('outputs/samples', exist_ok=True)
    save_image(grid, f'outputs/samples/epoch_{epoch:04d}.png')

    model_raw.train()
    return grid


def train_one_epoch(model, diffusion, dataloader, optimizer, scaler, epoch, config, device, local_rank):
    """Train for one epoch. Returns average loss."""
    model.train()
    model_raw = model.module if hasattr(model, 'module') else model

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', disable=(local_rank != 0))

    for batch_idx, (conditions, floor_plans) in enumerate(pbar):
        conditions = conditions.to(device)
        floor_plans = floor_plans.to(device)

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.float16):
            # Encode floor plan to latent (frozen VAE, no grad)
            x0 = model_raw.encode_floorplan(floor_plans)

            # Encode condition (trainable)
            cond_latent = model_raw.encode_condition(conditions)

            # Sample random timesteps
            t = torch.randint(0, config['num_timesteps'], (x0.shape[0],), device=device)

            # Compute diffusion loss
            loss = diffusion.p_losses(model_raw.denoiser, x0, cond_latent, t)

        scaler.scale(loss).backward()

        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model_raw.get_trainable_parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if local_rank == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log every 10 steps
            global_step = epoch * len(dataloader) + batch_idx
            if batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/epoch': epoch,
                    'train/global_step': global_step,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'system/gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1e9,
                    'system/gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1e9,
                }, step=global_step)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, diffusion, dataloader, config, device, local_rank):
    """Compute validation loss. Returns average loss."""
    model.eval()
    model_raw = model.module if hasattr(model, 'module') else model

    total_loss = 0.0
    num_batches = 0

    for conditions, floor_plans in dataloader:
        conditions = conditions.to(device)
        floor_plans = floor_plans.to(device)

        with autocast('cuda', dtype=torch.float16):
            x0 = model_raw.encode_floorplan(floor_plans)
            cond_latent = model_raw.encode_condition(conditions)
            t = torch.randint(0, config['num_timesteps'], (x0.shape[0],), device=device)
            loss = diffusion.p_losses(model_raw.denoiser, x0, cond_latent, t)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    # Reduce across all ranks
    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()

    return avg_loss


def main():
    args = parse_args()
    config = load_config(args.config)

    # Override for debug mode
    if args.debug:
        config['epochs'] = 2
        config['batch_size'] = 1
        config['num_workers'] = 0
        config['save_every'] = 1
        config['sample_every'] = 1
        config['num_samples'] = 2

    local_rank, world_size, is_distributed = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # Seed for reproducibility
    torch.manual_seed(config['seed'] + local_rank)

    # wandb (rank 0 only)
    if local_rank == 0:
        wandb.init(
            project=config.get('wandb_project', 'floorplan-diffusion'),
            entity=config.get('wandb_entity'),
            config=config,
            name=f"train-{config['project_name']}",
            resume='allow' if args.resume else None,
        )

    # Model
    model = FloorPlanDiffusionModel(config).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Diffusion
    diffusion = GaussianDiffusion(
        num_timesteps=config['num_timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
    )

    # Optimizer (only trainable params)
    model_raw = model.module if hasattr(model, 'module') else model
    optimizer = torch.optim.AdamW(
        model_raw.get_trainable_parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # Mixed precision scaler
    scaler = GradScaler('cuda')

    # Datasets
    train_dataset = FloorPlanDataset(config['data_dir'], split='train')
    val_dataset = FloorPlanDataset(config['data_dir'], split='val')

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler)
        if local_rank == 0:
            print(f"Resumed from epoch {start_epoch}")

    # Training loop
    if local_rank == 0:
        total_params = sum(p.numel() for p in model_raw.parameters())
        trainable_params = sum(p.numel() for p in model_raw.get_trainable_parameters())
        print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")
        print(f"Training for {config['epochs']} epochs, batch_size={config['batch_size']}/GPU, "
              f"world_size={world_size}, effective_batch={config['batch_size'] * world_size}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(start_epoch, config['epochs']):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_one_epoch(
            model, diffusion, train_loader, optimizer, scaler, epoch, config, device, local_rank
        )

        # Validate
        val_loss = validate(model, diffusion, val_loader, config, device, local_rank)

        if local_rank == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            wandb.log({
                'epoch/train_loss': train_loss,
                'epoch/val_loss': val_loss,
                'epoch/epoch': epoch,
            })

            # Generate and log samples
            if epoch % config['sample_every'] == 0:
                grid = generate_samples(model, diffusion, val_dataset, config, device, epoch)
                wandb.log({
                    'samples': wandb.Image(grid.permute(1, 2, 0).cpu().numpy(),
                                          caption=f'Epoch {epoch}: condition | generated | ground truth'),
                })

            # Save checkpoint
            if epoch % config['save_every'] == 0:
                save_checkpoint(model, optimizer, scaler, epoch, val_loss, config)

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scaler, epoch, val_loss, config, is_best=True)
                print(f"  New best val_loss: {val_loss:.4f}")

    if local_rank == 0:
        wandb.finish()
        print("Training complete!")

    cleanup_distributed()


if __name__ == '__main__':
    main()
