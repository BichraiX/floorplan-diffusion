import torch
import torch.nn as nn
from diffusers import AutoencoderKL

from src.condition_encoder import ConditionEncoder
from src.vit_denoiser import ViTDenoiser


class FloorPlanDiffusionModel(nn.Module):
    """
    Full model combining:
    - Frozen SD 2.1 VAE (encoder + decoder) for latent space
    - Trainable ConditionEncoder for condition images
    - Trainable ViTDenoiser for noise prediction
    """

    def __init__(self, config):
        super().__init__()

        # Load and freeze VAE
        vae_model = config.get('vae_model', 'stabilityai/sd-vae-ft-mse')
        self.vae = AutoencoderKL.from_pretrained(vae_model)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # VAE scaling factor
        self.vae_scaling_factor = self.vae.config.scaling_factor

        # Trainable condition encoder
        cond_cfg = config.get('condition_encoder', {})
        self.condition_encoder = ConditionEncoder(
            in_channels=cond_cfg.get('in_channels', 3),
            out_channels=cond_cfg.get('out_channels', 4),
            base_channels=cond_cfg.get('base_channels', 64),
        )

        # Trainable ViT denoiser
        vit_cfg = config.get('vit', {})
        self.denoiser = ViTDenoiser(
            in_channels=vit_cfg.get('input_channels', 8),
            out_channels=vit_cfg.get('output_channels', 4),
            patch_size=vit_cfg.get('patch_size', 2),
            embed_dim=vit_cfg.get('embed_dim', 768),
            depth=vit_cfg.get('depth', 28),
            num_heads=vit_cfg.get('num_heads', 16),
            mlp_ratio=vit_cfg.get('mlp_ratio', 4.0),
        )

    @torch.no_grad()
    def encode_floorplan(self, x):
        """
        Encode 512x512 floor plan image to 64x64x4 latent using frozen VAE.

        Args:
            x: (B, 3, 512, 512) floor plan image, normalized to [-1, 1]
        Returns:
            (B, 4, 64, 64) latent
        """
        latent_dist = self.vae.encode(x).latent_dist
        latent = latent_dist.sample()
        latent = latent * self.vae_scaling_factor
        return latent

    @torch.no_grad()
    def decode_latent(self, z):
        """
        Decode 64x64x4 latent to 512x512 floor plan image using frozen VAE.

        Args:
            z: (B, 4, 64, 64) latent
        Returns:
            (B, 3, 512, 512) reconstructed image
        """
        z = z / self.vae_scaling_factor
        decoded = self.vae.decode(z).sample
        return decoded

    def encode_condition(self, cond_img):
        """
        Encode 512x512 condition image to 64x64x4 using trainable encoder.

        Args:
            cond_img: (B, 3, 512, 512) condition image, normalized to [-1, 1]
        Returns:
            (B, 4, 64, 64) condition latent
        """
        return self.condition_encoder(cond_img)

    def get_trainable_parameters(self):
        """Return only trainable parameters (condition encoder + ViT denoiser)."""
        params = list(self.condition_encoder.parameters()) + list(self.denoiser.parameters())
        return params
