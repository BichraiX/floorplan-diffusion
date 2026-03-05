"""ViT-based denoiser for the floorplan diffusion model.

Implements the noise-prediction network described in Section 3.2.3 of
"Generating accessible multi-occupancy floor plans with fine-grained
control using a diffusion model".

Architecture: 28 transformer blocks, 16 attention heads, patch size 2,
embedding dimension 768.  Input is the channel-wise concatenation of
the noisy latent and the condition latent (8 channels, 64x64).
Output is the predicted noise (4 channels, 64x64).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by a two-layer MLP."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timesteps.
        Returns:
            (B, dim) timestep embeddings.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # If dim is odd, pad with a zero column
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head self-attention and FFN."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim)
        Returns:
            (B, N, dim)
        """
        # Self-attention with pre-norm and residual
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm2(x))
        return x


class ViTDenoiser(nn.Module):
    """Vision Transformer denoiser for latent diffusion.

    Takes the concatenation of the noisy latent and condition latent
    (8 channels, 64x64) and predicts the noise (4 channels, 64x64).
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 4,
        img_size: int = 64,
        patch_size: int = 2,
        embed_dim: int = 768,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.img_size = img_size
        num_patches = (img_size // patch_size) ** 2  # 1024

        # Patch embedding via convolution
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Timestep embedding
        self.time_embed = TimestepEmbedding(embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
             for _ in range(depth)]
        )

        # Final projection
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        self._init_weights()

    def _init_weights(self):
        # Initialise positional embedding with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Zero-init the final projection so the model starts near identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 8, 64, 64) concatenated noisy latent + condition latent.
            t: (B,) integer diffusion timesteps.
        Returns:
            (B, 4, 64, 64) predicted noise.
        """
        B = x.shape[0]
        p = self.patch_size
        h = w = self.img_size // p  # 32

        # 1. Patch embed: (B, 8, 64, 64) -> (B, embed_dim, 32, 32) -> (B, 1024, embed_dim)
        tokens = self.patch_embed(x)                     # (B, embed_dim, h, w)
        tokens = tokens.flatten(2).transpose(1, 2)       # (B, num_patches, embed_dim)

        # 2. Add positional embedding
        tokens = tokens + self.pos_embed

        # 3. Add timestep embedding (broadcast across all tokens)
        t_emb = self.time_embed(t)                       # (B, embed_dim)
        tokens = tokens + t_emb.unsqueeze(1)

        # 4. Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # 5. Final layer norm
        tokens = self.norm(tokens)

        # 6. Linear projection to patch pixels
        tokens = self.proj(tokens)                       # (B, h*w, p*p*out_channels)

        # 7. Unpatchify: reshape back to image
        # tokens: (B, h*w, p*p*C) -> (B, C, H, W)
        tokens = tokens.reshape(B, h, w, p, p, self.out_channels)
        # (B, h, w, p, p, C) -> (B, C, h, p, w, p)
        tokens = tokens.permute(0, 5, 1, 3, 2, 4)
        out = tokens.reshape(B, self.out_channels, h * p, w * p)

        return out
