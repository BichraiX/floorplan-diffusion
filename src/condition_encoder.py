import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    """
    Encodes a 512×512×3 condition image to a 64×64×4 latent representation.
    Uses strided convolutions to downsample 8× spatially.
    """
    def __init__(self, in_channels=3, out_channels=4, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            # 512 → 256
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            # 256 → 128
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            # 128 → 64
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            # Project to latent channels: 64 → 64 (spatial), 256 → out_channels
            nn.Conv2d(base_channels * 4, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 512, 512) condition image, normalized to [-1, 1]
        Returns:
            (B, 4, 64, 64) condition latent
        """
        return self.encoder(x)
