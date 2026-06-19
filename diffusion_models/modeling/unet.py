"""U-Net model definitions for Diffusion Models."""

import torch
from torch import nn
from diffusers import UNet2DModel

from diffusion_models.config import UNET_CONFIG


class BasicUNet(nn.Module):
    """A minimal UNet implementation for educational purposes."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2),
        ])
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = []
        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))
            if i < 2:
                h.append(x)
                x = self.downscale(x)

        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)
                x += h.pop()
            x = self.act(layer(x))

        return x


def build_unet() -> UNet2DModel:
    """
    Build a UNet2DModel from diffusers with project configuration.

    Returns:
        Configured UNet2DModel instance.
    """
    model = UNet2DModel(**UNET_CONFIG)
    return model