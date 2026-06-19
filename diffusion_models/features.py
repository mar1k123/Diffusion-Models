"""Feature engineering and utility functions for Diffusion Models."""

import torch


def corrupt(x: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """
    Corrupt the input `x` by mixing it with noise according to `amount`.

    Args:
        x: Clean input tensor of shape (B, C, H, W).
        amount: Noise amount per sample, shape (B,).

    Returns:
        Corrupted tensor of same shape as x.
    """
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount
