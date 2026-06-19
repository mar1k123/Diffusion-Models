"""Visualization functions for Diffusion Models."""

import torch
import torchvision
from matplotlib import pyplot as plt


def show_images(images: torch.Tensor, title: str = "", cmap: str = "Greys"):
    """
    Display a grid of images.

    Args:
        images: Tensor of shape (B, C, H, W).
        title: Plot title.
        cmap: Colormap for grayscale images.
    """
    grid = torchvision.utils.make_grid(images)[0].clip(0, 1)
    plt.imshow(grid, cmap=cmap)
    plt.title(title)
    plt.axis("off")


def plot_losses(losses: list, ylim: tuple = (0, 0.1)):
    """
    Plot training loss curve.

    Args:
        losses: List of loss values.
        ylim: Y-axis limits.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.ylim(*ylim)
    plt.title("Loss over time")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)


def plot_noising_process(x: torch.Tensor, noisy_x: torch.Tensor):
    """
    Show clean and noisy images side by side.

    Args:
        x: Clean images tensor.
        noisy_x: Noisy images tensor.
    """
    fig, axs = plt.subplots(3, 1, figsize=(16, 10))

    axs[0].imshow(torchvision.utils.make_grid(x[:8])[0].detach().cpu(), cmap="Greys")
    axs[0].set_title("Clean X")

    axs[1].imshow(
        torchvision.utils.make_grid(noisy_x[:8])[0].detach().cpu().clip(-1, 1),
        cmap="Greys",
    )
    axs[1].set_title("Noisy X (clipped to (-1, 1))")

    axs[2].imshow(
        torchvision.utils.make_grid(noisy_x[:8])[0].detach().cpu(), cmap="Greys"
    )
    axs[2].set_title("Noisy X")

    plt.tight_layout()
    plt.show()


def plot_scheduler_curves(scheduler):
    """
    Plot scheduler alpha curves.

    Args:
        scheduler: DDPMScheduler instance.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(
        scheduler.alphas_cumprod.cpu() ** 0.5,
        label=r"$\sqrt{\bar{\alpha}_t}$",
    )
    plt.plot(
        (1 - scheduler.alphas_cumprod.cpu()) ** 0.5,
        label=r"$\sqrt{1 - \bar{\alpha}_t}$",
    )
    plt.legend(fontsize="x-large")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.grid(True)


def plot_predictions(x: torch.Tensor, noised_x: torch.Tensor, preds: torch.Tensor):
    """
    Show input, corrupted, and predicted images.

    Args:
        x: Original clean images.
        noised_x: Corrupted images.
        preds: Model predictions.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 7))

    axs[0].set_title("Input data")
    axs[0].imshow(
        torchvision.utils.make_grid(x)[0].clip(0, 1), cmap="Greys"
    )

    axs[1].set_title("Corrupted data")
    axs[1].imshow(
        torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap="Greys"
    )

    axs[2].set_title("Network Predictions")
    axs[2].imshow(
        torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap="Greys"
    )

    plt.tight_layout()
    plt.show()


def plot_sampling_process(step_history: list, pred_output_history: list, n_steps: int):
    """
    Visualize the iterative sampling process.

    Args:
        step_history: List of tensors at each sampling step.
        pred_output_history: List of model outputs at each step.
        n_steps: Number of sampling steps.
    """
    fig, axs = plt.subplots(n_steps, 2, figsize=(9, 4), sharex=True)
    axs[0, 0].set_title("x (model input)")
    axs[0, 1].set_title("model prediction")
    for i in range(n_steps):
        axs[i, 0].imshow(
            torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap="Greys"
        )
        axs[i, 1].imshow(
            torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1),
            cmap="Greys",
        )
    plt.tight_layout()
    plt.show()


def show_generated_samples(images: torch.Tensor, nrow: int = 8):
    """
    Display generated samples in a grid.

    Args:
        images: Tensor of generated images.
        nrow: Number of images per row.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(
        torchvision.utils.make_grid(images.detach().cpu(), nrow=nrow)[0].clip(0, 1),
        cmap="Greys",
    )
    ax.set_title("Generated Samples")
    ax.axis("off")
    plt.show()