"""Inference and sampling for Diffusion Models."""

import torch
import torchvision
from matplotlib import pyplot as plt

from diffusion_models.config import (
    DEVICE,
    SAMPLING_STEPS,
    NUM_SAMPLES,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
)
from diffusion_models.modeling.unet import build_unet


class Sampler:
    """Handles image generation using a trained diffusion model."""

    def __init__(self, model_path: str = None):
        self.device = DEVICE
        self.model = build_unet().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def sample(self, n_steps: int = SAMPLING_STEPS, n_samples: int = NUM_SAMPLES) -> torch.Tensor:
        """
        Generate samples by iteratively denoising random noise.

        Args:
            n_steps: Number of sampling steps.
            n_samples: Number of images to generate.

        Returns:
            Tensor of generated images.
        """
        x = torch.rand(n_samples, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(self.device)

        with torch.no_grad():
            for i in range(n_steps):
                noise_amount = torch.ones((x.shape[0],)).to(self.device) * (1 - i / n_steps)
                pred = self.model(x, 0).sample
                mix_factor = 1 / (n_steps - i)
                x = x * (1 - mix_factor) + pred * mix_factor

        return x

    def sample_with_history(self, n_steps: int = 5) -> tuple:
        """
        Generate samples and return the full denoising history.

        Args:
            n_steps: Number of sampling steps.

        Returns:
            Tuple of (step_history, pred_output_history).
        """
        x = torch.rand(8, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(self.device)
        step_history = [x.detach().cpu()]
        pred_output_history = []

        with torch.no_grad():
            for i in range(n_steps):
                pred = self.model(x)
                pred_output_history.append(pred.detach().cpu())
                mix_factor = 1 / (n_steps - i)
                x = x * (1 - mix_factor) + pred * mix_factor
                step_history.append(x.detach().cpu())

        return step_history, pred_output_history


def show_sampling_process():
    """Visualize the iterative denoising process."""
    from diffusion_models.plots import plot_sampling_process

    sampler = Sampler()
    step_history, pred_output_history = sampler.sample_with_history(n_steps=5)
    plot_sampling_process(step_history, pred_output_history, n_steps=5)


def show_predictions():
    """Show model predictions on corrupted inputs."""
    from diffusion_models.dataset import get_dataloader
    from diffusion_models.features import corrupt
    from diffusion_models.plots import plot_predictions

    dataloader = get_dataloader(batch_size=8)
    x, _ = next(iter(dataloader))
    x = x[:8]

    amount = torch.linspace(0, 1, x.shape[0])
    noised_x = corrupt(x, amount)

    sampler = Sampler()
    with torch.no_grad():
        preds = sampler.model(noised_x.to(DEVICE)).detach().cpu()

    plot_predictions(x, noised_x, preds)


def main():
    """Generate and display samples."""
    sampler = Sampler()
    samples = sampler.sample()

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(
        torchvision.utils.make_grid(samples.detach().cpu(), nrow=8)[0].clip(0, 1),
        cmap="Greys",
    )
    ax.set_title("Generated Samples")
    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    main()