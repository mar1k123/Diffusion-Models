"""Training script for Diffusion Models."""

import torch
from torch import nn
from diffusers import DDPMScheduler

from diffusion_models.config import (
    DEVICE,
    BATCH_SIZE,
    N_EPOCHS,
    LEARNING_RATE,
    SAMPLING_STEPS,
    NUM_SAMPLES,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
)
from diffusion_models.dataset import get_dataloader
from diffusion_models.features import corrupt
from diffusion_models.plots import plot_losses, show_generated_samples
from diffusion_models.modeling.unet import build_unet


class Trainer:
    """Handles model training for diffusion models."""

    def __init__(self):
        self.device = DEVICE
        self.train_dataloader = get_dataloader(batch_size=BATCH_SIZE, train=True)
        self.model = build_unet().to(self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE
        )
        self.losses = []

    def train_epoch(self, epoch: int):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number.
        """
        for x, _ in self.train_dataloader:
            x = x.to(self.device)
            noise_amount = torch.rand(x.shape[0]).to(self.device)
            noisy_x = corrupt(x, noise_amount)

            pred = self.model(noisy_x, 0).sample
            loss = self.loss_fn(pred, x)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.losses.append(loss.item())

        avg_loss = sum(self.losses[-len(self.train_dataloader):]) / len(self.train_dataloader)
        print(f"Finished epoch {epoch}. Average loss: {avg_loss:.5f}")

    def train(self):
        """Run full training loop."""
        print(f"Training on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(N_EPOCHS):
            self.train_epoch(epoch)

    def generate_samples(self) -> torch.Tensor:
        """
        Generate samples from the trained model.

        Returns:
            Tensor of generated images.
        """
        self.model.eval()
        x = torch.rand(NUM_SAMPLES, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(self.device)

        with torch.no_grad():
            for i in range(SAMPLING_STEPS):
                noise_amount = torch.ones((x.shape[0],)).to(self.device) * (1 - i / SAMPLING_STEPS)
                pred = self.model(x, 0).sample
                mix_factor = 1 / (SAMPLING_STEPS - i)
                x = x * (1 - mix_factor) + pred * mix_factor

        self.model.train()
        return x


def main():
    """Main entry point for training."""
    trainer = Trainer()
    trainer.train()

    # Plot losses
    plot_losses(trainer.losses)

    # Generate and show samples
    samples = trainer.generate_samples()
    show_generated_samples(samples)


if __name__ == "__main__":
    main()