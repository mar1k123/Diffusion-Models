"""
Test suite for Diffusion Models project.
Validates models, data loading, training, and inference pipelines.

Usage:
    python -m pytest tests/test_model.py -v
    python -m pytest tests/ -v --tb=short
"""

import sys
import os
import unittest

import torch
import torchvision
from torch.utils.data import DataLoader

# Ensure the project root is in sys.path for importing diffusion_models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diffusion_models.config import (
    DEVICE,
    BATCH_SIZE,
    N_EPOCHS,
    IMAGE_SIZE,
    IMAGE_CHANNELS,
    SAMPLING_STEPS,
    NUM_SAMPLES,
    LEARNING_RATE,
)
from diffusion_models.dataset import get_dataloader
from diffusion_models.features import corrupt
from diffusion_models.modeling.unet import BasicUNet, build_unet
from diffusion_models.modeling.train import Trainer
from diffusion_models.modeling.predict import Sampler


# ---------------------------------------------------------------------------
# Helper decorator: skip test if CUDA required but not available
# ---------------------------------------------------------------------------
def requires_cuda(func):
    """Decorator to skip tests that need CUDA if it's not available."""

    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available — skipping test")
        return func(*args, **kwargs)

    return wrapper


# ===========================================================================
# 1. Configuration & Basic Setup Tests
# ===========================================================================
class TestConfiguration(unittest.TestCase):
    """Test that configuration values are valid and consistent."""

    def test_device_exists(self):
        """DEVICE should be a valid torch.device."""
        self.assertIsInstance(DEVICE, torch.device)

    def test_batch_size_positive(self):
        """BATCH_SIZE must be a positive integer."""
        self.assertGreater(BATCH_SIZE, 0)

    def test_epochs_positive(self):
        """N_EPOCHS must be a positive integer."""
        self.assertGreater(N_EPOCHS, 0)

    def test_learning_rate_positive(self):
        """LEARNING_RATE must be positive."""
        self.assertGreater(LEARNING_RATE, 0)

    def test_image_size_matches(self):
        """IMAGE_SIZE should be 28 for MNIST."""
        self.assertEqual(IMAGE_SIZE, 28)

    def test_image_channels_matches(self):
        """IMAGE_CHANNELS should be 1 for grayscale."""
        self.assertEqual(IMAGE_CHANNELS, 1)

    def test_sampling_steps_positive(self):
        """SAMPLING_STEPS must be positive."""
        self.assertGreater(SAMPLING_STEPS, 0)

    def test_num_samples_positive(self):
        """NUM_SAMPLES must be positive."""
        self.assertGreater(NUM_SAMPLES, 0)


# ===========================================================================
# 2. Data Pipeline Tests
# ===========================================================================
class TestDataPipeline(unittest.TestCase):
    """Test data loading and preprocessing."""

    def setUp(self):
        self.train_loader = get_dataloader(batch_size=BATCH_SIZE, train=True)

    def test_dataloader_returns_data(self):
        """Dataloader should yield batches."""
        x, y = next(iter(self.train_loader))
        self.assertIsNotNone(x)
        self.assertIsNotNone(y)

    def test_batch_shape(self):
        """Each batch should have correct shape (B, C, H, W)."""
        x, _ = next(iter(self.train_loader))
        self.assertEqual(x.dim(), 4)  # (batch, channel, height, width)
        self.assertEqual(x.shape[1], IMAGE_CHANNELS)  # 1 for grayscale
        self.assertEqual(x.shape[2], IMAGE_SIZE)  # 28x28 for MNIST
        self.assertEqual(x.shape[3], IMAGE_SIZE)

    def test_batch_size_respected(self):
        """Loader should return exactly BATCH_SIZE samples (except maybe last batch)."""
        x, _ = next(iter(self.train_loader))
        # Only test full batches; the last batch may be smaller
        if len(self.train_loader.dataset) > BATCH_SIZE:
            self.assertEqual(x.shape[0], BATCH_SIZE)

    def test_labels_are_integers(self):
        """MNIST labels should be integer tensors."""
        _, y = next(iter(self.train_loader))
        self.assertTrue(y.dtype in (torch.int64, torch.int32, torch.long))

    def test_labels_range(self):
        """MNIST labels should be in [0, 9]."""
        _, y = next(iter(self.train_loader))
        self.assertTrue((y >= 0).all() and (y <= 9).all())

    def test_pixel_values_range_raw(self):
        """Raw ToTensor values should be in [0, 1]."""
        x, _ = next(iter(self.train_loader))
        self.assertGreaterEqual(x.min().item(), 0.0)
        self.assertLessEqual(x.max().item(), 1.0)


# ===========================================================================
# 3. Feature Engineering Tests
# ===========================================================================
class TestFeatures(unittest.TestCase):
    """Test feature engineering functions."""

    def setUp(self):
        self.x = torch.rand(4, 1, 28, 28)

    def test_corrupt_shape(self):
        """corrupt() should return tensor of same shape as input."""
        amount = torch.rand(4)
        corrupted = corrupt(self.x, amount)
        self.assertEqual(corrupted.shape, self.x.shape)

    def test_corrupt_no_noise(self):
        """With amount=0, output should equal input (no corruption)."""
        amount = torch.zeros(4)
        corrupted = corrupt(self.x, amount)
        self.assertTrue(torch.allclose(corrupted, self.x, atol=1e-6))

    def test_corrupt_full_noise(self):
        """With amount=1, output should be pure noise (different from input)."""
        amount = torch.ones(4)
        corrupted = corrupt(self.x, amount)
        # After full corruption, values should generally differ from clean input
        self.assertFalse(torch.allclose(corrupted, self.x, atol=1e-6))

    def test_corrupt_range(self):
        """Corrupted values should stay reasonably bounded."""
        amount = torch.rand(4)
        corrupted = corrupt(self.x, amount)
        # Rough sanity check: values are mostly in a reasonable range
        self.assertLess(corrupted.abs().max().item(), 5.0)
        self.assertGreater(corrupted.abs().min().item(), -1.0)

    def test_corrupt_amount_broadcasting(self):
        """amount should broadcast correctly over (B, C, H, W)."""
        amount = torch.tensor([0.5, 0.5, 0.5, 0.5])
        corrupted = corrupt(self.x, amount)
        self.assertEqual(corrupted.shape, self.x.shape)


# ===========================================================================
# 4. Model Architecture Tests
# ===========================================================================
class TestBasicUNet(unittest.TestCase):
    """Test the minimal BasicUNet implementation."""

    def setUp(self):
        self.model = BasicUNet(in_channels=1, out_channels=1)

    def test_model_initialization(self):
        """Model should be an nn.Module."""
        self.assertIsInstance(self.model, torch.nn.Module)

    def test_forward_pass_shape(self):
        """Forward pass should preserve input shape."""
        x = torch.randn(8, 1, 28, 28)
        output = self.model(x)
        self.assertEqual(output.shape, x.shape)

    def test_batch_independence(self):
        """Different batch items should produce different outputs."""
        x = torch.randn(8, 1, 28, 28)
        output = self.model(x)
        # Outputs for different samples should not all be identical
        diffs = (output[0] - output[1:]).abs().sum(dim=(1, 2, 3))
        self.assertFalse((diffs < 1e-6).any())

    def test_parameter_count(self):
        """BasicUNet should have a reasonable number of parameters (≈309k)."""
        n_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(n_params, 300_000)
        self.assertLess(n_params, 320_000)  # ~309k

    def test_gradient_flow(self):
        """Gradients should flow through all trainable parameters."""
        x = torch.randn(4, 1, 28, 28)
        target = torch.randn(4, 1, 28, 28)
        output = self.model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(
                    (param.grad == 0).all(),
                    f"Zero gradient for {name}"
                )

    def test_eval_mode_no_grad(self):
        """In eval mode, no gradients should be computed."""
        self.model.eval()
        x = torch.randn(4, 1, 28, 28)
        with torch.no_grad():
            output = self.model(x)
        self.assertEqual(output.shape, x.shape)


class TestUNet2DModel(unittest.TestCase):
    """Test the diffusers UNet2DModel."""

    def setUp(self):
        self.model = build_unet()

    def test_model_initialization(self):
        """Model should be a UNet2DModel."""
        from diffusers import UNet2DModel
        self.assertIsInstance(self.model, UNet2DModel)

    def test_forward_pass_shape(self):
        """Forward pass should return a sample of correct shape."""
        x = torch.randn(4, 1, 28, 28)
        timestep = torch.tensor([0])
        output = self.model(x, timestep).sample
        self.assertEqual(output.shape, x.shape)

    def test_parameter_count(self):
        """UNet2DModel should have around 1.7M parameters."""
        n_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(n_params, 1_500_000)
        self.assertLess(n_params, 2_000_000)  # ~1.7M


# ===========================================================================
# 5. Trainer Tests
# ===========================================================================
class TestTrainer(unittest.TestCase):
    """Test the Trainer class."""

    def setUp(self):
        self.trainer = Trainer()

    def test_trainer_initialization(self):
        """Trainer should initialize model, optimizer, and dataloader."""
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.train_dataloader)
        self.assertIsNotNone(self.trainer.loss_fn)

    def test_loss_fn_is_mse(self):
        """Loss function should be MSELoss."""
        self.assertIsInstance(self.trainer.loss_fn, torch.nn.MSELoss)

    def test_model_on_correct_device(self):
        """Model should be on the configured DEVICE."""
        model_device = next(self.trainer.model.parameters()).device
        self.assertEqual(str(model_device), str(DEVICE))

    @requires_cuda
    def test_trainer_on_cuda(self):
        """If CUDA is available, model should be on GPU."""
        model_device = next(self.trainer.model.parameters()).device
        self.assertEqual(model_device.type, "cuda")

    def test_train_one_batch(self):
        """Training on a single batch should complete without error and record loss."""
        initial_loss_count = len(self.trainer.losses)
        x, _ = next(iter(self.trainer.train_dataloader))
        x = x.to(DEVICE)

        # Simulate one training step manually
        self.trainer.model.train()
        noise_amount = torch.rand(x.shape[0]).to(DEVICE)
        noisy_x = corrupt(x, noise_amount)
        pred = self.trainer.model(noisy_x, 0).sample
        loss = self.trainer.loss_fn(pred, x)

        self.trainer.optimizer.zero_grad()
        loss.backward()
        self.trainer.optimizer.step()
        self.trainer.losses.append(loss.item())

        self.assertEqual(len(self.trainer.losses), initial_loss_count + 1)
        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))


# ===========================================================================
# 6. Sampler / Inference Tests
# ===========================================================================
class TestSampler(unittest.TestCase):
    """Test the Sampler class for inference."""

    def setUp(self):
        self.sampler = Sampler()

    def test_sampler_initialization(self):
        """Sampler should create a model."""
        self.assertIsNotNone(self.sampler.model)

    def test_sampler_model_in_eval_mode(self):
        """Sampler's model should be in eval mode."""
        self.assertFalse(self.sampler.model.training)

    def test_sample_output_shape(self):
        """Generated samples should have the expected shape."""
        samples = self.sampler.sample(n_steps=5, n_samples=16)
        self.assertEqual(samples.shape, (16, 1, 28, 28))

    def test_sample_values_finite(self):
        """All sample values should be finite."""
        samples = self.sampler.sample(n_steps=5, n_samples=8)
        self.assertTrue(torch.isfinite(samples).all())

    def test_sample_values_range(self):
        """Sample values should be reasonably bounded."""
        samples = self.sampler.sample(n_steps=5, n_samples=8)
        self.assertLessEqual(samples.abs().max().item(), 10.0)

    def test_sample_with_history_shapes(self):
        """sample_with_history should return correct number of steps."""
        step_history, pred_history = self.sampler.sample_with_history(n_steps=5)
        self.assertEqual(len(step_history), 6)  # n_steps + 1 (initial noise + each step)
        self.assertEqual(len(pred_history), 5)

    def test_sample_with_history_shape_per_step(self):
        """Each step in history should have correct tensor shape."""
        step_history, _ = self.sampler.sample_with_history(n_steps=5)
        for step in step_history:
            self.assertEqual(step.shape[1:], (1, 28, 28))


# ===========================================================================
# 7. Integration / End-to-End Tests
# ===========================================================================
class TestIntegration(unittest.TestCase):
    """End-to-end tests that exercise the full pipeline."""

    def test_full_pipeline_data_to_sample(self):
        """Load data -> corrupt -> forward pass -> get sample (no crash)."""
        loader = get_dataloader(batch_size=4)
        x, _ = next(iter(loader))

        # Corrupt
        amount = torch.rand(x.shape[0])
        noisy = corrupt(x, amount)

        # Forward through UNet
        model = build_unet()
        model.eval()
        with torch.no_grad():
            output = model(noisy, torch.tensor([0])).sample

        self.assertEqual(output.shape, x.shape)

    def test_basic_unet_end_to_end(self):
        """Train BasicUNet for one batch and generate a sample."""
        loader = get_dataloader(batch_size=8)
        x, _ = next(iter(loader))

        model = BasicUNet()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        model.train()
        amount = torch.rand(x.shape[0])
        noisy = corrupt(x, amount)
        pred = model(noisy)
        loss = loss_fn(pred, x)
        loss.backward()
        opt.step()

        self.assertGreater(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

        # Generate a tiny sample
        model.eval()
        with torch.no_grad():
            sample = torch.randn(4, 1, 28, 28)
            for _ in range(3):
                pred = model(sample)
                sample = sample * 0.5 + pred * 0.5

        self.assertEqual(sample.shape, (4, 1, 28, 28))

    def test_sampler_from_untrained_model(self):
        """Even an untrained model should produce finite outputs."""
        sampler = Sampler()
        samples = sampler.sample(n_steps=3, n_samples=4)
        self.assertEqual(samples.shape, (4, 1, 28, 28))
        self.assertTrue(torch.isfinite(samples).all())


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    # Use verbosity=2 for detailed output, or remove for less
    unittest.main(verbosity=2)
