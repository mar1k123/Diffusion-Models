"""Data loading and preparation for Diffusion Models."""

import torchvision
from torch.utils.data import DataLoader

from diffusion_models.config import DATA_ROOT, BATCH_SIZE, DATASET_NAME


def get_dataloader(batch_size: int = BATCH_SIZE, train: bool = True) -> DataLoader:
    """
    Load dataset and return a DataLoader.

    Args:
        batch_size: Batch size for training.
        train: Whether to load training or test set.

    Returns:
        DataLoader for the dataset.
    """
    if DATASET_NAME == "MNIST":
        dataset = torchvision.datasets.MNIST(
            root=DATA_ROOT,
            train=train,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif DATASET_NAME == "FashionMNIST":
        dataset = torchvision.datasets.FashionMNIST(
            root=DATA_ROOT,
            train=train,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise ValueError(f"Unsupported dataset: {DATASET_NAME}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return dataloader