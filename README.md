# Diffusion Models

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/🤗_Diffusers-0.21%2B-FF9D00)](https://huggingface.co/docs/diffusers)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v1.0.0-brightgreen)](https://github.com/mar1k123/Diffusion-Models/releases)
[![Tests](https://img.shields.io/badge/Tests-27%2F27%20passed-success)](https://github.com/mar1k123/Diffusion-Models)

A from-scratch implementation of diffusion models for image generation. Built with PyTorch and `diffusers`.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI (Recommended)](#cli-recommended)
  - [Training](#training)
  - [Inference](#inference)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Docker](#docker)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results](#results)
- [Libraries Used](#libraries-used)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Diffusion models are a class of generative models that learn to reverse a gradual noising process. Starting from pure noise, they iteratively denoise towards a clean image.

This project implements:

| Component | File | Description |
|-----------|------|-------------|
| **U-Net** | `modeling/unet.py` | Neural network that predicts noise at each step |
| **DDPM Scheduler** | Used via `diffusers` | Controls how noise is added and removed |
| **Trainer** | `modeling/train.py` | Handles the training loop and loss computation |
| **Sampler** | `modeling/predict.py` | Generates new images from random noise |

---

## Features

- ✅ Minimal U-Net (`BasicUNet`) for learning the fundamentals
- ✅ Advanced `UNet2DModel` with attention blocks
- ✅ DDPM noise scheduler with configurable timesteps
- ✅ Modular, OOP-based code structure
- ✅ Visualization tools for every stage (noising, denoising, sampling)
- ✅ Jupyter notebook support
- ✅ CLI interface with argparse
- ✅ Docker support
- ✅ 27 unit tests
- ✅ Easy configuration via `config.py`

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mar1k123/Diffusion-Models.git
cd Diffusion-Models
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python -c "import torch; import diffusers; print('Setup OK')"
```

---

## Usage

### CLI (Recommended)

The simplest way to train and generate is via the command-line interface:

```bash
# Train the model
python cli.py train

# Train with custom parameters
python cli.py train --epochs 5 --batch-size 64

# Generate images
python cli.py predict

# Generate with custom parameters
python cli.py predict --steps 20 --samples 16
```

Run `python cli.py --help` to see all available commands.

### Training

Train the model on MNIST (default) or FashionMNIST:

```bash
python -m diffusion_models.modeling.train
```

### Inference

Generate new images from a trained model:

```bash
python -m diffusion_models.modeling.predict
```

### Jupyter Notebooks

Launch Jupyter Lab to explore the notebooks:

```bash
jupyter lab
```

### Docker

```bash
# Build the image
docker build -t diffusion-models .

# Train with GPU
docker run --gpus all diffusion-models

# Interactive shell
docker run -it diffusion-models bash
```

---

## Project Structure

```
├── LICENSE                  <- Open-source license
├── Makefile                 <- Convenience commands (make data, make train)
├── README.md                <- This file
├── cli.py                   <- Command-line interface
├── Dockerfile               <- Docker image definition
│
├── data/
│   ├── external/            <- Third-party data
│   ├── interim/             <- Intermediate transformed data
│   ├── processed/           <- Final datasets for modeling
│   └── raw/                 <- Original, immutable data dump
│
├── docs/                    <- MkDocs documentation
│
├── models/
│   └── checkpoints/         <- Saved model weights (.pt, .pth)
│
├── notebooks/               <- Jupyter notebooks
│
├── references/              <- Papers, manuals, explanatory materials
│
├── reports/
│   └── figures/             <- Generated plots and figures
│
├── tests/                   <- Unit tests
│
├── .gitignore
├── pyproject.toml           <- Project metadata & tool config
├── requirements.txt         <- Pinned dependencies
├── setup.cfg                <- Flake8 configuration
│
└── diffusion_models/        <- Main source code package
    │
    ├── __init__.py          <- Makes this a Python module
    ├── config.py            <- Global settings and hyperparameters
    ├── dataset.py           <- Data loading and preprocessing
    ├── features.py          <- Utility functions (e.g., noise corruption)
    ├── plots.py             <- All visualization code
    │
    └── modeling/
        ├── __init__.py
        ├── unet.py          <- U-Net model definitions
        ├── train.py         <- Training loop and Trainer class
        └── predict.py       <- Inference and Sampler class
```

---

## Configuration

All settings are centralized in `diffusion_models/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEVICE` | `cuda` / `cpu` | Auto-detected |
| `DATASET_NAME` | `"MNIST"` | Dataset (`MNIST` or `FashionMNIST`) |
| `IMAGE_SIZE` | `28` | Input image resolution |
| `BATCH_SIZE` | `128` | Training batch size |
| `N_EPOCHS` | `3` | Number of training epochs |
| `LEARNING_RATE` | `1e-3` | Adam learning rate |
| `SAMPLING_STEPS` | `40` | Denoising steps during generation |
| `NUM_SAMPLES` | `64` | How many images to generate |

---

## Results

After training for 3 epochs on MNIST:

| Stage | Description |
|-------|-------------|
| **Loss curve** | Shows convergence over training steps |
| **Noising process** | Visualizes how clean images become noise |
| **Denoising process** | Step-by-step reconstruction from noise |
| **Generated samples** | 8×8 grid of model-generated digits |

Plots are saved in `reports/figures/` and displayed during training.

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0.0 | Deep learning framework |
| `torchvision` | ≥ 0.15.0 | Datasets and image utilities |
| `diffusers` | ≥ 0.21.0 | Diffusion models components |
| `matplotlib` | ≥ 3.7.0 | Plotting and visualization |
| `jupyterlab` | ≥ 4.0.0 | Interactive notebooks |
| `notebook` | ≥ 7.0.0 | Notebook support |
| `ipython` | ≥ 8.0.0 | Interactive Python shell |

### Dev Dependencies

| Library | Purpose |
|---------|---------|
| `pytest` | Unit testing |
| `ruff` | Linting and formatting |

---

## Roadmap

- [ ] Add DDIM scheduler for faster sampling
- [ ] Support RGB datasets (CIFAR-10, Butterfly)
- [ ] Add mixed precision training (AMP)
- [ ] Add experiment tracking (MLflow / TensorBoard)
- [ ] Add Gradio web demo
- [ ] Publish to PyPI

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Original DDPM paper
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM paper
- [PyTorch](https://pytorch.org/) — Deep learning framework