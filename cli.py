"""Command-line interface for Diffusion Models."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Diffusion Models CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Batch size")

    # predict
    predict_parser = subparsers.add_parser("predict", help="Generate images")
    predict_parser.add_argument("--steps", type=int, default=40, help="Sampling steps")
    predict_parser.add_argument("--samples", type=int, default=64, help="Number of samples")

    args = parser.parse_args()

    if args.command == "train":
        from diffusion_models.modeling.train import Trainer
        trainer = Trainer()
        trainer.train()
    elif args.command == "predict":
        from diffusion_models.modeling.predict import Sampler
        sampler = Sampler()
        samples = sampler.sample(n_steps=args.steps, n_samples=args.samples)
        print(f"Generated {args.samples} samples with {args.steps} steps")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()