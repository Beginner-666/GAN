import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcgan_a import DCGANConfig, resolve_device
from dcgan_b import DCGANGenerator
from dcgan_b.artifacts import denormalize_images, load_checkpoint, save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate latent-space interpolation images with a trained DCGAN generator.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint containing generator weights.")
    parser.add_argument("--output", default="outputs/dcgan/interpolation.png")
    parser.add_argument("--steps", type=int, default=11)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--feature-maps-g", type=int, default=64)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = DCGANConfig(device=str(device))
    generator = DCGANGenerator(config, feature_maps_g=args.feature_maps_g).to(device)
    load_checkpoint(args.checkpoint, generator, device=device)
    generator.eval()

    z1 = torch.randn(1, config.noise_dim, 1, 1, device=device)
    z2 = torch.randn(1, config.noise_dim, 1, 1, device=device)
    alphas = torch.linspace(0.0, 1.0, steps=args.steps, device=device).view(-1, 1, 1, 1)
    z = (1.0 - alphas) * z1 + alphas * z2

    with torch.no_grad():
        images = generator(z)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(denormalize_images(images), output, nrow=args.steps)
    print(f"saved={output} steps={args.steps} checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
