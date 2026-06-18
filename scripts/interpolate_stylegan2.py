from __future__ import annotations

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
from dcgan_b.artifacts import denormalize_images, load_checkpoint, save_image_grid
from dcgan_b.stylegan2 import StyleGAN2Generator64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate StyleGAN2 latent interpolation images.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/stylegan2/interpolation_w.png")
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--space", default="w", choices=["z", "w", "w+"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--endpoint-scale", type=float, default=1.0, help="Use values below 1.0 to sample the second endpoint near the first endpoint.")
    parser.add_argument("--noise-mode", default="fixed", choices=["fixed", "random", "zero"], help="fixed gives the smoothest interpolation by sharing per-layer noise across frames.")
    parser.add_argument("--style-dim", type=int, default=128)
    parser.add_argument("--mapping-layers", type=int, default=4)
    parser.add_argument("--generator-channels", type=int, default=128)
    parser.add_argument("--truncation-psi", type=float, default=0.7)
    parser.add_argument("--truncation-cutoff", type=int, default=0)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def make_interpolation_noise(generator: StyleGAN2Generator64, steps: int, mode: str, device: torch.device) -> list[torch.Tensor] | None:
    if mode == "random":
        return None
    if mode == "zero":
        return [torch.zeros_like(noise) for noise in generator.make_noise(steps, device=device)]
    single_noise = generator.make_noise(1, device=device)
    return [noise.expand(steps, -1, -1, -1).contiguous() for noise in single_noise]


def main() -> None:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("--steps must be at least 2")
    if args.endpoint_scale < 0:
        raise ValueError("--endpoint-scale must be non-negative")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    truncation_cutoff = None if args.truncation_cutoff <= 0 else args.truncation_cutoff
    config = DCGANConfig(device=str(device))
    generator = StyleGAN2Generator64(config, style_dim=args.style_dim, base_channels=args.generator_channels, mapping_layers=args.mapping_layers).to(device)
    load_checkpoint(args.checkpoint, generator, device=device, use_ema=not args.no_ema)
    generator.eval()

    z1 = torch.randn(1, config.noise_dim, 1, 1, device=device)
    z2 = z1 + args.endpoint_scale * torch.randn_like(z1)
    alphas = torch.linspace(0.0, 1.0, steps=args.steps, device=device)
    noise = make_interpolation_noise(generator, args.steps, args.noise_mode, device)
    with torch.no_grad():
        if args.space == "z":
            z = (1.0 - alphas.view(-1, 1, 1, 1)) * z1 + alphas.view(-1, 1, 1, 1) * z2
            images = generator(z, truncation_psi=args.truncation_psi, truncation_cutoff=truncation_cutoff, noise=noise)
        elif args.space == "w":
            w1 = generator.get_latent(z1)
            w2 = generator.get_latent(z2)
            styles = (1.0 - alphas.view(-1, 1)) * w1 + alphas.view(-1, 1) * w2
            images = generator.forward_from_style(styles, truncation_psi=args.truncation_psi, truncation_cutoff=truncation_cutoff, noise=noise)
        else:
            w1 = generator.get_latent(z1).unsqueeze(1).expand(-1, generator.num_style_layers, -1)
            w2 = generator.get_latent(z2).unsqueeze(1).expand(-1, generator.num_style_layers, -1)
            styles = (1.0 - alphas.view(-1, 1, 1)) * w1 + alphas.view(-1, 1, 1) * w2
            images = generator.forward_from_style(styles, truncation_psi=args.truncation_psi, truncation_cutoff=truncation_cutoff, noise=noise)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image_grid(denormalize_images(images), output, nrow=args.steps)
    print(
        f"saved={output} steps={args.steps} space={args.space} endpoint_scale={args.endpoint_scale} "
        f"noise_mode={args.noise_mode} checkpoint={args.checkpoint} ema={not args.no_ema}"
    )


if __name__ == "__main__":
    main()
