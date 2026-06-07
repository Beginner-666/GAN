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
from dcgan_a.metrics import evaluate_fid_and_is
from dcgan_b.artifacts import load_checkpoint
from dcgan_b.stylegan import StyleGANGenerator64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a StyleGAN checkpoint with FID and Inception Score.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="lfw", choices=["imagefolder", "lfw", "celeba"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--num-images", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--style-dim", type=int, default=128)
    parser.add_argument("--mapping-layers", type=int, default=4)
    parser.add_argument("--generator-channels", type=int, default=128)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    config = DCGANConfig(batch_size=args.batch_size, num_workers=args.num_workers, device=str(device))

    try:
        from dcgan_a.data import build_dataloader, load_face_dataset
    except ImportError as exc:
        raise ImportError("Evaluation requires torchvision. Install dependencies from requirements-cu128.txt.") from exc

    dataset = load_face_dataset(args.data_root, args.dataset, image_size=config.image_size, train=False, download=args.download)
    dataloader = build_dataloader(dataset, config, shuffle=True, drop_last=False)

    generator = StyleGANGenerator64(
        config,
        style_dim=args.style_dim,
        base_channels=args.generator_channels,
        mapping_layers=args.mapping_layers,
    ).to(device)
    load_checkpoint(args.checkpoint, generator, device=device)
    generator.eval()

    real_batches: list[torch.Tensor] = []
    fake_batches: list[torch.Tensor] = []
    collected = 0
    with torch.no_grad():
        for real_images, _targets in dataloader:
            current = min(real_images.size(0), args.num_images - collected)
            if current <= 0:
                break
            real = real_images[:current]
            z = torch.randn(current, config.noise_dim, 1, 1, device=device)
            fake = generator(z).cpu()
            real_batches.append(real)
            fake_batches.append(fake)
            collected += current
            if collected >= args.num_images:
                break

    if collected == 0:
        raise RuntimeError("No images collected for evaluation.")

    metrics = evaluate_fid_and_is(real_batches, fake_batches, device=device, batch_size=args.batch_size)
    print(f"num_images={collected}")
    for key, value in metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
