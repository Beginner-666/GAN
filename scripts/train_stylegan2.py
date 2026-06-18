from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcgan_a import DCGANConfig, resolve_device
from dcgan_b.artifacts import append_metrics_csv, denormalize_images, load_checkpoint, save_checkpoint, save_image_grid, write_loss_svg
from dcgan_b.stylegan2 import StyleGAN2Discriminator64, StyleGAN2Generator64, build_stylegan2_optimizers, copy_stylegan2_for_ema, update_ema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 64x64 StyleGAN2 face generator.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="celeba", choices=["imagefolder", "lfw", "celeba"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--style-dim", type=int, default=128)
    parser.add_argument("--mapping-layers", type=int, default=4)
    parser.add_argument("--generator-channels", type=int, default=128)
    parser.add_argument("--discriminator-channels", type=int, default=64)
    parser.add_argument("--style-mixing-prob", type=float, default=0.9)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--r1-gamma", type=float, default=10.0)
    parser.add_argument("--r1-every", type=int, default=16)
    parser.add_argument("--pl-weight", type=float, default=2.0)
    parser.add_argument("--pl-every", type=int, default=4)
    parser.add_argument("--truncation-psi", type=float, default=0.7)
    parser.add_argument("--truncation-cutoff", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", default="outputs/stylegan2")
    parser.add_argument("--checkpoint-dir", default="checkpoints/stylegan2")
    parser.add_argument("--resume", default="")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--fixed-samples", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def d_logistic_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()


def g_nonsaturating_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(-fake_logits).mean()


def r1_regularization(discriminator: torch.nn.Module, real_images: torch.Tensor) -> torch.Tensor:
    real_images = real_images.detach().requires_grad_(True)
    real_logits = discriminator(real_images)
    grad = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad.pow(2).flatten(1).sum(1).mean()


def path_length_regularization(
    fake_images: torch.Tensor,
    styles: torch.Tensor,
    mean_path_length: torch.Tensor,
    decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(fake_images) / math.sqrt(fake_images.shape[2] * fake_images.shape[3])
    grad = torch.autograd.grad(
        outputs=(fake_images * noise).sum(),
        inputs=styles,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    path_lengths = grad.pow(2).sum(2).mean(1).sqrt()
    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
    penalty = (path_lengths - path_mean.detach()).pow(2).mean()
    return penalty, path_mean.detach(), path_lengths.detach().mean()


@torch.no_grad()
def save_ema_grid(
    generator: StyleGAN2Generator64,
    fixed_noise: torch.Tensor,
    output_path: Path,
    nrow: int,
    truncation_psi: float,
    truncation_cutoff: int | None,
) -> None:
    generator.eval()
    images = generator(fixed_noise, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    save_image_grid(denormalize_images(images), output_path, nrow=nrow)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    truncation_cutoff = None if args.truncation_cutoff <= 0 else args.truncation_cutoff

    config = DCGANConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        device=str(device),
        pin_memory=str(device) == "cuda",
        persistent_workers=args.num_workers > 0,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
    )

    try:
        from dcgan_a.data import build_dataloader, load_face_dataset
    except ImportError as exc:
        raise ImportError("Training requires torchvision. Install dependencies from requirements-cu128.txt.") from exc

    dataset = load_face_dataset(args.data_root, args.dataset, image_size=config.image_size, train=True, download=args.download)
    dataloader = build_dataloader(dataset, config)
    if len(dataloader) == 0:
        raise RuntimeError("Dataloader is empty. Reduce batch size or check dataset path.")

    generator = StyleGAN2Generator64(config, style_dim=args.style_dim, base_channels=args.generator_channels, mapping_layers=args.mapping_layers).to(device)
    discriminator = StyleGAN2Discriminator64(config, base_channels=args.discriminator_channels).to(device)
    generator_ema = copy_stylegan2_for_ema(generator).to(device)
    optimizer_g, optimizer_d = build_stylegan2_optimizers(generator, discriminator, args.lr, args.beta1, args.beta2)

    fixed_count = max(1, args.fixed_samples)
    fixed_noise = torch.randn(fixed_count, config.noise_dim, 1, 1, device=device)
    start_epoch = 1
    if args.resume:
        loaded_epoch, loaded_noise = load_checkpoint(args.resume, generator, discriminator, optimizer_g, optimizer_d, device, generator_ema=generator_ema)
        start_epoch = loaded_epoch + 1
        if loaded_noise is not None and loaded_noise.shape[1:] == fixed_noise.shape[1:]:
            fixed_noise = loaded_noise[:fixed_count]

    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    samples_dir = output_dir / "samples"
    metrics_csv = output_dir / "metrics.csv"
    loss_svg = output_dir / "loss.svg"
    global_step = 0
    mean_path_length = torch.zeros((), device=device)

    print(f"device={device} dataset={args.dataset} samples={len(dataset)} batches={len(dataloader)} model=stylegan2_64")
    for epoch in range(start_epoch, args.epochs + 1):
        generator.train()
        discriminator.train()
        for batch_idx, (real_images, _targets) in enumerate(dataloader, start=1):
            if args.max_batches and batch_idx > args.max_batches:
                break
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            z = torch.randn(batch_size, config.noise_dim, 1, 1, device=device)
            mixing_z = torch.randn_like(z) if args.style_mixing_prob > 0.0 else None
            with torch.no_grad():
                fake_images = generator(z, mixing_z=mixing_z, style_mixing_prob=args.style_mixing_prob)
            real_logits = discriminator(real_images)
            fake_logits = discriminator(fake_images)
            d_loss = d_logistic_loss(real_logits, fake_logits)

            r1_penalty_value = torch.zeros((), device=device)
            if args.r1_every > 0 and global_step % args.r1_every == 0:
                r1_penalty_value = r1_regularization(discriminator, real_images)
                d_loss = d_loss + (args.r1_gamma * 0.5 * args.r1_every) * r1_penalty_value

            optimizer_d.zero_grad(set_to_none=True)
            d_loss.backward()
            optimizer_d.step()

            z = torch.randn(batch_size, config.noise_dim, 1, 1, device=device)
            mixing_z = torch.randn_like(z) if args.style_mixing_prob > 0.0 else None
            fake_images, styles = generator(z, return_styles=True, mixing_z=mixing_z, style_mixing_prob=args.style_mixing_prob)
            fake_logits_for_g = discriminator(fake_images)
            g_loss = g_nonsaturating_loss(fake_logits_for_g)

            pl_penalty_value = torch.zeros((), device=device)
            path_length_value = torch.zeros((), device=device)
            if args.pl_weight > 0.0 and args.pl_every > 0 and global_step % args.pl_every == 0:
                pl_penalty_value, mean_path_length, path_length_value = path_length_regularization(fake_images, styles, mean_path_length)
                g_loss = g_loss + args.pl_weight * args.pl_every * pl_penalty_value

            optimizer_g.zero_grad(set_to_none=True)
            g_loss.backward()
            optimizer_g.step()
            update_ema(generator_ema, generator, args.ema_decay)

            global_step += 1
            row = {
                "epoch": epoch,
                "batch": batch_idx,
                "step": global_step,
                "d_loss": float(d_loss.detach().cpu()),
                "g_loss": float(g_loss.detach().cpu()),
                "d_real_logit": float(real_logits.detach().mean().cpu()),
                "d_fake_logit": float(fake_logits.detach().mean().cpu()),
                "r1_penalty": float(r1_penalty_value.detach().cpu()),
                "pl_penalty": float(pl_penalty_value.detach().cpu()),
                "path_length": float(path_length_value.detach().cpu()),
                "style_mixing_prob": args.style_mixing_prob,
                "ema_decay": args.ema_decay,
            }
            append_metrics_csv(metrics_csv, row)

            if batch_idx == 1 or batch_idx % args.log_interval == 0:
                print(
                    f"epoch={epoch}/{args.epochs} batch={batch_idx}/{len(dataloader)} "
                    f"d_loss={row['d_loss']:.4f} g_loss={row['g_loss']:.4f} "
                    f"real_logit={row['d_real_logit']:.4f} fake_logit={row['d_fake_logit']:.4f} "
                    f"r1={row['r1_penalty']:.4f} pl={row['pl_penalty']:.4f}"
                )

        if epoch % args.sample_every == 0:
            save_ema_grid(generator_ema, fixed_noise, samples_dir / f"epoch_{epoch:04d}_ema.png", 8, args.truncation_psi, truncation_cutoff)
            write_loss_svg(metrics_csv, loss_svg)

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(checkpoint_dir / f"stylegan2_epoch_{epoch:04d}.pt", epoch, generator, discriminator, optimizer_g, optimizer_d, config, fixed_noise, generator_ema=generator_ema)

    save_checkpoint(checkpoint_dir / "stylegan2_latest.pt", args.epochs, generator, discriminator, optimizer_g, optimizer_d, config, fixed_noise, generator_ema=generator_ema)
    save_ema_grid(generator_ema, fixed_noise, samples_dir / "latest_ema.png", 8, args.truncation_psi, truncation_cutoff)
    write_loss_svg(metrics_csv, loss_svg)
    print(f"done checkpoints={checkpoint_dir} outputs={output_dir}")


if __name__ == "__main__":
    main()
