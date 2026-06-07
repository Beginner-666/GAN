import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcgan_a import DCGANConfig, DCGANDiscriminator, resolve_device
from dcgan_a.losses import build_discriminator_optimizer, train_discriminator_step
from dcgan_b import DCGANGenerator
from dcgan_b.artifacts import append_metrics_csv, save_checkpoint, save_generated_grid, write_loss_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an intentionally unstable DCGAN setting to induce mode collapse.")
    parser.add_argument("--data-root", default="data/celeba_imagefolder")
    parser.add_argument("--dataset", default="imagefolder", choices=["imagefolder", "lfw", "celeba"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--noise-dim", type=int, default=8, help="Small latent dimension strongly reduces diversity.")
    parser.add_argument("--lr-g", type=float, default=1e-5, help="Low G lr makes it easier for D to dominate.")
    parser.add_argument("--lr-d", type=float, default=1e-3, help="High D lr makes it easier for D to dominate.")
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--feature-maps-g", type=int, default=32)
    parser.add_argument("--feature-maps-d", type=int, default=128)
    parser.add_argument("--d-steps", type=int, default=5, help="Number of discriminator updates per generator update.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", default="outputs/dcgan_mode_collapse_forced")
    parser.add_argument("--checkpoint-dir", default="checkpoints/dcgan_mode_collapse_forced")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--fixed-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = DCGANConfig(
        noise_dim=args.noise_dim,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr_d,
        beta1=args.beta1,
        beta2=args.beta2,
        feature_maps_d=args.feature_maps_d,
        device=str(device),
        pin_memory=str(device) == "cuda",
        persistent_workers=args.num_workers > 0,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
    )

    from dcgan_a.data import build_dataloader, load_face_dataset

    dataset = load_face_dataset(args.data_root, args.dataset, image_size=config.image_size, train=True, download=False)
    dataloader = build_dataloader(dataset, config)
    if len(dataloader) == 0:
        raise RuntimeError("Dataloader is empty. Reduce batch size or check dataset path.")

    generator = DCGANGenerator(config, feature_maps_g=args.feature_maps_g).to(device)
    discriminator = DCGANDiscriminator(config).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    optimizer_d = build_discriminator_optimizer(discriminator, config)
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(max(1, args.fixed_samples), config.noise_dim, 1, 1, device=device)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    samples_dir = output_dir / "samples"
    metrics_csv = output_dir / "metrics.csv"
    loss_svg = output_dir / "loss.svg"
    global_step = 0

    print(
        f"device={device} dataset={args.dataset} samples={len(dataset)} batches={len(dataloader)} "
        f"noise_dim={args.noise_dim} g_lr={args.lr_g} d_lr={args.lr_d} d_steps={args.d_steps}"
    )
    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        for batch_idx, (real_images, _targets) in enumerate(dataloader, start=1):
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            d_stats = None
            for _ in range(args.d_steps):
                z = torch.randn(batch_size, config.noise_dim, 1, 1, device=device)
                with torch.no_grad():
                    fake_images = generator(z)
                d_stats = train_discriminator_step(discriminator, optimizer_d, real_images, fake_images, device, criterion)

            z = torch.randn(batch_size, config.noise_dim, 1, 1, device=device)
            fake_images = generator(z)
            optimizer_g.zero_grad(set_to_none=True)
            pred_fake = discriminator(fake_images)
            g_loss = criterion(pred_fake, torch.ones_like(pred_fake))
            g_loss.backward()
            optimizer_g.step()

            assert d_stats is not None
            global_step += 1
            row = {
                "epoch": epoch,
                "batch": batch_idx,
                "step": global_step,
                "d_loss": d_stats["d_loss"],
                "g_loss": float(g_loss.detach().cpu()),
                "d_loss_real": d_stats["d_loss_real"],
                "d_loss_fake": d_stats["d_loss_fake"],
                "d_real_prob": d_stats["d_real_prob"],
                "d_fake_prob": d_stats["d_fake_prob"],
            }
            append_metrics_csv(metrics_csv, row)

            if batch_idx == 1 or batch_idx % args.log_interval == 0:
                print(
                    f"epoch={epoch}/{args.epochs} batch={batch_idx}/{len(dataloader)} "
                    f"d_loss={row['d_loss']:.4f} g_loss={row['g_loss']:.4f} "
                    f"real={row['d_real_prob']:.4f} fake={row['d_fake_prob']:.4f}"
                )

        if epoch % args.sample_every == 0:
            save_generated_grid(generator, fixed_noise, samples_dir / f"epoch_{epoch:04d}.png", nrow=8)
            write_loss_svg(metrics_csv, loss_svg)

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir / f"dcgan_epoch_{epoch:04d}.pt",
                epoch,
                generator,
                discriminator,
                optimizer_g,
                optimizer_d,
                config,
                fixed_noise,
            )

    save_checkpoint(checkpoint_dir / "dcgan_latest.pt", args.epochs, generator, discriminator, optimizer_g, optimizer_d, config, fixed_noise)
    save_generated_grid(generator, fixed_noise, samples_dir / "latest.png", nrow=8)
    write_loss_svg(metrics_csv, loss_svg)
    print(f"done checkpoints={checkpoint_dir} outputs={output_dir}")


if __name__ == "__main__":
    main()
