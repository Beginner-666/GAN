from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BEST_DCGAN = {
    "batch_size": 32,
    "lr": 0.0002,
    "feature_maps_g": 128,
    "feature_maps_d": 64,
    "noise_dim": 128,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run natural long-training mode-collapse/degeneration and mitigation comparison from the tuned best DCGAN config."
    )
    parser.add_argument("--data-root", default="data/celeba_imagefolder")
    parser.add_argument("--dataset", default="imagefolder", choices=["imagefolder", "lfw", "celeba"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-epochs", default="20,40,60,80,100,120,140,160,180,200", help="Comma-separated checkpoint epochs to evaluate.")
    parser.add_argument("--eval-images", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--output-root", default="outputs/dcgan_mode_collapse_natural")
    parser.add_argument("--checkpoint-root", default="checkpoints/dcgan_mode_collapse_natural")
    parser.add_argument("--log-root", default="logs/dcgan_mode_collapse_natural")
    parser.add_argument("--skip-training", action="store_true", help="Only evaluate existing checkpoints.")
    parser.add_argument("--skip-eval", action="store_true", help="Only train variants; do not run FID/IS evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    log_root = Path(args.log_root)
    for path in (output_root, checkpoint_root, log_root):
        path.mkdir(parents=True, exist_ok=True)

    variants = [
        {
            "name": "best_dcgan_longtrain",
            "description": "Tuned best DCGAN, trained longer without extra anti-collapse regularization.",
            "extra_train_args": [],
        },
        {
            "name": "best_dcgan_longtrain_smoothed_noise",
            "description": "Same tuned best DCGAN plus one-sided label smoothing and decayed instance noise.",
            "extra_train_args": [
                "--real-label",
                "0.9",
                "--instance-noise-std",
                "0.05",
                "--instance-noise-decay-epochs",
                str(max(1, args.epochs // 2)),
            ],
        },
    ]

    if not args.skip_training:
        for variant in variants:
            train_variant(args, variant, output_root, checkpoint_root, log_root)

    rows: list[dict[str, Any]] = []
    if not args.skip_eval:
        eval_epochs = parse_eval_epochs(args.eval_epochs)
        validate_eval_epochs(eval_epochs, args.checkpoint_every, args.epochs)
        for variant in variants:
            rows.extend(evaluate_variant(args, variant, output_root, checkpoint_root, eval_epochs))
        write_csv(output_root / "mode_collapse_natural_comparison.csv", rows)
        write_plots(output_root, rows)

    write_readme(output_root, variants, rows)


def parse_eval_epochs(text: str) -> list[int]:
    epochs: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            epochs.append(int(part))
    return sorted(set(epochs))


def validate_eval_epochs(eval_epochs: list[int], checkpoint_every: int, total_epochs: int) -> None:
    missing = [epoch for epoch in eval_epochs if epoch != total_epochs and epoch % checkpoint_every != 0]
    if missing:
        values = ",".join(str(epoch) for epoch in missing)
        raise ValueError(
            f"eval epochs must match saved checkpoints. These epochs are not saved by --checkpoint-every {checkpoint_every}: {values}"
        )


def train_variant(args: argparse.Namespace, variant: dict[str, Any], output_root: Path, checkpoint_root: Path, log_root: Path) -> None:
    name = variant["name"]
    output_dir = output_root / name
    checkpoint_dir = checkpoint_root / name
    log_path = log_root / f"{name}.log"
    if (checkpoint_dir / "dcgan_latest.pt").exists():
        print(f"skip existing training: {name}", flush=True)
        return

    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "train_dcgan.py"),
        "--dataset",
        args.dataset,
        "--data-root",
        args.data_root,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(BEST_DCGAN["batch_size"]),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(BEST_DCGAN["lr"]),
        "--feature-maps-g",
        str(BEST_DCGAN["feature_maps_g"]),
        "--feature-maps-d",
        str(BEST_DCGAN["feature_maps_d"]),
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--sample-every",
        str(args.sample_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ] + list(variant.get("extra_train_args", []))
    run_logged(cmd, log_path)


def evaluate_variant(
    args: argparse.Namespace,
    variant: dict[str, Any],
    output_root: Path,
    checkpoint_root: Path,
    eval_epochs: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_dir = output_root / variant["name"]
    checkpoint_dir = checkpoint_root / variant["name"]
    for epoch in eval_epochs:
        checkpoint = checkpoint_dir / f"dcgan_epoch_{epoch:04d}.pt"
        if not checkpoint.exists() and epoch == max(eval_epochs):
            checkpoint = checkpoint_dir / "dcgan_latest.pt"
        row = evaluate_checkpoint(args, checkpoint, output_dir, f"epoch_{epoch:04d}")
        row.update(
            {
                "experiment": variant["name"],
                "epoch": epoch,
                "batch_size": BEST_DCGAN["batch_size"],
                "lr": BEST_DCGAN["lr"],
                "feature_maps_g": BEST_DCGAN["feature_maps_g"],
                "feature_maps_d": BEST_DCGAN["feature_maps_d"],
                "noise_dim": BEST_DCGAN["noise_dim"],
                "regularization": "label_smoothing+instance_noise" if "smoothed_noise" in variant["name"] else "none",
                "checkpoint": str(checkpoint),
            }
        )
        rows.append(row)
    return rows


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path, output_dir: Path, name: str) -> dict[str, Any]:
    if not checkpoint.exists():
        return {
            "name": name,
            "eval_status": "missing_checkpoint",
            "num_images": "",
            "fid": "",
            "inception_score_mean": "",
            "inception_score_std": "",
            "pixel_diversity": "",
            "pairwise_l2_diversity": "",
        }

    import torch

    from dcgan_a import DCGANConfig, resolve_device
    from dcgan_a.metrics import evaluate_fid_and_is
    from dcgan_b import DCGANGenerator
    from dcgan_b.artifacts import load_checkpoint

    device = resolve_device(args.device)
    config = DCGANConfig(
        noise_dim=BEST_DCGAN["noise_dim"],
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=str(device),
    )
    from dcgan_a.data import build_dataloader, load_face_dataset

    dataset = load_face_dataset(args.data_root, args.dataset, image_size=config.image_size, train=False, download=False)
    dataloader = build_dataloader(dataset, config, shuffle=True, drop_last=False)
    generator = DCGANGenerator(config, feature_maps_g=BEST_DCGAN["feature_maps_g"]).to(device)
    load_checkpoint(checkpoint, generator, device=device)
    generator.eval()

    real_batches: list[torch.Tensor] = []
    fake_batches: list[torch.Tensor] = []
    collected = 0
    with torch.no_grad():
        for real_images, _targets in dataloader:
            current = min(real_images.size(0), args.eval_images - collected)
            if current <= 0:
                break
            z = torch.randn(current, BEST_DCGAN["noise_dim"], 1, 1, device=device)
            fake = generator(z).cpu()
            real_batches.append(real_images[:current])
            fake_batches.append(fake)
            collected += current
            if collected >= args.eval_images:
                break

    metrics = evaluate_fid_and_is(real_batches, fake_batches, device=device, batch_size=args.eval_batch_size)
    fake_all = torch.cat(fake_batches, dim=0)
    diversity = calculate_diversity(fake_all)
    result = {
        "name": name,
        "eval_status": "ok",
        "num_images": collected,
        "fid": metrics["fid"],
        "inception_score_mean": metrics["inception_score_mean"],
        "inception_score_std": metrics["inception_score_std"],
        "pixel_diversity": diversity["pixel_diversity"],
        "pairwise_l2_diversity": diversity["pairwise_l2_diversity"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_eval_fid_is_diversity.txt").write_text(
        "\n".join(
            [
                f"num_images={collected}",
                f"fid={metrics['fid']:.6f}",
                f"inception_score_mean={metrics['inception_score_mean']:.6f}",
                f"inception_score_std={metrics['inception_score_std']:.6f}",
                f"pixel_diversity={diversity['pixel_diversity']:.6f}",
                f"pairwise_l2_diversity={diversity['pairwise_l2_diversity']:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"evaluated {checkpoint}: fid={metrics['fid']:.6f} is={metrics['inception_score_mean']:.6f} "
        f"pixel_div={diversity['pixel_diversity']:.6f}",
        flush=True,
    )
    return result


def calculate_diversity(images: torch.Tensor, max_pairwise: int = 256) -> dict[str, float]:
    import torch

    images = images.float()
    pixel_diversity = float(images.std(dim=0).mean().item())
    flat = images[:max_pairwise].flatten(1)
    if flat.size(0) < 2:
        pairwise = 0.0
    else:
        pairwise = float(torch.pdist(flat, p=2).mean().item() / (flat.size(1) ** 0.5))
    return {"pixel_diversity": pixel_diversity, "pairwise_l2_diversity": pairwise}


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("running:", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plots(output_root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip comparison plots", flush=True)
        return

    valid = [r for r in rows if isinstance(r.get("fid"), float)]
    if not valid:
        return
    metrics = [
        ("fid", "FID (lower is better)", "mode_collapse_natural_fid.png", "{:.1f}"),
        ("inception_score_mean", "Inception Score", "mode_collapse_natural_is.png", "{:.2f}"),
        ("pixel_diversity", "Pixel diversity", "mode_collapse_natural_pixel_diversity.png", "{:.3f}"),
        ("pairwise_l2_diversity", "Pairwise L2 diversity", "mode_collapse_natural_pairwise_diversity.png", "{:.3f}"),
    ]
    for key, ylabel, filename, value_fmt in metrics:
        plt.figure(figsize=(8.4, 4.8), dpi=180)
        for experiment in sorted({r["experiment"] for r in valid}):
            series = sorted([r for r in valid if r["experiment"] == experiment], key=lambda r: int(r["epoch"]))
            xs = [int(r["epoch"]) for r in series]
            ys = [float(r[key]) for r in series]
            plt.plot(xs, ys, marker="o", linewidth=2.0, markersize=4.5, label=experiment)
            for idx, (x, y) in enumerate(zip(xs, ys)):
                if idx % 2 == 0 or idx == len(xs) - 1:
                    plt.text(x, y, value_fmt.format(y), ha="center", va="bottom", fontsize=6.5)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(ylabel + " under long DCGAN training")
        plt.xticks(sorted({int(r["epoch"]) for r in valid}))
        plt.grid(True, alpha=0.28)
        plt.legend(fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(output_root / filename)
        plt.close()


def write_readme(output_root: Path, variants: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Natural Mode Collapse / Mitigation Comparison",
        "",
        "All runs start from the tuned best DCGAN configuration:",
        "",
        f"- batch_size={BEST_DCGAN['batch_size']}",
        f"- lr={BEST_DCGAN['lr']}",
        f"- feature_maps_g={BEST_DCGAN['feature_maps_g']}",
        f"- feature_maps_d={BEST_DCGAN['feature_maps_d']}",
        f"- noise_dim={BEST_DCGAN['noise_dim']}",
        "",
        "## Variants",
        "",
    ]
    for variant in variants:
        lines.append(f"- `{variant['name']}`: {variant['description']}")
    lines.extend(
        [
            "",
            "## How to judge collapse",
            "",
            "Natural mode collapse is supported if longer training makes samples visually less diverse while FID worsens and diversity metrics drop.",
            "The regularized run tests a mitigation strategy: one-sided label smoothing plus decayed instance noise.",
            "",
            "## Outputs",
            "",
            "- `mode_collapse_natural_comparison.csv`",
            "- `mode_collapse_natural_fid.png`",
            "- `mode_collapse_natural_is.png`",
            "- `mode_collapse_natural_pixel_diversity.png`",
            "- `mode_collapse_natural_pairwise_diversity.png`",
            "",
            "Default evaluation epochs are 20,40,60,80,100,120,140,160,180,200. Keep `--checkpoint-every` aligned with these epochs.",
            "- `<variant>/samples/`",
            "- `<variant>/loss.svg`",
        ]
    )
    output_root.joinpath("README_MODE_COLLAPSE_NATURAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
