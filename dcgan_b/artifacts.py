from __future__ import annotations

import csv
import math
import struct
import zlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from torchvision.utils import save_image as torchvision_save_image
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torchvision_save_image = None


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def save_generated_grid(
    generator: nn.Module,
    fixed_noise: torch.Tensor,
    output_path: str | Path,
    nrow: int = 8,
) -> None:
    generator.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fake_images = generator(fixed_noise)
    save_image_grid(denormalize_images(fake_images), output_path, nrow=nrow)


def save_image_grid(images: torch.Tensor, output_path: str | Path, nrow: int = 8) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if torchvision_save_image is not None:
        torchvision_save_image(images, str(output_path), nrow=nrow)
        return

    images = images.detach().cpu().clamp(0.0, 1.0)
    if images.ndim != 4:
        raise ValueError("Expected images shaped [N, C, H, W]")
    n, c, h, w = images.shape
    if c not in (1, 3):
        raise ValueError("Expected 1 or 3 image channels")
    nrow = max(1, min(nrow, n))
    ncol = int(math.ceil(n / nrow))
    canvas = torch.ones(c, ncol * h, nrow * w)
    for idx, image in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        canvas[:, row * h : (row + 1) * h, col * w : (col + 1) * w] = image
    array = canvas.mul(255).byte().permute(1, 2, 0).numpy()
    if c == 1:
        array = array.repeat(3, axis=2)
    _write_png_rgb(output_path, array.tobytes(), width=array.shape[1], height=array.shape[0])


def _write_png_rgb(path: str | Path, rgb_bytes: bytes, width: int, height: int) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    stride = width * 3
    raw = b"".join(b"\x00" + rgb_bytes[y * stride : (y + 1) * stride] for y in range(height))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=6))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)


def append_metrics_csv(csv_path: str | Path, row: dict[str, Any]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_loss_svg(csv_path: str | Path, svg_path: str | Path) -> None:
    rows = _read_metric_rows(csv_path)
    svg_path = Path(svg_path)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        svg_path.write_text(_empty_svg("No metrics yet"), encoding="utf-8")
        return

    steps = [row["step"] for row in rows]
    series = {
        "d_loss": [row["d_loss"] for row in rows],
        "g_loss": [row["g_loss"] for row in rows],
    }
    svg_path.write_text(_line_chart_svg(steps, series), encoding="utf-8")


def save_checkpoint(
    path: str | Path,
    epoch: int,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    config: Any,
    fixed_noise: torch.Tensor,
    generator_ema: nn.Module | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "config": getattr(config, "__dict__", config),
        "fixed_noise": fixed_noise.detach().cpu(),
    }
    if generator_ema is not None:
        checkpoint["generator_ema"] = generator_ema.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    generator: nn.Module,
    discriminator: nn.Module | None = None,
    optimizer_g: torch.optim.Optimizer | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
    use_ema: bool = False,
    generator_ema: nn.Module | None = None,
) -> tuple[int, torch.Tensor | None]:
    checkpoint = torch.load(path, map_location=device)
    generator_key = "generator_ema" if use_ema and "generator_ema" in checkpoint else "generator"
    generator.load_state_dict(checkpoint[generator_key])
    if generator_ema is not None:
        ema_key = "generator_ema" if "generator_ema" in checkpoint else "generator"
        generator_ema.load_state_dict(checkpoint[ema_key])
    if discriminator is not None and "discriminator" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator"])
    if optimizer_g is not None and "optimizer_g" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    if optimizer_d is not None and "optimizer_d" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    fixed_noise = checkpoint.get("fixed_noise")
    if fixed_noise is not None:
        fixed_noise = fixed_noise.to(device)
    return int(checkpoint.get("epoch", 0)), fixed_noise


def _read_metric_rows(csv_path: str | Path) -> list[dict[str, float]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    rows: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "step": float(row["step"]),
                        "d_loss": float(row["d_loss"]),
                        "g_loss": float(row["g_loss"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def _empty_svg(message: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="360">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="360" y="180" text-anchor="middle" font-family="Arial" font-size="16">{message}</text>'
        "</svg>"
    )


def _line_chart_svg(steps: list[float], series: dict[str, list[float]]) -> str:
    width, height = 720, 360
    left, right, top, bottom = 58, 24, 24, 46
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [v for points in series.values() for v in points if math.isfinite(v)]
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(values), max(values)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad

    def xy(x: float, y: float) -> tuple[float, float]:
        px = left + (x - x_min) / (x_max - x_min) * plot_w
        py = top + (y_max - y) / (y_max - y_min) * plot_h
        return px, py

    colors = {"d_loss": "#2563eb", "g_loss": "#dc2626"}
    lines = []
    for name, points in series.items():
        coords = [xy(x, y) for x, y in zip(steps, points)]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        color = colors.get(name, "#111827")
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{path}"/>')
        lines.append(f'<text x="{left + 12}" y="{top + 18 + 18 * len(lines)}" fill="{color}" font-family="Arial" font-size="13">{name}</text>')

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#9ca3af"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#9ca3af"/>',
            f'<text x="{left}" y="{height - 14}" font-family="Arial" font-size="12">step {int(x_min)} to {int(x_max)}</text>',
            f'<text x="{left}" y="{top - 6}" font-family="Arial" font-size="12">loss {y_min:.3f} to {y_max:.3f}</text>',
            *lines,
            "</svg>",
        ]
    )
