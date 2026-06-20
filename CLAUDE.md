# CLAUDE.md

## Project Scope

This repository implements AI2602 Task D: GAN-based face image generation.

The current baseline is DCGAN on 64x64 RGB face images. LFW is used for quick
experiments, and CelebA is used for longer final training when available.

## Directory Convention

```text
dcgan_a/          A-student modules: data, discriminator, losses, metrics
dcgan_b/          B-student modules: generator and reusable training utilities
scripts/          runnable entry points
data/             local datasets, ignored by git
checkpoints/      model checkpoints, ignored by git
outputs/          generated samples, interpolation images, logs, ignored by git
```

## Interface Contract

Keep these defaults compatible across modules:

```python
NOISE_DIM = 128
IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 128
```

Images passed between modules use PyTorch tensor layout `[N, C, H, W]` and are
normalized to `[-1, 1]`. The generator must end with `Tanh()`.

## Development Rules

- Prefer small, importable modules under `dcgan_a/` and `dcgan_b/`.
- Keep scripts thin: parse arguments, call module functions, write artifacts.
- Do not commit datasets, checkpoints, generated images, or logs.
- Add command-line defaults that run on CPU for smoke tests and CUDA for real
  training when available.
- Avoid optional heavy dependencies unless the project already declares them.
- Validate changes with a lightweight smoke test before longer training.

## Validation Commands

Minimal import and shape checks:

```powershell
python scripts\sanity_check_a.py --device cpu
python scripts\sanity_check_b.py --device cpu
python scripts\sanity_check_stylegan.py --device cpu
python scripts\sanity_check_stylegan2.py --device cpu
```

Short training smoke test with a local ImageFolder dataset:

```powershell
python scripts\train_dcgan.py --dataset imagefolder --data-root data\custom_faces --epochs 1 --batch-size 8 --num-workers 0 --device cpu
python scripts\train_stylegan.py --dataset imagefolder --data-root data\custom_faces --epochs 1 --batch-size 4 --num-workers 0 --device cpu --max-batches 1
python scripts\train_stylegan2.py --dataset imagefolder --data-root data\custom_faces --epochs 1 --batch-size 4 --num-workers 0 --device cpu --max-batches 1
```

LFW quick run after downloading data:

```powershell
python scripts\download_data.py --dataset lfw --data-root data
python scripts\train_dcgan.py --dataset lfw --data-root data --epochs 5 --batch-size 64 --num-workers 0
python scripts\train_stylegan.py --dataset lfw --data-root data --epochs 1 --batch-size 64 --num-workers 0 --max-batches 10
python scripts\train_stylegan2.py --dataset lfw --data-root data --epochs 1 --batch-size 64 --num-workers 0 --max-batches 10
```

Interpolation test after training:

```powershell
python scripts\interpolate_stylegan.py --checkpoint checkpoints\stylegan\stylegan_latest.pt --space w --truncation-psi 0.7
python scripts\interpolate_stylegan2.py --checkpoint checkpoints\stylegan2\stylegan2_latest.pt --space w --truncation-psi 0.7
```
