# Linux Training Plan for CelebA DCGAN

This plan assumes the large CelebA dataset is used and all training/tuning runs
are executed sequentially on one RTX 4090. The other teammate can focus on the
experiment report while training results are produced.

## 0. Data Layout

Do not point `ImageFolder` directly at the HuggingFace download directory if it
contains `.cache`. Create a clean ImageFolder entry instead:

```bash
mkdir -p data/celeba_imagefolder
ln -sfn /inspire/hdd/project/robot-reasoning/xiangyushun-p-xiangyushun/zichun/GAN/GAN/data/celeba_hf/img_align_celeba data/celeba_imagefolder/faces
```

Check the image count:

```bash
find data/celeba_imagefolder/faces -iname "*.jpg" | wc -l
```

Expected CelebA aligned image count is about `202599`.

## 1. Environment Checks

```bash
conda activate dcgan-face
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 2. Quick Smoke Test

Run a short CelebA job before launching automated tuning:

```bash
mkdir -p logs

python -u scripts/train_dcgan.py \
  --dataset imagefolder \
  --data-root data/celeba_imagefolder \
  --epochs 1 \
  --batch-size 64 \
  --num-workers 8 \
  --device cuda \
  --output-dir outputs/celeba_smoke \
  --checkpoint-dir checkpoints/celeba_smoke \
  2>&1 | tee logs/celeba_smoke.log
```

Check that sample images and a checkpoint were created:

```bash
ls outputs/celeba_smoke/samples
ls checkpoints/celeba_smoke
```

## 3. Automated Tuning and Final Training

The automation script runs several candidate configurations, evaluates each with
FID/IS, selects the lowest-FID candidate, then runs a longer final training job
using the winning parameters.

Default candidates:

- `baseline_bs128_lr2e4_g64_d64`
- `stable_bs128_lr1e4_g64_d64`
- `smallbatch_bs64_lr2e4_g64_d64`
- `largerg_bs64_lr2e4_g128_d64`

Recommended command:

```bash
mkdir -p logs

python -u scripts/auto_tune_dcgan.py \
  --dataset imagefolder \
  --data-root data/celeba_imagefolder \
  --device cuda \
  --num-workers 8 \
  --tune-epochs 20 \
  --final-epochs 100 \
  --eval-images 1024 \
  --eval-batch-size 32 \
  --sample-every 5 \
  --checkpoint-every 5 \
  --output-root outputs/celeba_autotune \
  --checkpoint-root checkpoints/celeba_autotune \
  --log-root logs/celeba_autotune \
  2>&1 | tee logs/celeba_autotune_master.log
```

If time is short, reduce the tuning and final epochs:

```bash
python -u scripts/auto_tune_dcgan.py \
  --dataset imagefolder \
  --data-root data/celeba_imagefolder \
  --device cuda \
  --num-workers 8 \
  --tune-epochs 10 \
  --final-epochs 50 \
  --eval-images 512 \
  --eval-batch-size 32 \
  --output-root outputs/celeba_autotune_fast \
  --checkpoint-root checkpoints/celeba_autotune_fast \
  --log-root logs/celeba_autotune_fast \
  2>&1 | tee logs/celeba_autotune_fast_master.log
```

## 4. Outputs to Inspect

Tuning summary:

```bash
cat outputs/celeba_autotune/summary.csv
cat outputs/celeba_autotune/winner.json
cat outputs/celeba_autotune/final_result.json
```

Generated samples:

```bash
find outputs/celeba_autotune -path "*/samples/latest.png"
```

Interpolation images:

```bash
find outputs/celeba_autotune -name "interpolation.png"
```

Evaluation logs:

```bash
find outputs/celeba_autotune -name "eval_fid_is.txt" -print -exec cat {} \;
```

The final model is stored under a directory named like:

```text
checkpoints/celeba_autotune/final_<winning_candidate>_e100/dcgan_latest.pt
```

The final generated images and interpolation are under:

```text
outputs/celeba_autotune/final_<winning_candidate>_e100/
```

## 5. Manual Final Evaluation

If you want to rerun final evaluation with more images:

```bash
python scripts/evaluate_dcgan.py \
  --checkpoint checkpoints/celeba_autotune/final_<winning_candidate>_e100/dcgan_latest.pt \
  --dataset imagefolder \
  --data-root data/celeba_imagefolder \
  --num-images 4096 \
  --batch-size 32 \
  --num-workers 8 \
  --feature-maps-g <winning_feature_maps_g> \
  --device cuda \
  2>&1 | tee outputs/celeba_autotune/final_eval_4096.txt
```

Replace `<winning_candidate>` and `<winning_feature_maps_g>` using
`outputs/celeba_autotune/winner.json`.

## 6. Report Division of Work

4090 owner:

- Run smoke test, automated tuning, final training, interpolation, and FID/IS.
- Send `summary.csv`, `winner.json`, `final_result.json`, sample grids,
  interpolation images, and loss curves to the report writer.

Report writer:

- Write GAN/DCGAN background and model structure.
- Describe CelebA preprocessing and ImageFolder layout.
- Create tables from `summary.csv`.
- Compare sample images, FID, IS, loss curves, and interpolation results.
- Explain why the final hyperparameters were selected.

