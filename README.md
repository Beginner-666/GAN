# AI2602 GAN 人脸生成项目

本仓库对应 AI2602 深度学习课程项目任务 D：基于 GAN 的人脸图像生成。项目以 CelebA aligned face images 为主要数据集，实现并比较了 DCGAN、轻量版 StyleGAN-like 和课程规模 StyleGAN2，同时补充了潜变量插值、FID/IS 评估和模式崩溃分析。

完整实验报告见 [REPORT_TASK_D.tex](REPORT_TASK_D.tex)，报告图片位于 [report_images/](report_images/)。

## 主要结果

| 实验 | Epoch | Eval Images | FID ↓ | IS ↑ | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| DCGAN 最佳配置 | 50 | 4096 | 16.394 | 2.093 | 基础模型最终结果 |
| DCGAN 长训无缓解 | 400 | 2048 | 31.052 | 2.241 | FID 上升，多样性下降 |
| DCGAN 缓解组 | 160 | 2048 | 15.589 | 2.303 | label smoothing + instance noise 早期有效 |
| StyleGAN-like | 100 | 4096 | 23.114 | 2.400 | 引入 mapping、AdaIN、noise injection 与 R1 |
| StyleGAN2 | 50 | 4096 | 24.926 | 1.845 | 引入 modulated conv、demodulation、style mixing 与 PL 正则 |
| 强制模式崩溃 | 10 | 2048 | 336.690 | 1.000 | 判别器过强导致严重退化 |

结论上，最终 DCGAN 在同一评估协议下取得最低 FID；StyleGAN-like 的 IS 更高，但 FID 未超过 DCGAN；StyleGAN2 结构更接近原版核心设计，但在当前 64x64、50 epoch 和轻量通道配置下，FID/IS 仍未超过最佳 DCGAN。

## 项目结构

```text
dcgan_a/                 数据、判别器、损失、指标等基础模块
dcgan_b/                 DCGAN 生成器、StyleGAN-like、StyleGAN2 与保存工具
scripts/                 训练、评估、插值、调参与模式崩溃实验脚本
report_images/           报告使用的样本图、曲线、架构图和插值图
REPORT_TASK_D.tex        任务 D 实验报告 LaTeX 源文件
README.md                项目说明
```

## 环境配置

推荐使用 conda：

```bash
conda env create -f environment.yml
conda activate dcgan-face
```

如果服务器上的 conda pip 阶段失败，可以手动创建：

```bash
conda create -n dcgan-face python=3.12 numpy scipy pillow tqdm pip -y
conda activate dcgan-face
pip install -r requirements-cu128.txt
```

验证 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 数据准备

训练主要使用 CelebA aligned face images，约 202599 张人脸图片。由于 `torchvision` 的 CelebA/LFW 自动下载经常不可用，实际训练推荐使用 `imagefolder` 入口：

```text
data/celeba_imagefolder/
  faces/
    000001.jpg
    000002.jpg
    ...
```

如果图片解压在 `data/celeba_hf/img_align_celeba/`，可以建立软链接：

```bash
mkdir -p data/celeba_imagefolder
ln -sfn "$PWD/data/celeba_hf/img_align_celeba" data/celeba_imagefolder/faces
find data/celeba_imagefolder/faces -iname "*.jpg" | wc -l
```

图像预处理统一为 `64 x 64` RGB，并归一化到 `[-1, 1]`；生成器最后使用 `Tanh`，输出范围与真实图像保持一致。

## 快速运行

自检：

```bash
python scripts/sanity_check_a.py --device cpu
python scripts/sanity_check_b.py --device cpu
python scripts/sanity_check_stylegan.py --device cpu
python scripts/sanity_check_stylegan2.py --device cpu
```

DCGAN 最佳配置训练：

```bash
python -u scripts/train_dcgan.py \
  --dataset imagefolder \
  --data-root data/celeba_imagefolder \
  --epochs 50 \
  --batch-size 32 \
  --num-workers 8 \
  --lr 0.0002 \
  --feature-maps-g 128 \
  --feature-maps-d 64 \
  --output-dir outputs/celeba_final_bs32_g128_d64_e50 \
  --checkpoint-dir checkpoints/celeba_final_bs32_g128_d64_e50 \
  --sample-every 5 \
  --checkpoint-every 5 \
  --device cuda
```

DCGAN 评估：

```bash
python scripts/evaluate_dcgan.py \
  --checkpoint checkpoints/celeba_final_bs32_g128_d64_e50/dcgan_latest.pt \
  --dataset imagefolder \
  --data-root data/celeba_imagefolder \
  --num-images 4096 \
  --batch-size 32 \
  --num-workers 8 \
  --feature-maps-g 128 \
  --device cuda
```

潜变量插值：

```bash
python scripts/interpolate.py \
  --checkpoint checkpoints/celeba_final_bs32_g128_d64_e50/dcgan_latest.pt \
  --output outputs/celeba_final_bs32_g128_d64_e50/interpolation.png \
  --feature-maps-g 128 \
  --device cuda
```

## 实验脚本

常用入口如下：

```text
scripts/auto_tune_dcgan.py                     DCGAN 自动调参
scripts/run_dcgan_mode_collapse_comparison.py  长训退化与缓解策略对比
scripts/train_dcgan_mode_collapse.py           强制模式崩溃实验
scripts/train_stylegan.py                      轻量版 StyleGAN-like 训练
scripts/evaluate_stylegan.py                   StyleGAN-like FID/IS 评估
scripts/interpolate_stylegan.py                StyleGAN-like 潜变量插值
scripts/train_stylegan2.py                     StyleGAN2 训练
scripts/interpolate_stylegan2.py               StyleGAN2 W 空间插值
```

报告中的主要实验包括：

- DCGAN 调参：比较 batch size、学习率和生成器通道数，最终选择 `batch_size=32, lr=2e-4, G=128, D=64`。
- DCGAN 长训分析：最佳配置继续训练到 400 epoch 后 FID 上升、多样性下降，说明存在生成分布退化。
- 缓解实验：one-sided label smoothing 与 instance noise 在 80/160 epoch 明显改善结果，但不能彻底阻止后期退化。
- StyleGAN-like：实现 mapping network、learned constant、AdaIN、noise injection、R1 regularization 和 EMA。
- StyleGAN2：实现 modulated convolution、demodulation、style mixing、path length regularization、R1 regularization 和 EMA，并完成 W 空间插值与 4096 张图像 FID/IS 评估。
- 强制模式崩溃：通过削弱生成器、增强判别器和增加判别器更新步数，展示训练失衡导致的严重退化。

## 评估指标

FID 使用 Inception v3 特征空间中的均值和协方差衡量真实图像分布与生成图像分布的距离，越低越好。IS 衡量生成图像的清晰度和类别多样性，越高越好；但 CelebA 属于单一人脸领域，因此本文主要依据 FID 和可视化结果分析，IS 作为辅助指标。

## 注意事项

- `data/`、`checkpoints/`、`logs/` 等大文件目录不应提交。
- 使用 `imagefolder` 时，`--data-root` 必须指向包含类别子目录的根目录，例如 `data/celeba_imagefolder`。
- StyleGAN-like 与 StyleGAN2 是课程项目规模实现，不等同于官方高分辨率完整复现。