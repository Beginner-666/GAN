# 任务D A同学基础部分

本目录实现分工文档中 A 同学负责的基础模块：数据加载与预处理、DCGAN 判别器、判别器 BCE loss/Adam optimizer、FID 与 Inception Score 评估函数。

## 环境

推荐使用 conda：

```bash
conda env create -f environment.yml
conda activate dcgan-face
```

当前 `environment.yml` 按 Python 3.12 + CUDA 12.8 配置，PyTorch 使用官方 cu128 wheel。若你已经有 Python 3.12 环境，也可以直接安装：

```bash
pip install -r requirements-cu128.txt
```

验证 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

## 数据

推荐目录结构：

```text
data/
  lfw/
  celeba/
  custom_faces/
    faces/
      000001.jpg
      000002.jpg
```

本地无标签图片可用 `imagefolder` 模式读取，只需把所有图片放进同一个子目录，例如 `custom_faces/faces/`。

可以先单独下载数据，不启动训练：

```bash
python scripts/download_data.py --dataset lfw --data-root data
```

CelebA 也保留了自动下载入口：

```bash
python scripts/download_data.py --dataset celeba --data-root data
```

如果 CelebA 自动下载失败，通常是 Google Drive 限流或文件校验问题，建议手动下载后放到 `data/celeba/` 再读取。

## 快速自检

```bash
python scripts/sanity_check_a.py
```

该脚本只检查接口、网络输出形状、loss 反传和 FID/IS 数学函数，不会下载数据或加载 Inception 权重。

默认优先使用 GPU；没有 CUDA 时会自动回退 CPU。也可以显式指定：

```bash
python scripts/sanity_check_a.py --device cpu
```

## 判别器训练侧接入

B 同学训练主循环中可以直接使用：

```python
from dcgan_a import DCGANConfig, DCGANDiscriminator, resolve_device
from dcgan_a.losses import build_discriminator_optimizer, train_discriminator_step

device = resolve_device("cuda")
config = DCGANConfig(device=str(device))
discriminator = DCGANDiscriminator(config).to(device)
optimizer_d = build_discriminator_optimizer(discriminator, config)

stats = train_discriminator_step(
    discriminator=discriminator,
    optimizer=optimizer_d,
    real_images=real_images,
    fake_images=fake_images,
    device=device,
)
```

也提供了一个只训练判别器接口的最小例子：

```bash
python scripts/train_discriminator_example.py --data-root data/custom_faces --dataset imagefolder --steps 100
```
