# B同学接续说明：Generator、训练框架、可视化

本文档说明如何接入 A 同学已完成的模块，继续实现任务D中 B 同学负责的生成器、训练主循环、checkpoint、生成图片保存与插值实验。

## 已完成模块

A 同学代码位于 `dcgan_a/`：

| 文件 | 作用 |
|------|------|
| `dcgan_a/config.py` | 统一配置：`NOISE_DIM=128`、`IMAGE_SIZE=64`、`CHANNELS=3`、`BATCH_SIZE=128`、CUDA 设备选择 |
| `dcgan_a/data.py` | LFW / CelebA / ImageFolder 数据加载，图片预处理到 `[-1, 1]` |
| `dcgan_a/discriminator.py` | DCGAN Discriminator，输入图片，输出真假概率 |
| `dcgan_a/losses.py` | 判别器 BCE loss、Adam optimizer、单步训练函数 |
| `dcgan_a/metrics.py` | FID 和 Inception Score 计算 |

辅助脚本：

| 脚本 | 作用 |
|------|------|
| `scripts/download_data.py` | 下载 LFW / CelebA 并检查能否读取 |
| `scripts/sanity_check_a.py` | A 模块基础自检 |
| `scripts/train_discriminator_example.py` | 判别器接口示例，不是完整 GAN 训练 |

## 统一接口约定

B 同学实现 Generator 时请保持以下接口一致：

```python
NOISE_DIM = 128
IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 128
```

Generator 推荐输入输出：

```python
z = torch.randn(batch_size, 128, 1, 1, device=device)
fake_images = generator(z)
```

输出必须满足：

```python
fake_images.shape == (batch_size, 3, 64, 64)
fake_images.min() >= -1
fake_images.max() <= 1
```

因为真实图片在 `dcgan_a.data` 中已经通过 `Normalize((0.5,), (0.5,))` 归一化到 `[-1, 1]`，所以 Generator 最后一层应使用 `Tanh()`。

## 推荐 Generator 结构

可以新建：

```text
dcgan_b/
  __init__.py
  generator.py
```

推荐 DCGAN Generator：

```python
import torch
from torch import nn

from dcgan_a import DCGANConfig


class DCGANGenerator(nn.Module):
    def __init__(self, config: DCGANConfig = DCGANConfig(), feature_maps_g: int = 64):
        super().__init__()
        nz = config.noise_dim
        ngf = feature_maps_g
        c = config.channels
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, c, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
```

## 训练主循环接入方式

B 同学主训练脚本建议命名为：

```text
scripts/train_dcgan.py
```

核心接入代码：

```python
import torch
from torch import nn

from dcgan_a import DCGANConfig, DCGANDiscriminator, resolve_device
from dcgan_a.data import build_dataloader, load_face_dataset
from dcgan_a.losses import build_discriminator_optimizer, train_discriminator_step
from dcgan_b.generator import DCGANGenerator

device = resolve_device("cuda")
config = DCGANConfig(device=str(device))

dataset = load_face_dataset("data", name="lfw", image_size=config.image_size, train=True)
dataloader = build_dataloader(dataset, config)

discriminator = DCGANDiscriminator(config).to(device)
generator = DCGANGenerator(config).to(device)

optimizer_d = build_discriminator_optimizer(discriminator, config)
optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
criterion = nn.BCELoss()

for real_images, _ in dataloader:
    batch_size = real_images.size(0)
    z = torch.randn(batch_size, config.noise_dim, 1, 1, device=device)
    fake_images = generator(z)

    d_stats = train_discriminator_step(
        discriminator=discriminator,
        optimizer=optimizer_d,
        real_images=real_images,
        fake_images=fake_images,
        device=device,
        criterion=criterion,
    )

    optimizer_g.zero_grad(set_to_none=True)
    pred_fake = discriminator(fake_images)
    g_targets = torch.ones_like(pred_fake)
    g_loss = criterion(pred_fake, g_targets)
    g_loss.backward()
    optimizer_g.step()
```

注意：`train_discriminator_step()` 内部会对 `fake_images.detach()`，因此不会更新 Generator；Generator 更新要单独写。

## 数据下载命令

先用 LFW 跑通：

```bash
python scripts/download_data.py --dataset lfw --data-root data
```

正式训练可尝试 CelebA：

```bash
python scripts/download_data.py --dataset celeba --data-root data
```

如果 CelebA 自动下载失败，手动下载后放入 `data/celeba/`。

## 评估接入

训练若干 epoch 后，生成一批 fake images，并从 dataloader 中取一批 real images：

```python
from dcgan_a.metrics import evaluate_fid_and_is

metrics = evaluate_fid_and_is(
    real_images=[real_batch],
    fake_images=[fake_batch],
    device=device,
)
print(metrics)
```

正式报告中建议用更多图片计算 FID/IS，少量 batch 的结果只适合调试。

## B 同学待完成清单

- 实现 `DCGANGenerator`。
- 实现 `scripts/train_dcgan.py` 主训练脚本。
- 每隔若干 epoch 保存 checkpoint：Generator、Discriminator、两个 optimizer、epoch。
- 每隔若干 epoch 保存生成图片网格。
- 记录 `d_loss`、`g_loss`、`d_real_prob`、`d_fake_prob`。
- 实现潜变量线性插值实验。
- 跑通 LFW 后再切换 CelebA。

## 当前可用验证命令

检查 A 模块：

```bash
python scripts/sanity_check_a.py
```

检查 CUDA：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```
