# AI2602 深度学习大作业说明

本课程大作业共有4个任务可供选择，具体如下：

---

## 任务A：Transformer 在气象数据时序预测中的应用

### 背景

气象数据时序预测在天气预报、灾害预警、农业生产等领域具有重要的应用价值。传统方法如ARIMA、LSTM等在处理长序列数据时存在一定的局限性。Transformer模型由于其自注意力机制（Self-Attention）能够捕捉长距离依赖关系，近年来在自然语言处理（NLP）领域取得了显著成功，并逐渐被应用于时序数据预测任务中。本大作业旨在探索Transformer模型在气象数据时序预测中的应用，并评估其性能。

### 说明

- **模型选择**：标准Transformer或改进版本（如Informer、Temporal Fusion Transformer）。
- **数据集**：气象数据（如Kaggle上的Weather Dataset）。

### 基本要求

- 实现一个基于Transformer的气象数据时序预测模型。
- 使用所选择的数据集进行训练和测试，并记录模型的预测效果。
- 探索模型在不同气象变量（如温度、降水量）上的预测效果，并分析其适用性。
- 进行超参数调优（如学习率、层数、头数等）并分析其对模型性能的影响。
- 实现一个基于Transformer的时序预测模型。
- 在选定的时序数据集上训练模型，并评估其预测性能。

### Bonus 要求

- 对比Transformer模型与传统时序预测模型（如LSTM、ARIMA）的性能差异。
- 尝试对Transformer模型进行改进，例如引入其他注意力机制（如稀疏注意力、局部注意力）或结合其他模型（如CNN、Graph Neural Networks）进行联合训练。

### 参考资料

- Transformer：Attention is All You Need [arXiv:1706.03762]
- Informer：Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting [arXiv:2012.07436]

---

## 任务B：Vision Transformer (ViT) 用于图像分类任务

### 背景

图像分类是计算机视觉领域的核心任务之一，传统方法如卷积神经网络（CNN）在该任务上取得了显著成功。然而，近年来，基于Transformer的视觉模型（如Vision Transformer, ViT）逐渐展现出强大的性能，尤其是在大规模数据集上的表现优于传统CNN模型。本大作业旨在探索ViT在图像分类任务中的应用，并评估其性能。

### 说明

**模型**

- **基础模型**：使用标准的Vision Transformer (ViT)模型，将图像分割为固定大小的patch，并将这些patch线性嵌入后输入Transformer编码器进行特征提取。
- **改进模型**：可以尝试对ViT进行改进，例如引入混合模型（Hybrid Model），即先用ResNet提取局部特征，再用ViT进行全局特征提取。

**数据集**

- **数据集选择**：使用公开的图像分类数据集，例如CIFAR-10或Tiny ImageNet。
- **数据预处理**：对图像进行标准化处理，并将其划分为训练集、验证集和测试集。

### 基本要求

- 实现一个基于Vision Transformer (ViT)的图像分类模型。
- 使用所选择的数据集进行训练和测试，并记录模型的分类准确率（Accuracy）及其他评估指标（如Top-1/Top-5准确率）。
- 对比ViT模型与传统CNN模型（如ResNet）的性能差异。

### Bonus 要求

- 尝试对ViT模型进行改进，例如引入数据增强技术（如MixUp、RandAugment）或正则化方法（如Stochastic Depth、Dropout）以提升模型性能。
- 进行超参数调优（如patch大小、层数、头数等）并分析其对模型性能的影响。

### 参考论文

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Steiner et al., "How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers"

---

## 任务C：自监督学习（Self-Supervised Learning）在图像分类中的应用

### 背景

自监督学习（Self-Supervised Learning, SSL）是一种无需人工标注数据即可学习有效特征表示的机器学习方法。它通过设计"前置任务"（pretext tasks）从数据本身生成监督信号，从而在大规模无标签数据上进行预训练。自监督学习在计算机视觉、自然语言处理等领域取得了显著成果，能够显著降低对标注数据的依赖，提高模型在下游任务中的表现。本大作业的目标是探索自监督学习中的预训练方法，包括旋转预测（Rotation）、拼图（Jigsaw Puzzle）等前置任务。

### 说明

**模型**

使用轻量级卷积神经网络（如ResNet-18）作为骨干网络，参考以下自监督预训练方法实现：

- **旋转预测（Rotation Prediction）**
  对输入图像随机旋转0°、90°、180°或270°，训练模型预测旋转角度。
  参考论文：Gidaris et al., "Unsupervised Representation Learning by Predicting Image Rotations", ICLR 2018.

- **拼图任务（Jigsaw Puzzle）**
  将图像分割为多个小块并打乱顺序，训练模型恢复原始排列。
  参考论文：Noroozi & Favaro, "Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles", ECCV 2016.

- **相对块位置预测（Relative Patch Position）**
  预测图像中两个随机块之间的相对位置关系。
  参考论文：Doersch et al., "Unsupervised Visual Representation Learning by Context Prediction", ICCV 2015.

**数据集**

- **数据**：Tiny ImageNet，当作预训练的无标签数据，大小 246M
- **数据预处理**：对图像进行标准化处理，并使用数据增强技术（如随机裁剪、水平翻转）以提升模型泛化能力。

### 基本要求

- 实现一个基于自监督学习的预训练模型，完成预训练。
- 实现一个自监督预训练任务。
- 进行超参数调优（如batch size、学习率、预训练轮数等）并分析其对模型性能和资源消耗的影响。

### Bonus 要求

- 将预训练模型迁移到图像分类任务中，使用少量标注数据进行微调，并记录模型的分类准确率（Accuracy）及其他评估指标（如Top-1/Top-5准确率）。
- 对比自监督预训练模型与随机初始化模型在图像分类任务中的性能差异。
- 实现不同的自监督预训练任务。

---

## 任务D：基于 GAN 的人头图像生成

### 背景

生成对抗网络（GAN）是一种强大的生成模型，通过生成器与判别器的对抗训练，能够生成高质量的数据样本。本项目旨在利用GAN模型实现人头图像的生成，探索其在图像生成领域的应用。

### 项目目标

- 理解GAN的基本原理，包括生成器与判别器的对抗训练过程。
- 复现一个基于GAN的人头图像生成系统。
- 在选定的数据集上训练模型，并生成高质量的人头图像。

### 说明

**模型选择**：可以参考以下GAN模型或其改进版本：

- **DCGAN**（Deep Convolutional GAN）：适用于图像生成的经典GAN模型。
- **StyleGAN**：能够生成高质量且多样化的图像。
- **CycleGAN**：适用于图像风格迁移任务，可尝试生成特定风格的人头图像。

**数据集**：可以使用以下公开的人脸数据集：

- **CelebA**：包含超过20万张名人头像的图像数据集，大小 1.3G
- **LFW**（Labeled Faces in the Wild）：包含超过1.3万张人脸图像，适用于小规模实验，大小 118M

**评估指标**：使用以下指标评估生成图像的质量：

- **FID**（Fréchet Inception Distance）：衡量生成图像与真实图像之间的分布距离。
- **IS**（Inception Score）：衡量生成图像的多样性与清晰度。

### 基本要求

- 实现基础的GAN模型（如DCGAN）并进行人头图像生成。
- 在选定数据集上训练模型，生成高质量的人头图像。
- 测试两张人头图像之间线性插值的一系列结果。
- 使用FID或IS评估生成图像的质量。

### Bonus 任务

- 对比改进的GAN模型（如StyleGAN或CycleGAN）与基础模型的性能差异。
- 探索解决GAN训练中的模式崩溃问题，并提出改进方法。

### 参考资料与论文

- 论文 "Generative Adversarial Nets" [arXiv:1406.2661]
- DCGAN：论文 "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" [arXiv:1511.06434]
- StyleGAN：论文 "A Style-Based Generator Architecture for Generative Adversarial Networks" [arXiv:1812.04948]
- CycleGAN：论文 "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" [arXiv:1703.10593]
