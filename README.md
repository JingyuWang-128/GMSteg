---

# 🚀 GenMamba-INN: Robust Generative Steganography via Frequency-Gated Mamba in Latent Space

## 🎯 1. 背景与痛点 (Pain Points in Robust Steganography)

目前的鲁棒性图像隐写（Robust Image Steganography）领域，尽管在生成式方法（Generative Methods）上取得了一定进展，但仍面临着不可调和的**“三难困境” (Trilemma)**：**不可感知性 (Imperceptibility)**、**鲁棒性 (Robustness)** 和 **容量 (Capacity)** 之间的权衡。具体痛点如下：

1. **痛点一：特征提取的局限性 (Limitation of Feature Extraction)**
* **问题：** 现有的基于 CNN 或 Transformer 的方法在潜空间（Latent Space）进行特征建模时，要么感受野受限（CNN），难以捕捉全局依赖；要么计算复杂度随序列长度二次增长（Transformer），难以高效处理高分辨率特征。
* **后果：** 导致隐写模型在保持高图像质量的同时，难以嵌入大量秘密信息，或者生成的图像存在明显的伪影。


2. **痛点二：嵌入策略的盲目性 (Blindness of Embedding Strategy)**
* **问题：** 传统方法往往对潜空间的所有区域“一视同仁”，忽视了图像纹理的分布差异。
* **后果：** 秘密信息被错误地嵌入到平滑区域（如天空、纯色背景），导致肉眼可见的噪点或伪影，极大地降低了隐写的不可感知性（安全性）。


3. **痛点三：有损重构的脆弱性 (Vulnerability to Lossy Reconstruction)**
* **问题：** 基于 Stable Diffusion (SD) 等生成式模型的方法，必须经过 VAE 的解码（Decode）和重编码（Encode）过程，或者面临信道传输中的噪声攻击。
* **后果：** VAE 的重构本身是有损的（Lossy），这种微小的“潜空间漂移” (Latent Drift) 会导致基于可逆神经网络（INN）的精确提取失效，秘密信息无法正确还原。



---

## 💡 2. 我们的创新点 (Key Innovations)

针对上述痛点，我们提出了 **GenMamba-INN**，一种结合了状态空间模型（Mamba）与可逆神经网络的新型生成式隐写框架。

### 🌟 创新点一：基于 Mamba 的潜空间可逆隐写框架 (Latent Mamba-INN Framework)

* **针对痛点：** 痛点一（特征提取局限性）
* **核心思想：** 利用 Mamba (State Space Model) 的线性复杂度优势，替代传统 CNN/Transformer 作为 INN 的变换核心。
* **具体实现：**
* 构建了 `LatentINNBlock`，采用可逆神经网络架构保证信息理论上的无损嵌入与提取。
* 在耦合层（Coupling Layer）的变换函数  和  中，引入了 Mamba 模块。
* **优势：** 能够以  的复杂度捕捉潜空间中的长距离全局依赖（Global Context），显著提升了模型对复杂图像内容的建模能力，从而在不损失生成质量的前提下提高了嵌入容量。



### 🌟 创新点二：频率自适应门控机制 (Frequency-Gated Embedding Mechanism)

* **针对痛点：** 痛点二（嵌入策略盲目性）
* **核心思想：** “把秘密藏在乱纹理里”。利用图像的频率特性指导嵌入过程，自动将信息分配到人眼不敏感的高频纹理区域。
* **具体实现：**
* 设计了 `FrequencyGatedSSM` 模块。
* 引入 **DCT 频率分析子网**，实时计算载体潜码（Cover Latent）的局部纹理复杂度，生成一张“频率热力图” (Frequency Heatmap)。
* 使用该热力图作为 **门控信号 (Gate)**，对 Mamba 的特征输出进行加权调制。
* **优势：** 模型学会了“自适应避让”平滑区域，只在纹理丰富区域进行修改，实现了极致的不可感知性。



### 🌟 创新点三：潜空间漂移矫正模块 (Latent Drift Rectifier, LDR)

* **针对痛点：** 痛点三（有损重构的脆弱性）
* **核心思想：** 在提取端增加一个“去噪前置护盾”，专门对抗 VAE 重构损耗和外部攻击。
* **具体实现：**
* 设计了 `DriftRectifier` 模块，置于提取网络的最前端。
* 它是一个轻量级的 Mamba 残差网络，专门训练用于预测并抵消潜码的漂移量。
* **训练策略：** 采用分阶段课程学习（Curriculum Learning），在 Stage 2 专门引入强噪声训练该模块。
* **优势：** 使得脆弱的 INN 架构能够适应有损的生成环境，在经受 VAE 重构甚至 JPEG 压缩/高斯噪声攻击后，仍能高精度恢复秘密信息。



---

## 🛠️ 3. 实现细节与代码结构 (Implementation Details)

### 🏗️ 整体架构 (Architecture)

* **Encoder:** Pre-trained Stable Diffusion VAE (frozen).
* **Backbone:** Multi-scale Invertible Neural Network (INN) with Mamba blocks.
* **Optimization:** * **Stage 1:** Warm-up INN (Focus on capacity & imperceptibility).
* **Stage 2:** Train Rectifier (Focus on robustness against noise).
* **Stage 3:** Joint Fine-tuning.



### 📂 核心文件说明

* `models/inn_block.py`: 实现了带有 **Tanh Clamping** 的可逆耦合层，防止梯度爆炸。
* `models/mamba_block.py`: 实现了 **Frequency-Gated Mamba**，融合频率特征进行自适应特征提取。
* `models/gen_mamba.py`: 实现了 **Channel Permutation (通道交换)**，确保 Cover 和 Secret 充分融合。
* `models/rectifier.py`: 实现了漂移矫正器，保障鲁棒性。

---
