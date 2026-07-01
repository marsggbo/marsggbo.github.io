---
layout: post
title: "从图像到文本：Diffusion 模型原理全解——数学、结构、训练、推理一次讲清"
date: 2026-06-30
tags: [Diffusion, LLM, 生成模型, 数学推导]
---

# 从图像到文本：Diffusion 模型原理全解——数学、结构、训练、推理一次讲清

---

## 1. 前言：从一个直觉开始

你有没有想过，往一张清晰的照片里慢慢加噪声，直到变成纯噪声——这个过程能不能反过来？

Diffusion 模型的核心思路就是：**学习「加噪」的逆过程**。先定义一个确定性的加噪过程（从真实数据走向纯噪声），再训练一个神经网络学会逆转它（从纯噪声还原出真实数据）。

生成时，从随机噪声出发，一步步"去噪"，最终得到新的样本。

这个想法可以用于图像，也可以用于文本——但两者的模型结构、输入输出形式、计算流程差别很大。本文从图像 diffusion 讲起，然后过渡到文本 diffusion，把每一步的维度变化和计算逻辑讲清楚。

---

## 2. 先把核心数学搞懂

### 2.1 符号表：读公式前先认清这些变量

在进入公式之前，先把所有符号的物理含义说清楚。后面看到这些字母，脑子里要能立刻对应上它们代表什么。

| 符号 | 含义 | 直觉 |
|---|---|---|
| $x_0$ | 原始干净数据（图像像素、token embedding 等） | "未被污染的真相" |
| $x_t$ | 第 $t$ 步加噪后的数据 | "经过 $t$ 轮污染的样本" |
| $x_T$ | 最终完全噪声化的数据 | "几乎纯噪声，看不出原始内容" |
| $t$ | 时间步，$t \in \{1, 2, \ldots, T\}$ | "当前处于第几轮加噪" |
| $T$ | 总步数，通常取 1000 | "加噪的总轮数" |
| $\beta_t$ | 第 $t$ 步的**噪声强度**（noise schedule），$\beta_t \in (0,1)$ | "这一步要往里加多少噪声" |
| $\alpha_t$ | $\alpha_t = 1 - \beta_t$，信号保留比例 | "这一步保留多少原来的信号" |
| $\bar\alpha_t$ | $\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$，前 $t$ 步的累积信号保留比例 | "经过 $t$ 步，原始信号还剩多少" |
| $\epsilon$ | 标准高斯噪声，$\epsilon \sim \mathcal{N}(0, I)$ | "每次加入的随机干扰" |
| $\epsilon_\theta$ | 神经网络，参数为 $\theta$ | "用来预测噪声的模型" |

> **关于 noise schedule**：$\beta_t$ 不是随机的，而是人为预先设计好的序列。最简单的是线性 schedule：从 $\beta_1 = 10^{-4}$ 线性增大到 $\beta_T = 0.02$。意思是：**早期步骤（小 $t$）加噪很轻微，后期步骤（大 $t$）加噪越来越猛**。这样设计是为了让逆向去噪在每一步都是一个可处理的小问题，而不是一步从纯噪声跳回清晰图像。

### 2.2 前向过程（加噪）

前向过程是固定的，**不需要学习**。给定一张真实图像 $x_0$，逐步向它加高斯噪声：

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\; \sqrt{1-\beta_t}\, x_{t-1},\; \beta_t I\right)$$

**这个公式的白话翻译**：从 $x_{t-1}$ 生成 $x_t$ 时，先把 $x_{t-1}$ 缩小到 $\sqrt{1-\beta_t}$ 倍（信号衰减），然后加上方差为 $\beta_t$ 的高斯噪声。$\beta_t$ 越大，信号衰减越多，噪声越大。

> 这个公式用到了条件高斯分布：$\mathcal{N}(x; \mu, \sigma^2 I)$ 表示均值为 $\mu$、方差为 $\sigma^2$ 的高斯分布。从这个分布采样等价于：$x = \mu + \sigma \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。

这个过程有一个非常好用的性质：**可以直接从 $x_0$ 一步跳到任意时刻 $t$**，不需要逐步走 $t$ 次。

令 $\alpha_t = 1 - \beta_t$，$\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$，则：

$$q(x_t \mid x_0) = \mathcal{N}(x_t;\; \sqrt{\bar\alpha_t}\, x_0,\; (1-\bar\alpha_t) I)$$

写成采样形式（实际代码里用这个）：

$$\boxed{x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)}$$

**直觉解读**：
- $\sqrt{\bar\alpha_t}\, x_0$：原始信号按 $\sqrt{\bar\alpha_t}$ 缩放后的残留部分
- $\sqrt{1 - \bar\alpha_t}\, \epsilon$：加入的噪声部分
- $t$ 越大 → $\bar\alpha_t$ 越小 → 信号保留越少，噪声越多
- $t = T$ 时 $\bar\alpha_T \approx 0$，于是 $x_T \approx \epsilon \sim \mathcal{N}(0,I)$，彻底变成纯噪声

为什么可以一步跳到？因为多个高斯分布的叠加仍然是高斯分布，可以把递推关系合并成一个解析式。这个性质让训练极其高效——不需要真的走 $t$ 步，只需算一次 $\bar\alpha_t$。

```python
# 预计算 noise schedule（在训练开始前做一次）
T = 1000
betas = torch.linspace(1e-4, 0.02, T)       # β_1, β_2, ..., β_T，线性增长
alphas = 1 - betas                            # α_t = 1 - β_t
alpha_bar = torch.cumprod(alphas, dim=0)      # ᾱ_t = α_1 * α_2 * ... * α_t

# 给定 x0 和时间步 t，一步得到 x_t（即上面的采样公式）
def q_sample(x0, t):
    # x0: [B, C, H, W]，t: [B]
    eps = torch.randn_like(x0)                           # ε ~ N(0,I)
    sqrt_ab    = alpha_bar[t][:, None, None, None].sqrt()  # √ᾱ_t
    sqrt_1_ab  = (1 - alpha_bar[t])[:, None, None, None].sqrt()  # √(1-ᾱ_t)
    x_t = sqrt_ab * x0 + sqrt_1_ab * eps                # 公式直接翻译
    return x_t, eps
```

### 2.3 反向过程（去噪）

反向过程是我们要学习的。我们想要 $q(x_{t-1} \mid x_t)$——已知加了 $t$ 步噪声的 $x_t$，推断出少加一步噪声的 $x_{t-1}$。

这个分布依赖于整个数据集，**无法直接计算**。但当 $\beta_t$ 很小时，反向过程也近似是高斯的（这是 diffusion 模型一个关键的数学洞见），于是我们用神经网络来拟合它：

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1};\; \mu_\theta(x_t, t),\; \Sigma_\theta(x_t, t))$$

**训练策略**：不直接预测均值 $\mu_\theta$，而是训练神经网络 $\epsilon_\theta$ 来预测被加入的噪声 $\epsilon$。已知 $x_t$ 和预测的噪声 $\hat\epsilon$，可以反推出 $x_0$，进而推出均值：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\, \epsilon_\theta(x_t, t)\right)$$

**这个公式的来源**：把 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ 代入贝叶斯后验 $q(x_{t-1} \mid x_t, x_0)$ 的均值表达式，再用 $\hat\epsilon$ 替换 $\epsilon$，就得到上式。

### 2.4 训练目标

经过变分推导，最终的损失函数简化为一个极其简洁的形式：

$$\boxed{\mathcal{L} = \mathbb{E}_{x_0,\, \epsilon,\, t}\!\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]}$$

**说人话**：

1. 从数据集随机取一个真实样本 $x_0$
2. 随机取一个时间步 $t$
3. 采样随机噪声 $\epsilon \sim \mathcal{N}(0,I)$
4. 算出加噪后的 $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$
5. 把 $x_t$ 和 $t$ 喂给网络，让它预测 $\hat\epsilon = \epsilon_\theta(x_t, t)$
6. 用 MSE 衡量预测误差

整个训练目标就是一个**预测噪声的回归问题**，比 GAN 的博弈稳定得多，比 VAE 的 ELBO 也更简洁。

---

## 3. 图像 Diffusion：UNet 结构

### 3.1 为什么用 UNet

图像 diffusion（DDPM、Stable Diffusion 的噪声预测部分）用 UNet 作为去噪网络 $\epsilon_\theta$。

**输入**：加了噪声的图像 $x_t$，形状 `[B, C, H, W]`
**输出**：预测的噪声 $\hat\epsilon$，形状 `[B, C, H, W]`（与输入完全相同）

UNet 是编码器-解码器结构，有跳跃连接（skip connections），能同时处理全局语义（"这里是人脸"）和局部细节（"这几个像素是眼睛"），这正好符合图像去噪的需求。

### 3.2 时间步 t 怎么注入

网络需要知道当前是第几步（$t$），因为不同时刻的噪声程度不同，去噪策略也不同（$t=999$ 几乎全是噪声，需要大幅修改；$t=1$ 接近干净图像，只需微调）。

做法：把标量 $t$ 编码成一个向量，加进网络的每一层。编码方式和 Transformer 里的位置编码一样——Sinusoidal 编码。

```python
def timestep_embedding(t, dim):
    # t: [B]，整数时间步
    # 输出: [B, dim]，每个时间步对应一个 dim 维向量
    half = dim // 2
    # 生成 half 个频率（从高频到低频）
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args = t[:, None].float() * freqs[None]   # [B, half]，时间步 × 频率
    # 用 cos 和 sin 各表示一半维度，拼在一起
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]
    return emb

# 在 UNet 的每个 ResBlock 里，把时间 embedding 加到特征图上
class ResBlock(nn.Module):
    def __init__(self, channels, t_dim):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        # 把 t_dim 维的时间向量投影到 channels 维，然后加到特征图
        self.time_proj = nn.Linear(t_dim, channels)

    def forward(self, x, t_emb):
        # x:     [B, C, H, W]  当前特征图
        # t_emb: [B, t_dim]    时间步 embedding
        scale = self.time_proj(t_emb)[:, :, None, None]  # [B, C, 1, 1]
        return self.conv(x) + scale   # 广播加到每个空间位置
```

### 3.3 完整训练：公式对照代码

下面把训练流程的数学和代码并排放，方便对照：

**数学步骤：**

$$t \sim \text{Uniform}\{1,\ldots,T\}, \quad \epsilon \sim \mathcal{N}(0,I)$$
$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$$
$$\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

**对应代码：**

```python
T = 1000
betas = torch.linspace(1e-4, 0.02, T)   # β_t，线性 schedule
alphas = 1 - betas                        # α_t = 1 - β_t
alpha_bar = torch.cumprod(alphas, dim=0)  # ᾱ_t = ∏α_s

def train_step(x0, unet):
    # x0: [B, C, H, W]，像素值归一化到 [-1, 1]
    B = x0.shape[0]

    # 步骤1：t ~ Uniform{1,...,T}
    t = torch.randint(0, T, (B,))               # [B]，每个样本随机一个时间步

    # 步骤2：ε ~ N(0,I)，然后算 x_t（前向公式的直接实现）
    eps = torch.randn_like(x0)                  # [B, C, H, W]
    sqrt_ab   = alpha_bar[t][:, None, None, None].sqrt()      # √ᾱ_t，形状 [B,1,1,1]
    sqrt_1_ab = (1 - alpha_bar[t])[:, None, None, None].sqrt()
    x_t = sqrt_ab * x0 + sqrt_1_ab * eps        # [B, C, H, W]，加噪后的图像

    # 步骤3：网络预测噪声 ε_θ(x_t, t)
    eps_pred = unet(x_t, t)                     # [B, C, H, W]

    # 步骤4：MSE 损失，即 ||ε - ε_θ||²
    loss = F.mse_loss(eps_pred, eps)
    return loss
```

### 3.4 推理：从噪声还原图像

**数学步骤：**

从 $x_T \sim \mathcal{N}(0,I)$ 出发，每一步：

$$\mu_t = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon\right), \quad \hat\epsilon = \epsilon_\theta(x_t, t)$$

$$x_{t-1} = \mu_t + \sqrt{\beta_t}\, z, \quad z \sim \mathcal{N}(0,I) \quad (t > 0)$$

**对应代码：**

```python
@torch.no_grad()
def ddpm_sample(unet, shape):
    # shape: (B, C, H, W)
    x = torch.randn(shape)   # x_T ~ N(0, I)，从纯噪声出发

    for t in reversed(range(T)):    # t = T-1, T-2, ..., 1, 0
        t_batch = torch.full((shape[0],), t)

        # 用网络预测噪声 ε̂ = ε_θ(x_t, t)
        eps_pred = unet(x, t_batch)         # [B, C, H, W]

        # 计算均值 μ_t（对应上面公式）
        a  = alphas[t]        # α_t
        ab = alpha_bar[t]     # ᾱ_t
        mu = (x - (1-a) / (1-ab).sqrt() * eps_pred) / a.sqrt()

        # x_{t-1} = μ_t + √β_t * z（最后一步 t=0 不加噪声）
        if t > 0:
            z = torch.randn_like(x)
            x = mu + betas[t].sqrt() * z
        else:
            x = mu    # t=0 时直接返回均值，不加额外噪声

    return x   # [B, C, H, W]，去噪后的图像
```

**维度流**（以 `C=3, H=W=64` 为例）：

```
输入：[B, 3, 64, 64]（纯噪声 x_T）
  ↓ UNet encoder（下采样，增加通道数）
  [B,  64, 64, 64]
  [B, 128, 32, 32]
  [B, 256, 16, 16]
  ↓ bottleneck（可加 self-attention）
  [B, 256, 16, 16]
  ↓ UNet decoder（上采样 + skip connection，恢复空间分辨率）
  [B, 128, 32, 32]
  [B,  64, 64, 64]
  ↓ 输出头（1×1 卷积，把通道数变回 C=3）
输出：[B, 3, 64, 64]（预测的噪声 ε̂，与输入同形状）
```

---

## 4. 从图像到文本：最大的区别在哪里

图像是**连续空间**里的信号，像素值是实数，加高斯噪声天然合法。

文本是**离散空间**——token id 是整数（比如"猫"对应 id=3421），**你不能在两个整数之间做线性插值**。想象一下 `"猫" × 0.3 + "狗" × 0.7`——这在整数空间毫无意义。

这就是文本 diffusion 最核心的挑战。目前主要有两条技术路线：

| | 图像 Diffusion | 连续文本 Diffusion | 离散 Masked Diffusion |
|---|---|---|---|
| 操作对象 | 像素值（连续） | token embedding（连续） | token id（离散） |
| 噪声类型 | 高斯噪声 | 高斯噪声 | Mask / 替换 |
| 网络结构 | UNet | Transformer | Transformer |
| 输出 | 预测噪声 $\hat\epsilon$，形状同输入 | 预测噪声（或 $x_0$），形状同输入 | 每个位置的 token logits |
| 损失函数 | MSE | MSE | Cross Entropy |
| 解码方式 | 直接读像素值 | 找最近邻 token | argmax |

---

## 5. 连续空间文本 Diffusion（Embedding Space）

### 5.1 思路：在 embedding 上加噪声

既然 token id 是离散的，加不了噪声，那就把每个 token 先转换成连续的 embedding 向量，然后在 embedding 空间里做扩散。

```
输入序列（token ids）：[B, L]
      ↓ embedding 层（查表，每个 id 变成 d 维向量）
token embeddings：    [B, L, d_model]   ← 在这里做 diffusion
      ↓ 加噪 / 去噪（T 步）
去噪后的 embeddings：  [B, L, d_model]
      ↓ 解码回 token id（见 5.5 节）
输出序列（token ids）：[B, L]
```

**注意 embedding 归一化**：训练时通常把 embedding 归一化到单位球面（$\|x_0\| = 1$），这样能让信号和噪声的量级相当——否则如果 embedding 的模很大，噪声占比会很小，信噪比失衡，训练不稳定。

### 5.2 前向加噪（文本版）：数学和代码对照

文本版的前向过程数学公式和图像版**完全相同**，只是维度从 `[B, C, H, W]` 变成了 `[B, L, d]`：

$$q(x_t \mid x_0) = \mathcal{N}(x_t;\; \sqrt{\bar\alpha_t}\, x_0,\; (1-\bar\alpha_t) I)$$
$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

这里 $x_0 \in \mathbb{R}^{L \times d}$ 是归一化后的 token embedding 矩阵，$\epsilon \in \mathbb{R}^{L \times d}$ 是同形状的高斯噪声。序列中每个位置、每个维度独立地加噪声。

```python
def forward_diffuse_text(x0, t, alpha_bar):
    # x0: [B, L, d]，归一化的 token embeddings
    # t:  [B]
    eps = torch.randn_like(x0)              # [B, L, d]，ε ~ N(0,I)
    ab  = alpha_bar[t][:, None, None]       # [B, 1, 1]，广播到 [B, L, d]
    # 公式直接翻译：x_t = √ᾱ_t * x0 + √(1-ᾱ_t) * ε
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    return x_t, eps   # x_t: [B, L, d]，形状始终不变
```

> **图像 vs 文本**：形状的区别是 `[B, C, H, W]` vs `[B, L, d]`。图像的空间结构是二维的（H×W），文本的是一维序列（L）。加噪公式和 $\alpha/\beta$ 的含义完全一样，只是张量维度不同。

### 5.3 去噪网络：为什么换成 Transformer

图像 UNet 用局部卷积（每个像素只看周围的邻居）处理空间结构，而文本需要**全局依赖**——理解"猫"这个词的含义，需要看整句话的上下文，不是只看左右两个位置。

因此去噪网络 $\epsilon_\theta$ 换成 Transformer（BERT 风格的双向 Encoder），用 self-attention 让序列中每个位置都能看到所有其他位置。

**数学上**，去噪网络 $\epsilon_\theta(x_t, t)$ 的接口没有变：输入是加噪后的 $x_t$ 和时间步 $t$，输出是预测的噪声（形状与 $x_t$ 相同）。变的只是内部实现（从 UNet 换成 Transformer）。

```python
class TextDiffusionTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, vocab_size, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)   # token id → embedding
        self.pos_embed = nn.Embedding(max_len, d_model)  # 位置 embedding
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        # 输出仍是 embedding 空间（d_model 维），不是 vocab logits
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x_t, t):
        # x_t: [B, L, d]  加噪后的 embedding（连续值，不是 token id）
        # t:   [B]         时间步
        B, L, d = x_t.shape

        # 时间步编码：标量 t → d 维向量，广播加到序列每个位置
        t_emb = timestep_embedding(t, d)        # [B, d]
        t_emb = self.time_embed(t_emb)          # [B, d]，经过 MLP 变换
        t_emb = t_emb[:, None, :]               # [B, 1, d] → 广播到 [B, L, d]

        # 位置编码
        pos = torch.arange(L, device=x_t.device)
        p_emb = self.pos_embed(pos)[None]       # [1, L, d]

        # 融合：加噪 embedding + 时间信息 + 位置信息
        h = x_t + t_emb + p_emb                # [B, L, d]

        # 双向 self-attention：每个位置都能看到整个序列
        h = self.transformer(h)                 # [B, L, d]

        # 输出：预测噪声 ε̂（也在 embedding 空间，和 x_t 同维度）
        return self.out_proj(h)                 # [B, L, d]
```

### 5.4 训练（文本连续 diffusion）：数学和代码对照

**数学：**

$$x_0 = \text{normalize}(E[\text{token\_ids}]) \quad \in \mathbb{R}^{L \times d}$$
$$t \sim \text{Uniform}\{1,\ldots,T\}, \quad \epsilon \sim \mathcal{N}(0,I)$$
$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon$$
$$\mathcal{L} = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$$

**代码：**

```python
def train_step_text_continuous(token_ids, model, alpha_bar):
    # token_ids: [B, L]，整数 token id
    B, L = token_ids.shape

    # 步骤1：token id → embedding，并归一化（让量级和噪声匹配）
    x0 = model.embed(token_ids)              # [B, L, d]
    x0 = F.normalize(x0, dim=-1)            # 归一化到单位球面

    # 步骤2：随机时间步 + 加噪
    t = torch.randint(0, T, (B,))
    x_t, eps = forward_diffuse_text(x0, t, alpha_bar)   # [B, L, d]

    # 步骤3：预测噪声
    eps_pred = model(x_t, t)                # [B, L, d]

    # 步骤4：MSE 损失（和图像版完全相同的公式）
    loss = F.mse_loss(eps_pred, eps)
    return loss
```

### 5.5 推理 + 解码回 token：最近邻是怎么回事？

**去噪过程**（和图像版数学完全相同，只是维度不同）：

$$\mu_t = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\hat\epsilon\right), \quad x_{t-1} = \mu_t + \sqrt{\beta_t}\, z$$

```python
@torch.no_grad()
def text_diffusion_sample(model, B, L, d):
    x = torch.randn(B, L, d)    # 从噪声出发，x_T ~ N(0, I)

    for t in reversed(range(T)):
        t_batch = torch.full((B,), t)
        eps_pred = model(x, t_batch)         # [B, L, d]

        # DDPM 去噪公式（与图像版完全相同）
        a  = alphas[t]
        ab = alpha_bar[t]
        mu = (x - (1-a)/(1-ab).sqrt() * eps_pred) / a.sqrt()
        if t > 0:
            x = mu + betas[t].sqrt() * torch.randn_like(x)
        else:
            x = mu
    # x: [B, L, d]，最终去噪后的连续 embedding，但还不是 token id
    ...
```

**解码阶段：为什么要找最近邻？**

去噪完成后，我们得到的是 `[B, L, d]` 形状的连续向量——每个位置是 $d$ 维实数向量。但输出需要是离散的 token id（整数）。怎么从连续向量映射回 token id？

方法是：把这个向量和词表中**所有 token 的 embedding 比较相似度**，找最相近的那个 token。

$$\hat{w}_i = \arg\max_{v \in \text{Vocab}} \cos(x_i, e_v) = \arg\max_{v} \frac{x_i \cdot e_v}{\|x_i\| \|e_v\|}$$

归一化后，余弦相似度等价于点积，而"对所有 token 做点积取 argmax"正好就是一个**线性层 + argmax**：

```python
# all_embeddings: [V, d]，词表中所有 token 的 embedding 矩阵（即 Embedding 层的权重）
def decode_to_tokens(x, all_embeddings):
    # x:              [B, L, d]，去噪后的连续 embedding
    # all_embeddings: [V, d]

    # 分别归一化
    x_norm   = F.normalize(x, dim=-1)                  # [B, L, d]
    emb_norm = F.normalize(all_embeddings, dim=-1)     # [V, d]

    # 矩阵乘法：[B, L, d] × [d, V] = [B, L, V]
    # logits[b, i, v] = 第 b 个样本、第 i 个位置与第 v 个 token 的余弦相似度
    logits = x_norm @ emb_norm.T                       # [B, L, V]

    # 对每个位置，取相似度最高的 token id
    token_ids = logits.argmax(dim=-1)                  # [B, L]
    return token_ids
```

**这个操作一点都不复杂**——它本质上和普通语言模型最后一层（`lm_head`）的操作是一样的：都是把隐状态向量和词表做矩阵乘法，找分数最高的 token。区别只是：普通 LM 的 `lm_head` 是一个**单独训练的投影矩阵**，而这里直接复用了 **embedding 矩阵本身**（因为加噪/去噪都在 embedding 空间进行，所以解码时也用同一套坐标系）。

**为什么不直接用 softmax 采样？** 也可以。`logits.softmax(-1)` 之后用 `multinomial` 采样，效果通常更好（保留多样性），而不是每次都取 argmax（会导致重复和退化输出）。

**"最近邻解码"有没有问题？** 有，这是连续文本 diffusion 的已知弱点：
1. **Rounding Problem（取整问题）**：去噪得到的连续向量可能处于 embedding 空间的"夹缝"里，和任何真实 token 都不够近，强行取最近邻会产生错误。这也是很多论文直接改用**离散 masked diffusion** 的原因。
2. **计算量**：词表 $V$ 通常有 3~5 万个 token，每次解码要和所有 token 计算相似度，算一个 `[B, L, V]` 的矩阵，计算量不小（虽然可以用矩阵乘法高效实现）。

---

## 6. 离散 Masked Diffusion（更主流）

连续 embedding diffusion 存在"rounding problem"：embedding 空间的 MSE 损失和最终的 token 准确率对不上，训练信号不够直接。

**Masked Diffusion**（MDLM、Absorbing Diffusion 等）把 diffusion 直接做在 token 空间：用 `[MASK]` 替换 token，而不是加高斯噪声。这避免了连续空间和离散 token 之间的转换问题。

### 6.1 前向过程（Masking）：数学和代码对照

**数学**：在时刻 $t$，每个位置独立地以概率 $1-\bar\alpha_t$ 被替换为 `[MASK]`：

$$q(x_t^{(i)} \mid x_0^{(i)}) = \begin{cases} x_0^{(i)} & \text{以概率 } \bar\alpha_t \\ \texttt{[MASK]} & \text{以概率 } 1-\bar\alpha_t \end{cases}$$

其中 $x_t^{(i)}$ 表示第 $i$ 个位置的 token。同样，$\bar\alpha_t$ 越小（$t$ 越大），被 mask 的比例越高。

```
t=0（原始）：  ["The", "cat",  "sat",  "on",  "mat"]     → ᾱ_t ≈ 1，保留所有
t=T/3：       ["The", "[M]",  "sat",  "[M]", "mat"]      → ᾱ_t ≈ 0.6，保留 60%
t=2T/3：      ["[M]", "[M]",  "sat",  "[M]", "[M]"]      → ᾱ_t ≈ 0.2，保留 20%
t=T（全噪声）：["[M]", "[M]", "[M]",  "[M]", "[M]"]      → ᾱ_t ≈ 0，全 mask
```

**代码：**

```python
def mask_forward(token_ids, t, alpha_bar, mask_id):
    # token_ids: [B, L]，原始 token id（整数）
    # 返回：x_t: [B, L]，部分位置被替换为 mask_id
    ab = alpha_bar[t][:, None]                          # [B, 1]，广播到 [B, L]
    # 每个位置以概率 ᾱ_t 保留（1 = 保留，0 = mask）
    keep = torch.bernoulli(ab.expand_as(token_ids.float())).bool()  # [B, L]
    x_t = token_ids.clone()
    x_t[~keep] = mask_id                                # 未保留的位置 → [MASK]
    return x_t                                          # [B, L]，dtype=long
```

### 6.2 去噪网络：BERT 风格，输出是 logits

离散 diffusion 的网络接口和连续版本有根本区别：

- **连续版**：输入 `[B, L, d]`（连续 embedding），输出 `[B, L, d]`（预测的噪声，也是连续 embedding）
- **离散版**：输入 `[B, L]`（含 MASK 的 token id 序列），输出 `[B, L, V]`（每个位置的 token 概率 logits）

```python
class MaskedDiffusionModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.time_mlp  = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers)
        # 关键区别：输出是词表大小的 logits，不是 embedding
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x_t, t):
        # x_t: [B, L]  含 MASK 的 token id 序列（整数）
        # t:   [B]
        B, L = x_t.shape

        h = self.embed(x_t)                              # [B, L, d]，id 转 embedding
        h = h + self.pos_embed(torch.arange(L))[None]   # [B, L, d]，加位置编码

        t_emb = self.time_mlp(timestep_embedding(t, d_model))  # [B, d]
        h = h + t_emb[:, None, :]                        # [B, L, d]，加时间信息

        h = self.transformer(h)                          # [B, L, d]，双向 attention
        logits = self.out_proj(h)                        # [B, L, V]，每位置的 token 分布
        return logits
```

### 6.3 训练（Masked Diffusion）：数学和代码对照

**数学**：损失只在被 mask 的位置计算，让网络预测被遮住的原始 token：

$$\mathcal{L} = -\sum_{i: x_t^{(i)} = \texttt{[MASK]}} \log p_\theta(x_0^{(i)} \mid x_t, t)$$

这和 BERT 的 Masked Language Modeling（MLM）损失几乎一模一样！区别只是：BERT 固定 mask 15% 的位置，而 masked diffusion 的 mask 比例由时间步 $t$（即 $1-\bar\alpha_t$）控制。

```python
def train_step_masked(token_ids, model, alpha_bar, mask_id):
    # token_ids: [B, L]
    B, L = token_ids.shape

    # 步骤1：随机时间步 + 按比例 mask
    t = torch.randint(0, T, (B,))
    x_t = mask_forward(token_ids, t, alpha_bar, mask_id)    # [B, L]

    # 步骤2：网络预测每个位置的 token 概率
    logits = model(x_t, t)                                  # [B, L, V]

    # 步骤3：只在被 mask 的位置计算损失（预测原始 token）
    mask = (x_t == mask_id)                                 # [B, L]，bool
    loss = F.cross_entropy(
        logits[mask],       # [N_masked, V]，被 mask 位置的预测
        token_ids[mask],    # [N_masked]，被 mask 位置的真实 token
    )
    return loss
```

### 6.4 推理（Masked Diffusion）

**数学**：从全 MASK 序列出发，每一步用反向去噪分布 $p_\theta(x_{t-1} \mid x_t)$ 逐步揭开 MASK。具体策略：在 $t \to t-1$ 这一步，有比例为 $\frac{\bar\alpha_{t-1} - \bar\alpha_t}{1 - \bar\alpha_t}$ 的 MASK 位置被填上（从模型预测的分布中采样），其余 MASK 位置继续保持 MASK。

```python
@torch.no_grad()
def masked_diffusion_sample(model, B, L, mask_id, vocab_size):
    x = torch.full((B, L), mask_id)     # 全 MASK 序列，[B, L]

    for t in reversed(range(T)):
        t_batch = torch.full((B,), t)
        logits = model(x, t_batch)       # [B, L, V]

        # 对所有位置采样（但只对 MASK 位置生效）
        probs   = logits.softmax(dim=-1)                        # [B, L, V]
        sampled = torch.multinomial(probs.view(-1, vocab_size), 1).view(B, L)

        still_masked = (x == mask_id)   # [B, L]，当前还是 MASK 的位置

        if t > 0:
            # 这一步要揭开的比例：ᾱ_{t-1} / ᾱ_t（每步只揭开一点点）
            # 剩余不揭开的比例：1 - ᾱ_{t-1} / ᾱ_t
            remask_prob = 1 - alpha_bar[t-1] / alpha_bar[t]
            # 以概率 (1 - remask_prob) 揭开 MASK，以概率 remask_prob 继续保持 MASK
            reveal = torch.bernoulli(
                torch.full_like(sampled.float(), 1 - remask_prob)
            ).bool()
            x = torch.where(still_masked & reveal, sampled, x)
        else:
            # 最后一步：把所有剩余 MASK 全部填上
            x = torch.where(still_masked, sampled, x)

    return x   # [B, L]，完整的 token 序列
```

**对比连续版解码**：离散版的解码无需"最近邻查找"，输出直接就是 logits，`argmax` 或 `multinomial` 一步得到 token id，干净利落。这是离散 diffusion 的主要优势。

---

## 7. 三条路线的维度总对比

```
【图像 DDPM】
  训练输入：
    x0        [B, C, H, W]   真实图像，像素 ∈ [-1, 1]
    eps       [B, C, H, W]   标准高斯噪声
    x_t       [B, C, H, W]   加噪图像（由 x0 和 eps 计算得到）
    t         [B]             时间步
  网络：UNet(x_t, t) → eps_pred [B, C, H, W]
  损失：MSE(eps_pred, eps)
  推理：x_T [B,C,H,W] → T步DDPM公式 → x_0 [B,C,H,W]

【连续文本 Diffusion】
  训练输入：
    token_ids  [B, L]         token id 序列
    x0         [B, L, d]      归一化 token embeddings
    eps        [B, L, d]      标准高斯噪声
    x_t        [B, L, d]      加噪 embedding
    t          [B]
  网络：Transformer(x_t, t) → eps_pred [B, L, d]
  损失：MSE(eps_pred, eps)
  推理：x_T [B,L,d] → T步DDPM → x_0 [B,L,d] → nearest neighbor → [B,L]

【离散 Masked Diffusion】
  训练输入：
    token_ids  [B, L]         原始 token id（整数，long）
    x_t        [B, L]         部分 masked 的 token id（整数，long）
    t          [B]
  网络：Transformer(x_t, t) → embed → [B,L,d] → out_proj → logits [B, L, V]
  损失：CrossEntropy(logits[mask], token_ids[mask])
  推理：全MASK [B,L] → T步逐步揭开 → [B,L]（完整序列）
```

---

## 8. 和 AR（自回归）LLM 的本质区别

GPT 这类自回归模型：**从左到右逐 token 生成，每个 token 只能看到左边的上下文**。

文本 Diffusion：**并行生成整个序列，每个位置可以看到全局上下文**（双向 attention）。

| | 自回归 LLM | 文本 Diffusion |
|---|---|---|
| 生成方式 | 左到右，串行 | 全局并行，多步去噪 |
| Attention | 单向（causal mask） | 双向（全局可见） |
| 推理步数 | = 序列长度 L | = 去噪步数 T（通常 10~1000） |
| 条件建模 | 直接：prefix → next token | 间接：condition embedding 注入 |
| 编辑能力 | 差（需要重新生成） | 强（可以只对部分位置去噪） |

文本 diffusion 的最大优势：**全局一致性更好**，**可编辑性强**（局部重新去噪）。劣势：推理慢（需要多步），质量目前仍不如 AR 大模型。

---

## 9. 条件生成：怎么做文生图、文生文

实际应用中需要**条件生成**：根据 prompt 生成图像或文本。条件信息通过 cross-attention 注入去噪网络。

**数学**：在去噪网络中，每一层先做 self-attention（序列内部交互），再做 cross-attention（与条件交互）：

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- Self-attention：$Q, K, V$ 均来自噪声序列 $x_t$（序列自身的全局上下文）
- Cross-attention：$Q$ 来自 $x_t$，$K, V$ 来自条件 $c$（将条件信息融入每个位置）

```python
class ConditionalDiffusionLayer(nn.Module):
    def forward(self, x_t, condition, t_emb):
        # x_t:       [B, L, d]    噪声序列（query 来源）
        # condition: [B, Lc, d]   文本 encoder 输出（Lc 可以不等于 L）
        # t_emb:     [B, d]       时间步 embedding

        h = x_t + t_emb[:, None, :]      # [B, L, d]，注入时间信息

        # 1. Self-attention：序列内部的全局交互
        #    Q = K = V 均来自 h，输出形状仍为 [B, L, d]
        h = self.self_attn(h, h, h)       # [B, L, d]

        # 2. Cross-attention：从条件中提取信息
        #    Q 来自 h，K/V 来自 condition
        #    Q: [B, L, d_head]，K/V: [B, Lc, d_head]
        #    attn(Q, K^T): [B, L, Lc]（每个噪声位置对所有条件位置的注意力权重）
        #    乘以 V: [B, L, d_head] → 融入条件信息
        h = self.cross_attn(h, condition, condition)  # [B, L, d]

        h = self.ffn(h)                   # [B, L, d]
        return h
```

**维度变化**：
```
x_t       [B, L, d]     噪声序列
condition [B, Lc, d]    文本 condition（Lc 可以不等于 L）

cross-attention:
  Q = h @ W_Q           [B, L,  d_head]
  K = condition @ W_K   [B, Lc, d_head]
  V = condition @ W_V   [B, Lc, d_head]
  score = Q @ K^T       [B, L, Lc]         ← L 个噪声位置 × Lc 个条件 token
  out = score @ V       [B, L, d_head]     ← 每个噪声位置融入条件信息
  输出: [B, L, d]       ← 序列长度不变，每个位置都吸收了 condition
```

---

## 10. 完整流程总结

**图像 Diffusion 训练**：
```
随机 x0 [B,C,H,W] → 随机 t → 加噪得 x_t → UNet 预测 eps → MSE 损失
```

**图像 Diffusion 推理**：
```
x_T [B,C,H,W]（纯噪声）→ T步去噪（每步 UNet 预测 eps，DDPM公式更新 x）→ x_0
```

**文本连续 Diffusion 训练**：
```
token_ids [B,L] → embed+normalize → x0 [B,L,d] → 随机 t → 加噪 x_t
                → Transformer 预测 eps [B,L,d] → MSE 损失
```

**文本连续 Diffusion 推理**：
```
x_T [B,L,d]（纯噪声）→ T步DDPM去噪 → x_0 [B,L,d] → nearest neighbor → [B,L]
```

**文本 Masked Diffusion 训练**：
```
token_ids [B,L] → 随机 t → mask部分token → x_t [B,L]
                → Transformer → logits [B,L,V] → 只在mask位置算 CE 损失
```

**文本 Masked Diffusion 推理**：
```
全MASK序列 [B,L] → T步逐步揭开（每步预测 logits，按概率揭开部分 MASK）→ 完整序列
```

三条路线的本质是一样的：**定义一个把数据破坏掉的过程，训练网络学会修复它**。不同的只是破坏方式（加高斯 vs mask）、网络结构（UNet vs Transformer）、以及输出形式（连续 embedding vs 离散 logits）。

---

> 如果这篇文章涉及的 Diffusion 和生成模型效率优化你想深入研究，可以看看我们团队出版的[《动手学 AutoML：从 NAS 到大语言模型优化实战》](https://item.jd.com/14945889.html)，书里 LLM 效率优化那章和 Diffusion 的加速思路有些呼应，感兴趣可以翻翻。
>
> ![动手学AutoML书籍封面](https://github.com/marsggbo/marsggbo.github.io/blob/master/assets/img/book_cover_automl.png?raw=true)
