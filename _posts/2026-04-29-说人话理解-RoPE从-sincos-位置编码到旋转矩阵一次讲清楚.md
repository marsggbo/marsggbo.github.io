---
layout: post
title: 说人话理解 RoPE：从 sin/cos 位置编码到旋转矩阵，一次讲清楚
date: '2026-04-29'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/2032881588019663675
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/2032881588019663675>

## **1. 前言**

你有没有过这种感觉——看了五六篇 RoPE 的解析文章，每篇都说"RoPE 利用了旋转矩阵来编码相对位置"，然后给你列一堆公式，最后贴一段 `rotate_half` 的代码，你点点头，好像懂了，但第二天回想起来，还是不知道那个旋转矩阵是从哪儿来的，也不明白代码里的 `chunk(2, dim=-1)` 到底在干什么。

我就是这样。

作为研究 LLM 推理效率的牛马，KV cache 管理、prefix sharing 这些东西几乎每天都要打交道，RoPE 的性质直接决定了 KV cache 能不能共享（不能共享就白搭）。所以最近花时间把这条线从头捋了一遍，从最朴素的 sin/cos 位置编码开始，一路推到 RoPE 的实现，争取把每个"为什么"都说清楚。

有几个问题在网上真的很难找到好答案：

* `sin` 和 `cos` 为什么要**交替配对**用？全用 `sin` 不行吗？
* position embedding 和 hidden dimension 到底是什么关系？维度那么多，为什么每个维度还要用不同的"频率"？
* RoPE 从 2D 扩展到 d 维的时候，到底发生了什么？很多文章直接就跳过去了
* 代码里的 `rotate_half` 为什么直接前后切两半，而不是像 sinusoidal 那样 sin/cos 间隔处理？

下面一个一个说清楚。

不过为了防止后面乱，先把本文用到的所有变量统一说明一下：

![](/assets/img/marsggbo/2026-04-29-说人话理解-RoPE从-sincos-位置编码到旋转矩阵一次讲清楚/0c682e69.jpg)

注：后面为了简洁，有时用 $d$ 统一表示 hidden dim 或 head dim，具体含义从上下文判断。

---

## **2. 问题的起点：Attention 根本不知道顺序**

先确认一件事：Attention 机制本身是**置换不变（permutation-invariant）的**。

说人话：给 Attention 层看 `[A, B, C]` 和看 `[C, A, B]`，如果没有位置编码，**它的输出完全一模一样**。

为什么？来直接算一遍。

假设序列只有 2 个 token，embedding 维度 $d=2$。两个 token 的 embedding 分别是：

![\\\mathbf{x}_A = [1, 0],\quad \mathbf{x}_B = [0, 1]](https://www.zhihu.com/equation?tex=%5C%5C%5Cmathbf%7Bx%7D_A+%3D+%5B1%2C+0%5D%2C%5Cquad+%5Cmathbf%7Bx%7D_B+%3D+%5B0%2C+1%5D)

为了简化，令 $W_Q = W_K = W_V = I$（单位矩阵），即 $Q = K = V = X$（直接用 embedding 自身）。

**情形 1：输入顺序是 `[A, B]`**

$$
\\X = \begin{bmatrix}1 &amp; 0 \\ 0 &amp; 1\end{bmatrix},\quad Q = K = V = \begin{bmatrix}1 &amp; 0 \\ 0 &amp; 1\end{bmatrix}
$$

注意力分数矩阵（缩放后）：

$$
\\\text{scores} = QK^T = \begin{bmatrix}1\cdot1+0\cdot0 &amp; 1\cdot0+0\cdot1 \\ 0\cdot1+1\cdot0 &amp; 0\cdot0+1\cdot1\end{bmatrix} = \begin{bmatrix}1 &amp; 0 \\ 0 &amp; 1\end{bmatrix}
$$

经过 softmax（按行）：

$$
\\\text{attn} = \text{softmax}\begin{bmatrix}1 &amp; 0 \\ 0 &amp; 1\end{bmatrix} = \begin{bmatrix}0.731 &amp; 0.269 \\ 0.269 &amp; 0.731\end{bmatrix}
$$

输出 $O = \text{attn} \cdot V$：

![\\O_A = 0.731 \cdot [1,0] + 0.269 \cdot [0,1] = [0.731,\ 0.269]](https://www.zhihu.com/equation?tex=%5C%5CO_A+%3D+0.731+%5Ccdot+%5B1%2C0%5D+%2B+0.269+%5Ccdot+%5B0%2C1%5D+%3D+%5B0.731%2C%5C+0.269%5D)

![\\O_B = 0.269 \cdot [1,0] + 0.731 \cdot [0,1] = [0.269,\ 0.731]](https://www.zhihu.com/equation?tex=%5C%5CO_B+%3D+0.269+%5Ccdot+%5B1%2C0%5D+%2B+0.731+%5Ccdot+%5B0%2C1%5D+%3D+%5B0.269%2C%5C+0.731%5D)

**情形 2：输入顺序是 `[B, A]`**

$$
\\X = \begin{bmatrix}0 &amp; 1 \\ 1 &amp; 0\end{bmatrix},\quad Q = K = V = \begin{bmatrix}0 &amp; 1 \\ 1 &amp; 0\end{bmatrix}
$$

注意力分数：

$$
\\\text{scores} = QK^T = \begin{bmatrix}0\cdot0+1\cdot1 &amp; 0\cdot1+1\cdot0 \\ 1\cdot0+0\cdot1 &amp; 1\cdot1+0\cdot0\end{bmatrix} = \begin{bmatrix}1 &amp; 0 \\ 0 &amp; 1\end{bmatrix}
$$

和情形 1 的分数矩阵完全一样！softmax 后也一样，输出是：

![\\O_B = 0.731 \cdot [0,1] + 0.269 \cdot [1,0] = [0.269,\ 0.731]](https://www.zhihu.com/equation?tex=%5C%5CO_B+%3D+0.731+%5Ccdot+%5B0%2C1%5D+%2B+0.269+%5Ccdot+%5B1%2C0%5D+%3D+%5B0.269%2C%5C+0.731%5D)

![\\O_A = 0.269 \cdot [0,1] + 0.731 \cdot [1,0] = [0.731,\ 0.269]](https://www.zhihu.com/equation?tex=%5C%5CO_A+%3D+0.269+%5Ccdot+%5B0%2C1%5D+%2B+0.731+%5Ccdot+%5B1%2C0%5D+%3D+%5B0.731%2C%5C+0.269%5D)

两种顺序下，$A$ 的输出都是 ![[0.731, 0.269]](https://www.zhihu.com/equation?tex=%5B0.731%2C+0.269%5D)，$B$ 的输出都是 ![[0.269, 0.731]](https://www.zhihu.com/equation?tex=%5B0.269%2C+0.731%5D)。**换个顺序输入，只是输出行的顺序变了，每个 token 对应的输出向量本身没有任何变化。**

这就是所谓的置换不变：Attention 不在乎你是先说 A 还是先说 B，它对每个 token 的处理方式取决于 token 的内容，和位置无关。

所以我们必须人为地把"位置"信息注入进去。问题是：**怎么注入，才算"注入对了"？**

---

## **3. 位置编码应该满足什么条件？**

在设计具体方案之前，先想清楚"好的位置编码"应该有什么性质。

最核心的一条：**位置 $m$ 的 query 向量 $\mathbf{q}_m$ 和位置 $n$ 的 key 向量 $\mathbf{k}_n$ 的内积，应该只依赖相对位置 $(m-n)$，而不依赖 $m$、$n$ 的绝对值。**

用公式写就是：

$$
\\\mathbf{q}_m \cdot \mathbf{k}_n = f(m - n)
$$

其中 $f$ 是某个只和差值 $(m-n)$ 有关的函数。

**为什么这个性质很重要？**

因为语言中"两个词的关系"取决于它们的相对距离，而不是它们在文章里的绝对位置。"他打了我"里"打"和"我"的关系，不会因为这句话出现在第 1 段还是第 100 段而改变。

有了这个目标，接下来的问题就变成了：怎么设计位置编码，让内积满足这个条件？

---

## **4. Sinusoidal 位置编码：sin 和 cos 为什么要搭配用？**

原始 Transformer（Vaswani et al., 2017）的方案是对每个位置索引 $t$（第几个 token）和维度索引 $i$ 计算一个标量，直接加到 token embedding 上：

$$
\\\text{PE}(t,\ 2i) = \sin(t \cdot \theta_i),\qquad \text{PE}(t,\ 2i+1) = \cos(t \cdot \theta_i),\qquad \theta_i = 10000^{-2i/d}
$$

偶数维度用 sin，奇数维度用 cos，两个维度共用同一频率 $\theta_i$，成对出现。

这个设计有两个很自然的问题：

* 为什么一个维度用 sin，另一个用 cos，不能全用 sin 吗？
* 为什么要每**两个**维度组成一对，而不是其他分组方式？

搞懂这两点之后，会有种豁然开朗的感觉，下面一一说清楚。

### 4.1 为什么一个用 sin，一个用 cos？

PE 是加到 token embedding 上再做投影的，注入 attention 后效果是什么？以位置 $s$ 的 token 和位置 $t$ 的 token 为例（注意：$s, t$ 是位置索引整数，不是 query/key 向量）：

$$
\\\tilde{\mathbf{x}}_s = \mathbf{x}_s + \text{PE}(s),\qquad \tilde{\mathbf{x}}_t = \mathbf{x}_t + \text{PE}(t)
$$

经过线性投影后计算 attention score：

$$
\\Q_s \cdot K_t = \underbrace{(W_Q \mathbf{x}_s) \cdot (W_K \mathbf{x}_t)}_{\text{语义交互}} + \underbrace{(W_Q \mathbf{x}_s) \cdot (W_K \text{PE}(t)) + (W_Q \text{PE}(s)) \cdot (W_K \mathbf{x}_t)}_{\text{语义 × 位置交叉项}} + \underbrace{(W_Q\,\text{PE}(s)) \cdot (W_K\,\text{PE}(t))}_{\text{位置 × 位置}}
$$

前三项涉及 $W_Q, W_K$ 的参数，模型在训练中可以灵活学习如何处理。**最后一项 $(W_Q\,\text{PE}(s)) \cdot (W_K\,\text{PE}(t))$ 完全由位置编码决定**——如果它只依赖相对距离 $(s-t)$，就给模型一个稳定的"位置锚点"。

为了分析最方便，令 $W_Q = W_K = I$（单位矩阵），此时这一项简化为 $\text{PE}(s) \cdot \text{PE}(t)$。

**先看全用 sin 会怎样**。两个位置的 PE 内积：

$$
\\\text{PE}_{\sin}(s) \cdot \text{PE}_{\sin}(t) = \sum_i \sin(s\theta_i)\sin(t\theta_i)
$$

用积化和差 ![\sin(a)\sin(b) = \frac{1}{2}[\cos(a-b) - \cos(a+b)]](https://www.zhihu.com/equation?tex=%5Csin%28a%29%5Csin%28b%29+%3D+%5Cfrac%7B1%7D%7B2%7D%5B%5Ccos%28a-b%29+-+%5Ccos%28a%2Bb%29%5D)：

![\\= \frac{1}{2}\sum_i \bigl[\cos\!\bigl((s-t)\theta_i\bigr) - \cos\!\bigl((s+t)\theta_i\bigr)\bigr]](https://www.zhihu.com/equation?tex=%5C%5C%3D+%5Cfrac%7B1%7D%7B2%7D%5Csum_i+%5Cbigl%5B%5Ccos%5C%21%5Cbigl%28%28s-t%29%5Ctheta_i%5Cbigr%29+-+%5Ccos%5C%21%5Cbigl%28%28s%2Bt%29%5Ctheta_i%5Cbigr%29%5Cbigr%5D)

里面有 $\cos((s-t)\theta_i)$（相对位置），**同时还有** $\cos((s+t)\theta_i)$（绝对位置之和）。两者都在，内积同时依赖相对和绝对位置，没办法给模型纯粹的"相对位置信号"。

**换成 sin + cos 配对**：

![\\\text{PE}(s) \cdot \text{PE}(t) = \sum_i \bigl[\sin(s\theta_i)\sin(t\theta_i) + \cos(s\theta_i)\cos(t\theta_i)\bigr] = \sum_i \cos\!\bigl((s-t)\theta_i\bigr)](https://www.zhihu.com/equation?tex=%5C%5C%5Ctext%7BPE%7D%28s%29+%5Ccdot+%5Ctext%7BPE%7D%28t%29+%3D+%5Csum_i+%5Cbigl%5B%5Csin%28s%5Ctheta_i%29%5Csin%28t%5Ctheta_i%29+%2B+%5Ccos%28s%5Ctheta_i%29%5Ccos%28t%5Ctheta_i%29%5Cbigr%5D+%3D+%5Csum_i+%5Ccos%5C%21%5Cbigl%28%28s-t%29%5Ctheta_i%5Cbigr%29)

利用 $\sin(a)\sin(b) + \cos(a)\cos(b) = \cos(a-b)$，**绝对位置完全消掉，只剩相对位置差 $(s-t)$**。

这就是 sin/cos 必须配对的根本原因：$\cos(a-b)$ 的恒等式展开式天然需要 sin 和 cos 各一个。全用 sin 缺少 cos 项，内积里就会多出 $\cos(s+t)$ 这个绝对位置污染项。

### 4.2 为什么要每两个维度一组，且每组频率不同？

上面的结论直接决定了分组方式：**一个 sin + 一个 cos 凑在一起，内积才能只依赖 $(s-t)$**，两者必须成对出现，不能单打独斗。

attention 的内积计算是逐维度累加的：

$$
\\Q_s \cdot K_t = \sum_{j=0}^{d-1} Q_s^{(j)} \cdot K_t^{(j)} = \underbrace{Q_s^{(0)} K_t^{(0)} + Q_s^{(1)} K_t^{(1)}}_{\text{第 0 对}} + \underbrace{Q_s^{(2)} K_t^{(2)} + Q_s^{(3)} K_t^{(3)}}_{\text{第 1 对}} + \cdots
$$

sinusoidal PE 的奇偶设计使得维度 $2i$ 和维度 $2i+1$ 天然成一对，其内积之和恰好构成 $\cos((s-t)\theta_i)$：

$$
\\\sin(s\theta_i)\sin(t\theta_i) + \cos(s\theta_i)\cos(t\theta_i) = \cos\!\bigl((s-t)\theta_i\bigr)
$$

**这就是"每两个维度一组"的根本来源**：内积的每一对加法项对应一个 $\cos(a-b)$ 恒等式，强制要求 sin 和 cos 必须成对落在相邻维度。

---

**那为什么 $d/2$ 对要各用不同的频率 $\theta_i$？**

想象极端情形：$d=64$，但全部 32 对都用同一频率 $\theta_0 = 1$。

* 位置 $t = 0$：![\text{PE} = [\sin 0, \cos 0, \sin 0, \cos 0, \ldots] = [0, 1, 0, 1, \ldots]](https://www.zhihu.com/equation?tex=%5Ctext%7BPE%7D+%3D+%5B%5Csin+0%2C+%5Ccos+0%2C+%5Csin+0%2C+%5Ccos+0%2C+%5Cldots%5D+%3D+%5B0%2C+1%2C+0%2C+1%2C+%5Cldots%5D)
* 位置 $t = 2\pi \approx 6.28$：![\text{PE} = [\sin 2\pi, \cos 2\pi, \ldots] = [0, 1, 0, 1, \ldots]](https://www.zhihu.com/equation?tex=%5Ctext%7BPE%7D+%3D+%5B%5Csin+2%5Cpi%2C+%5Ccos+2%5Cpi%2C+%5Cldots%5D+%3D+%5B0%2C+1%2C+0%2C+1%2C+%5Cldots%5D)

两个完全不同的位置，PE 向量**一模一样**——模型根本分不清。sin/cos 的值在 ![[-1, 1]](https://www.zhihu.com/equation?tex=%5B-1%2C+1%5D) 之间震荡，周期是 $2\pi / \theta_i$。同一频率意味着超过一个周期就会"碰撞"。

解决办法：让每对维度用不同的频率，就像时钟三根指针各自独立旋转：

![](/assets/img/marsggbo/2026-04-29-说人话理解-RoPE从-sincos-位置编码到旋转矩阵一次讲清楚/a982ff7d.jpg)

秒针（高频）区分秒级差距，时针（低频）区分小时级差距，三根指针组合才能唯一确定时刻。同理，高频维度对区分相邻 token，低频维度对感知长程位置，$d/2$ 对频率合在一起唯一表示任意位置。

**结论：$d$ 维 embedding 里有 $d/2$ 个计时器，每个计时器占 2 个维度（sin + cos 各一），每个计时器转速不同。sin/cos 必须配对，是 $\cos(a-b)$ 公式的要求；频率必须各异，是为了避免碰撞、覆盖所有距离范围。**

---

## **5. RoPE：不加，而是转**

Sinusoidal PE 把位置加到 embedding 再做投影，内积展开后有 4 项，位置和语义信息混在一起，相对位置的保证并不精确。苏剑林（苏大神，@苏建林）在 2021 年提出了 RoPE（Rotary Position Embedding），核心思路翻转：**不在 embedding 上加位置，直接对 Q 和 K 施加依赖于位置的旋转变换**。

直觉：如果位置 $m$ 的 Q 被旋转了 $m\theta$ 角度，位置 $n$ 的 K 被旋转了 $n\theta$ 角度，内积里自然包含角度差 $(m-n)\theta$——相对位置，而且是**精确地**只有相对位置，没有绝对位置的污染。

### 5.1 从 2D 推到 d_h 维

先把问题化到最简单：head dimension $d_h = 2$，query ![\mathbf{q} = [q_1, q_2]^T](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bq%7D+%3D+%5Bq_1%2C+q_2%5D%5ET)，key ![\mathbf{k} = [k_1, k_2]^T](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bk%7D+%3D+%5Bk_1%2C+k_2%5D%5ET)，只有一个频率 $\theta$。

RoPE 对位置 $m$ 的 query 做旋转：

$$
\\\mathbf{q}_m^{\text{rope}} = \begin{bmatrix}\cos m\theta &amp; -\sin m\theta \\ \sin m\theta &amp; \cos m\theta\end{bmatrix} \begin{bmatrix}q_1 \\ q_2\end{bmatrix} = \begin{bmatrix}q_1\cos m\theta - q_2\sin m\theta \\ q_1\sin m\theta + q_2\cos m\theta\end{bmatrix}
$$

这是平面几何里的**逆时针旋转**：把向量 $\mathbf{q}$ 旋转 $m\theta$ 弧度，模长不变。同样对位置 $n$ 的 key 旋转 $n\theta$，然后算内积（展开后利用 $\cos(a)\cos(b)+\sin(a)\sin(b)=\cos(a-b)$）：

$$
\\\langle \mathbf{q}_m^{\text{rope}},\ \mathbf{k}_n^{\text{rope}} \rangle = (q_1 k_1 + q_2 k_2)\cos\!\bigl((m-n)\theta\bigr) + (q_1 k_2 - q_2 k_1)\sin\!\bigl((m-n)\theta\bigr)
$$

只含 $(m-n)\theta$，**精确地只依赖相对位置**，绝对位置 $m, n$ 完全消掉了。

扩展到 $d_h$ 维也是同样的道理：把 $d_h$ 维 query 切成 $d_h/2$ 对，每对独立做一次 2D 旋转，各自用不同的频率 $\theta_i = 10000^{-2i/d_h}$：

$$
\\\text{第 } i \text{ 对} \quad (q_{2i+1}, q_{2i+2}) \xrightarrow{\;\text{旋转}\; m\theta_i\;} \begin{bmatrix}q_{2i+1}\cos m\theta_i - q_{2i+2}\sin m\theta_i \\ q_{2i+1}\sin m\theta_i + q_{2i+2}\cos m\theta_i\end{bmatrix}
$$

全部 $d_h/2$ 对旋转完拼在一起，等价于施加一个**块对角旋转矩阵** $R_{\Theta,m}$。内积性质对全部维度精确成立：

$$
\\\langle \mathbf{q}_m^{\text{rope}},\ \mathbf{k}_n^{\text{rope}} \rangle = \mathbf{q}^T R_{\Theta,m}^T R_{\Theta,n}\,\mathbf{k} = \mathbf{q}^T R_{\Theta,n-m}\,\mathbf{k}
$$

### 5.2 代码实现

实际代码不会构造那个稀疏块对角矩阵，而是把矩阵乘法等价分解成两次逐元素乘法。以 $d_h=4$、相邻配对为例展开：

$$
\\\mathbf{q}^{\text{rope}} = \begin{bmatrix}q_1 \\ q_2 \\ q_3 \\ q_4\end{bmatrix} \odot \begin{bmatrix}\cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1\end{bmatrix} + \begin{bmatrix}-q_2 \\ q_1 \\ -q_4 \\ q_3\end{bmatrix} \odot \begin{bmatrix}\sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1\end{bmatrix}
$$

"邻位交换取负"——把每对 $(q_{2i+1}, q_{2i+2})$ 变成 $(-q_{2i+2}, q_{2i+1})$——是 HuggingFace LLaMA 实现里的相邻配对方式。还有另一种等价实现：把整个向量**前半段和后半段**对应配对，维度 $i$ 和维度 $i + d_h/2$ 配成一对：

```python
import torch

def rotate_half(x):
    # x shape: [B, heads, T, d_h]
    # 前后半段配对：维度 i 和维度 i+d_h/2 配对旋转
    x1 = x[..., : x.shape[-1] // 2]   # 前半段
    x2 = x[..., x.shape[-1] // 2 :]   # 后半段
    return torch.cat([-x2, x1], dim=-1)  # [-后半, 前半]

def apply_rope(q, k, cos, sin):
    # cos/sin shape: [1, 1, T, d_h]
    return (q * cos) + (rotate_half(q) * sin), \
           (k * cos) + (rotate_half(k) * sin)
```

以 $d_h = 4$，位置 $m$ 为例展开：前半 `x1 = [q₁, q₂]`，后半 `x2 = [q₃, q₄]`，`rotate_half(q) = [-q₃, -q₄, q₁, q₂]`。cos 表为 `[cos(mθ₀), cos(mθ₁), cos(mθ₀), cos(mθ₁)]`，结果：$(q_1, q_3)$ 配对旋转 $m\theta_0$，$(q_2, q_4)$ 配对旋转 $m\theta_1$。和相邻配对只是分组方式不同，**数学性质完全一致**。

**为什么用前后切两半，而不像 sinusoidal 那样奇偶间隔？**

sinusoidal 的相邻配对（维度 $2i$ 和 $2i+1$）需要"隔一个取一个"，实现稍复杂。前后切两半只需要一次 chunk 操作，逻辑更清晰——因为频率查找表是把 $d_h/2$ 个频率直接重复两遍，前半段和后半段天然共享相同频率，只需做整块交换。

cos/sin 查找表的预计算：

```python
def build_rope_cache(head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
    # 频率：theta_i = base^(-2i/d_h)，共 d_h/2 个
    # i=0 -&gt; 最大频率（旋转最快），i=d_h/2-1 -&gt; 最小频率（旋转最慢）
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    positions = torch.arange(max_seq_len, dtype=torch.float)    # [max_len]
    freqs = torch.outer(positions, inv_freq)                     # [max_len, d_h/2]
    # freqs[t, i] = t * theta_i，即位置 t、第 i 对维度的旋转角度

    # 前后半段配对：把 d_h/2 个频率重复两遍 -&gt; [max_len, d_h]
    # 前 d_h/2 列 = 后 d_h/2 列，保证前后半段共享相同频率
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()
```

### 5.3 与 KV Cache 的配合

RoPE 和 KV cache 配合得天衣无缝。Prefill 阶段一次性处理 prompt，每个位置 $t$ 的 K 在生成时编码了位置 $t$ 的旋转并缓存。Decode 阶段新 token 在位置 `past_len`，只需对新的 Q、K 施加该位置的旋转：

```python
cos_table, sin_table = build_rope_cache(head_dim)

# Decode 阶段：past_len = 已缓存的 K 序列长度
cos = cos_table[past_len: past_len + 1].unsqueeze(0).unsqueeze(0)  # [1,1,1,d_h]
sin = sin_table[past_len: past_len + 1].unsqueeze(0).unsqueeze(0)
q_rope, k_rope_new = apply_rope(q, k_new, cos, sin)

# 新 K 追加到历史 KV cache，历史 K 保持不动
k_full = torch.cat([k_cache, k_rope_new], dim=2)
```

历史 K 不动（已编好各自位置的旋转），新 Q 旋转到当前位置，做 attention 时：

$$
\\\langle \mathbf{q}_m^{\text{rope}},\ \mathbf{k}_n^{\text{rope}} \rangle = f(\mathbf{q}, \mathbf{k},\ m - n)
$$

相对距离 $(m-n)$ 自动从旋转角度之差中来，**不需要对历史 K 重新编码，cache 可以直接复用**。这也是为什么 prefix sharing（前缀 KV 共享）在 RoPE 模型里能正确工作：只要前缀从位置 0 开始，其 KV 就和任何共享同一前缀的请求完全一致。

---

## **6. 对比总结**

![](/assets/img/marsggbo/2026-04-29-说人话理解-RoPE从-sincos-位置编码到旋转矩阵一次讲清楚/91328362.jpg)

**为什么 sin/cos 要配对用，不能全用 sin？** 因为 $\sin(a)\sin(b) + \cos(a)\cos(b) = \cos(a-b)$，配对后内积只剩相对位置差。全用 sin 的话，积化和差会多出 $\cos(a+b)$ 项，绝对位置混进来了。

**为什么 position embedding 跟维度索引 $i$ 也有关？** 每对维度充当不同频率的计时器，高频区分相邻 token，低频感知长程结构。如果所有维度用同一频率，不同位置的编码会碰撞。

**RoPE 从 2D 到 $d_h$ 维怎么扩展？** 把 $d_h$ 维切成 $d_h/2$ 对，每对独立做 2D 旋转，每对频率不同——就是 2D 的简单重复，合在一起是块对角矩阵。

**代码里的 `rotate_half` 为什么直接前后切两半？** 前后半段配对是一种等价的分组方式（维度 $i$ 和 $i+d_h/2$ 配对），`chunk(2, dim=-1)` + `cat([-x2, x1])` 比相邻配对实现更简洁，数学性质完全一样。

欢迎评论区讨论，有理解不对的地方也欢迎指出。