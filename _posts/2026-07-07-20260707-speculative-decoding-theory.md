---
layout: post
title: "投机解码到底能快多少？从理论建模到 closed-form 的完整推导"
date: 2026-07-07
tags: [LLM, 推理优化, 投机解码, 理论分析]
---

# 投机解码到底能快多少？从理论建模到 closed-form 的完整推导

---

## 1. 为什么要认真建模这件事

我最近在看 Speculative Decoding（投机解码）和 Self-Speculative Decoding（自投机解码，也叫 SSD）的相关工作，有一个感受：**大家都知道它快，但"快多少"这件事没有被认真量化过**。

图里的那个公式（我看到有人分享的参考）给了一个很漂亮的 closed form：

$$\text{Speedup} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}$$

其中 $\alpha$ 是逐 token 接受率，$\gamma$ 是草稿长度，$c$ 是草稿模型 vs. 验证（大）模型单 token 推理的计算成本比。

这个公式乍看很简洁，但我觉得它藏了不少假设，而且在 SSD（自投机解码）的场景下，这个建模需要做一些修正。今天我想从头把这两条路的性能建模捋一遍，看看各自的 closed form 长什么样，以及哪些因素会让理论和实际产生偏差。

---

## 2. 先把基准搞清楚：标准自回归的代价

标准自回归解码生成 $N$ 个 token，每次都要跑一次完整的大模型前向：

$$T_\text{baseline} = N \cdot t_\text{big}$$

其中 $t_\text{big}$ 是大模型生成一个 token 的时间（包含 prefill 之后的 decode latency，KV cache 命中的情况下可以近似为一个常数）。

这是我们的参照系，所有加速比都相对于它来算。

---

## 3. 投机解码（Speculative Decoding，SD）的性能建模

### 3.1 一轮的结构

经典 SD 的一轮操作是：

1. **Draft 阶段**：用小的 draft model 串行生成 $\gamma$ 个候选 token，耗时 $\gamma \cdot t_\text{draft}$
2. **Verify 阶段**：把这 $\gamma$ 个 token 送给大模型（target model）做一次**并行**前向，耗时约 $t_\text{big}(\gamma)$

注意第 2 步不是 $\gamma \cdot t_\text{big}$，因为大模型可以对 $\gamma$ 个 token 同时做 prefill 式的并行计算。但也不完全等于 $t_\text{big}(1)$，因为序列长了 attention 和 KV cache 的 overhead 会稍微变大。

**简化假设**（也是上面那个公式的隐含假设）：
- 大模型验证 $\gamma$ 个 token 的时间 $\approx t_\text{big}$（一次 decode cost），即序列长度对单次 forward 耗时影响忽略不计
- 令 $c = t_\text{draft} / t_\text{big}$（草稿比）

那一轮的总耗时：

$$T_\text{round} = \gamma \cdot t_\text{draft} + t_\text{big} = (\gamma c + 1) \cdot t_\text{big}$$

### 3.2 一轮能接受多少 token？

设逐 token 接受率为 $\alpha$（独立同分布假设）。若生成草稿序列为 $d_1, d_2, \ldots, d_\gamma$：

- $d_1$ 被接受的概率 $= \alpha$
- $d_1, d_2$ 都被接受的概率 $= \alpha^2$
- ...
- 全部 $\gamma$ 个都被接受的概率 $= \alpha^\gamma$

每轮接受 token 数的期望（别忘了最后大模型还会贡献一个新 token）：

$$\mathbb{E}[\#\text{tokens per round}] = \sum_{k=0}^{\gamma} \alpha^k = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

推导：设 $X$ = 接受的 token 数（从 0 到 $\gamma$），最后大模型总会产出一个 token，所以：

$$\mathbb{E}[\#] = 1 + \sum_{k=1}^{\gamma} P(\text{前 } k \text{ 个都接受}) = 1 + \alpha + \alpha^2 + \cdots + \alpha^\gamma = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

### 3.3 Speedup 的 closed form

每轮耗时 $(\gamma c + 1) \cdot t_\text{big}$，产出 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 个 token。

标准自回归产出同样数量 token 需要 $\frac{1-\alpha^{\gamma+1}}{1-\alpha} \cdot t_\text{big}$。

因此加速比为：

$$\boxed{\text{Speedup}_\text{SD} = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}}$$

这就是图里的那个公式，完整推导到这里。

### 3.4 这个公式说了什么？

对 $\gamma$ 求导，令 $\partial \text{Speedup} / \partial \gamma = 0$，可以找到最优草稿长度 $\gamma^*$。这个导数不太好直接解析出来，但有几个定性结论值得记住：

**结论 1**：$c$ 越小（草稿模型越轻量），最优 $\gamma^*$ 越大，加速上限越高。这就是图里左右两张图的直觉：$c=0.05$ 时 $\gamma^* \approx 8$，$c=0.0005$ 时 $\gamma^*$ 更大且加速比可以到 20x。

**结论 2**：当 $\gamma \to \infty$ 时，若 $\alpha < 1$，Speedup $\to 0$。草稿越长，浪费的验证成本越高，最终得不偿失。

**结论 3**：$\alpha$ 对加速的影响是非线性的。$\alpha$ 从 0.9 到 0.95 的提升，带来的加速远比 0.7 到 0.75 大。原因是 $\alpha^\gamma$ 在 $\alpha$ 接近 1 时衰减很慢，能撑住更长的草稿序列。

---

## 4. 自投机解码（Self-Speculative Decoding，SSD）

### 4.1 SSD 和 SD 的本质区别

SSD 的思路是：**不用另一个小模型，直接用大模型自己的 skip layers 来当 draft**。

最简单的实现是把大模型的前 $L_d$ 层作为 draft 网络（contiguous prefix），前向到第 $L_d$ 层就直接 decode，生成 $\gamma$ 个草稿 token；然后跑完整的 $L$ 层做验证。但更一般的变体（如 LayerSkip、Draft&Verify 等）会跳过中间某些层，形成**非连续的 skip 模式**，这两种情况的 KV cache 复用行为截然不同——这也是建模里最容易踩坑的地方。

好处是：
- 不需要维护两套权重，内存 footprint 比 SD 低
- draft 和 verify 共享词表，分布天然对齐
- 在显存受限场景下，省出来的内存可以换更大的 batch size

代价是：draft 不再是独立模型，接受率 $\alpha$ 往往低于精心选型的 draft model。

### 4.2 KV cache 复用的前提：激活路径必须一致

这里有个经常被忽视的细节，值得单独强调。

**KV cache 能被 verify 阶段复用，当且仅当对应层的中间激活（hidden state）在 draft 和 verify 路径下完全相同。**

对于 contiguous prefix（取前 $L_d$ 层）：
- draft 路径：层 $1 \to 2 \to \cdots \to L_d$，激活 $h_1, h_2, \ldots, h_{L_d}$
- verify 路径：层 $1 \to 2 \to \cdots \to L_d \to L_d+1 \to \cdots \to L$，前 $L_d$ 层的激活与 draft **完全相同**

所以 verify 阶段可以直接复用 draft 计算过的 KV，只需从第 $L_d+1$ 层开始补算。这是原始 SSD 论文里的场景。

**但如果 skip 的是非连续的层，情况就不一样了。**

假设 draft 跳过了第 $s$ 层（$s < L_d$），那么：
- draft 路径里，第 $s+1$ 层的输入是第 $s-1$ 层的输出（跳过了 $s$）
- verify 路径里，第 $s+1$ 层的输入是第 $s$ 层的输出（没有跳过）

两者输入不同，所以第 $s+1$ 层及其之后所有层的 KV 值均不匹配。**一旦某层被跳过，其后所有层的 KV cache 全部作废，无法给 verify 复用。**

### 4.3 一般化建模：$r_\text{draft}$ 和 $r_\text{reuse}$ 拆开

针对一般的 skip 模式，需要引入两个独立参数：

- $r_\text{draft} = |\mathcal{A}| / L$：draft 实际参与计算的层数比例（$\mathcal{A}$ 是 active 层的集合）
- $r_\text{reuse}$：verify 阶段可以跳过（复用 KV）的层数比例

$r_\text{reuse}$ 的计算规则：设 $s_1 = \min(\{1,\ldots,L\} \setminus \mathcal{A})$ 是第一个被跳过的层，则：

$$r_\text{reuse} = \frac{s_1 - 1}{L}$$

即：只有第一个 skip 层**之前**的那些层，其 KV 才对 verify 有效。

几个特殊情况：
- **Contiguous prefix**（$\mathcal{A} = \{1, \ldots, L_d\}$，无跳过）：$s_1 = L_d + 1$，所以 $r_\text{reuse} = r_\text{draft} = r$
- **第一层就跳**（$1 \notin \mathcal{A}$）：$s_1 = 1$，$r_\text{reuse} = 0$，KV cache 完全无法复用
- **中间跳层**（如跳掉第 $s$ 层，$s < L_d$）：$r_\text{reuse} = (s-1)/L < r_\text{draft}$

一轮 SSD 总耗时的一般形式：

$$T_\text{round}^\text{SSD} = \gamma \cdot r_\text{draft} \cdot t_\text{big} + (1 - r_\text{reuse}) \cdot t_\text{big} = [\gamma r_\text{draft} + (1 - r_\text{reuse})] \cdot t_\text{big}$$

对应的 closed form：

$$\boxed{\text{Speedup}_\text{SSD-general} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)[\gamma r_\text{draft} + (1 - r_\text{reuse})]}}$$

这个公式涵盖了所有 SSD 变体：

| Skip 模式 | $r_\text{draft}$ | $r_\text{reuse}$ | 分母 | 退化到 |
|-----------|-----------------|-----------------|------|--------|
| Contiguous prefix | $r$ | $r$ | $1 + (\gamma-1)r$ | 原始 SSD 公式 |
| 完全不 skip（$r_\text{draft}=1$） | $1$ | $1$ | $\gamma$ | 纯串行，无加速 |
| 首层即 skip | $r_\text{draft}$ | $0$ | $\gamma r_\text{draft} + 1$ | 和 SD 同构，$c = r_\text{draft}$ |
| 中间跳层 | $r_\text{draft}$ | $r_\text{reuse} < r_\text{draft}$ | 介于上两者之间 | — |

**关键结论**：非连续 skip 的 SSD，KV 复用收益会大幅缩水，当 $r_\text{reuse} \ll r_\text{draft}$ 时，其性能模型接近标准 SD（$c = r_\text{draft}$），KV 复用带来的额外优势几乎消失。

### 4.4 和 SD 的对比

并排放三个公式：

$$\text{Speedup}_\text{SD} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}$$

$$\text{Speedup}_\text{SSD-contiguous} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)[1 + (\gamma-1)r]}$$

$$\text{Speedup}_\text{SSD-general} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)[\gamma r_\text{draft} + (1 - r_\text{reuse})]}$$

Contiguous SSD 之所以分母更小（$1 + (\gamma-1)r$ vs $\gamma c + 1$，当 $c \approx r$ 时差了 $1 - r$ 项），正是 KV 复用的贡献。非连续 skip 一旦让 $r_\text{reuse}$ 接近 0，这个优势就彻底没了。

---

## 5. 把更多因素塞进来

上面的 closed form 都基于几个理想化假设。现实里有哪些因素会让这个模型失准？

### 5.1 验证阶段的 overhead 不是常数

我们假设大模型验证 $\gamma$ 个 token 的时间 $= t_\text{big}$（1 个 token 的时间）。但实际上：

$$t_\text{verify}(\gamma) = t_\text{big} \cdot f(\gamma)$$

其中 $f(\gamma) \geq 1$，且随 $\gamma$ 增大。原因：
- KV cache 长了，attention 的 memory bandwidth 开销增加
- batch size = $\gamma$ 的 prefill 比 decode 慢（decode 是 memory-bound，prefill 是 compute-bound，两者 kernel 调度不同）

一个简单的修正：

$$f(\gamma) \approx 1 + \epsilon \cdot (\gamma - 1)$$

其中 $\epsilon \ll 1$ 是单 token 增量开销。代入后分母变为：

$$\gamma c + f(\gamma) \approx \gamma c + 1 + \epsilon(\gamma-1) = (\gamma-1)(c + \epsilon) + c + 1$$

$\epsilon$ 通常很小（0.01-0.05 量级），但当 $\gamma$ 很大时不可忽略。

### 5.2 接受率 $\alpha$ 不是 i.i.d. 的

实际上第 $k$ 个 token 的接受率往往与前面的接受历史相关，而且不同位置的 $\alpha$ 本身不同。更精确的建模是：

$$\mathbb{E}[\#\text{tokens}] = 1 + \sum_{k=1}^{\gamma} \prod_{j=1}^{k} \alpha_j$$

当 $\alpha_j$ 单调递减（靠后位置的草稿越来越不准）时，最优 $\gamma^*$ 会比 i.i.d. 假设下更小。很多实验论文观察到接受率随位置衰减，这就是实际最优草稿长度往往不超过 8 的原因。

### 5.3 计算效率比 $c$ 在 batch 推理时会变

单请求场景下，$c \approx t_\text{draft} / t_\text{big}$ 主要由参数量决定（memory bandwidth bound 时近似线性于参数量之比）。但在 batch serving 场景：

- 大模型 batch size 增大时，吞吐趋向 compute-bound，$t_\text{big}$ 在 batch 变大时减慢更显著
- draft model 参数少，更容易在小 batch 下就达到 compute bound 上限
- 结果：$c_\text{batch} > c_\text{single}$，加速比下降

这是 batch serving 场景下投机解码加速效果不如单请求的核心原因。

### 5.4 SSD 的层比例 $r$ 和接受率 $\alpha$ 之间的 trade-off

SSD 里，$r_\text{draft}$ 越大（draft 用更多层），draft model 能力越强，$\alpha$ 也越高。但草稿成本也高了。这构成一个联合优化问题：

$$\text{Speedup}(r_\text{draft}, r_\text{reuse}, \gamma) = \frac{1 - \alpha(r_\text{draft})^{\gamma+1}}{(1-\alpha(r_\text{draft}))[\gamma r_\text{draft} + (1 - r_\text{reuse})]}$$

其中 $\alpha(r_\text{draft})$ 是 $r_\text{draft}$ 的单调递增函数，而 $r_\text{reuse}$ 由 skip 模式决定（见第 4 节）。最优 $(r^*, \gamma^*)$ 需要联合搜索，没有解析解，但 grid search 配合实测 $\alpha$ 可以很快找到。

---

## 6. SSD 的内存优势：能建模进加速比吗？

这个问题值得单独拎出来讨论，因为很多文章提到"SSD 省内存"但没有说清楚**省出来的内存到底有没有加速价值**。

### 6.1 省内存本身不直接加速单请求

先把这个坑踩清楚：在单请求场景下，SSD 省掉的 draft model 显存（大约 $r_\text{draft} \times M_\text{big}$，其中 $M_\text{big}$ 是大模型显存占用）对延迟**没有直接帮助**。省出来的显存闲着，加速比还是那个公式。

### 6.2 内存优势通过 batch size 转化为 throughput 提升

真正的价值在 serving 场景，尤其是 **KV cache 受限时**。

设 GPU 总显存为 $M_\text{total}$，模型权重占 $M_\text{weight}$，剩余显存全给 KV cache：

$$M_\text{kv} = M_\text{total} - M_\text{weight}$$

KV cache 的大小正比于 $B \times L_\text{seq} \times d_\text{kv} \times L$（batch size × 序列长度 × 头维度 × 层数），所以在相同硬件上能支撑的最大 batch size：

$$B_\text{max} \propto \frac{M_\text{kv}}{L_\text{seq} \cdot d_\text{kv} \cdot L}$$

对于 SD：需要额外维护 draft model 权重，$M_\text{weight}^\text{SD} = M_\text{big} + M_\text{draft}$

对于 SSD：只有大模型一套权重，$M_\text{weight}^\text{SSD} = M_\text{big}$

因此 SSD 能支撑的 batch size 更大：

$$\frac{B_\text{max}^\text{SSD}}{B_\text{max}^\text{SD}} = \frac{M_\text{total} - M_\text{big}}{M_\text{total} - M_\text{big} - M_\text{draft}}$$

这个比值在 $M_\text{draft}$ 较大时可以相当显著。举个例子：70B 大模型在 A100 80G 上量化后约占 35-40GB，如果 draft model 是 7B（约 3.5-4GB），释放这部分显存可以让 batch size 多出 ~10%。

### 6.3 Throughput 建模

把 batch size 优势代入 throughput：

$$\text{Throughput} = \frac{B \times \mathbb{E}[\#\text{tokens per round}]}{T_\text{round}}$$

定义 $\beta = B_\text{max}^\text{SSD} / B_\text{max}^\text{SD} \geq 1$，SSD 相对 SD 的 throughput 比：

$$\frac{\text{Throughput}_\text{SSD}}{\text{Throughput}_\text{SD}} = \beta \cdot \frac{\text{Speedup}_\text{SSD}}{\text{Speedup}_\text{SD}}$$

当 $\beta > 1$ 时（显存确实受限），SSD 的 throughput 优势 = 加速比的差距 + 显存带来的 batch 增益。

但注意：**batch 增大了，$\alpha$ 会下降**（不同请求的分布不同，草稿的命中率降低），这是一个负反馈。实际 $\beta$ 的有效价值需要在特定负载下实测。

### 6.4 SSD 在企业部署中适用吗？

直接给结论：**适用场景很窄，但确实存在。**

**SSD 相对 SD 有优势的场景**：
1. **显存极度受限**：单机多卡塞不下大模型 + draft model，但 SSD 能省出 draft model 的权重开销，换来更大的 KV cache 空间或更高的并发
2. **单请求低延迟优先**（如代码补全、交互式对话）：此时 batch size = 1，SSD 的 KV 复用优势最大，$\beta = 1$，纯靠 Speedup 公式的分母优势
3. **无合适 draft model**：目标模型是定制 fine-tuned 版本，没有对应的小模型；SSD 不需要配对的 draft model，开箱即用

**SSD 相对 SD 没有优势的场景**：
1. **大 batch 高吞吐 serving**：$\alpha$ 随 batch 增大而下降，SSD 的 $r_\text{draft}$ 带来的接受率本就低于专用 draft model，大 batch 下差距更大
2. **显存充裕**：$\beta \approx 1$，内存优势消失，SSD 只剩 KV 复用这一点，而这要求 skip 模式是 contiguous prefix，限制了 $\alpha$ 的上限
3. **对话历史长**：长序列时 KV cache 本身就很大，verify 的 attention overhead $f(\gamma)$ 增大，SSD 节省的验证成本被其他 overhead 稀释

一句话总结：**SSD 是"没有好 draft model 时的替代方案"，而不是"有了 SD 之后还要叠加的优化"**。企业部署如果有资源维护专用 draft model（比如 Llama-3-70B 配 Llama-3-8B），SD 的综合表现通常更好；反之，SSD 的零配置优势确实有价值。

---

## 7. SSD 的 CUDA Graph：连续 skip 好做，非连续 skip 怎么办？

这一节纯讲工程，和上面的理论建模互补。如果你只关心公式推导可以跳过，但如果你要真的实现 SSD，这里是踩坑最密的地方。

### 7.1 CUDA Graph 的本质约束

先说清楚 CUDA Graph（CG）是什么以及它为什么和 skip layer 有矛盾。

CG 的工作方式：先"录制"一次完整的 kernel 执行序列（包括 kernel 类型、参数地址、依赖关系），生成一个固定的 DAG，之后每次推理直接 replay 这个 DAG，绕过 CPU 端的 kernel launch overhead。

这带来了一个硬约束：**CG 捕获的是拓扑固定的 kernel 图，不允许图在 replay 时动态增删节点。**

skip layer 从计算图角度看就是"某些 transformer block 的 kernel 不执行"，一旦要运行时决定跳哪层，图的拓扑就在变——这和 CG 的设计正面冲突。

### 7.2 连续 skip（contiguous prefix）：没有问题

对于原始 SSD（取前 $L_d$ 层做 draft），draft 路径和 verify 路径都是拓扑固定的：

- Draft 路径：执行层 $1 \to 2 \to \cdots \to L_d$，固定 $L_d$ 个 block
- Verify 路径：执行层 $1 \to 2 \to \cdots \to L$，固定 $L$ 个 block

两条路径分别捕获成两个 CG，draft 阶段 replay draft-graph $\gamma$ 次，verify 阶段 replay verify-graph 1 次。整个生命周期里图的拓扑**从未改变**，CG 完全适用。

这也是为什么 vLLM / SGLang 对标准投机解码的 CG 支持相对完善——两个模型各自一套图，没有特殊处理。

### 7.3 固定的非连续 skip：同样没问题

如果 skip 模式是静态决定的（比如"永远跳第 4、8、12 层"），那虽然跳的不连续，但拓扑依然固定。捕获时只录制实际执行的 kernel，不录制被跳过的层，replay 就按这个固定子图走。

唯一需要注意的是 **residual connection 的处理**：跳过第 $l$ 层时，第 $l+1$ 层的输入必须直接连到第 $l-1$ 层的输出（identity pass-through），这通常用一个 no-op copy kernel 或直接共享 tensor 指针来实现，不影响 CG 捕获。

所以：**连续 skip 和固定非连续 skip，CG 都没问题，本质上是同一类情况——图拓扑是静态的。**

### 7.4 动态 skip：这才是真正的麻烦

真正和 CG 冲突的是**运行时自适应决定跳哪层**，典型场景是基于 token confidence 的 early exit：

```python
# 伪代码：动态 early exit，每个 token 可能在不同层退出
for layer_idx in range(L):
    hidden = transformer_block[layer_idx](hidden)
    if confidence(hidden) > threshold:  # 运行时判断，拓扑在变！
        break  # 提前退出
```

每次前向退出的层数不同，图的拓扑每次都可能不一样，CG 无法直接捕获。

### 7.5 解法 1：Topology-Preserving No-op（主流做法）

思路：**不改变图的拓扑，改变数据流**。

把"跳过某层"改成"这层做一个 identity 操作"：

```python
# 改造后：拓扑固定，但通过 mask 控制层是否生效
for layer_idx in range(L):
    if skip_mask[layer_idx]:   # skip_mask 是 GPU 上的 tensor，不是 Python bool
        hidden = hidden        # identity，实现上是零乘或直接传递
    else:
        hidden = transformer_block[layer_idx](hidden)
```

关键是 `skip_mask` 必须是 **GPU tensor**，不能是 Python 控制流——Python `if` 会让 CG 捕获时走固定分支，replay 时永远走同一条路。

具体实现方式有两种：

**方式 A：乘以 0/1 mask（最简单但有额外计算）**

```python
output = transformer_block[layer_idx](hidden)
hidden = hidden + skip_mask[layer_idx] * (output - hidden)
# skip_mask[i] = 0 → hidden 不变（跳过）
# skip_mask[i] = 1 → hidden = output（正常执行）
```

坏处：即使跳过，transformer block 的 forward 还是跑了一遍，只是最后结果被 mask 掉，**计算资源没有节省**。

**方式 B：cudaStreamWaitEvent + conditional copy（细粒度控制）**

只 launch 需要执行的 kernel，但用 CUDA event 保持图的 DAG 结构完整，跳过的层用 memcpy no-op 保持依赖边。这种方式能真正节省被跳层的计算，但实现复杂，需要手动管理 event。

vLLM 对类似场景（MoE 的 expert routing）的处理和方式 B 思路接近，但具体实现会根据 backend 有所不同。

### 7.6 解法 2：Multi-Graph 预捕获（适合枚举可控的场景）

如果 skip 模式的数量有限（比如只有几种固定的 skip 配置），可以预先把所有可能的图都捕获好，运行时根据当前 skip 配置选对应的图 replay：

```python
# 离线预捕获
graphs = {}
for skip_config in all_skip_configs:
    graphs[skip_config] = capture_cuda_graph(model, skip_config)

# 在线 replay
current_config = decide_skip_config(...)  # 可以是 CPU 逻辑
graphs[current_config].replay()
```

这个方案的优势：每个图都只包含实际执行的 kernel，**没有方式 A 的无效计算**。

限制：skip_config 的枚举空间不能太大。$L=32$ 层的模型如果每层都能独立跳，理论上有 $2^{32}$ 种配置，显然无法全部预捕获。但如果限制"只能在 4 个固定位置的层选一个退出"，配置数就是 4，完全可以。

这也是为什么很多实际 SSD 实现会把 skip 模式约束成离散的几档（比如退出层只能是 $L/4, L/2, 3L/4, L$），既能做 CG，又给了 $r$ 一定的调节自由度。

### 7.7 解法 3：Persistent CUDA Graph + pointer swap（适合 verify 阶段的复用）

这个方法专门针对 SSD 的一个特殊需求：verify 阶段需要跑完整 $L$ 层，但前 $L_d$ 层的权重和 draft 阶段完全一样，能不能在 verify 时复用 draft 阶段的 CG 节点？

答案是**不能直接复用图节点**，但可以通过 **tensor pointer swap** 实现 KV cache 的零拷贝共享：

1. Draft CG 捕获时，KV cache 写入到一个预分配的 buffer（地址 $P_\text{kv}$）
2. Verify CG 捕获时，前 $L_d$ 层的 KV cache 输入指针指向同一个 $P_\text{kv}$
3. Replay 时：先 replay draft CG 写 KV，再 replay verify CG 读 KV——中间不需要任何数据搬运

`cudaGraphExecKernelNodeSetParams` 允许在不重新捕获的情况下更新 kernel 参数（包括指针），这是实现这个 trick 的 API 基础。vLLM 的 CG 实现里大量用到了这个 API 来在不同 batch size 下复用同一个图。

### 7.8 小结：什么情况下 SSD 能用 CG

| Skip 模式 | CG 可行性 | 推荐方案 |
|-----------|-----------|----------|
| Contiguous prefix（固定 $L_d$） | 完全可行 | 两张图分别捕获，verify 复用 KV |
| 固定非连续 skip | 完全可行 | 单图捕获实际执行路径 |
| 少量离散配置（如 4-8 档） | 可行，有开销 | Multi-graph 预捕获 + 运行时选图 |
| 动态 early exit（基于 confidence） | 受限 | No-op mask（有冗余计算）或放弃 CG |
| 完全自由的运行时 skip | 不可行 | 必须用 eager mode，接受 launch overhead |

**核心结论**：SSD 想用 CG 的关键是**提前固定 skip 模式**。如果 skip 模式在离线阶段确定（无论连续还是非连续），CG 都没问题；如果要运行时动态决定，要么接受 no-op 的冗余计算，要么把配置空间限制到可预捕获的范围内。这也反过来约束了 SSD 里 skip 策略的设计空间——**越灵活的 skip 策略，越难和 CG 兼容，工程上往往只能选有限档位的静态配置**。

### 7.9 和 MoE 的对比：同是"条件计算"，动态性的维度不一样

这里值得单独展开，因为 MoE 模型同样有动态计算的问题，和 SSD skip 放在一起对比能把 CG 适配思路讲得更清楚。

**MoE 是"宽度动态"，SSD skip 是"深度动态"**：

| | MoE | SSD dynamic skip |
|---|---|---|
| 固定的是 | 层数 $L$ | token 数 |
| 变化的是 | 每层激活哪些 expert、每个 expert 收多少 token | 执行哪些层 |
| 动态性来源 | router 的 top-k 输出（数据依赖） | confidence/early-exit 判断（数据依赖） |
| 图拓扑变化 | expert kernel 的 input size 在变，但 kernel 集合固定 | kernel 集合本身在变（某些层整个不执行） |

理解这个维度差异很重要，因为它决定了 CG 适配难度的本质差异。

**MoE 的标准 CG 适配方案：padding to fixed shape**

MoE 的动态在于每个 expert 收到的 token 数不固定。比如 8 个 expert、batch=64，理论上每个 expert 平均收 16 个 token，但实际可能是 5、30、0、21……直接按实际 token 数 launch kernel，shape 每次都变，CG 无法捕获。

标准解法：**把每个 expert 的输入 padding 到一个固定上限 $K_\text{max}$**，保证所有 expert 的 kernel 每次都以相同 shape 执行。对应的 CG 拓扑完全固定，只需在 replay 前更新 token 分配的 mask tensor。

```python
# 伪代码：MoE CG 适配
# 捕获时：expert_inputs shape = [num_experts, K_max, hidden_dim]，固定
# replay 时：更新 routing_mask（GPU tensor），无需重新捕获
graph.replay()  # kernel shape 不变，routing_mask 控制哪些 slot 有效
```

padding 引入的开销：无效 token 的 expert forward 算了但不用（类似 SSD no-op mask）。但 MoE 的 router 通常做 load balancing，每个 expert 的负载比较均匀，padding 浪费的算力有限（通常 < 5%）。vLLM 和 SGLang 对 MoE 的 CG 支持基本都走这条路。

**SSD dynamic skip 的类比方案：layer no-op padding**

用 MoE padding 的思路类比到 SSD：把"跳过某层"改成"这层做 no-op forward"，保持所有层的 kernel 都执行，用 GPU tensor mask 控制结果是否生效。

```python
# SSD no-op mask 方案（类比 MoE padding）
for layer_idx in range(L):
    candidate = transformer_block[layer_idx](hidden)
    hidden = torch.where(skip_mask[layer_idx], hidden, candidate)
    # skip_mask 是 GPU tensor，replay 前更新，不重新捕获
```

但这里有一个 MoE 没有的问题：**MoE padding 浪费的是"部分 expert 的少量 slot"，而 SSD no-op 浪费的是"整个 transformer block 的 forward"**。一个 transformer block 包含完整的 self-attention + FFN，相当于整层的计算全白做了。如果 SSD 设计的初衷是跳过 30-50% 的层（$r_\text{draft} = 0.5$-$0.7$），no-op 方案把跳层省下的计算全部还回去，**加速效果清零**。

所以 SSD 的 no-op 方案只适合一种情况：**你在乎的是吞吐而不是单请求延迟**，需要 CG 带来的 launch overhead 节省（通常几十到几百 µs），而不在乎 GPU 算力的利用率。

**MoE + SSD 叠加：双重动态**

当大模型本身是 MoE 架构（比如 Mixtral、DeepSeek-MoE），然后在上面做 SSD，两种动态就叠在一起了：

- 每层内部：expert routing 动态（宽度动态）
- 跨层：skip 决策动态（深度动态）

分别处理：
- 宽度动态：每层内部继续用 MoE padding 方案，CG 可以捕获
- 深度动态：如果 skip 是静态的（离线确定），直接从图里去掉被 skip 的层；如果是动态的，要么 no-op（但开销更大，因为被跳的是整个 MoE block），要么预捕获多档

组合下来，**MoE + 动态 SSD skip 的最实用方案**通常是：
1. 把 SSD 的 skip 配置限制为 2-4 个静态档位（比如草稿用前 $L/4$、$L/2$、$3L/4$ 层三档）
2. 每档各预捕获一张 CG，每档内部用 MoE padding 处理 expert 的宽度动态
3. 档位切换在 CPU 端做（选哪张图 replay），不进入 CG 的捕获范围

这样既保住了 CG 的 launch overhead 收益，又保住了 SSD skip 真正节省计算的优势（没有 no-op 冗余）。代价是 skip 策略的灵活性受限——但从上面第 4 节的分析可以知道，$r_\text{draft}$ 在几个离散档位之间搜索，本来也足够覆盖大部分的 Speedup 最优点。

**一句话总结两者的差异**：MoE 的动态是"图中每个节点的 input size 在变"，CG 用 padding 填平就行；SSD dynamic skip 的动态是"图中的节点集合在变"，padding 没法解决，只能要么固化节点集合（静态 skip），要么接受 no-op 的冗余计算，要么预捕获多图切换。工程取舍的核心永远是：**动态性换来多少加速收益，值不值得为了 CG 兼容性付出对应的代价**。

### 7.10 实际案例：SparseSpec 怎么做的

上面讨论的都是理论层面的方案，来看一个真实实现：[SparseSpec](https://github.com/sspec-project/SparseSpec)，一个针对 RLM（推理大模型，即做 CoT 推理的那类）的批量推理加速框架，主打"稀疏自推测解码"（Sparse Self-Speculative Decoding），声称最高 2.3x 吞吐量提升。

#### SparseSpec 的核心思路：动态不是层的 skip，而是 attention 的 KV 范围

SparseSpec 的 SSD 变体和传统 skip layer 有一个本质区别：**它 skip 的不是 transformer block，而是 attention 的 KV 访问范围**。

具体做法叫 PillarAttn：
1. **验证阶段**：跑完整 attention，同时把每个 token 的 attention score 转储到全局内存（`dump_logits`）
2. **Top-K 筛选**：从这批 attention score 里找出对当前 query 贡献最大的 Top-K 个 KV 位置，作为下一轮草稿的"重要 token"集合
3. **草稿阶段**：不访问完整 KV cache，只访问这 Top-K 个稀疏位置

这样 skip 的是"大量不重要的 KV"，而不是"整个 transformer 层"。skip 的粒度细得多，也不会破坏激活路径的一致性（每层都还是完整执行的）。

#### CUDA Graph 适配：同一张图处理三种请求类型

SparseSpec 的 CG 方案是本文讨论思路里最干净的实现之一——**不改变图的拓扑，把所有动态性压进 GPU tensor，通过 `request_type` 在一张图里同时处理 NORMAL / DRAFT / VERIFY 三种请求**。

核心在 attention kernel 的 InputTransform（`serve/attention/backend.py`）：

```cpp
// CUDA JIT attention kernel 内部
if (request_type == 1) {   // DRAFT：用稀疏索引
    indices = params.flatten_indices + kv_head_idx * MAX_TOTAL_DRAFT_KV_LEN;
} else {                   // NORMAL / VERIFY：用完整索引
    indices = params.kv_indices;
}
```

`request_type` 是一个 GPU tensor，每次 replay 前按实际请求更新，图拓扑本身不动。同一张图，草稿请求走稀疏 KV 路径，验证请求走完整 KV 路径，混在一个 batch 里一起跑。

验证阶段的 attention score 转储也类似（`LogitsTransform`）：

```cpp
if (request_type == 2) {   // VERIFY 才写分数
    params.dump_logits[offset + kv_idx] = __float2half(logits / packed_qo_len);
}
```

只有 VERIFY 请求的 kernel 执行会写入 `dump_logits`，DRAFT 请求执行时写入目标不同，结果被自然隔离。

#### 双缓冲 CG：消除 CPU 准备和 GPU 执行之间的同步气泡

SparseSpec 的另一个工程细节值得记一下：**双缓冲 CUDA Graph**（`serve/model/model.py` 的 `KVPool` 类）。

```python
class KVPool:
    def __init__(self, ...):
        self.phase_idx = 0
        self._wrappers = []           # 两份 wrapper
        self._attn_fwd_metadata = []  # 两份 metadata
    
    def step(self):
        self.phase_idx ^= 1           # 0 ↔ 1 交替
```

标准 CG 的流程是：CPU 准备元数据 → GPU replay 图 → 等 GPU 完成 → CPU 准备下一批 → …，中间有等待的 bubble。双缓冲的做法：两张预捕获的图交替使用，第 0 张在 GPU 上 replay 时，CPU 已经在准备第 1 张图的输入参数，GPU 执行完第 0 张立刻 replay 第 1 张，不需要等待。

捕获时也预捕获了一系列 batch size 的图（余弦分桶：1, 2, 4, 8, 16, 32, 64, 128, 256...），每个 batch size 下捕获两张（双缓冲各一张）：

```python
def eager_cuda_graph_mode(self, max_batch_size=256, num_cuda_graphs=16):
    captured_args = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640]
    for batch_size in captured_args:
        for phase in range(2):      # 双缓冲
            runner = _CUDAGraphModelRunner(self.model)
            runner.capture(batch_size=batch_size, ...)
            self.graph_runners[batch_size].append(runner)
```

实际 forward 时，按当前 batch size 选最近的已捕获档位（允许 padding 到上一档），再按当前 `phase` 选双缓冲里的哪一张：

```python
def forward(self, **kwargs):
    batch_size = kwargs["token_ids"].size(0)
    idx = np.searchsorted(self.captured_bsz, batch_size, side="left")
    # 在 padding 比例允许范围内走 CG，否则 eager fallback
    model_executor = self.graph_runners[padded_bsz][self.kv_cache.phase]
    return model_executor(**kwargs)
```

#### SparseSpec 和本文讨论方案的对应关系

回头看 7.5-7.8 节讨论的几个方案，SparseSpec 实际走的是**方案 1（topology-preserving）的精细版**：

- **不是 layer-level 的 no-op**，而是 **attention KV-range 的条件访问**——代价比整层 no-op 小得多，被跳过的只是 attention 里不重要的 KV 读取，transformer block 本身的 MLP、归一化等还是完整执行
- **同一张图支持三种请求类型**，不需要 multi-graph 切换，靠 `request_type` tensor 在运行时路由
- **双缓冲消除同步气泡**，这是在 7.7 节 pointer swap 思路之外的另一个工程优化维度

这个设计也解释了为什么 SparseSpec 不需要显式处理 skip layer 的 KV 复用问题（第 4.2 节）：它的稀疏性在 attention 的 KV 索引层面而非 transformer 层层面，$r_\text{reuse}$ 的分析框架对它并不直接适用——它的"跳过"不改变激活路径，所有层的 hidden state 仍然是 full-path 计算出来的。

| | 本文理论方案 | SparseSpec 实现 |
|---|---|---|
| 动态性来源 | transformer layer 级别的 skip | attention KV range 的稀疏访问 |
| CG 适配思路 | no-op mask / multi-graph | 单图 + `request_type` tensor 路由 |
| 冗余计算 | 整层 no-op（代价大） | 仅 attention 的冗余 KV 访问（代价小） |
| KV 复用 | 依赖 contiguous prefix | 不适用（层不 skip，复用问题不存在） |
| batch 内混合 | 需要多图切换 | 天然支持（三种类型共享一张图） |

---

## 8. 最优草稿长度的 closed-form 近似

对于标准 SD 公式，对 $\gamma$ 求导：

$$\frac{\partial}{\partial \gamma}\text{Speedup} = \frac{(1-\alpha)[-\alpha^{\gamma+1}\ln\alpha(\gamma c + 1) - c(1-\alpha^{\gamma+1})]}{(1-\alpha)^2(\gamma c+1)^2}$$

令分子中括号 = 0：

$$-\alpha^{\gamma+1}\ln\alpha(\gamma c + 1) = c(1-\alpha^{\gamma+1})$$

这个方程没有解析解。但若 $\alpha^{\gamma+1} \ll 1$（即 $\gamma$ 足够大使接受率衰减显著）时，右边 $\approx c$，左边变成 $-\alpha^{\gamma+1}\ln\alpha \cdot \gamma c$（近似 $\gamma^* c \gg 1$），化简得：

$$\alpha^{\gamma+1} \approx \frac{-1}{\gamma^* \ln\alpha}$$

因为 $\ln\alpha < 0$（$\alpha < 1$），令 $|\ln\alpha| = -\ln\alpha$：

$$\gamma^* \approx \frac{1}{|\ln\alpha|} \cdot \frac{1}{\alpha^{\gamma^*+1}}$$

这是个隐式方程，但可以用一阶近似：令右边的 $\gamma^*$ 用初始猜测 $\gamma_0 = \frac{1}{|\ln\alpha|}$ 代入，迭代一次：

$$\gamma^* \approx \frac{1}{|\ln\alpha|} \cdot e^{(\gamma_0+1)|\ln\alpha|} = \frac{e^{(\frac{1}{|\ln\alpha|}+1)|\ln\alpha|}}{|\ln\alpha|}$$

这个近似在 $\alpha$ 接近 1 时比较准，但说实话不够优雅。工程上直接 grid search 更实用。

---

## 8. 数值校验：两个极端场景

### 场景 A：$c = 0.05, \alpha = 0.9, \gamma = 8$

$$\text{Speedup} = \frac{1 - 0.9^9}{(1-0.9)(8 \times 0.05 + 1)} = \frac{1 - 0.387}{0.1 \times 1.4} = \frac{0.613}{0.14} \approx 4.4$$

和图里 $\alpha=0.90, \gamma=8$ 那条线的值（约 4.5）吻合，没问题。

### 场景 B：$c = 0.0005, \alpha = 0.95, \gamma = 64$

$$\text{Speedup} = \frac{1 - 0.95^{65}}{(1-0.95)(64 \times 0.0005 + 1)} = \frac{1 - 0.0356}{0.05 \times 1.032} = \frac{0.964}{0.0516} \approx 18.7$$

图里显示约 16-17，稍有偏差，说明在 $\gamma = 64$ 时验证 overhead $f(\gamma) > 1$ 的影响已经不可忽略了，实际比理想值低一点，这是预期内的。

---

## 9. 一张对比表总结

| 因素 | SD | SSD-Contiguous | SSD-Skip |
|------|-----|----------------|----------|
| 草稿成本比 | $c = t_\text{draft}/t_\text{big}$ | $r_\text{draft} = L_d/L$ | $r_\text{draft} = |\mathcal{A}|/L$ |
| KV 可复用比例 | 0（两套模型） | $r_\text{reuse} = r_\text{draft}$ | $r_\text{reuse} = (s_1-1)/L \leq r_\text{draft}$ |
| Speedup 分母 | $\gamma c + 1$ | $1 + (\gamma-1)r$ | $\gamma r_\text{draft} + (1-r_\text{reuse})$ |
| 接受率 $\alpha$ | 依赖 draft model 质量 | 依赖前缀层能力，偏低 | 依赖 skip 模式，更低 |
| 内存占用 | 两套权重 | 一套权重 | 一套权重 |
| Batch 并发优势 | 无（内存较高） | 有（显存受限时） | 有（显存受限时） |
| 企业部署适用性 | 有专用 draft model 时最优 | 显存受限 / 无 draft model | 通常不如 Contiguous |

---

## 10. 结语

把所有 closed form 并排放一起：

$$\text{Speedup}_\text{SD} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}$$

$$\text{Speedup}_\text{SSD-contiguous} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)[1 + (\gamma-1)r]}$$

$$\text{Speedup}_\text{SSD-general} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)[\gamma r_\text{draft} + (1 - r_\text{reuse})]}$$

$$\text{Throughput}_\text{SSD} = \beta \cdot \text{Speedup}_\text{SSD} \cdot \text{Throughput}_\text{baseline}, \quad \beta = \frac{M_\text{total} - M_\text{big}}{M_\text{total} - M_\text{big} - M_\text{draft}}$$

这几个公式说清楚了几件事：

**第一，KV 复用的前提是激活路径一致**。非连续 skip 一旦让 $r_\text{reuse} \ll r_\text{draft}$，SSD 的 KV 复用优势彻底消失，退化成和 SD 等价的形式。所以很多 skip-layer SSD 的实现里，"KV cache 可以复用"这个说法要打问号，得看 skip 的是哪些层。

**第二，内存优势不是"直接加速"，是"换 batch size"**。SSD 省出来的 draft model 显存，在显存受限时能换成更高的并发量，通过 $\beta > 1$ 提升 throughput。但 $\alpha$ 会随 batch size 增大而下降，这个收益不是线性的，需要实测。

**第三，SSD 是"没有 draft model 时的 fallback"，不是"叠在 SD 上的加分项"**。如果资源允许维护专用 draft model，SD + 好 draft model 的 $\alpha$ 优势几乎必然超过 SSD 的 KV 复用优势。SSD 真正发光的场景是：显存极度受限、或者目标模型是自定义 fine-tune 版本、找不到合适的 draft model。

真正有意思的后续问题是：SSD 的 skip 层怎么搜？随机跳、等间隔跳、还是用 NAS 方法学出来的 skip 模式？不同 skip 模式下 $r_\text{draft}$ 和 $r_\text{reuse}$ 的差距有多大？这些直接影响 SSD 的实际加速，感觉可以专门写一篇。欢迎评论区交流。

---

> 如果这篇文章涉及的 LLM 推理效率和加速方法你想系统深入，可以看看我之前出版的[《动手学 AutoML：从 NAS 到大语言模型优化实战》](https://item.jd.com/14945889.html)，书里有专章讲 LLM 推理效率优化，和本文讨论的理论框架有直接关联。
>
> ![动手学AutoML书籍封面](https://github.com/marsggbo/marsggbo.github.io/blob/master/assets/img/book_cover_automl.png?raw=true)
