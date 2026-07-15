---
layout: post
title: "vllm v1 源码精读（五）：从 generate() 到 Speculative Decoding 的完整计算流"
date: 2026-07-09
tags: [LLM, vLLM, Speculative Decoding, 推理优化, 源码解析]
---

# vllm v1 源码精读（五）：从 generate() 到 Speculative Decoding 的完整计算流

> 本文基于 vllm [commit ba22152](https://github.com/vllm-project/vllm/tree/ba22152)，所有代码均来自本地 clone 的真实源文件，标注了文件路径和行号。

**系列文章**：
- [（一）为什么要重写，以及 LLM() 这行代码背后发生了什么](https://marsggbo.github.io/blog/2026/20260706-vllm-v1-01-arch/)
- [（二）generate() 计算流——model.forward() 在哪里被调用？](https://marsggbo.github.io/blog/2026/20260706-vllm-v1-02-generate/)
- [（三）KV Cache 管理、Chunked Prefill 与异步架构](https://marsggbo.github.io/blog/2026/20260706-vllm-v1-03-kvcache/)
- [（四）插件系统——用 Python entry_points 实现零侵入扩展](https://marsggbo.github.io/blog/2026/20260706-vllm-v1-04-plugins/)
- **（五）从 generate() 到 Speculative Decoding 的完整计算流**（本文）

---

## 1. 前言

Speculative Decoding（SD）的算法原理不难理解：用小模型提前猜几个 token，大模型一次性验证，接受就保留，拒绝就从第一个错误位置重新采样。

但落到 vllm 的工程实现里，有一串问题论文里不会告诉你：**draft model 在哪里初始化？它和 target model 是进程间通信还是同进程？input_ids 怎么打包 draft tokens？propose() 返回什么、传给谁？**

这篇文章从这些问题出发，自上而下把整个计算流讲清楚。

---

## 2. 先建立概念：SD 涉及的几个关键词

在看代码之前，先把 SD 里反复出现的几个术语说清楚。

**Draft model 和 propose**：SD 需要一个"起草者"（drafter），负责提前猜接下来的 K 个 token——这个猜测过程叫 **propose**，猜出来的 token 叫 **draft tokens**。起草者可以是独立的小模型（DraftModel），也可以是 n-gram 匹配、并行预测头等不同方案，第 6 节展开。

**Target model 和 verify**：大模型（target model）一次 forward 把这 K 个 draft token 全部喂进去，验证它们对不对——这个过程叫 **verify**。

**Rejection sampling**：verify 的判断标准不是简单的对/错，而是按概率比 `min(1, p(x)/q(x))` 来决定接受还是拒绝（p 是 target 分布，q 是 draft 分布）。这样保证了：不管 draft 猜得准不准，最终输出的 token 分布和原始自回归采样**完全一样**，不损精度。

**Bookkeeping（账本同步）**：rejection sampling 的结果在 GPU 上算出来，但 KV Cache 的 block table 在 CPU 侧维护。bookkeeping 就是把"接受了几个 draft token"的结果同步到 CPU，更新 block table，回收被拒绝的 draft token 占用的 KV Cache slot。

---

## 3. 整体架构：GPUModelRunner 里住着什么

`GPUModelRunner` 是 SD 相关逻辑的宿主。下图展示了它内部的对象层级、draft model 在哪里、以及运行时数据怎么在 target model 和 drafter 之间流动：

![GPUModelRunner 内部结构与数据流](/assets/img/posts/20260709-vllm-v1-05-speculative-decoding/sd_internal_structure.png)

三个成员的职责：

- **`self.model`**：target model，标准 `nn.Module`，attention 层已替换为 FlashAttention 版本，在 `load_model()` 加载
- **`self.drafter`**：Proposer 对象，内部持有 `self.model`（draft model），负责 propose，有四种实现（第 6 节展开）
- **`self.rejection_sampler`**：做 rejection sampling，在 `vllm/v1/sample/` 下而不是 `spec_decode/`——因为 rejection sampling 属于采样层而非 draft 算法层

**target model 和 draft model 的关系**：两者都是独立的 `nn.Module`，加载在同一 Worker 进程的同一块 GPU 上，**没有进程间通信，没有跨 GPU 传输**。它们之间的"信息交换"就是普通的 Python 函数调用——把一个 GPU tensor 当参数传过去，零拷贝，零开销。

---

## 4. 初始化：draft model 怎么加载进来

搞清楚架构之后，先来看 draft model 是怎么初始化的。

`GPUModelRunner.__init__()` 第 565 行根据配置创建 Proposer 对象：

```python
# vllm/v1/worker/gpu_model_runner.py
# class GPUModelRunner
# def __init__(self, vllm_config, ...)  第 565~632 行
if spec_config.method == "ngram":
    self.drafter = NgramProposer(self.vllm_config)
elif spec_config.uses_draft_model():
    self.drafter = DraftModelProposer(self.vllm_config, self.device, self)
elif spec_config.use_eagle():
    self.drafter = EagleProposer(self.vllm_config, self.device, self)
elif spec_config.method == "medusa":
    self.drafter = MedusaProposer(...)
```

以 EagleProposer 为例，完整初始化链路如下：

```
GPUModelRunner.__init__()
  └─ EagleProposer.__init__()              # eagle.py:10
       └─ SpecDecodeBaseProposer.__init__()    # llm_base_proposer.py
            # 此时 __init__ 不加载权重，只做配置初始化

GPUModelRunner.load_model()               # 加载 target model 权重后
  └─ if hasattr(self, "drafter"):
       self.drafter.load_model(self.model)  # 把 target model 传进去
            # ↑ SpecDecodeBaseProposer.load_model(target_model)
            └─ self.model = self._get_model()    # 加载 draft model 权重
                 # EAGLE/Medusa: set_model_tag("eagle_head") + get_model()
                 # DraftModelProposer: set_model_tag("draft_model") + get_model()
            └─ self._maybe_share_embeddings(target_model)  # 共享 target 的 embedding
            └─ self._maybe_share_lm_head(target_model)     # 共享 target 的 lm_head
```

**为什么 `load_model()` 需要传入 target model？** EAGLE 的 draft head 通常共用 target model 的 embedding 层和 lm_head——draft 不需要自己学词表映射，直接复用 target 的，既节省显存又保证两者语义空间对齐。`set_model_tag` 的作用是给 `torch.compile` 做命名空间区分，避免 target 和 draft 的编译 graph 混淆（`DraftModelProposer` 用 `"draft_model"` tag，EAGLE/Medusa 基类默认用 `"eagle_head"` tag）。

NgramProposer 的 `load_model()` 是空实现——它不需要任何模型，不占显存。

---

## 5. 运行时主线：execute_model() → sample_tokens() → propose()

有了架构和初始化的认知，再来看每轮推理里代码按什么顺序跑。

### 5.1 execute_model()：draft tokens 怎么打包进 input_ids（关键）

普通 decode 每步只送 1 个 token 给 target model；开启 SD 后，**从第二轮起，`input_ids` 里会包含上一轮 propose 出的 K 个 draft tokens，加上 1 个 bonus token（来自上一轮 target 自己采样的结果），共 K+1 个 token 一次性送进 target forward**。

这是 SD 提速的核心工程实现——把原来需要串行跑 K 次的 decode，压缩成一次 batched prefill。prefill 是 compute-bound，GPU 利用率远高于 memory-bound 的 decode。

`_prepare_inputs()` 里负责拼接这段 input_ids（第 2212 行），同时构建 `SpecDecodeMetadata`：

```python
# vllm/v1/worker/gpu_model_runner.py
# class GPUModelRunner
# def _prepare_inputs(self, scheduler_output)  第 2212 行

# 对于 SD 请求，input_ids 的拼接逻辑（伪代码）：
# [prompt tokens]  +  [上一轮 draft_token_ids[0..K-1]]  +  [bonus token]
#                      ↑ 来自上一轮 propose() 的返回值
#                      存在 spec_decode_metadata.draft_token_ids 里

spec_decode_metadata = SpecDecodeMetadata(
    draft_token_ids=draft_token_ids,      # [total_draft_tokens] 所有请求的 draft 扁平拼接
    num_draft_tokens=num_draft_tokens,    # list[int] 每个请求 draft 了多少步
    logits_indices=logits_indices,        # 在 target logits 里 gather 各 draft 位置的索引
    bonus_logits_indices=bonus_logits_indices,
)
```

target forward 完成后，把 logits 和 hidden states 打包，返回 None，等 `sample_tokens()` 来取：

```python
# vllm/v1/worker/gpu_model_runner.py
# class GPUModelRunner
# def execute_model(self, scheduler_output)  第 4418~4437 行
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,           # [N_tokens, vocab_size]，包含所有 draft 位置的 logits
    spec_decode_metadata,
    hidden_states,    # [N_tokens, d_model]，EAGLE/Medusa 的 propose 要用
    ...
)
return None  # sampling 留给 sample_tokens() 做，GPU/CPU 真正并行
```

### 5.2 sample_tokens()：三步走——rejection → bookkeeping → propose

`sample_tokens()` 拿到 `execute_model_state` 后，顺序做三件事：

![sample_tokens() 调用流程](/assets/img/posts/20260709-vllm-v1-05-speculative-decoding/diagram_flow.png)

**第一步：`_sample()` 做 rejection**

从 `execute_model_state` 取出 target logits，从 `SpecDecodeMetadata` 取出 draft token ids，传给 `self.rejection_sampler`，逐 token 按概率比检验。接受的 draft 直接保留，第一个被拒绝的位置截断，从 target 分布重新采样一个修正 token。全程在 GPU 上跑 Triton kernel，不涉及 CPU 同步。

**第二步：`_bookkeeping_sync()` 做 CPU 同步**

把每个请求"接受了几个 draft token"的结果从 GPU 同步到 CPU，更新 KV Cache 的 block table，回收被拒绝的 draft token 对应的 slot。这是纯 CPU 操作。

**第三步：`propose_draft_token_ids()` 生成下一批 draft**

调 `self.drafter.propose()`，为下一轮推理准备 K 个 draft token。不同 drafter 依赖的输入不同：

- **EAGLE / Medusa**：依赖 target forward 刚产出的 `hidden_states`（从 `execute_model_state` 取），消费 GPU tensor，可以在 bookkeeping 之前提前开跑（第 4554 行）
- **DraftModel**：不依赖 target hidden_states，独立 autoregressive decode，同样消费 GPU tensor，可以提前 propose（第 4554 行）
- **ngram**：需要 CPU 侧完整的 token ids，必须等 bookkeeping 完成后才能 propose（第 4626 行）

### 5.3 多轮流水线时序

下图展示了多轮推理中 propose 和 verify 如何跨轮流转：

![多轮推理时序图](/assets/img/posts/20260709-vllm-v1-05-speculative-decoding/diagram_seq.png)

沿着时序图从第 0 轮往下走：

**第 0 轮（冷启动）**：没有 draft，target model 正常跑一次 forward，采样出第一个 token。同时 drafter 做第一次 propose，预测接下来 K 个 draft token 存起来备用。

**第 1 轮 execute_model()**：上一轮的 K 个 draft token 被 `_prepare_inputs()` 打包进 `input_ids`，一次 forward 全部送进 target model 验证。

**第 1 轮 rejection → bookkeeping → propose**：rejection 决定接受哪些 draft，bookkeeping 同步到 CPU，propose 为第 2 轮准备新一批 draft。EAGLE/DraftModel 的 propose 和 bookkeeping 并行，ngram 串行等待。

**三个值得记住的结论**：
- **propose 生成的是下一轮的 draft**，是流水线而非同轮闭环
- **target forward 一次吃 K 个 draft token**，把 K 步串行 decode 压缩成一次 prefill
- **部分 drafter 的 propose 和 bookkeeping 可以重叠**，隐藏 CPU 同步延迟

---

## 6. Proposer 层：四种实现的接口与差异

### 6.1 统一接口：propose() 返回什么，传给谁

四种算法生成 draft 的方式差异极大，但对外暴露统一的接口：`propose()` 接受 target forward 的中间结果，返回 draft token ids 列表。

返回值是一个 `list[Tensor]`，长度为 K（speculative_tokens 数），每个 Tensor 的 shape 是 `[batch_size]`，对应 batch 里所有请求在该步的 draft token：

```
propose() 返回:
  [
    draft_token_ids_step0,   # shape [batch_size]，第 1 个 draft token
    draft_token_ids_step1,   # shape [batch_size]，第 2 个 draft token
    ...
    draft_token_ids_stepK-1, # shape [batch_size]，第 K 个 draft token
  ]
```

这个返回值经过 `propose_draft_token_ids()`（`gpu_model_runner.py` 第 4893 行）存入 `execute_model_state`，下一轮被 `_prepare_inputs()` 读取，拼进 `input_ids` 参与 target forward。这就是 draft 跨轮流转的完整链路。

### 6.2 SpecDecodeBaseProposer：autoregressive 循环的输入输出

EAGLE 和 DraftModel 共用这个基类。核心是一个 autoregressive 循环，每步用上一步采样出的 draft token 作为输入，循环 K-1 次（第一个 draft token 在循环外先算出来）：

```python
# vllm/v1/worker/spec_decode/llm_base_proposer.py
# class SpecDecodeBaseProposer
# def propose(self, execute_model_req, ...)  第 682 行

draft_token_ids_list = [initial_draft_token_ids]  # 第一步在循环外产出

for token_index in range(self.num_speculative_tokens - 1):
    input_ids = draft_token_ids_list[-1].int()   # 上一步的 draft token 作为这步输入

    model_kwargs = {"input_ids": input_ids, "positions": positions}
    if self.pass_hidden_states_to_model:
        # EAGLE 走这里：把 target hidden states 拼入 draft model 输入
        # DraftModel 不走这里：只用 input_ids，完全独立
        model_kwargs["hidden_states"] = self.hidden_states[:batch_size]

    draft_hidden = self.model(**model_kwargs)   # draft model forward
    draft_token_ids, _ = self._sample_draft_tokens(draft_hidden, ...)
    draft_token_ids_list.append(draft_token_ids)

return draft_token_ids_list  # list[Tensor]，长度 K，每个 shape [batch_size]
# ↑ 这个返回值最终进入 execute_model_state，下一轮被打包进 input_ids
```

### 6.3 EAGLE vs DraftModel：运行时的关键差异

从 propose 循环的角度看，两者的差异只有一个布尔开关 `pass_hidden_states_to_model`：

```python
# vllm/v1/worker/spec_decode/eagle.py
# class EagleProposer(SpecDecodeBaseProposer)  第 10 行
def __init__(self, vllm_config, device, runner):
    super().__init__(
        vllm_config, device,
        pass_hidden_states_to_model=True,   # ← EAGLE：draft model 需要 target hidden_states
        runner=runner,
    )

# vllm/v1/spec_decode/draft_model.py
# class DraftModelProposer(SpecDecodeBaseProposer)
def __init__(self, ...):
    super().__init__(
        ...,
        pass_hidden_states_to_model=False,  # ← DraftModel：完全独立，不消费 target 特征
    )
```

这个开关在基类循环（6.2 节）的 `if self.pass_hidden_states_to_model` 分支产生实质差异：

| | EAGLE | DraftModel |
|---|---|---|
| draft model 的输入 | `input_ids` + target `hidden_states` 拼接 | 只有 `input_ids` |
| 依赖 target forward | 是，消费 target 最后一层特征 | 否，完全独立 |
| 接受率 | 更高（站在 target 肩膀上猜） | 较低（独立猜） |
| `_get_model()` | 使用基类默认实现，tag = `"eagle_head"` | **override**，tag = `"draft_model"`，配置构建逻辑也不同 |
| embedding/lm_head | 与 target 共享 | **override** 为不共享 |

`DraftModelProposer` 除了这个开关外，还 override 了 `_get_model()`（加载时用 `"draft_model"` tag）、`_create_draft_vllm_config()`（独立的 parallel config）、以及 `_maybe_share_embeddings` 和 `_maybe_share_lm_head`（独立小模型不共享 target 词表）。EAGLE 的 `EagleProposer` 整个文件只有 23 行，全靠基类默认行为跑通。

EAGLE 的 insight 是：不让 draft model 从零开始猜，而是让它"寄生"在 target 的 residual stream 上——把 target hidden states 和 draft model 的 embedding 拼在一起作为输入，draft model 相当于在 target 已经理解的语义基础上再细化预测，接受率因此高得多。

### 6.4 MedusaProposer：并行多头，无 autoregressive 循环

Medusa 不需要 autoregressive 循环，直接用 target 最后一层的 `hidden_states` 驱动 K 个独立预测头并行输出：

```python
# vllm/v1/worker/spec_decode/medusa.py
# class MedusaProposer
# def propose(self, num_speculative_tokens, target_hidden_states, ...)
blocks = self.model(target_hidden_states)   # K 个 head 并行 forward
logits = self.model.compute_logits(blocks)  # [batch, num_heads, vocab]
draft_token_ids = logits.argmax(dim=-1)     # [batch, num_heads] = [batch, K]
return [draft_token_ids[:, i] for i in range(num_heads)]  # list[Tensor]，每个 shape [batch]
```

一次 forward 出所有 K 个 draft，代价是各 head 之间完全独立，无法利用前几步的 draft 信息，接受率通常比 EAGLE 低。

### 6.5 NgramProposer：无模型，纯 CPU 匹配

`load_model()` 是空实现，直接在 context token 序列里找重复 pattern——把 context 翻转后用 KMP lps 数组找最长后缀匹配，取匹配位置后 K 个 token 作为 draft。整个 batch 用 Numba `@njit(parallel=True)` 并行加速，零额外显存。对 RAG、code completion 等有大量重复 pattern 的场景接受率相当高。

---

## 7. RejectionSampler：四阶段 Triton kernel

`rejection_sample()`（`rejection_sampler_utils.py` 第 864 行）分四个 Triton kernel 执行：

![RejectionSampler 四阶段 kernel](/assets/img/posts/20260709-vllm-v1-05-speculative-decoding/diagram_rejection.png)

数学原理：给定 target 分布 p 和 draft 分布 q，对每个 draft token x，以 `min(1, p(x)/q(x))` 的概率接受；拒绝时从残差分布 `max(p - q, 0)` 里重新采样，保证输出分布仍是 p，和原始自回归采样等价。

Step 3 的核心 kernel（第 459 行）：

```python
# vllm/v1/sample/rejection_sampler_utils.py
# 第 459 行，Triton kernel（模块级函数）
@triton.jit
def _rejection_kernel(...):
    for i in range(num_draft_tokens):
        # 非 greedy：概率比检验
        accepted &= target_log_prob > tl.log(u) + draft_log_prob
        # greedy：精确匹配
        target_argmax = compute_global_target_argmax(...)
        accepted &= (target_argmax == draft_sampled)

    rejected_steps[req] = first_rejected_index  # 记录第一个被拒绝的位置
```

一旦某个位置被拒绝，后续 draft 全部作废（基于错误前缀生成的），顺序检查并记录第一个拒绝位置即可。

---

## 8. SpecDecodeMetadata：贯穿三步的数据胶水

`SpecDecodeMetadata` 在 `_prepare_inputs()` 第 2212 行构建，是 execute_model 和 sample_tokens 之间信息传递的核心：

| 字段 | 形状 | 作用 |
|---|---|---|
| `draft_token_ids` | `[total_num_draft_tokens]` | 所有请求的 draft tokens 扁平拼接 |
| `num_draft_tokens` | `list[int]` | 每个请求各自 draft 了多少步 |
| `logits_indices` | `[num_draft + batch_size]` | 在 target logits 中 gather 对应行的索引 |
| `bonus_logits_indices` | `[batch_size]` | 每个请求 bonus token 在 target logits 中的位置 |

`logits_indices` 的设计原因：batch 里不同请求的 draft 步数可以不一样（context 太长或提前停止），target logits 的位置对应关系不规则。把所有请求的 draft 位置和 bonus 位置拼成一个索引，`RejectionSampler` 一次 gather 拿到所有需要的 logits，不用按请求循环。

---

## 9. 延伸：Self-Speculative Decoding 为什么难接进 vllm

前面讲的都是"独立 draft 模型"方案。另一个思路是 Self-Speculative Decoding（SSD）——不引入额外模型，用同一个模型本身的子网络生成 draft。

| 方案 | 核心思路 |
|---|---|
| Early Exit | 前几层 transformer 输出直接过 LM head |
| Layer Skip | 跳过部分中间层，用稀疏子网络做 draft |
| Sparse Attention | draft 阶段只加载部分 KV，verify 阶段全量 |

vllm v1 目前原生不支持 SSD，有两个根本冲突：`self.drafter` 和 target model 是独立对象（SSD 需要两用同一个模型）；KV cache 假设 draft 和 target 各自独立（SSD 的稀疏/全量切换和 block table 设计不兼容）。

SparseSpec（arXiv 2512.01278）是稀疏 attention 路线的代表，直接绕开 vllm 实现了整套 serving stack：

![SparseSpec 自定义 serving stack 架构](/assets/img/posts/20260709-vllm-v1-05-speculative-decoding/diagram_sparsespec.png)

这套设计目前还是 PoC，README 明确说"We plan to upstream a subset of features to vLLM in the future"。

---

## 10. 小结

**初始化**：draft model 在 `GPUModelRunner.__init__()` 里随 Proposer 一起创建，加载到同一 Worker 进程的同一块 GPU，和 target model 之间没有 IPC，只是普通函数调用传 GPU tensor。

**运行时主线**：每轮推理的核心是 `_prepare_inputs()` 把上一轮 propose 出的 K 个 draft token 打包进 `input_ids`，target forward 一次验证全部，替代了 K 次串行 decode。`sample_tokens()` 随后做 rejection → bookkeeping → propose，为下一轮准备新的 draft。

**Proposer 设计**：四种实现对外统一调 `propose()`，返回 `list[Tensor]`（长度 K，每个 shape `[batch_size]`）。EAGLE 和 DraftModel 共用基类 autoregressive 循环，唯一差异是 `pass_hidden_states_to_model` 开关——EAGLE 把 target hidden states 拼入 draft model 输入，DraftModel 完全独立；Medusa 并行多头一次 forward 出 K 个 draft；ngram 无模型，纯 CPU 匹配。

**RejectionSampler**：四阶段 Triton kernel，数学上保证输出分布和原始自回归采样等价。`SpecDecodeMetadata` 是三步之间的数据胶水，存储 draft token ids 和 logits 索引。

如果你在做 Self-Speculative Decoding 方向，如何优雅 fit 进 vllm 的 block table 架构是个值得深挖的工程问题，欢迎评论区交流。

---

> 我们团队做 LLM 推理效率和 NAS 方向的研究，把一些积累整理成了[《动手学 AutoML：从 NAS 到大语言模型优化实战》](https://item.jd.com/14945889.html)。Speculative Decoding 这篇聊的是工程实现，书里更多是从 AutoML 和 NAS 的视角看大模型优化，角度不同，但如果你对系统性地理解大模型效率这件事感兴趣，可以看看。
>
> ![动手学AutoML书籍封面](https://github.com/marsggbo/marsggbo.github.io/blob/master/assets/img/book_cover_automl.png?raw=true)
