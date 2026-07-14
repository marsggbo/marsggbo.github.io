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

但落到 vllm 的工程实现里，有一个问题论文里不会告诉你：**这套逻辑具体怎么嵌进 vllm 的推理流水线？谁调用谁？数据怎么流？**

这篇文章从这个问题出发，先给完整时序，再逐层拆模块细节。

---

## 2. 全局视角：SD 在 vllm 里的计算流

### 2.1 先把几个概念说清楚

在看代码之前，先把 SD 里反复出现的几个术语说清楚，不然后面容易懵。

**Draft model 和 propose**：SD 需要一个"起草者"（drafter），负责提前猜接下来的 K 个 token——这个猜测过程叫 **propose**，猜出来的 token 叫 **draft tokens**。起草者可以是独立的小模型（DraftModel），也可以是 n-gram 匹配、并行预测头等不同方案，后面第 6 节会展开。

**Target model 和 verify**：大模型（target model）一次 forward 把这 K 个 draft token 全部喂进去，同时计算出"这些 token 对不对"——这个验证过程叫 **verify**。

**Rejection sampling**：verify 的判断标准不是简单的对/错，而是按照一个概率比 `min(1, p(x)/q(x))` 来决定接受还是拒绝（p 是 target 分布，q 是 draft 分布）。这样做的好处是：不管 draft 猜得准不准，最终输出的 token 分布和原始自回归采样**完全一样**，不损精度。这叫 rejection sampling（拒绝采样）。

**Bookkeeping（账本同步）**：rejection sampling 的结果在 GPU 上算出来，但 KV Cache 的 block table 在 CPU 侧维护。"bookkeeping" 就是把 GPU 上"接受了几个 draft token"的结果同步到 CPU，更新 block table，把被拒绝的 draft token 占用的 KV Cache slot 回收——就像账本对账一样。

有了这几个概念，再来看整个计算流。

### 2.2 接入点：两步执行模型

系列文章（二）讲过，`GPUModelRunner` 的推理逻辑拆成两步：

- `execute_model()`：target model forward，结果存入 `execute_model_state`，返回 None
- `sample_tokens()`：从 `execute_model_state` 取 logits，采样输出 token

SD 就嵌在这两步里，没有引入新的函数：
- `execute_model()` 把**上一轮 propose 出的 draft tokens** 打包进 input，一次 forward 全部 verify
- `sample_tokens()` 顺序做三件事：rejection sampling → bookkeeping → 为下一轮 propose

### 2.3 多轮时序：逐步走读

下图是多轮推理的完整时序，这是理解后面所有细节的基础：

![多轮推理时序图](/assets/img/posts/20260709-vllm-v1-05-speculative-decoding/diagram_seq.png)

沿着时序图从第 0 轮往下走：

**第 0 轮（冷启动）**：第一轮没有 draft，target model 正常跑一次 forward，采样出第一个 token。同时，drafter 用 target forward 产出的 hidden states 做第一次 propose，预测接下来 K 个 draft token，存起来备用。

**第 1 轮 execute_model()**：把上一轮 propose 的 K 个 draft token 拼进 `input_ids`，一次 forward 全部送进 target model。注意：原本需要串行跑 K 次 decode，现在变成一次 batched prefill，这是 SD 提速的根本——prefill 是 compute-bound，GPU 利用率远高于 memory-bound 的 decode。

**第 1 轮 rejection sampling**：target forward 算出每个位置的 logits，和 draft model 当时的预测概率做比对，按 `min(1, p/q)` 决定接受还是拒绝。第一个被拒绝的位置之后的 draft 全部截断，从 target 分布重新采样一个修正 token。

**第 1 轮 bookkeeping**：rejection 结果（接受了几个 token）从 GPU 同步到 CPU，更新 block table，回收被拒绝的 draft token 占用的 KV Cache slot。这步是纯 CPU 操作。

**第 1 轮 propose**：drafter 预测下一批 K 个 draft token，供第 2 轮使用。不同的 drafter 依赖的输入不同：

- **EAGLE / Medusa**：依赖 target model 刚算出的 hidden states。EAGLE 把 target hidden states 和 draft model 的 embedding 拼在一起作为输入；Medusa 直接用 target 最后一层的 hidden states 驱动多个预测头并行输出。因为它们消费的都是 GPU tensor，不需要等 CPU 侧的 bookkeeping 完成就可以提前开跑，和 bookkeeping 形成 GPU/CPU 的并行重叠。
- **DraftModel（独立小模型）**：小模型自己做 autoregressive decode，每步用上一步采样出的 draft token 作为输入，**不依赖 target 的 hidden states**，和 target model 是两个完全独立的模型。同样消费 GPU tensor，可以提前 propose。
- **ngram**：纯 CPU 文本匹配，在 context token 序列里找最长后缀匹配，预测接下来的 token。需要 CPU 侧完整的 token ids，必须等 bookkeeping 完成后才能 propose。

**之后每轮循环上述流程**，直到生成 EOS 或达到最大长度。

三个值得记住的结论：
- **propose 生成的是下一轮的 draft**，第 N 轮 propose，第 N+1 轮 verify，是流水线而非同轮闭环
- **target forward 一次吃 K 个 draft token**，相当于把 K 步串行 decode 压缩成一次 prefill
- **propose 和 bookkeeping 可以重叠**，进一步隐藏 CPU 同步延迟

### 2.4 模块全景

`GPUModelRunner` 是 SD 相关逻辑的宿主，持有两个 SD 专用成员和一个数据载体：

![整体架构：模块与依赖关系](/assets/img/posts/20260709-vllm-v1-05-speculative-decoding/diagram_arch.png)

- **`self.drafter`**：Proposer，负责 propose，有四种实现（第 6 节展开）
- **`self.rejection_sampler`**：负责 rejection sampling，在 `vllm/v1/sample/` 下而不是 `spec_decode/`——因为 rejection sampling 在概念上属于"采样"而非"draft 算法"
- **`SpecDecodeMetadata`**：贯穿 execute_model 和 sample_tokens 的数据胶水，在 `_prepare_inputs()` 构建

有了这个全景，接下来逐层看细节。

---

## 3. execute_model()：target forward 吃 draft tokens

非第一轮推理时，`input_ids` 里包含上一轮 propose 出的 draft tokens（第 4352 行）：

```python
# vllm/v1/worker/gpu_model_runner.py
# class GPUModelRunner
# def execute_model(self, scheduler_output)  第 4352 行
model_output = self._model_forward(
    input_ids=input_ids,   # 含上一轮的 K 个 draft tokens + 1 个 bonus token
    positions=positions,
    ...
)
```

forward 完之后，把 logits 和 hidden states 打包进 `execute_model_state`，返回 None。此时 rejection 和 propose 都还没跑：

```python
# vllm/v1/worker/gpu_model_runner.py
# class GPUModelRunner
# def execute_model(self, scheduler_output)  第 4418~4437 行
self.execute_model_state = ExecuteModelState(
    scheduler_output,
    logits,
    spec_decode_metadata,
    hidden_states,
    ...
)
return None  # sampling 留给 sample_tokens() 做
```

---

## 4. sample_tokens()：rejection → bookkeeping → propose

`sample_tokens()` 拿到 `execute_model_state` 后，顺序做三件事：

![sample_tokens() 调用流程](/assets/img/posts/20260709-vllm-v1-05-speculative-decoding/diagram_flow.png)

**第一步：`_sample()` 做 rejection**

把 target logits 和 draft token ids（来自 `SpecDecodeMetadata`）传给 `self.rejection_sampler`，逐 token 按概率比检验：接受的 draft 直接保留，第一个被拒绝的位置截断，从 target 分布采样一个修正 token。全程在 GPU 上跑 Triton kernel，不涉及 CPU 同步。

**第二步：`_bookkeeping_sync()` 做 CPU 同步**

把每个请求实际接受了几个 draft token 同步到 CPU，更新 KV cache 的 block table，回收被拒绝的 draft token 对应的 slot。

**第三步：`propose_draft_token_ids()` 生成下一批 draft**

调 `self.drafter.propose()`，用 target forward 的 hidden states 猜下一批 draft tokens。EAGLE/DraftModel 在 `_bookkeeping_sync()` 之前就可以开跑（第 4554 行），ngram 要等 bookkeeping 之后（第 4626 行）。

---

## 5. SpecDecodeMetadata：三步之间的数据胶水

`SpecDecodeMetadata` 在 `_prepare_inputs()` 第 2212 行构建：

| 字段 | 形状 | 作用 |
|---|---|---|
| `draft_token_ids` | `[total_num_draft_tokens]` | 所有请求的 draft tokens 扁平拼接 |
| `num_draft_tokens` | `list[int]` | 每个请求各自 draft 了多少步 |
| `logits_indices` | `[num_draft + batch_size]` | 在 target logits 中 gather 对应行的索引 |
| `bonus_logits_indices` | `[batch_size]` | 每个请求 bonus token 在 target logits 中的位置 |

`logits_indices` 把所有请求的 draft 位置和 bonus 位置拼在一起，让 `RejectionSampler` 一次 gather 拿到所有需要的 logits，不用按请求循环——因为 batch 里不同请求的 draft 步数可以不一样，位置对应关系不规则。

---

## 6. Proposer 层：接口设计与四种实现

### 6.1 为什么要抽象 Proposer 接口

四种算法生成 draft 的方式差异极大，但 `propose_draft_token_ids()` 只需要调一个 `self.drafter.propose()`，不关心里面是什么算法。

`self.drafter` 在 `GPUModelRunner.__init__()` 第 565~632 行根据 `speculative_config.method` 初始化：

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

四种实现的继承关系：

![Proposer 继承关系](/assets/img/posts/20260709-vllm-v1-05-speculative-decoding/diagram_proposer.png)

**EAGLE 和 DraftModel 继承同一个基类，Medusa 和 ngram 独立实现**——因为三种范式的根本差异：autoregressive 循环（EAGLE/DraftModel）、并行多头（Medusa）、纯 CPU 文本匹配（ngram），强行统一基类会扭曲设计。

### 6.2 SpecDecodeBaseProposer：autoregressive 循环

基类核心是一个 autoregressive 循环（第 682 行），每步用上一步的 draft token 作为输入，循环 K-1 次：

```python
# vllm/v1/worker/spec_decode/llm_base_proposer.py
# class SpecDecodeBaseProposer
# def propose(self, ...)  第 682 行
for token_index in range(self.num_speculative_tokens - 1):
    input_ids = draft_token_ids_list[-1].int()
    model_kwargs = {"input_ids": input_ids, "positions": ...}
    if self.pass_hidden_states_to_model:
        model_kwargs["hidden_states"] = self.hidden_states[:input_batch_size]
    ret_hidden_states = self.model(**model_kwargs)
    draft_token_ids, _ = self._sample_draft_tokens(sample_hidden_states, ...)
    draft_token_ids_list.append(draft_token_ids)
```

子类唯一的定制点是构造函数里的 `pass_hidden_states_to_model` 开关。

### 6.3 EagleProposer：23 行的秘密

EAGLE 的 insight 是 draft model 不自己独立预测，而是"寄生"在 target 的 residual stream 上——把 target hidden states 和 draft model 的 embedding 拼接起来作为输入，接受率因此高得多。

代码层面就体现在一个布尔开关：

```python
# vllm/v1/worker/spec_decode/eagle.py
# class EagleProposer(SpecDecodeBaseProposer)  第 10 行
def __init__(self, vllm_config, device, runner):
    super().__init__(
        vllm_config, device,
        pass_hidden_states_to_model=True,  # ← EAGLE 相比 DraftModel 的全部差异
        runner=runner,
    )
```

整个文件 23 行，没有任何业务逻辑，全靠基类跑通。

### 6.4 MedusaProposer：并行多头

Medusa 一次 forward 出所有 K 个 draft，不需要 autoregressive 循环：

```python
# vllm/v1/worker/spec_decode/medusa.py
# class MedusaProposer
# def propose(self, num_speculative_tokens, target_hidden_states, ...)
blocks = self.model(target_hidden_states)   # 多 head 并行 forward
logits = self.model.compute_logits(blocks)  # [batch, num_heads, vocab]
return logits.argmax(dim=-1)                # [batch, num_heads]
```

代价是各 head 完全独立，接受率通常比 EAGLE 低。

### 6.5 NgramProposer：连 draft model 都不需要

`load_model()` 是空实现，直接在 context 里找重复 pattern。核心是 KMP lps 数组找最长后缀匹配，整个 batch 用 Numba `@njit(parallel=True)` 并行加速，零额外显存。对 RAG、code completion 等有大量重复 pattern 的场景接受率相当高。

---

## 7. RejectionSampler：四阶段 Triton kernel

数学原理：给定 target 分布 p 和 draft 分布 q，对每个 draft token x，以 `min(1, p(x)/q(x))` 的概率接受；拒绝时从残差分布 `max(p - q, 0)` 里重新采样，保证输出分布仍是 p，和原始自回归采样等价。

`rejection_sample()`（`rejection_sampler_utils.py` 第 864 行）分四个 Triton kernel 执行：

![RejectionSampler 四阶段 kernel](/assets/img/posts/20260709-vllm-v1-05-speculative-decoding/diagram_rejection.png)

Step 3 的核心 kernel（第 459 行）：

```python
# vllm/v1/sample/rejection_sampler_utils.py
# 第 459 行，Triton kernel（不在任何 class 下，是模块级函数）
@triton.jit
def _rejection_kernel(...):
    for i in range(num_draft_tokens):
        accepted &= target_log_prob > tl.log(u) + draft_log_prob  # 非 greedy
        target_argmax = compute_global_target_argmax(...)
        accepted &= (target_argmax == draft_sampled)               # greedy

    rejected_steps[req] = first_rejected_index
```

一旦某个位置被拒绝，后续 draft 全部作废（基于错误前缀生成的），所以顺序检查并记录第一个拒绝位置即可。

---

## 8. 另一套路径：Speculator（TP/EP 场景）

前面讲的是主路径（`gpu_model_runner.py`），drafter 只管算法逻辑，KV cache 由外层统一管。

在 `vllm/v1/worker/gpu/model_runner.py`（DeepSeek 等 TP/EP 多 GPU 场景），SD 改用 `self.speculator`，**speculator 同时负责 draft forward 和 KV cache 管理**：

```python
# vllm/v1/worker/gpu/model_runner.py
# class GPUModelRunner（TP/EP 版本，与主路径的 gpu_model_runner.py 不同）
# def sample_tokens(self, ...)  第 1466 行
draft_tokens = self.speculator.propose(
    input_batch,
    attn_metadata,
    slot_mappings_by_layer,
    spec_hidden_states,
    num_sampled,    # 上一轮 rejection 结果
    num_rejected,   # 用于 KV cache 回退
    ...
)
```

`num_rejected` 出现在 propose 参数里，是因为 speculator 上一轮已经把 draft KV 写入 GPU——被拒掉的 KV slot 需要在下一轮 propose 前回退，speculator 自己负责这块。

继承链：`EagleSpeculator → AutoRegressiveSpeculator → DraftModelSpeculator → BaseSpeculator`

---

## 9. 延伸：Self-Speculative Decoding

前面讲的都是"独立 draft 模型"方案。另一个思路是 Self-Speculative Decoding（SSD）——不引入额外模型，用同一个模型本身的子网络生成 draft。

| 方案 | 核心思路 |
|---|---|
| Early Exit | 前几层 transformer 输出直接过 LM head |
| Layer Skip | 跳过部分中间层，用稀疏子网络做 draft |
| Sparse Attention | draft 阶段只加载部分 KV，verify 阶段全量 |

vllm v1 目前原生不支持 SSD，有两个根本冲突：`self.drafter` 和 target model 是独立对象（SSD 需要两用同一个模型）；KV cache 假设 draft 和 target 各自独立（SSD 的稀疏/全量切换和 block table 不兼容）。

SparseSpec（arXiv 2512.01278）是稀疏 attention 路线的代表，直接绕开 vllm 实现了整套 serving stack：

![SparseSpec 自定义 serving stack 架构](/assets/img/posts/20260709-vllm-v1-05-speculative-decoding/diagram_sparsespec.png)

这套设计目前还是 PoC，README 明确说"We plan to upstream a subset of features to vLLM in the future"。

---

## 10. 小结

**接入点**：SD 嵌在 execute_model() + sample_tokens() 两步里。execute_model 把 draft tokens 打包进 target forward，sample_tokens 按顺序做 rejection → bookkeeping → propose。**propose 生成的是下一轮的 draft**，是流水线关系。

**模块归属**：`self.drafter` 和 `self.rejection_sampler` 挂在 `GPUModelRunner` 下。`SpecDecodeMetadata` 是数据胶水，在 `_prepare_inputs()` 构建。

**Proposer 设计**：四种实现对外统一调 `propose()`。EAGLE/DraftModel 共用 `SpecDecodeBaseProposer`（autoregressive 循环），核心差异只有 `pass_hidden_states_to_model` 一个开关；Medusa 并行多头，独立实现；ngram 不需要模型。

**RejectionSampler**：四阶段 Triton kernel，数学上保证输出分布和原始自回归采样等价。

如果你在做 Self-Speculative Decoding 方向，怎么 fit 进 vllm 的 block table 架构是个值得深挖的工程问题，欢迎评论区交流。

---

> 另外，我们团队最近出版了[《动手学 AutoML：从 NAS 到大语言模型优化实战》](https://item.jd.com/14945889.html)，感兴趣的话可以看看。
>
> ![动手学AutoML书籍封面](https://github.com/marsggbo/marsggbo.github.io/blob/master/assets/img/book_cover_automl.png?raw=true)
