---
layout: post
title: "vllm v1 源码精读（五）：Speculative Decoding 的接口设计、计算流与自投机推理"
date: 2026-07-09
tags: [LLM, vLLM, Speculative Decoding, 推理优化, 源码解析]
---

# vllm v1 源码精读（五）：Speculative Decoding 的接口设计、计算流与自投机推理

> 本文基于 vllm [`https://github.com/vllm-project/vllm/tree/ba22152`](https://github.com/vllm-project/vllm/tree/ba22152)，源码链接均指向该 commit 的固定行号。

**系列文章**：
- [（一）为什么要重写，以及 LLM() 这行代码背后发生了什么](https://marsggbo.github.io/blog/2026/20260706-vllm-v1-01-arch/)
- [（二）generate() 计算流——model.forward() 在哪里被调用？](https://marsggbo.github.io/blog/2026/20260706-vllm-v1-02-generate/)
- [（三）KV Cache 管理、Chunked Prefill 与异步架构](https://marsggbo.github.io/blog/2026/20260706-vllm-v1-03-kvcache/)
- [（四）插件系统——用 Python entry_points 实现零侵入扩展](https://marsggbo.github.io/blog/2026/20260706-vllm-v1-04-plugins/)
- **（五）Speculative Decoding 的接口设计、计算流与自投机推理**（本文）

---

## 1. 前言

Speculative Decoding（投机推理）的算法原理并不复杂——用小模型先生成一批 draft tokens，再用大模型一次性验证，接受则保留，拒绝则回退，理论上能在不损失输出质量的前提下显著提升 throughput。

但一旦落到工程实现里，问题就多了：

- batch 里不同请求的 draft 步数不一样，怎么对齐？
- draft 模型和 target 模型的 KV cache 怎么管理？
- EAGLE、Medusa、ngram 这些算法风格差异这么大，接口怎么统一？
- rejection sampling 的 Triton kernel 里到底跑的是什么逻辑？

这篇就结合 vllm v1 的源码把这些问题一一拆开来看。代码主要在两个目录：

- [`vllm/v1/spec_decode/`](https://github.com/vllm-project/vllm/tree/ba22152/vllm/v1/spec_decode)：proposer 层，负责生成 draft tokens
- [`vllm/v1/worker/gpu/spec_decode/`](https://github.com/vllm-project/vllm/tree/ba22152/vllm/v1/worker/gpu/spec_decode)：speculator/worker 层，负责 draft 执行和 rejection sampling

---

## 2. 整体架构：三层分工

vllm v1 的 SD 实现分三层，职责非常清晰：

```
[Proposer 层]      负责"生成 draft tokens"的算法逻辑
      ↓
[Speculator 层]    负责在 GPU 上执行 draft forward，管理 KV cache
      ↓
[RejectionSampler] 负责对比 draft/target 分布，接受或拒绝
```

对应代码：

- Proposer：[`llm_base_proposer.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/llm_base_proposer.py)、[`eagle.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/eagle.py)、[`draft_model.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/draft_model.py)、[`medusa.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/medusa.py)、[`ngram_proposer.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/ngram_proposer.py)
- Speculator：[`worker/gpu/spec_decode/speculator.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/speculator.py)、[`autoregressive/speculator.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py)
- RejectionSampler：[`rejection_sampler.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py)、[`rejection_sampler_utils.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py)

---

## 3. 计算流：一轮推理里发生了什么

先把整个 pipeline 的调用链看清楚，再拆细节。

`GPUModelRunner.execute_model()` 里，一轮 SD 推理的顺序是这样的：

```
① target_model.forward(input_ids=上一轮的 draft_tokens)
      ↓ 输出 target_logits 和 target_hidden_states
② RejectionSampler.__call__(target_logits, draft_logits)
      ↓ 输出 sampled_token_ids, num_sampled, num_rejected
③ speculator.propose(
       last_hidden_states=target_hidden_states,
       num_sampled=..., num_rejected=...,
       last_sampled=sampled_token_ids)
      ↓ 输出 draft_tokens [num_reqs, num_spec_steps]（供下一轮使用）
```

有几个值得注意的地方：

**target 模型 forward 的 input 是 draft tokens，不是真实 tokens**。target 模型一次性处理上一轮所有 draft tokens（相当于一次 prefill），这正是 SD 加速的核心——把 K 步串行 decode 换成 1 次 batched prefill，prefill 是 compute-bound 的，GPU 利用率远高于 decode 的 memory-bound。

**propose 在 rejection 之后**，用的是 target 模型刚跑出来的 hidden states。也就是说，draft 是为**下一轮**准备的，而不是当前轮。这个 pipeline 设计让 GPU 几乎没有空闲：target forward 出来的 hidden states 立刻喂给 draft model，draft model 在生成 tokens 的同时，下一批请求的 prefill 已经可以开始了。

**`SpecDecodeMetadata`** 是连接三个阶段的数据结构（[`metadata.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/metadata.py)），核心字段：

| 字段 | 形状 | 作用 |
|---|---|---|
| `draft_token_ids` | `[num_tokens]` | 所有请求的 draft tokens 扁平拼接 |
| `num_draft_tokens` | `list[int]` | 每个请求各自 draft 了多少步 |
| `cu_num_draft_tokens` | `[batch_size]` | inclusive cumsum，供 kernel 按请求切分 |
| `logits_indices` | `[num_tokens + batch_size]` | 在 target logits 中 gather 对应行的索引 |
| `bonus_logits_indices` | `[batch_size]` | 每个请求 bonus token 在 target logits 中的位置 |

`logits_indices` 是 `target_logits_indices` 和 `bonus_logits_indices` 的拼接，让 rejection sampler 一次 gather 就能拿到所有需要的 logits，不用两次 IO。

---

## 4. Proposer 层：五种算法，两套继承体系

vllm v1 目前支持五种 draft 算法，分两个体系：

### 4.1 基于 LLM 的 Proposer：SpecDecodeBaseProposer

[`SpecDecodeBaseProposer`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/llm_base_proposer.py#L69) 是 EAGLE、DraftModel 等所有需要神经网络的 proposer 的公共基类，提供完整的 autoregressive drafting 循环。

`propose()` 方法（[第 502 行](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/llm_base_proposer.py#L502)）的主循环结构：

```python
# vllm/v1/spec_decode/llm_base_proposer.py
class SpecDecodeBaseProposer:
    def propose(self, ...):
        # 第一步 forward
        self.set_inputs_first_pass(...)         # 准备第一步的 input_ids / positions
        self.model(...)                         # draft 模型第一步 forward
        self._sample_draft_tokens(...)          # 采样第一个 draft token

        # 剩余 K-1 步循环
        for token_index in range(K - 1):
            self._update_positions_dependent_metadata(...)  # positions / slot_mapping + 1
            self.model(...)
            self._sample_draft_tokens(...)

        return torch.stack(draft_tokens)        # [batch_size, K]
```

**`EagleProposer`**（[`eagle.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/eagle.py)）继承自这个基类，整个文件只有 23 行，唯一的区别是初始化时传了 `pass_hidden_states_to_model=True`。

这个 `True` 影响深远：基类在每次 forward 时会把 target model 的 hidden states 拼入 EAGLE head 的输入（[第 730 行](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/llm_base_proposer.py#L730)），这正是 EAGLE 架构的核心——draft head 消费 target 的 residual stream，而不是独立跑一个小模型。

**`DraftModelProposer`**（[`draft_model.py`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/draft_model.py)）传的是 `pass_hidden_states_to_model=False`（[第 29 行](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/draft_model.py#L29)），有自己完整的 LM head，不共享任何 target 权重（`_maybe_share_embeddings` 和 `_maybe_share_lm_head` 均为空实现）。

这两种方案的本质区别：

| | EAGLE | DraftModel |
|---|---|---|
| Draft 来源 | 消费 target hidden states | 独立小模型，自给自足 |
| 额外显存 | 只有 EAGLE head（轻量） | 完整小模型（较重） |
| 依赖关系 | 强耦合 target 架构 | 与 target 架构无关 |
| 典型场景 | 与 target 同架构系列 | 跨模型族加速 |

### 4.2 MedusaProposer：并行多头预测

[`MedusaProposer`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/medusa.py#L40) 完全不继承基类，是独立实现。

原因很简单：Medusa 不是 autoregressive 的。它直接拿 target hidden states，让多个独立的 Medusa head 并行预测 K 个位置，没有循环、没有 KV cache、没有 slot mapping，`propose()` 的核心就三行：

```python
# vllm/v1/spec_decode/medusa.py
class MedusaProposer:
    def propose(self, ...):
        blocks = self.model(target_hidden_states)
        logits = self.model.compute_logits(blocks)
        return argmax(logits)  # [batch_size, num_heads]
```

极简，但有局限：各 head 之间完全独立，接受率通常比 EAGLE 低，尤其是 speculative steps 多的时候。

### 4.3 NgramProposer：不需要模型，纯 CPU

[`NgramProposer`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/ngram_proposer.py) 的 `load_model()` 是空实现——它根本不需要神经网络。

核心算法是 **KMP 变体的 n-gram 匹配**（[`_find_longest_matched_ngram_and_propose_tokens`，第 207 行](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/ngram_proposer.py#L207)）：

1. 把 context token 序列翻转，把"找最长后缀匹配"变成"找最长前缀匹配"
2. 用 KMP 的 lps（Longest Proper Suffix）数组在整个 context 里找最长满足 `[min_n, max_n]` 的匹配
3. 取匹配位置之后 K 个 token 作为 draft

整个 batch 并行处理用了 `@njit(parallel=True)`（Numba JIT，[第 177 行](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/spec_decode/ngram_proposer.py#L177)）。

**什么场景适合 ngram？** RAG、code completion、文档摘要这类有大量重复 pattern 的场景，接受率相当高，而且零额外显存开销，成本极低。

---

## 5. Speculator 层：Worker 上如何执行 Draft

[`BaseSpeculator`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/speculator.py#L31) 定义了三个抽象方法：

```python
# vllm/v1/worker/gpu/spec_decode/speculator.py
class BaseSpeculator(ABC):
    def init_cudagraph_manager(self, cudagraph_mode): ...
    def capture(self, attn_states): ...
    def propose(self,
                last_hidden_states, aux_hidden_states,
                num_sampled, num_rejected,
                last_sampled, next_prefill_tokens,
                temperature, seeds, ...): ...
```

`propose` 的 signature 包含了 `num_sampled`/`num_rejected`——上一轮 rejection sampling 的结果，draft model 需要用这个信息来"回退"被拒绝的 KV cache slots，把 positions 和 slot mapping 重置到正确状态。

[`AutoRegressiveSpeculator`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py#L30) 是核心实现，`propose()` 的执行流（[第 127 行](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py#L127)）：

```python
# vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
class AutoRegressiveSpeculator(DraftModelSpeculator):
    def propose(self, ...):
        self.prepare_prefill_inputs(...)    # Triton kernel：shift input_ids，定位 last_token_indices
        self._prefill(...)                  # draft 模型第 0 步 forward，写 KV，生成 draft_tokens[:,0]
        self.prepare_decode_inputs(...)     # positions/seq_lens + 1，input_ids ← draft_tokens[:,0]
        self._multi_step_decode(...)

    def _multi_step_decode(self, ...):
        for step in range(1, K):
            self._build_draft_attn_metadata(...)  # 重算 slot mapping
            self._run_model(...)                   # draft forward，写新 KV
            self.sample_draft(...)                 # argmax / gumbel
            self.update_draft_inputs(...)          # Triton kernel：更新 input_ids/positions
```

**KV cache 怎么管的？** Prefill 阶段直接复用 target 模型的 `attn_metadata` 和 `slot_mappings`（代码注释在 [第 223 行](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py#L223)），因为 draft 的 batch shape 与 target 完全一致。Decode 阶段每步调 `block_tables.compute_slot_mappings()` 重算，draft 模型把新 KV 写入自己专属的 layer（`draft_attn_layer_names`），不覆盖 target 的 KV。

[`EagleSpeculator`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py#L12) 继承自 `AutoRegressiveSpeculator`，同样只重写了一个方法 `load_draft_model()`，用 `load_eagle_model()` 加载 EAGLE 专用架构。所有 draft 执行逻辑完全继承，没有多余代码。

---

## 6. RejectionSampler：四个 Triton Kernel

[`RejectionSampler.__call__()`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py#L101) 的核心在 [`rejection_sample()`](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L864)，分四个 Triton kernel 阶段执行（[第 923 行起](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L923)）：

**Step 1：`_compute_local_logits_stats_kernel`**
按 `VOCAB_BLOCK_SIZE=8192` 分块，并行计算每个位置的 logit 统计量：greedy 模式下求 block-local argmax；非 greedy 模式下求 max 和 sum-exp，为后续 log-sum-exp 准备。

**Step 2（block verification 模式专有）**
- `_compute_cumulative_log_p_kernel`：计算前缀联合接受率 `log_p_i = sum min(log(p/q), 0)`
- `_compute_local_residual_mass_kernel`：计算残差质量 `max(p_i * M_b(x) - M_s(x), 0)` 的分块偏积

**Step 3：`_rejection_kernel`**（[第 459 行](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L459)）
每个请求一个 warp，顺序检查每个 draft token：

```python
# vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
# _rejection_kernel（Triton kernel，第 459 行，每个请求一个 warp）

@triton.jit
def _rejection_kernel(...):
    for i in range(num_draft_tokens):
        # 标准 rejection sampling（非 greedy，Leviathan et al. 2023）
        accepted &= target_log_prob > log(u) + draft_log_prob  # 概率比检验

        # greedy 模式
        target_argmax = compute_global_target_argmax(...)
        accepted &= (target_argmax == draft_sampled)            # 精确匹配

    rejected_steps[req] = first_rejected_index
```

**Step 4：`_resample_kernel` + `_insert_resampled_kernel`**
对被拒绝位置和 bonus token 重采样：
- 有 draft logits → 从残差分布 `max(p - q, 0)` 采样（[第 752 行](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L752)）
- 无 draft logits（one-hot draft，如 ngram）→ 把被拒绝 token 在 target 分布里置 `-inf`，再采样（[第 768 行](https://github.com/vllm-project/vllm/blob/ba22152/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L768)）

Resample 用的是 **gumbel block argmax**（`RESAMPLE_BLOCK_SIZE=1024`），并行分块，不需要全 vocab 的 softmax 展开。

---

## 7. 如何扩展自定义 SD 算法

看完这套架构，扩展点很清晰。举个最简单的例子：实现一个**基于 prompt 前缀重复模式的 draft proposer**——如果当前 context 末尾 token 和 prompt 里某段相同，就拿后续 token 作为 draft，不需要任何模型推理。

这种算法不依赖神经网络，也不需要 autoregressive 循环，最适合直接独立实现 `propose()` 接口，和 `NgramProposer` 的扩展方式一致：

```python
# my_proposer.py
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

class MyPrefixProposer:
    """
    在 prompt token 里找与 context 末尾匹配的前缀，
    取后续 num_speculative_tokens 个 token 作为 draft。
    """
    def __init__(self, vllm_config, device):
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
        )

    def load_model(self, target_model, target_attn_layer_names):
        pass  # 不需要模型

    def propose(
        self,
        requests,              # List[CachedRequestState]
        token_ids_cpu,         # numpy array，所有请求的 token 序列
        position_ids_cpu,
        input_batch,
        ...
    ):
        draft_token_ids = []
        num_draft_tokens = []

        for req in requests:
            ctx = token_ids_cpu[req.start : req.end]  # 当前 context
            prompt = ctx[: req.prompt_len]            # prompt 部分
            suffix = ctx[-3:]                         # 取末尾 3 token 做匹配 key

            # 在 prompt 里找 suffix，取后续 token 作为 draft
            drafts = self._find_continuation(prompt, suffix,
                                             self.num_speculative_tokens)
            draft_token_ids.extend(drafts)
            num_draft_tokens.append(len(drafts))

        return SpecDecodeMetadata(
            draft_token_ids=torch.tensor(draft_token_ids, device="cuda"),
            num_draft_tokens=num_draft_tokens,
            ...
        )

    def _find_continuation(self, prompt, suffix, k):
        for i in range(len(prompt) - len(suffix)):
            if (prompt[i : i + len(suffix)] == suffix).all():
                return prompt[i + len(suffix) : i + len(suffix) + k].tolist()
        return []
```

核心只有两步：在 prompt 里找匹配，拼 `SpecDecodeMetadata` 返回。rejection sampling 和 verify 完全复用 vllm 已有的逻辑，不需要动任何其他文件。

如果算法是 autoregressive 的（需要 draft model forward），则继承 `SpecDecodeBaseProposer`，覆写三个方法：

```python
# vllm/v1/spec_decode/draft_model.py（参考）
class DraftModelProposer(SpecDecodeBaseProposer):

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        # 把 draft 模型的 model_config / parallel_config 替换进去
        return replace(base, model_config=spec.draft_model_config, ...)

    @override
    def _get_model(self) -> nn.Module:
        # 加载 draft 模型权重
        return get_model(vllm_config=self._create_draft_vllm_config(), ...)

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        pass  # DraftModel 不共享权重；EAGLE 子类这里会做权重绑定
```

`propose()` 的主循环不需要动，基类已经实现好了。

**目前的限制**：vllm v1 的 SD 实现强假设 draft 和 target 共享同一套 block table，如果要实现完全独立的 KV 管理（如稀疏 KV），需要改动 `BlockSpaceManager` 和 `DraftModelSpeculator.set_attn()`，改动面较大。

---

## 8. 聊聊 Self-Speculative Decoding

上面说的都是 **draft model 是独立模型** 的方案。另一个思路是 Self-Speculative Decoding（SSD）——不引入额外模型，用同一个模型生成 draft。

主要有三条路：

| 方案 | 核心思路 | 代表工作 |
|---|---|---|
| Early Exit | 前几层 transformer 输出直接过 LM head 得 draft | 多篇论文均有实现 |
| Layer Skip | 跳过部分中间层，用稀疏子网络做 draft | Self-SD (Tang et al.) |
| Sparse Attention | draft 阶段只加载部分 KV cache，verify 全量 | SparseSpec |

vllm v1 目前原生不支持 SSD，原因正如上节说的——架构上强依赖独立 draft model 的抽象。

[SparseSpec](https://github.com/sspec-project/SparseSpec)（arXiv 2512.01278）是稀疏 attention 路线的一个工程实现，核心思路是：**draft 阶段只读 Top-K 重要 token 的 KV，verify 阶段才用全量 KV**，从而把 draft 的显存带宽消耗压缩到 5% 左右（`budget_ratio=0.05`），专门针对 Reasoning LLM 长序列 decode 的 memory-bound 瓶颈。

因为这套思路和 vllm 的标准 SD 接口不兼容（vllm 假设 draft 和 target 共用同一套 block table，没有稀疏索引的概念），SparseSpec 选择完全绕开 vllm，**重新实现了整个 serving stack**，主要改了三层：

**1. 注意力后端（`serve/attention/backend.py`）**

基于 FlashInfer JIT 扩展，实现了两个自定义注意力变体：

```python
# serve/attention/backend.py
class BatchTopKAttention:
    # Draft 模式：REGISTER_INPUT_TRANSFORM 把 KV index 重定向到 flatten_indices（稀疏索引）
    core_variant_decl = """
    REGISTER_INPUT_TRANSFORM(sparse_kv_transform, {
        if (request_type[request_idx] == DRAFT) {
            kv_idx = flatten_indices[kv_idx];  // 只读 Top-K token 的 KV
        }
    });
    """

class BatchAttentionScore:
    # Verify 模式：REGISTER_LOGITS_TRANSFORM 把注意力 logits dump 出来
    score_variant_decl = """
    REGISTER_LOGITS_TRANSFORM(dump_attn_scores, {
        if (request_type[request_idx] == VERIFY) {
            dump_logits[...] = logits;  // 供后续 Top-K 更新使用
        }
    });
    """
```

draft 和 verify 通过 `request_type` 枚举在同一个 batch 里混跑，不需要两次单独的 forward。

**2. KV Cache 管理（`serve/request/kv_cache_ptr/pillar.py`）**

实现了 `PillarCachePtr`，在标准 KV cache 之上维护一套动态更新的稀疏索引：

```python
# serve/request/kv_cache_ptr/pillar.py
class PillarCachePtr(StreamingCachePtr):

    def dispatch_prepare_fwd_metadata(self, ...):
        # 每步 draft 前：把 Top-K 索引打包成 flatten_indices，传给注意力 kernel
        self.concat_kv_indices_contiguous_kernel(
            self.selected_indices_buffer_gpu,  # Top-K token 索引
            self.recent_indices,               # 最近 token 索引（anchor）
            self.flatten_indices,              # 输出：拼接后的稀疏索引
        )

    def dispatch_update_selected_indices(self, ...):
        # 每次 verify 后：用 dump_logits 跑 Top-K，更新 Pillars
        self.topk_wrapper.run(
            self.dump_logits_snapshot,         # verify 阶段 dump 的注意力分数
            self.selected_indices_buffer_gpu,  # 更新 Top-K 索引
            k=self.num_selected_tokens,
        )
```

每隔 `spec_stride`（默认 16）步做一次全量 verify，同时更新 Pillars；draft 阶段只读当前 Pillars 对应的 KV。

**3. 调度器（`serve/scheduler/spec_scheduler.py`）**

引入了**错位编排**（staggered scheduling）：新请求进来时分配一个偏移 `cur_spec_idx`，使 batch 内不同请求的 verify 时机错开，把 verify 的计算峰值均摊到每一步。同时引入了 4 个请求状态：

```python
# serve/request/request.py
class ReqExecType(Enum):
    NORMAL = 0   # 普通 prefill
    DRAFT  = 1   # 稀疏 KV draft
    VERIFY = 2   # 全量 KV verify + rejection sampling
    STALL  = 3   # 等待 CPU 侧 Top-K 计算完成
```

`STALL` 是因为 Top-K 更新涉及 CPU→GPU 的 index 拷贝，需要一个缓冲状态等待结果就绪，才能进入下一轮 draft。

这套设计灵活，但代价是完全脱离了 vllm 生态——无法作为 vllm 插件直接部署，要用 SparseSpec 需要换掉整个 serving stack。README 里也明确说 "We plan to upstream a subset of features to vLLM in the future"，但目前还是 PoC 状态。

---

## 9. 小结

回顾一下这篇聊的内容：

- **整体分三层**：Proposer（算法逻辑）→ Speculator（GPU 执行 + KV 管理）→ RejectionSampler
- **计算流**：target forward → rejection sampling → speculator.propose（为**下一轮**准备 draft）
- **Proposer 两套体系**：基于 LLM 的（EAGLE/DraftModel 共用基类，`pass_hidden_states_to_model` 是核心开关）+ 独立实现的（Medusa 并行多头，ngram 纯 CPU）
- **RejectionSampler 四阶段 Triton kernel**：logit 统计 → block verification（可选）→ 逐 token 接受/拒绝 → 残差重采样
- **扩展点**：继承 `SpecDecodeBaseProposer` 改动最小；实现 SSD 类算法需要更深层改动

Self-Speculative Decoding 整体还很早期，尤其是怎么优雅地 fit 进 vllm 这套 block table 架构，是个值得深挖的工程问题。如果你也在做这个方向，欢迎评论区交流。

---

> 如果你对大模型方向感兴趣，我们团队也出了一本[《动手学 AutoML：从 NAS 到大语言模型优化实战》](https://item.jd.com/14945889.html)——书的核心是 AutoML 和 NAS，以及它们在大模型上的应用（NAS for LLM、AutoML for LLM Agent 等），和本文的推理优化方向相邻，感兴趣的话可以翻翻。
>
> ![动手学AutoML书籍封面](https://github.com/marsggbo/marsggbo.github.io/blob/master/assets/img/book_cover_automl.png?raw=true)
