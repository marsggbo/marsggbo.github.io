---
layout: post
title: "LLM Agent 真的有「记忆」吗？我们写了篇综述来说清楚这件事"
date: 2026-05-19
tags: [LLM, Agent, Memory, KV Cache, RAG, 论文解读]
---

# LLM Agent 真的有「记忆」吗？我们写了篇综述来说清楚这件事

> 原文：[LLM Agent Memory: A Survey from a Unified Representation–Management Perspective](https://www.preprints.org/frontend/manuscript/cb9b2d755639339002ad1d6da7a9e230/download_pub)

---

## 1. 前言

今天想和大家聊聊我们自己做的一篇综述——**LLM Agent Memory**。

是的，是「我们」。这篇文章有我（Xin He）的名字在上面，是跟 A\*STAR CFAR 和 HKUST 的同事们一起写的，所以这次解读算是第一视角了，不会有任何信息差，该踩的坑我都踩过（笑）。

写这篇综述的起因很简单：我们发现，**关于 LLM memory 的研究已经乱成一锅粥了**。

同一个东西，有人叫 "long-term memory"，有人叫 "episodic memory"，有人叫 "KV cache memory"，有人叫 "parametric memory"——不同文章用不同术语描述的可能是同一件事，也可能是完全不同的两件事，但你根本分不清楚。更糟的是，现有的综述要么只聚焦 RAG，要么只看 KV cache 优化，要么只谈 knowledge editing，**没有人把这些东西放在同一个框架下统一比较**。

这就是我们这篇综述要解决的核心问题：**给 LLM memory 建一套统一的分类体系，让不同方向的工作可以在同一语言下对话**。

---

## 2. 为什么 LLM 的「记忆」这么难搞？

先交代下背景。

大家都知道，LLM 有个根本性的限制：**context window 是有限的**。不管是 4K、128K 还是 1M token，总有个上限。但实际应用场景——多轮对话、长期任务、agent 系统——往往要求模型能跨会话保留信息，记住用户偏好，甚至在几十轮交互之后还能准确回忆起第 3 轮说过的事情。

这就产生了一个根本矛盾：**有限的上下文窗口 vs. 无限延伸的交互需求**。

为了解这个矛盾，学术界和工业界提出了各种各样的解法，大致可以分三类：

- **塞更多文字进 context**（RAG、agentic memory）
- **压缩/缓存中间状态**（KV cache 复用、streaming attention）
- **把知识直接写进模型权重**（continual learning、model editing）

问题是，这三条路子背后的哲学完全不同，工程实现也天差地别，但文献里经常混在一起讲，导致读者很难建立整体认知。

我们这篇综述的切入点，就是把这些路子用一个**统一的抽象**穿起来。

---

## 3. 核心框架：两个维度，一套接口

我们提出的框架沿着两个维度展开：

### 3.1 Memory Representation（存在哪、存成什么形式）

三种：

**① Token-level Memory（文本 token）**
最直觉的形式——把信息存成自然语言，塞回 context 窗口。RAG 是典型代表，把外部知识库检索出来的片段插进 prompt 里。透明度最高，最容易调试，但有两个硬伤：计算复杂度 $O(n^2)$，而且超长 context 下模型容易忽略中间的内容（"lost in the middle"问题）。

**② Intermediate Latent Memory（中间表示，KV Cache）**
把注意力机制的 KV 矩阵当作记忆载体。这是目前 LLM 推理优化最热的方向——KV cache 本质上记录了模型处理过的所有 token 的内部状态，可以跨 decode 步骤复用，甚至跨请求复用（prefix caching）。效率最高，latency 最低，但容量受限、不跨会话持久、也很难人工检查里面存了什么。

**③ Parameter-level Memory（模型参数）**
知识直接编码进权重——这是 pretraining 和 fine-tuning 的本质。知识最持久，不依赖任何外部存储，但写入代价极高，而且一旦更新容易产生 catastrophic forgetting。

如下图，三种 representation 各自对应不同的应用场景和工程约束：

![LLM memory 统一框架：从应用需求到记忆表示](/assets/img/posts/llm-agent-memory/emb_004_p3.png)

### 3.2 Memory Management（怎么操作这些记忆）

不管哪种 representation，都要经历三个管理操作，这也是我们的核心贡献之一——**用同一套接口描述所有 memory 机制**：

- **Construction（建立）**：往记忆里写什么、怎么组织索引
- **Update（更新）**：如何增量修改、如何避免遗忘旧知识
- **Query（查询）**：如何在推理时高效检索到相关内容

每种 representation 在这三个操作上的挑战都不一样：

| Memory 类型 | Construction | Update | Query | 战略价值 |
|---|---|---|---|---|
| Token-level | 中 | 中 | **高**（query 是瓶颈） | Editability |
| Intermediate latent | 低 | **高**（容量限制） | 低 | Efficiency |
| Parameter-level | **高**（训练成本） | **高**（遗忘风险） | 低 | Persistence |

这张表格是我觉得整篇综述最有价值的东西之一。**representation 的选择本质上是在把系统瓶颈移到不同的管理操作上**，而不是消灭瓶颈。

---

## 4. 逐层拆解：每种 Memory 具体是怎么做的

### 4.1 Token-level：RAG 和 Agentic Memory

Token-level 这部分涵盖两大方向：

**RAG** 大家都熟——建向量索引、检索、插 context。但细节里有很多门道：检索粒度（chunk 大小）影响 precision/recall 的 trade-off，混合稀疏-稠密检索能提升鲁棒性，query 改写（扩展、分解、step-back prompting）能显著提升召回质量。RAG 的更新策略也不是简单的重新 embed，要考虑索引结构的演化（从 flat chunking 到 hierarchical/graph-based）。

**Agentic Memory** 则更进一步——agent 把自己的历史行动、观察、结论都存成文本，通过摘要压缩和选择性遗忘来维护一个可用的长期记忆库。这类系统面临的核心挑战是：**随着 memory 越积越多，如何保证检索出来的还是高信噪比的内容**。

### 4.2 Intermediate Latent：KV Cache 的花式玩法

KV cache 这部分最接近我日常研究的方向，踩的坑也最多。

除了标准的 prefix caching，这里覆盖了一大批工作：

- **KV cache 压缩**（StreamingLLM、SnapKV 等）：selective eviction，只保留最重要的 token 对应的 KV，丢掉其他的
- **KV cache as persistent memory**（CAMELoT、MemOS、Memory3）：把 KV 外化成可存储的 memory pool，支持跨会话复用
- **Steering vectors**：用 activation space 里的方向向量控制模型行为，也算是一种 intermediate latent memory，只是存的不是知识而是"行为偏好"

这条路的核心矛盾：**KV cache 高度依赖特定模型架构，换个模型就得重新存**，不像文本 memory 那么通用。

### 4.3 Parameter-level：把知识写进权重

这部分最"重"也最受限：

- **Pretraining / Continual Pretraining**：从零或增量地把知识编码进权重。写入代价极高，动辄几百个 GPU 小时
- **Knowledge Editing**（ROME、MEMIT 等）：外科手术式地修改特定的事实知识，不重新训练
- **Continual Learning**（EWC 等）：在学新知识时通过正则化保护旧知识，对抗 catastrophic forgetting
- **LoRA / PEFT**：轻量级 adapter，用少量参数记录任务特定知识

**参数级记忆的核心矛盾**：写入代价太高，而且更新风险大——稍不注意就会破坏原有的能力。这也是为什么实际系统往往把稳定的、通用的知识放权重，把动态的、个性化的放外部 memory。

---

## 5. 最重要的 Takeaway：如何选择 Memory 机制

这是我觉得对工程师最有用的部分——**给定一个实际任务，我应该用哪种 memory**？

论文里提出了一个决策框架，基于两个维度：

- **Retention**：需要短期记住还是长期记住？
- **Functional form**：记的是 episodic（具体事件）、semantic（概念知识）还是 procedural（操作技能）？

如下图的 Decision-support matrix：

![Memory 选择决策矩阵：根据任务需求选择合适的 memory 类型](/assets/img/posts/llm-agent-memory/page_010.png)

实践建议（直接从论文里翻译过来，因为原文说得很精准）：

> - **Token memory**：用于频繁变化或需要外部接地的信息（RAG 外部知识库、用户历史对话）
> - **Latent memory（KV cache）**：用于一次会话内的短期推理状态
> - **Parametric memory**：用于稳定的、值得长期固化的知识

实际的健壮系统往往**三层都有**，组成一个 memory hierarchy，在灵活性、效率和持久性之间取得平衡。这其实和操作系统的存储层次结构（寄存器 → L1/L2 cache → RAM → 磁盘）有异曲同工之妙，只是换了个场景。

---

## 6. 个人 take：这篇综述解决了什么真问题

说实话，写综述这件事很磨人——要读的文章数量级完全不一样，还要在各种不同框架下找到统一的切入点。

但我觉得这篇综述做对了一件事：**把 memory 从"任务特定的工程技巧"提升到了"系统级能力"来看待**。

以前大家讲 RAG 就只讲 RAG，讲 KV cache 压缩就只讲 KV cache，两件事背后明明都是在解决同一个问题（如何让模型高效利用历史信息），但因为技术栈不同、社区不重叠，这两个方向的研究者几乎没有任何交流。

这篇综述的价值，更多是在**给领域提供公共语言**，而不是提出某个新算法。对于想系统理解 LLM memory 全景的人来说，这个框架（representation × management）应该能帮你快速对齐认知，避免"盲人摸象"的情况。

未来方向上，我们觉得几个问题值得关注：

1. **跨 memory 类型的联合优化**：token memory 和 KV cache memory 目前是割裂设计的，有没有统一的系统可以根据任务自动选择？
2. **memory 的评测标准缺失**：现在各个方向都用自己的 benchmark，很难横向比较
3. **OS/DB 领域的技术迁移**：内存管理、缓存替换、数据库索引这些成熟技术，在 LLM memory 里的应用才刚刚开始

好了，今天就聊到这里。

感兴趣的同学可以去看原文，链接在文章开头。欢迎评论区交流，或者直接找我（A\*STAR CFAR, hexin@cfar.a-star.edu.sg）。如果有哪里理解不对或者有不同看法，也欢迎指出来，毕竟作为作者之一，有机会出个 v3 的话还可以改（笑）。
