---
layout: post
title: "LMSYS'26 | Unified Radix Cache：同一个 prefix，为什么不能只用一个 cache boundary？"
date: 2026-08-12
tags: [LLM, 推理优化, Prefix Caching, SGLang, 系统解读]
---

# LMSYS'26 | Unified Radix Cache：同一个 prefix，为什么不能只用一个 cache boundary？

> 原文：[Unified Radix Cache: One Tree for Hybrid Model Prefix Caching](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache)
>
> 相关项目：[SGLang](https://github.com/sgl-project/sglang)

## 1. 先把问题说人话：缓存的不是“字符串”，而是模型状态

你有没有想过，为什么同一个 prompt，第一次请求很慢，第二次却可能快很多？

模型处理输入时，先有一段 **prefill**：把整段 prompt 一次性送进模型，计算每个 token 对应的中间状态；随后进入 **decode**，一个 token 一个 token 地生成答案。对于 Transformer 来说，注意力层会为历史 token 保存 key/value，后续 token 可以直接读取，这些历史状态就叫 **KV cache**。

**Prefix caching** 做的事情，就是把多个请求共同拥有的 prompt 前缀的 KV cache 留下来。第二个请求如果前 10,000 个 token 一样，就不必重新计算这 10,000 个 token 的 prefill。对长上下文、agent 多轮轨迹、代码仓库问答来说，这往往比再挤出一点 GPU FLOPS 更有价值。

传统实现通常隐含了一个很强的假设：

> 一段 token prefix 对应一个连续的、大家都认可的 reusable boundary（可复用边界）。

在纯 **full attention** Transformer 中，这个假设几乎没问题。full attention 指每个 token 都可以访问完整历史；因此前缀相同，前缀产生的 KV 就可以从头到尾接着用。

但模型开始混搭之后，事情就不再这么简单了。现在的 hybrid model 往往在同一个网络里同时出现 full attention、**sliding window attention（SWA，滑动窗口注意力）**，以及 **Mamba/SSM（用 recurrent state 压缩历史的状态空间模块）**。它们看到的是同一串 token，却不一定保存同一种状态，也不一定在同一个位置允许复用。

这正是 Unified Radix Cache 要解决的工程问题：**如何让不同组件共享一棵前缀树，但各自保留自己的复用语义。**

## 2. 读懂后面的设计，只需要先分清三个概念

原文默认读者已经熟悉 serving 系统，所以几个关键概念一闪而过。这里先把它们拆开，否则后面很容易把“树上的同一个节点”误解成“所有组件都能使用同一份物理 cache”。

### 2.1 Prefix identity：这段 token 在哪里

假设有三条请求：

```text
system, 你是助手, 解释 MoE
system, 你是助手, 解释 prefix cache
system, 你是助手, 解释 SWA
```

三条请求共享前两个片段。Radix tree（基数树）就是把共享前缀压缩成一条路径，分叉部分再各自展开。它维护的是一种**规范化的 token 坐标**：某个 token prefix 是谁，它在树上和哪些请求共享。

这棵树本身并不承诺“所有状态都能复用”。它更像缓存的目录和索引，而不是某一个模型组件的完整实现。

### 2.2 Reuse validity：这个组件能安全用到哪里

同一个树节点，对不同组件可能有不同答案。full attention 可能接受整条路径；SWA 只接受最近的窗口；Mamba 只接受某个合法的 recurrent checkpoint。

所以必须把“前缀相同”与“状态可复用”分开：前者是身份问题，后者是组件语义问题。

### 2.3 Payload location：状态现在放在哪

即使某份状态有效，它也可能在 GPU、CPU 内存或外部内存池。HiCache 的 L1/L2/L3 正是在管理这个维度：L1 通常是 GPU 显存，L2 是 Host 内存，L3 是更大的远端或分布式存储。

**换存储层不应该改变 prefix identity。** 一段前缀从 GPU 被淘汰到 Host，之后又被预取回 GPU，树上仍然是同一个节点，只是 payload 的住址变了。

这三个概念对应三个问题：

```text
prefix identity       这是谁？
reuse validity         谁能用？能用到哪？
payload location       现在放在哪？
```

Unified Radix Cache 的关键，就是不再用一个 cache class 同时回答这三个问题。

## 3. 为什么 FULL、SWA、MAMBA 共享同一个 prefix，却不能共享同一个边界

用一段具体前缀看最清楚。假设 token 是：

```text
h0  h1  h2  h3  h4  h5
```

### 3.1 FULL：整条路径都是状态

full attention 为每个位置保存 KV。可以把它想成自己的物理池：

```text
FULL: F0  F1  F2  F3  F4  F5
```

只要 token 序列从 `h0` 到 `h5` 完全一致，FULL 就可以把整条匹配路径交给新请求。

### 3.2 SWA：文本前缀很长，真正需要的可能只有尾窗

SWA 每层只看最近 `W` 个 token。假设窗口大小为 2，那么在处理后续 token 时，`h0` 到 `h3` 已经不在这层的可见窗口内了。它仍然存在于 canonical tree 中，却不代表 SWA 还需要或能够复用它。

SWA 可以维护自己的物理池：

```text
SWA:       S0  S1       （对应逻辑上的 h4, h5）
```

这里 `S0/S1` 不是 `F4/F5` 的别名，而是 SWA 自己的 slot。逻辑上两者对齐，物理上仍然独立。这样做是因为不同组件可能采用不同布局、页大小、生命周期或复制策略。

### 3.3 MAMBA：复用的是 checkpoint，不是任意一段 KV

Mamba/SSM 不把完整历史表示成 attention KV，而是递推一个 state：处理 `h0` 得到 `s0`，处理 `h1` 得到 `s1`，依次更新。新请求要从某个位置继续，必须拿到该位置对应的合法 state checkpoint。

更麻烦的是，后续 decode 会继续修改这个 recurrent state。多个请求不能直接共同写同一个 state，否则一个请求生成的 token 会污染另一个请求。因此常见做法是：命中共享 checkpoint 后，把它复制到请求私有的 slot，再继续更新。

于是，对同一段 `h0...h5`：

- FULL 的 reusable boundary 可以在 `h5`，并且整条路径都可用；
- SWA 可能只对 `h4,h5` 建立可用窗口；
- MAMBA 可能只在 `h5` 保存一个可复制的 state checkpoint。

**同一个 prefix identity，不等于同一个 reusable boundary，更不等于同一个物理 index space。** 这句话如果不提前说清楚，Figure 1 里三张卡片看起来就像“同一份 cache 被重复画了三遍”。

## 4. 一棵树到底统一了什么

既然三类组件的状态不同，最直接的做法似乎是分别实现三套 cache。但一旦模型组合变多，工程会迅速变成 class matrix explosion：FULL、FULL+SWA、FULL+MAMBA、FULL+SWA+MAMBA，每加一种组件还要和 HiCache 的每个层级再组合一次。

Unified Radix Cache 选择了一个更克制的拆分：

- **树核心统一 prefix identity**：所有请求沿同一套 token radix topology 匹配、分裂和插入；
- **组件自己验证 reuse validity**：FULL、SWA、MAMBA 各自决定某个位置是否安全；
- **HiCache 管理 payload location**：把有效 payload 在 L1/L2/L3 之间搬运；
- **sidecar 跟随 source pool**：附属数据跟着主组件移动，不额外制造复用语义。

因此，“Unified”不是强迫不同模块使用同一种 cache，而是让它们共享不需要重复实现的那部分。

![Unified Radix Cache 总览](/assets/img/posts/20260812-unified-radix-cache/image1.svg)

Figure 1 放在这里，应该当作这套拆分的总览，而不是某个抽象口号的插图：中间是一棵 canonical token tree，左边是不同组件的复用规则，右边是同一份 payload 的分层驻留位置。

## 5. 最长匹配不等于安全复用边界

有了统一树，匹配流程仍然要解决一个细节：**树可以继续往深处走，但组件的验证结果可能不一致。**

假设请求的 token 与树上的 `n1 -> n2 -> n3 -> n4` 一直匹配。遍历当然可以走到 `n4`，但系统要为所有 active components 找一个共同安全边界：

```text
             n1    n2    n3    n4
FULL          ✓     ✓     ✓     ✓
SWA           ✓     ✓     ✗     ✗
MAMBA         ✓     ✓     ✗     ✓
```

这里的 `✗` 不是“整棵树到此为止”。它只表示某个组件在这个候选位置没有可用状态。MAMBA 在更深的 `n4` 重新出现 `✓` 也说明了：组件的合法边界未必单调，不能遇到一次失败就停止遍历。

真正提交给请求的 reusable boundary，需要所有参与本次复用的组件都同意。因此上面的例子最终停在 `n2`，即使 traversal 已经看到了 `n4`。

这也是实现上最容易写错的地方：

1. **遍历层**负责沿 canonical token path 找候选节点；
2. **组件验证层**对候选节点给出自己的接受/拒绝；
3. **边界聚合层**取 active components 的共同安全结果。

如果只实现第 1 步，系统会把“最长文本匹配”误当成“最长状态匹配”，在 SWA 或 recurrent state 上产生错误复用。

![match_prefix 如何选安全边界](/assets/img/posts/20260812-unified-radix-cache/image2.svg)

Figure 2 的意义也就在这里：它不是在展示一次普通的树遍历，而是在强调“走得更深”和“能安全复用”是两件事。

## 6. Component、anchor、sidecar：三种对象不要混为一谈

组件化并不意味着每个数据结构都应该成为 component。更准确的分工是：

- **Component** 会参与边界判断，定义自己的状态池和生命周期；
- **Anchor** 是组件在 canonical tree 上的对齐位置，用来表达“这个状态对应哪段 token”；
- **Sidecar** 不定义新的复用边界，只跟随某个 source pool 的分配、迁移和淘汰。

仍然看 `h0...h5` 的例子：FULL 可以覆盖 `[0:6)`，SWA 只覆盖 `[4:6)`，并通过 `F4 -> S0、F5 -> S1` 建立逻辑对齐。`C4 KV`、`C128 KV`、indexer 或 compressor state 之类附属数据，则可以声明自己跟随 FULL 或 SWA。

![DeepSeek-V4 组件与 sidecar 关系](/assets/img/posts/20260812-unified-radix-cache/image3.svg)

为什么要这么严格？因为 sidecar 如果也向树里注册一套独立语义，系统就会把“压缩数据有没有搬到位”误认为“前缀是否可复用”。最后不仅类变多，边界判断也会被无关的数据结构绑架。

## 7. 树核心提供生命周期，组件提供语义钩子

统一树不是一个只读索引。真实 serving 系统要不断执行 match、split、insert、lock、evict：请求来了要匹配，前缀在中间分叉要切节点，新 token 要插入，正在使用的区域要加锁，空间不足要淘汰。

这些动作的树形 bookkeeping 可以统一，但每种组件在动作发生时要做的事情不同。例如：

- match 时，FULL 检查连续 KV，SWA 检查窗口范围，MAMBA 检查 checkpoint 是否存在且可复制；
- split 时，组件要决定自己的 pool 是否也需要切分，或者只保留一个 anchor；
- insert 时，SWA 可能只写入窗口，MAMBA 则更新 recurrent state；
- lock/evict 时，各组件根据自己的占用和引用关系处理物理 slot。

因此合理的接口不是“所有组件实现一棵树”，而是树核心提供统一生命周期，组件通过 hooks 注入语义。新增加一种 hybrid block，主要工作就变成实现这些 hooks，而不是复制整套 radix-tree 代码。

## 8. HiCache 改变的是 payload 位置，不是 prefix 身份

单层 GPU cache 的容量很容易被长上下文和多会话挤爆。HiCache 把缓存拆成多层：GPU L1 低延迟但容量小，Host L2 容量更大，external L3 可以进一步扩展容量。请求命中 L2/L3 时，需要把 payload 搬回更快的层级，或者直接在较慢层级上完成一部分准备工作。

这里最重要的不是“多了两层存储”，而是身份和位置被分开：

```text
同一个 radix node
    ├── FULL payload: 可能在 GPU L1
    ├── SWA payload: 可能在 Host L2
    └── MAMBA checkpoint: 可能在 external L3
```

它们仍然对应同一个 token prefix，只是不同组件、不同 tier 的 payload 状态不同。这样做也解释了为什么 sidecar 能自然接入：sidecar 只需要跟随 source pool 做迁移，而不必重新定义一棵树。

![HiCache 多轮 benchmark](/assets/img/posts/20260812-unified-radix-cache/image4.png)

原文的 multi-turn benchmark 测了 `DeepSeek-V4-Flash` 和 `Inkling-Small`。`TTFT`（Time To First Token）是从请求开始到第一个输出 token 的时间，越低越好；`prompt token hit rate` 是输入 token 中被缓存直接复用的比例，越高越好；`effective input token throughput` 还把省掉的前缀计算折算进去，更接近真实服务能力。

只有 L1 时，缓存很快被后续轮次挤掉，hit rate 下降、TTFT 上升；加入 L2 后能多撑一段；L3 则把不断增长的多轮共享前缀保留得更久。以图中 `DeepSeek-V4-Flash` 为例，effective throughput 从 L1 的 `9.4k`，经过 L1+L2 的 `14.3k`，到三层的 `145.5k`。这不是 GPU 突然变快了，而是重复 prefill 被大幅避免了。

## 9. LRU 只知道“刚才用过”，不知道“下一轮还会用”

传统 **LRU（least recently used）** 淘汰策略按最近访问时间排序。对独立请求这通常够用，但 agent 或多轮 session 有明显的时间结构：一个 session 可能刚完成一轮，下一轮还没到达，却大概率继续复用同一段前缀。

如果只看最近访问时间，这段数据会被当作冷数据淘汰。Unified Radix Cache 因此引入 **session reference**：每个请求携带稳定的 `session_id`，请求结束后，把该 session 仍可能复用的区域登记下来：FULL 记 path，SWA 记 trailing window，MAMBA 记 reusable frontier。

这里的 reference 是 eviction 的软信号，不是 pin。内存真的不足时，仍然可以淘汰所有节点；只是引用数低的区域优先被淘汰。关闭 session 时，先释放引用，不必立刻删除 cache entry，避免短暂的 session 生命周期把缓存反复抖动。generation 机制则用来防止旧请求晚到，把已经关闭的 session 错误地重新标活。

![Session-aware eviction 示意图](/assets/img/posts/20260812-unified-radix-cache/image6.svg)

Figure 5 里的 `ref=3/2/1` 应理解为组件本地的未来复用信号，而不是全局的“这个节点有几个人正在读”。FULL、SWA、MAMBA 统计的都是各自真正关心的 region。

## 10. 实验应该怎么看：命中率不是唯一答案

在 SWE-bench 这类 agent workload 上，session-aware eviction 的价值不一定体现为总 hit ratio 大幅增加。更重要的是：命中是否发生在 GPU 这种更快的层级。

![SWE-bench 上的 cache residency 和 TTFT](/assets/img/posts/20260812-unified-radix-cache/image7.png)

原文中，`DeepSeek-V4-Pro` 在 batch size 128 时 device hit ratio 从 `41.8%` 提升到 `51.0%`，总 hit ratio 只从 `96.3%` 到 `96.5%`，但 TTFT 仍下降 `11.0%`。这说明“总命中率几乎不变”并不等于“系统没有收益”：host 命中和 device 命中的延迟完全不同。

`Qwen3.5-397B-A17B` 的变化更明显，batch size 32 时 device hit ratio 从 `5.0%` 到 `33.6%`，batch size 64 时总 hit ratio 从 `57.2%` 到 `66.6%`，TTFT 分别下降 `13.5%` 和 `16.6%`。

不过这里要保留一个实验 caveat：这不是只切换 session-aware 开关的纯净 ablation，同时包含了 Unified Radix Cache 实现与普通 HiRadixCache + LRU 的整体对比。因此这些数字能证明方案在该 workload 上有效，但不能把全部增益严格归因到某一个模块。

## 11. Rust tree core：优化的是 bookkeeping，不是模型计算

当 prefix 很长、并发 session 很多时，树遍历、节点分裂、锁管理、LRU 更新和 eviction scan 会进入 scheduler 的关键路径。它们不是 GPU forward，却会影响请求什么时候真正开始 prefill。

![Rust vs Python tree core](/assets/img/posts/20260812-unified-radix-cache/image5.png)

原文的 Rust 实验是 L1-only prototype，测的是 tree core 的 native 化收益，并不是把整个 serving 系统重写成 Rust。`gpt-oss-20b` 的 SWA 场景平均 TTFT 降低约 `38%`，最后 25 轮降低约 `42%`；FULL 场景也有收益。hybrid SSM/Mamba 场景的 residual bookkeeping 开销下降明显，但总 TTFT 被更重的 GPU forward 稀释，所以端到端变化没有那么夸张。

这个结果的正确解读是：**当树管理成为瓶颈时，native tree core 值得做；但它不会消除模型本身的计算时间。**

## 12. 我的 take：这篇文章真正重新定义的是“缓存边界”

Unified Radix Cache 的价值不在于又发明了一种 radix tree，也不在于把某个 benchmark 刷到更高。它把过去经常揉在一起的三个问题拆开了：

1. token prefix 的身份由一棵 canonical tree 统一维护；
2. 每种模型组件自己决定状态在哪个边界可复用；
3. 有效 payload 放在 GPU、Host 还是 external tier，由 HiCache 管理；
4. sidecar 只跟随 source pool，不抢占复用语义；
5. session reference 给淘汰策略补上“未来还会不会用”的信息。

这套抽象直接回答了开头那个疑问：为什么 FULL、SWA、MAMBA 明明拥有同一个 prefix，KV cache 复用却不同？因为它们共享的只是 **prefix identity**，而不是状态表示、合法边界和物理内存布局。

我认为这是 hybrid model 时代更合理的 prefix caching 抽象。模型结构只会越来越混搭，如果每增加一种 attention/state 模块就复制一套 cache 实现，系统迟早会被组合数拖垮。**一棵树负责“这段前缀是谁”，组件负责“我能安全用到哪”，分层缓存负责“状态现在在哪”**。边界立对了，后面的 HiCache、session-aware eviction 和 Rust tree core 才能各自接上，而不必互相污染。

