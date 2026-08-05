---
layout: post
title: "LLM 学习笔记-Deepspeed-MoE 论文"
date: '2023-12-07'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/670968683
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/670968683>

论文 [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2201.05596.pdf)

## **1. Introduction**

现有的 MoE 方法在正式使用场景中存在的挑战：

1. 场景局限：大都是 encoder-decoder 模型或者 sequence-to-sequence 任务；
2. 训练时的内存需求巨大：
3. 推理性能还不太行：通常单个 GPU 放不下 MoE 做推理。另一方面多 GPU 的 MoE 推理方法还欠缺研究。MoE 推理还收到内存带宽的影响。

Deepspeed-MoE针对上述挑战做了下面的改进：

1. 把 MoE 的任务扩展到了各种自回归的 NLG 任务
2. 提出 PR-MoE 来减少 MoE 参数
3. 设计了 Deepspeed-MoE 推理系统，减少 7.3 倍推理延时和开销。

## **2. Related work**

* 先讲了 large-scale 的 dense model
* 然后引出 moe
* 最后介绍了一下现有的 MoE 训练和推理优化系统

+ 各种并行策略：dp, pp, zero, ep
+ fastmoe, tutel

## **3. 将 MoE 扩展到其他下游任务**

MoE 模型是基于 dense 模型设计的，即在 dense 模型的基础上把单个 MLP 替换成 多个 MLP，再加上 gate 等模块。例如 1.3B+MoE-128 就是基于参数量为 1.3B 的dense模型，每一层 MLP 替换成 128 个 expert 组成的模块。

![](/assets/img/marsggbo/2023-12-07-LLM-学习笔记-Deepspeed-MoE-论文/e4fc4578.jpg)

扩展的任务包括：

* completion prediction task (LAMBADA [16])
* common sense reasoning task (PIQA [35])
* 2 个 reading comprehension tasks (BoolQ [18], RACEh [36])
* 2 个 question answering tasks (TriviaQA [21], WebQs [20])

实验结果不赘述了，反正MoE赢过dense，而且开销更小就完事了。

## **4. PR-MoE & MoS：减少模型大小，提高parameter efficiency**

PR-MoE （Pyramid Residual MoE）结构如下

![](/assets/img/marsggbo/2023-12-07-LLM-学习笔记-Deepspeed-MoE-论文/3b55a380.jpg)

PR-MoE 的是基于两个现象设计的：

1. 现有的 MoE 每一层专家模块的 expert 数量都是一样，所以做了个对比试验：实验 A 指在模型前半部分添加专家模块，实验 B 则只在后半部分添加专家模块。实验结果显示 B 表现更高，所以说明模型深层（即后半部分）更需要专家模块；所以最终 PR-MoE 后半部分的专家数量会设置的更多一些。
2. 要提高 MoE 的泛化能力，有两种方法：

1. 增加 expert 数量，固定 expert capacity：这个的问题在于会增加阐述了和内存开销
2. 固定 expert 数量，增加 expert capacity：这个的问题在于会增加通信量

上面提到的 expert capacity 可以理解成每次选 top-k 个 expert 参与计算，K 越大，expert capacity 就越大，通常认为这样模型泛化能力越强。这是因为我们会认为不同 expert 彼此之间会做一个纠正，不至于预测结果偏差太大。所以一种思路是固定一个 expert，然后再动态选择一个 expert，这样就可以既减少了通信量（因为只有一个 expert 需要选择），又保证了 capacity。

MoE 训练加速的方法是专家并行，最简单和最高效的办法是每个 GPU 上均等平分 expert。例如总共有 64 个 expert，然后这个时候使用 64 个 GPU 的话，那就只需要 给每个 GPU 分配一个 expert就可以了。如果只有 32 个 GPU，那就每个 GPU 分配 2 个 expert。

但是现在PR-MoE 的训练的难点在于不同层 expert 的数量不一样了。其实，如果 GPU 数量少于最少的专家数量的话，还挺好划分的，简单平均分就好了。但是这种情况呢？假设有三层，每层专家数量分别是 {32, 64, 128}，GPU 数量是 128。按照原论文的说法，deepspeed-moe 支持这样的并行方式：

“a PR-MoE model running on 128 GPUs, with 32, 64, and 128 experts at different MoE layers, can be trained with 128-way data parallelism for the non-expert parallelism, and {32, 64, 128} expert parallelism plus {4, 2, 1} data parallelism for MoE parameters.”

简单来说，就是非 expert 模块采用 128 路的数据并行；但是对于专家模块，这才用专家并行和数据并行相结合的策略。例如第一层只有 32 个专家，那么我们就把第一层复制 4 遍，即对第一层做 4 路数据并行，每一路内部会做 32 路的专家并行。其他同理。

为进一步降低参数量，论文还用知识蒸馏得到更小的 MoE 模型，称作 Mixture of Students (MoS)。

## **5. 推理优化**

我们首先对于 1.3B+MoE-128 模型（总参数量是接近 52B）考虑两种推理的极端情况：

1. 只有一个输入的 token 数据做推理，那这个时候其实只会激活一条路径，所以只需要使用 1.3B 对应的内存开销
2. 当输入一个大 batch 的数据时，很可能每个 expert 都被激活了，那这个时候需要的是 52B 对应的内存开销

换言之，模型推理的开销介于 1.3～52B 之间。

Deepspeed-MoE 的优化思路是下面 3 条：

1. 支持各种并行策略混合，而且把路径相同的 token 打包一起做推理，这样每次就只需要激活一个路径了，即单个 dense 模型的开销。
2. 优化通信scheduling
3. 优化 transformer和 MoE 相关的 kernel

上面 3 条优化思路详细介绍如下：

## **5.1 灵活组合多种并行策略：TP，EP，DP**

简单理解就多种并行方式混合使用。下图是以单个 MoE 层为例解释了如何使用混合并行策略，假设总共有16 个 GPU，专家数量为 8 个，可以看到并行模式是针对专家和非专家模块分别设计的，具体如下：

* 对于非 expert 参数模块：

+ 使用4 路数据并行，也就是说该部分参数复制了 4 遍
+ 每一路数据并行的内部采用 4 路的tensor 并行

* 对于 expert 参数模块：

+ 使用 8 路专家并行
+ 每个专家通过 2 路的 tensor 并行进行参数拆分

![](/assets/img/marsggbo/2023-12-07-LLM-学习笔记-Deepspeed-MoE-论文/1babbfbe.jpg)

## **5.2 优化通信**

我们知道专家并行需要把专家分配到不同的 GPU 上，因此计算是时需要用到 all-to-all 通信来互相交换 token。论文作者发现使用基于 NCCL 的 torch.distributed 接口会有很大的 overhead，尤其是当规模上去的时候。于是他们改用了微软的 SCCL，实验结果显示比 NCCL 更好。作者还设计了两个新的通信优化策略：

### **5.2.1 Hierachical all-to-all**

简单来说，**分层 AlltoAll 设计**的工作原理如下：

1. 首先，每个 GPU 将其数据进行本地转换。这可以是任何类型的转换，例如将数据排序或压缩。
2. 然后，每个 GPU 将其转换后的数据与同一节点上的其他 GPU 进行交换。这可以理解成是**局部 AlltoAll**。
3. 最后，每个节点将其转换后的数据与其他节点进行交换。这可以理解成**全局 AlltoAll**。

假设总共有 $p$ 个 GPU，每个节点包含 $G$ 个GPU，所以总共有 $p/G$ 个节点。复杂度对比：

* 原本直接所有 GPU 之间做 all-to-all 的复杂度是 $O(p)$
* 现在分层 all-to-all 的复杂度为 $O(G+p/G)$包括两部分：

+ 局部通信复杂度：因为不同节点之间可以同时做通信，所以复杂度是$O(G)$
+ 全局通信复杂度：其实就是节点之间做通信，所以复杂度是 $O(p/G)$

![](/assets/img/marsggbo/2023-12-07-LLM-学习笔记-Deepspeed-MoE-论文/96475dc6.jpg)

### **5.2.2 Tensor 和 expert 协同并行的通信优化**

在插入 MoE 模块后的 transformer layer 通常长这样：inputs → MLP1 → MoE layer

为了加速我们可以对 MLP 层分别使用 tensor 并行，对 MoE layer做专家并行，这是前提。

* 我们先考虑在 MLP1 上执行 tensor 并行：我们知道 tensor 并行的特点是参与计算的 GPU 的输入数据是相同的，输出数据也是相同的（column parallel 需要用到 all-gather, row parallel 需要用到 all-reduce）。
* 接着在 MoE 上做专家并行：正如前面提到的，MLP1 最后在不同设备上的输出结果会保持一致，那么也就是说对于后面的 MoE layer 而言，同一个 tensor parallel group 的进程之间因为他们的输入数据是一样的了，所以就不需要再做all-to-all 通信了，也就是说 GPU0 和 GPU1 没必要在浪费时间做 all-to-all 了，同理 GPU2 和 GPU3 也不需要了。那换言之，为了节省通信开销，我们可以先在 GPU0 和 GPU2 之间做 all-to-all 通信，结束之后， 同一个tensor parallel group内部的 GPU 再做 all-gather即可，即 GPU0 把数据同步给 GPU1。

小结：这样一来，原本 all-to-all 的通信开销就从 $O(p)$降到了 $O(L)+O(p/L)$，其中 p 表示总共的 GPU 数量， L 表示tensor parallelism degree。对应到下面图中 p=4，L=2。当总的 GPU 数量增大的时候，这种方式的通行效率才能体现出来

![](/assets/img/marsggbo/2023-12-07-LLM-学习笔记-Deepspeed-MoE-论文/1c622eaf.jpg)

## **5.3 MoE-related Kernel优化**

基于前面的介绍我们知道 MoE 的核心主要是

* gate 模块
* 各种 data layout transformation，例如：

+ **稀疏张**量 one-hot vector，用来指示每个 token 分配给了哪个 expert
+ **Sparse-Dense Einsum Operations**，比如是根据上面onehot vector 和 inputs tokens 进行矩阵乘法，实现 token 的 masked 计算
+ **数据的重新排序和重排（Reordering and Rearranging of Data）**：MoE 模型中的最终 einsum 运算会对 tokens 进行缩放和重新排序，以使其回到原始的顺序。这个过程可能会涉及对数据进行重新排列和重组。

gate 模块包含了很多算子，例如生成 token 的 mask，top-k 操作，scatter token 到对应 expert等等，这些算子都需要启动 kernel call,而每一次都需要开销。所以论文 的优化思路是两点：

* 使用 dense representation代替稀疏张量表示，简单理解就是弄了一个 mapping table，记录每个 token和 expert 的对应关系
* 把 gate 模块内的所有算子融合成了单个 kernel 算子

实验结果显示优化后的 MoE kernel 能够降低 6 被的延迟。

## **疑问**

整篇文章主要是第 5 章读起来费劲，读完还是有些问题没有搞明白，希望和大家讨论一下：

* 图 8 和图 9 中的小方格分别表示什么含义，是模型参数还是 input token？
* all-to-all 具体是指下面的哪一种，因为看到了不同的说法：

1. 参与通信的 GPU 把自身所有数据都同步给其他 GPU，然后每个 GPU 通过 mask 执行计算
2. 参与通信的 GPU 根据 gate 的结果只传输需要的 token 给对应的 GPU

- 为什么图 8 和图 9 的 local transform 都是把 GPU 0 的 B 和 C 做交换，可以把 A 和 C 交换吗？
- 每个 GPU 都只传输了 3/4 的数据给其他 GPU，这是为什么？另外我理解的是不同的 expert 处理的数据量应该是不一样的

## **参考：**

* <https://zhuanlan.zhihu.com/p/466363675>
* [https://www.cnblogs.com/marsggbo/p/16871789.html](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/marsggbo/p/16871789.html)

### **微信公众号：AutoML机器学习**

http://weixin.qq.com/r/HD8gOHzEmiHlrThb92oO (二维码自动识别)

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**