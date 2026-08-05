---
layout: post
title: "深度学习并行训练算法一锅炖: DDP, TP, PP, ZeRO"
date: 2022-11-09
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/581677880
related_posts: false
toc:
  sidebar: left
---

> “ 本文主要参考ColossalAI论文 《Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training》  
>  ColossalAI框架开源提供了本文介绍的所有并行训练: [https://github.com/hpcaitech/ColossalAI](https://link.zhihu.com/?target=https%3A//github.com/hpcaitech/ColossalAI)  
>  ”

## **前言**

本文会介绍几种流行的深度学习并行方法，包括

* 数据并行（data parallel）
* 模型并行（model parallel）

+ tensor并行
+ pipeline并行
+ Sequence并行

* Zero Redundancy Data Parallelism （ZeRO）

下图给出了这些并行方法的示意图，非常直观好懂。

![](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/b99a41b5.jpg)

不过在介绍各种并行训练方法之前，我们首先对一些概念做一个声明，方便后面理解

模型训练过程中涉及到的参数主要包含两大类，*model data* 和 *non-model data*，具体表示如下：

* model data

+ 模型权重
+ 模型权重的梯度
+ 优化器的状态

* non-model data

+ 模型逐层的特征向量（也叫作activations）

## **1. Data parallelism (DP)**

经典的数据并行算法是在多个设备上都拷贝一份完整的模型参数，彼此之间可以独立计算，所以每个设备传入的输入数据不一样​，这也是为什么叫数据并行。只不过，每隔一段时间（比如一个batch或者若干个batch）后需要彼此之间同步模型权重的梯度​。随着模型大小不断增大，单个GPU的内存已经无法容纳现如今的大模型，所以便有了后面会介绍的模型并行​。

DP下有很多优化算法，具体的可以看看我优秀师弟 [@二猹树](https://www.zhihu.com/people/f97528ae4a73b5bf0504a7123453a37e)的相关综述

* 《[Communication-Efficient Distributed Deep Learning: A Comprehensive Survey](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2003.06307)》
* 《[A Quantitative Survey of Communication Optimizations in Distributed Deep Learning](https://link.zhihu.com/?target=https%3A//ieeexplore.ieee.org/abstract/document/9275615/)》

## **2. Model Parallelism (MP)**

### **2.1 Pipeline Parallelism (PP)**

pipeline parallelism是比较常见的模型并行算法，它是模型做层间划分，即inter-layer parallelism。以下图为例，如果模型原本有6层，你想在2个GPU之间运行pipeline，那么每个GPU只要按照先后顺序存3层模型即可。

![](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/e5f8ddb2.jpg)

已经有很多Pipeline相关的研究工作了，例如PipeDream，GPipe，和Chimera。它们的主要目的都是降低bubble time。这里不做过多介绍。

### **2.2 Tensor Parallelism (TP)**

前面介绍的Pipeline Parallelism是对模型层间做划分，叫inter-layer parallelism。那么另一种方式则是对模型层内做划分，即intra-layer Parallelism，也叫Tensor Parallelism。

![](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/d839956f.jpg)

Tensor Parallelism

**2.2.1 1D Tensor Parallelism**

Megatron-LM [1]是最早提出1D Tensor并行的工作。该工作主要是为了优化transformer训练效率，把线性层按照行或者列维度对权重进行划分。如下图所示，原本线性层为![Y=W_1W_2X](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/cc519cb2.jpg) ，这里将![W_1](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/71afb8d5.jpg)按列进行划分，将![W_2](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/027a1fcf.jpg)按行进行划分。这样，每个GPU只需要存一半的权重即可，最后通过All-reduce操作来同步Y的结果。当GPU数量为![N](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/39082460.jpg)时，每个GPU只需要存![\frac{1}{N}](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/6ea1df8f.jpg)的权重即可，只不过每层输出需要用All-reduce来补全结果之后才能继续下一层的计算。

![](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/4af39fbc.jpg)

对于土豪公司，可以使用NVLink来连接GPU（如下图a），从而提供高带宽来降低通信开销。但是土豪终归是少数的，大部分公司和个人是没法承担这昂贵的硬件费用，因此比较常见的GPU连接方式是下图b，即节点内花点钱实现NVLink连接，节点之间通过PCIe连接。

![](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/80aca5ec.jpg)

1D Tensor并行对通信速度要求较高，不过1D在每层的输入和输出都有冗余的内存开销。以Fig.4为例，我们可以看到虽然模型权重被划分了，但是每个GPU都有重复的输入![X](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/6ce37be6.jpg),另外All-reduce之后每个GPU也会有重复的输出![Y](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/ce020eb9.jpg)，所以后续一些工作尝试从这里做进一步改进,包括2D, 2.5D,和3D tensor并行。

**2.2.2 2D Tensor Parallelism**

2D Tensor Parallel [2] 基于SUMMA和Cannon矩阵相乘算法沿着两个不同的维度对 *输入数据*，*模型权重*，*每层的输出* 进行划分。给定![N](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/39082460.jpg)个GPU，tensor会被划分成![N](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/39082460.jpg)个chunk（使用`torch.chunk`），每个GPU保存一个chunk。这![N](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/39082460.jpg)个GPU呈方形网络拓扑结构，即每行每列均为![\sqrt{N}](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/ccb72633.jpg)个GPU。假设tensor的维度大小是![[P,Q]](https://www.zhihu.com/equation?tex=%5BP%2CQ%5D)，那么划分后每个GPU上存的chunk大小即为![[P/\sqrt{N},Q/\sqrt{N}]](https://www.zhihu.com/equation?tex=%5BP%2F%5Csqrt%7BN%7D%2CQ%2F%5Csqrt%7BN%7D%5D)。至此，每个GPU都只会保存部分的输入输出以及部分的权重。虽然相比于1D Tensor并行，2D额外增加了模型权重的通信，但是需要注意的是当GPU数量很多的时候，每个GPU上分配的模型权重就会小很多，而且因为使用的All-reduce通信方式，所以2D也还是要比1D更高效的。

**2.2.3 2.5D Tensor Parallelism**

2.5D Tensor Parallel [3] 是受2.5D矩阵乘法算法 [4] 启发进一步对2D Tensor并行的优化。具体来说2.5D增加了 *depth* 维度。当 *depth=1* 时等价于2D；

同样假设有![N](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/39082460.jpg)个GPU，其中![N=S^2*D](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/20f7c2d8.jpg)，![S](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/e6217fb4.jpg)类似于原来2D正方形拓扑结构的边长，而![D](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/1e9eba69.jpg) 则是新增加的维度 *depth* 。![D](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/1e9eba69.jpg)可以由用户指定，![S](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/e6217fb4.jpg) 则会自动计算出来了。所以一般来说至少需要8个GPU才能运行2.5D算法，即![S=2,D=2](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/37b35f22.jpg)。

**2.2.4 3D Tensor Parallelism**

3D Tensor Parallel [5]是基于3D矩阵乘法算法 [6]实现的。假设有 ![N](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/39082460.jpg)个 GPU，tensor维度大小为![[P, Q, K]](https://www.zhihu.com/equation?tex=%5BP%2C+Q%2C+K%5D)，那么每个chunk的大小即为 ![[P/\sqrt[3]{N},Q/\sqrt[3]{N},K/\sqrt[3]{N}]](https://www.zhihu.com/equation?tex=%5BP%2F%5Csqrt%5B3%5D%7BN%7D%2CQ%2F%5Csqrt%5B3%5D%7BN%7D%2CK%2F%5Csqrt%5B3%5D%7BN%7D%5D)。当tensor维度小于3时，以全连接层为例，假设权重维度大小为 ![[P, Q]](https://www.zhihu.com/equation?tex=%5BP%2C+Q%5D) ,那么可以对第一个维度划分两次，即每个chunk的维度大小为 ![[P/(\sqrt[3]{N})^2,Q/\sqrt[3]{N}]](https://www.zhihu.com/equation?tex=%5BP%2F%28%5Csqrt%5B3%5D%7BN%7D%29%5E2%2CQ%2F%5Csqrt%5B3%5D%7BN%7D%5D) 。3D Tensor并行的通信开销复杂度是 ![O(N^{1/3})](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/55a20232.jpg) ，计算和内存开销都均摊在所有GPU上。

**2.2.5 小结**

1D Tensor并行每一层的输出是不完整的，所以在传入下一层之前都需要做一次All-gather操作，从而使得每个GPU都有完整的输入，如下图a所示。

2D/2.5D/3D Tensor 并行算法因为在一开始就对输入进行了划分， 所以中间层不需要做通信，只需要在最后做一次通信即可。在扩展到大量设备（如GPU）时，通信开销可以降到很小。这3个改进的Tensor并行算法可以很好地和Pipeline并行方法兼容。

![](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/ca5344ce.jpg)

### **2.3 Sequential Parallelism**

Tensor parallelism主要是为了解决由 model data （模型权重，梯度和优化器状态）导致的内存瓶颈，但是 non-model data也可能成为性能瓶颈。比如像AlphaFold和NAS任务中会存在很多中间特征值（也叫activations）。

以DARTS算法为例，它的模型参数量其实并不多，但是它有很多分支，所以activations会消耗大量GPU内存，这也是为什么很多NAS算法只能在CIFAR-10上搜索到合适的模型结构后，再做人工扩展，最后应用到ImageNet上做性能验证。

同样地，在使用Transformer训练语言模型时，由于Transformer层中的Self-attention机制的复杂度是![O(n^2)](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/a5e2df1c.jpg)，其中 ![n](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/ce59f558.jpg) 是序列长度。换言之，长序列数据将增加中间activation内存使用量，从而限制设备的训练能力。

Sequential Parallelism （SP）就为了解决non-model data导致的性能瓶颈而提出的。下图给出了SP在Transform并行训练上的应用，具体的原理可以查看原论文[7]。

![](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/46cd1e32.jpg)

## **3. Zero Redundancy Data Parallelism (ZeRO)**

训练过程中GPU内存开销主要包含以下几个方面：

* 模型状态内存（Model State Memory）：

+ 梯度
+ 模型参数
+ 优化器状态：当使用像Adam这样的优化器时，优化器的状态会成为GPU内存开销的大头。前面介绍的DP，TP， PP算法并没有考虑这个问题。

* 激活内存（Activation Memory）：在优化了模型状态内存之后，人们发现激活函数也会导致瓶颈。激活函数计算位于前向传播之中，用于支持后向传播。
* 碎片内存（Fragmented Memory）：深度学习模型的低效有时是由于内存碎片所导致的。在模型之中，每个张量的生命周期不同，由于不同张量寿命的变化而会导致一些内存碎片。由于这些碎片的存在，会导致即使有足够的可用内存，也会因为缺少连续内存而使得内存分配失败。ZeRO 根据张量的不同寿命主动管理内存，防止内存碎片。

ZeRO针对模型状态的三部分都做了对应的内存改进方法：

* ZeRO1：只划分优化器状态(optimizer states, os)，即![P_{os}](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/52c36a99.jpg)
* ZeRO2：划分优化器状态和梯度(gradient, g)，即![P_{os+g}](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/a08dc187.jpg)
* ZeRO3：划分优化器状态和梯度和模型参数(parameters, p)，即![P_{os+g+p}](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/e823249c.jpg)

下图给出了三种方法带来的内存开销收益

![](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/ac2193cd.jpg)

不管采用三种方法的哪一种，ZeRO简单理解就是给定![N](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/39082460.jpg)个设备，然后把一堆data等分到这些设备上，每个设备只存![1/N](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/dd4fab2b.jpg)的数据量，并且每次也只负责更新这![1/N](/assets/img/marsggbo/2022-11-09-深度学习并行训练算法一锅炖-DDP-TP-PP-ZeRO/dd4fab2b.jpg)的数据。

因为对数据做了划分，ZeRO在每一层都需要有通信操作。我们考虑ZeRO在某一层的具体操作：

* 在forward的时候，会首先使用all-gather让每个设备拥有该层完整的模型权重，然后计算得到输出，最后每个设备会只保留原来的权重，即把all-gather过来的权重扔掉，这样可以节省开销。
* 在backward的时候，同样会先all-gather该层的所有权重，然后计算梯度，最后也会把梯度进行划分，每个设备上只会存1/N对应的梯度数据。

注意ZeRO对数据划分方式并没有什么具体的要求，可以是随意划分，因为最后反正会用all-gather使得所有设备商都有用完整的数据；当然，也可以使用前面提到的Tensor Parallelism的划分方式，这样一来可以有效降低通信开销，进一步提高效率。

> “ 关于ZeRO更详细的介绍可以查看原论文或者看看[这篇博客](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/rossiXYZ/p/15782054.html) ”

未完待续。。。

### **微信公众号：AutoML机器学习**

http://weixin.qq.com/r/HD8gOHzEmiHlrThb92oO (二维码自动识别)

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**