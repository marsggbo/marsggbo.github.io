---
layout: post
title: 机器学习超参数优化算法-Hyperband
date: '2018-12-23'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/53088201
related_posts: false
toc:
  sidebar: left
---

> 参考文献:[Hyperband: Bandit-Based Configuration Evaluation for Hyperparameter Optimization](https://link.zhihu.com/?target=https%3A//openreview.net/pdf%3Fid%3Dry18Ww5ee)

## **I. 传统优化算法**

机器学习中模型性能的好坏往往与超参数(如batch size,filter size等)有密切的关系。最开始为了找到一个好的超参数，通常都是靠人工试错的方式找到"最优"超参数。但是这种方式效率太慢，所以相继提出了**网格搜索(Grid Search, GS)** 和 **随机搜索(Random Search,RS)**。

但是GS和RS这两种方法总归是盲目地搜索，所以**贝叶斯优化(Bayesian Optimization,BO)** 算法闪亮登场。BO算法能很好地吸取之前的超参数的经验，更快更高效地最下一次超参数的组合进行选择。但是BO算法也有它的缺点，如下：

* 对于那些具有**未知平滑度**和**有噪声**的**高维**、**非凸**函数，BO算法往往很难对其进行拟合和优化，而且通常BO算法都有很强的假设条件，而这些条件一般又很难满足。
* 为了解决上面的缺点，有的BO算法结合了启发式算法(heuristics)，但是这些方法很难做到并行化

## **II. Hyperband算法**

## **1. Hyperband是什么**

为了解决上述问题，Hyperband算法被提出。在介绍Hyperband之前我们需要理解怎样的超参数优化算法才算是好的算法，如果说只是为了找到最优的超参数组合而不考虑其他的因素，那么我们那可以用穷举法，把所有超参数组合都尝试一遍，这样肯定能找到最优的。但是我们都知道这样肯定不行，因为**我们还需要考虑时间，计算资源等因素**。而这些因素我们可以称为**Budget**,用B表示。

**假设一开始候选的超参数组合数量是n，那么分配到每个超参数组的预算就是B/n。所以Hyperband做的事情就是在n与B/n做权衡(tradeoff)。**

上面这句话什么意思呢？也就是说如果我们希望候选的超参数越多越好，因为这样能够包含最优超参数的可能性也就越大，但是此时分配到每个超参数组的预算也就越少，那么找到最优超参数的可能性就降低了。反之亦然。所以Hyperband要做的事情就是**预设尽可能多的超参数组合数量，并且每组超参数所分配的预算也尽可能的多，从而确保尽可能地找到最优超参数**。

## **2. Hyperband算法**

Hyperband算法对 [Jamieson & Talwlkar(2015)](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1502.07943)提出的SuccessiveHalving算法做了扩展。所以首先介绍一下SuccessiveHalving算法是什么。

其实仔细分析SuccessiveHalving算法的名字你就能大致猜出它的方法了：假设有nn组超参数组合，然后对这nn组超参数均匀地分配预算并进行验证评估，根据验证结果淘汰一半表现差的超参数组，然后重复迭代上述过程直到找到最终的一个最优超参数组合。

基于这个算法思路，如下是Hyperband算法步骤：

![](/assets/img/marsggbo/2018-12-23-机器学习超参数优化算法-Hyperband/649f50de.jpg)

* r: 单个超参数组合**实际**所能分配的预算；
* R: 单个超参数组合所能分配的**最大**预算；
* $s_{max}$ : 用来控制总预算的大小。上面算法中 $s_{max}=⌊log_η(R)⌋$ *,当然也可以定义为* $s_{max}=⌊log_η(n_{max})⌋$
* B: 总共的预算,B=($s_{max}$+1)R
* η: 用于控制每次迭代后淘汰参数设置的比例
* **get\_hyperparameter\_configuration(n)**:采样得到n组不同的超参数设置
* **run\_then\_return\_val\_loss(t,** $r_i$ **)**:根据指定的参数设置和预算计算valid loss。LL表示在预算为$r_i$的情况下各个超参数设置的验证误差
* $topk(T,L,⌊n_i/η⌋)$ :第三个参数表示需要选择top k(k=***⌊n\_*i/η⌋**)参数设置。

> 注意上述算法中对超参数设置采样使用的是均匀随机采样，所以有算法在此基础上结合贝叶斯进行采样，提出了[BOHB:Practical Hyperparameter Optimization for Deep Learning](https://link.zhihu.com/?target=https%3A//openreview.net/forum%3Fid%3DHJMudFkDf)

## **3. Hyperband算法示例**

文中给出了一个基于MNIST数据集的示例，并将迭代次数定义为预算(Budget),即一个epoch代表一个预算。超参数搜索空间包括学习率，batch size,kernel数量等。

令 $R=81,η=3$ ，所以 $s_{max}=4,B=5,R=5×81$ 。

下图给出了需要训练的超参数组和数量和每组超参数资源分配情况。

由算法可以知道有两个loop，其中inner loop表示SuccessiveHalving算法。再结合下图左边的表格，**每次的inner loop，用于评估的超参数组合数量越来越少，与此同时单个超参数组合能分配的预算也逐渐增加**，所以这个过程能更快地找到合适的超参数。

右边的图给出了不同ss对搜索结果的影响，可以看到s=0或者s=4并不是最好的，所以并不是说s越大越好。

![](/assets/img/marsggbo/2018-12-23-机器学习超参数优化算法-Hyperband/b68df728.jpg)

**MARSGGBO♥原创**

### 微信公众号: 【AutoML机器学习】

![](/assets/img/marsggbo/2018-12-23-机器学习超参数优化算法-Hyperband/228f26c5.jpg)

AutoML机器学习
