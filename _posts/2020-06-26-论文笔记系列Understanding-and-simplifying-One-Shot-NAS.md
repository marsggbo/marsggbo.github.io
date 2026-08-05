---
layout: post
title: "【论文笔记系列】Understanding and simplifying One-Shot NAS"
date: '2020-06-26'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/150831133
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/150831133>

**论文：Understanding and Simplifying One-Shot Architecture Search**

## **search space 设计**

文章认为好的 search space 需要满足以下条件：

* search space 需要足够 large 和 expressive，这样才能探索更丰富多样的候选网络架构
* one-shot 模型在验证集上的准确率必须与 stand-alone 模型的准确率高度相关。也就是说相比于其他候选模型，A 模型在验证集上准确率高，那么对 A 模型 retrain 的之后，它在测试集上的准确率也要是最高，或者是靠前的。
* 在资源有限的情况下，one-shot 模型不能太大

## **One-shot 模型训练**

* **互相适应的鲁棒性（Robustness of co-adaptation）**：如果只是简单地直接训练整个 one-shot 模型，那么模型内各部分是高度耦合的，即使移除一些不重要的部分，也可能使得模型预测准确率大打折扣。所以文章引入 path dropout 策略来提高训练稳定性。具体做法就是最开始啥都不 drop，而后每个 batch 都随机 drop，而且 drop 的概率也线性增加。drop 的概率计算公式为 $r^{1/k}$,$r$ 是一个超参数，$k$ 表示操作数量。
* **训练模型的稳定性（Stabilizing Model Training.）**： - 虽然 Relu-BN-Conv 效果也差不多，实验使用更常用的 BN-Relu-Conv 顺序。另外我们知道在评估阶段我们会从 one-shot 模型里选择一个子模型来评估，也就是说我们会剔除一些操作，但是模型里的 BN 操作的统计量只是基于 one-shot 模型计算得到的，所以**在评估阶段 BN 的统计量每个 batch 都要重新计算。** - 另外在训练 one-shot 模型的时候，对于一个 batch 里的数据，我们 dropout 的操作都是一样的。换句话说这批数据都是在同一个子模型下训练的。文章称这种方式也会导致训练不稳定，所以他们将一个大小为 1024 的 batch 数据进一步划分成多个子 batch，称作**ghost batch**。比如 1024 批数据可以划分成 32 个大小为 32 的 ghost batch，然后每个 ghost batch 应用不同的 path dropout 操作（即对应不同的子模型）。
* **避免过度正则化**：在训练模型时，我们经常会用 L2 正则化。但是在这里只是对选择的子模型做正则化。不然一些没有被选择过的操作也被正则化的话就过分了啊~~

## **实验结果**

下面只介绍一个比较有意思的实验结果，即 Dropout rate 对结果的影响：

结果如下图示：

* 设置的概率值太小的话（最左），可以看到 one-shot 模型的整体准确率都不高，但是 retrain 之后的 stand-alone 模型性能好像都还可以。但是这样 one-shot 的准确率并不能很好的反映最后模型的性能。
* 设置的概率值太高的话（最右），虽然 one-shot 模型的准确率提高了，但是可以看到准确率的范围分布在了 0.6~0.8 之间，也就是说概率值过大使得模型可能把更多机会给到了一些表现可能

![](/assets/img/marsggbo/2020-06-26-论文笔记系列Understanding-and-simplifying-One-Shot-NAS/4d758d53.jpg)

## **理解 one-shot 模型**

由上图我们可以看到（以最左图为例），one-shot 模型的准确率从 0.1~0.8， 而 stand-alone（即 retrain 之后的子模型）的准确率范围却只是 0.92~0.945。为什么 one-shot 模型之间的准确率差别会更大呢？

文章对此给出了一个猜想：**one-shot 模型会学习哪一个操作对模型更加有用，而且最终的准确率也是依赖于这些操作的**。换句话说：

* 在移除一些不太重要的操作时，可能会使 one-shot 模型准确率有所降低，但是最后对 stand-alone 模型性能的预测影响不大。
* 而如果把一些最重要的操作移除之后，不仅对 one-shot 模型影响很大，对最后的

我理解是这个意思

**状态one-shot 模型准确率stand-alone 模型准确率**移除操作之前80%92%移除不太重要的操作78%91%移除重要的操作56%90%

为了验证这一猜想，文章做了如下实验：

首先将几乎保留了所有操作的(dropout 概率是 $r=10^-8$)模型叫做**reference architectures**，注意这里用的是复数，也就是说这个 reference architectures 有很多种，即有的是移除了不太重要的操作后的结构，有的时移除了非常重要的操作候的结构，那么不同结构的准确率应该是不一样的。不过在没有 retrain 的情况下，什么操作都没有移除的 one-shot 模型（All on）应该是最好的（或者是表现靠前的，这里我们认为是最好的）。

注意这里的移除某些操作后得到的模型还是 One-shot 模型，而不是采样后的模型。采样后的模型是指从这个完整的 one-shot 中按照某种策略得到的模型。文中把这种模型叫做**candidate architectures**。

我们以分类任务为例，假设**reference architectures**对某个样本的预测输出是 $(p_1,p_2,...,p_n)$,其中 $n$ 表示类别数量；而**candidate architectures**的输出为 $(q_1,q_2,...,q_n)$。注意 candidate architecture 的输出应该是没有 retrain 的结果。

所以如果上面的猜想是正确的，那么表现最好的**candidate architecture**的预测应该要和所有操作都保留的 one-shot 模型的预测结果要十分接近。文中使用**对称散度**来判断相似性，散度公式为 $D_{\mathrm{KL}}(p \| q)=\sum_{i=1}^{n} p_{i} \log \frac{p_{i}}{q_{i}}$，那么对称散度就是 $D_{\mathrm{KL}}(p \| q)+D_{\mathrm{KL}}(q \| p)$。对称散度结果是在 64 个随机样本上得到的平均值，散度值越接近于 0，表示二者输出越相近。

最后的结果如图示，可以看到在训练集上散度值低的模型（即预测值和保留大多数操作的完整模型很接近），在验证集上的准确率也相对高一些。

![](/assets/img/marsggbo/2020-06-26-论文笔记系列Understanding-and-simplifying-One-Shot-NAS/9dc8006e.jpg)
> 原论文各种名词用的很混乱，一下是 one-hot，一下是 reference architecture，看得很迷幻，结合了好几个博文归纳总结的此文，有问题欢迎评论区指出。 参考：

* **[【NAS-005】2018.07.11- One-Shot -ICML 2018](https://zhuanlan.zhihu.com/p/73539339)**
* **[AutoDL 论文解读（七）：基于 one-shot 的 NAS](https://link.zhihu.com/?target=https%3A//blog.csdn.net/u014157632/article/details/102600575)**

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-06-26-论文笔记系列Understanding-and-simplifying-One-Shot-NAS/029a9211.jpeg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2020-06-25 17:11:21**