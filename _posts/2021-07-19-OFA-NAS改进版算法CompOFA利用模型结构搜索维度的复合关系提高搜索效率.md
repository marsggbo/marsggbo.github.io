---
layout: post
title: OFA-NAS改进版算法：CompOFA利用模型结构搜索维度的复合关系提高搜索效率
date: '2021-07-19'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/391052412
related_posts: false
toc:
  sidebar: left
---

## **1. 长话短说**

这篇论文（简称CompOFA）主要是基于 [Once for all: Train One Network and Specialize it for Efficient Deployment](https://zhuanlan.zhihu.com/p/391042698) 论文的改进，其主要就是把原来OFA的搜索空间简化，带来的好处有

1. 实验结果显示搜索时间缩小了一半
2. 虽然搜索时间变短了，但是依旧能找到Pareto-front模型，而且模型性能和OFA搜到的很接近甚至更好

## **2. 动机**

前面提到了 CompOFA 这篇论文的主要贡献在于提出了精简OFA搜索空间的搜索策略，那在介绍这个精简策略之前，我们首先回顾一下OFA的搜索空间，如下图所示。

![](/assets/img/marsggbo/2021-07-19-OFA-NAS改进版算法CompOFA利用模型结构搜索维度的复合关系提高搜索效率/2e6cd17b.jpg)

经过计算OFA的整体搜索空间大小为 $2\times10^{19}$左右，而本篇论文的主要贡献就是简化搜索空间。它的依据是基于之前一些模型缩放探究的工作[1][2][3]得出的结论，这些结论其实也比较符合直觉，即 **width和depth不是互相独立的，相反它们之间存在着复合关系，即二者同时增加对模型性能都有提升作用**。作者特别做了个小实验进一步证明这一点，结果如下图是，可以看到随着depth和width同时增加，对角线的accuracy（左图）和latency（右图）也是逐步提升的。

![](/assets/img/marsggbo/2021-07-19-OFA-NAS改进版算法CompOFA利用模型结构搜索维度的复合关系提高搜索效率/7e980190.jpg)

文章花了一大段篇幅，引经据典地强调 width和depth存在复合关系，以此来证明他们工作的可行性，后面要介绍的采样策略也是基于此。

另外CompOFA还总结了一个比较有意思的点。我们都知道NAS任务的搜索空间一般都是巨大的，但是以OFA为例，它涵盖的模型的FLOP的范围是120~560 MFLOPS，即使我们以 1个FLOP为基本单位进行划分，假设模型是均匀分布的，那么每个FLOP的块里仍然有 $10^{11}$个模型。

那么一个很自然的问题就是，即使在指定大小的FLOP块里，去找到最优的模型结构也是非常困难的。而且实际场景中并不会精确到具体的FLOP值，是可以有一个上下区间的，这样一来搜索空间就更大了，所以如何简化搜索空间也是值得研究的问题。

## **3. 采样策略**

OFA论文中的采样策略是渐进式收缩的，采样的模型数量非常多。而CompOFA的采样策略则是让width和depth成对缩放，细节如下：

在介绍OFA的时候已经介绍过了，其搜索空间主要有三部分：

* Depth：[2,3,4]
* Width：[3,4,6]
* Kernel Size: [3,5,7]

CompOFA的做法是固定Kernel size，并将depth和width成对搜索，即 {[depth, width]}={[2,3],[3,4],[4,6]}。因为总共有5个block，所以搜索空间大小为 $3^5=243$。

下面是OFA和CompOFA的搜索流程对比，可以看到CompOFA的搜索时间大大减少。

![](/assets/img/marsggbo/2021-07-19-OFA-NAS改进版算法CompOFA利用模型结构搜索维度的复合关系提高搜索效率/181a9a3b.jpg)

## **4. 实验结果**

下图给出不同硬件平台的结果对比

![](/assets/img/marsggbo/2021-07-19-OFA-NAS改进版算法CompOFA利用模型结构搜索维度的复合关系提高搜索效率/8252a871.jpg)

以5ms延迟为一个块，分别在该块内随机采样50个模型，然后对比错误率的累积概率分布。可以看到以26%的错误率为例，CompOFA采样得到的模型错误率小于26%的数量比例要多于OFA。

![](/assets/img/marsggbo/2021-07-19-OFA-NAS改进版算法CompOFA利用模型结构搜索维度的复合关系提高搜索效率/0eb918aa.jpg)

OFA的一个主要贡献是提出了Progressive Shrinking （PS） 算法，但是在CompOFA中，作者在精简后的搜索空间上对比了使用PS和不使用PS策略，结果如下。可以看到PS对CompOFA最终模型性能影响不是很大

![](/assets/img/marsggbo/2021-07-19-OFA-NAS改进版算法CompOFA利用模型结构搜索维度的复合关系提高搜索效率/d3657a52.jpg)

## **参考**

* [1] Yu, Jiahui, et al. "**Bignas**: Scaling up neural architecture search with big single-stage models." European Conference on Computer Vision. Springer, Cham, 2020.
* [2] Tan, Mingxing, and Quoc Le. "**Efficientnet**: Rethinking model scaling for convolutional neural networks." International Conference on Machine Learning. PMLR, 2019.
* [3] Radosavovic, Ilija, et al. "Designing network design spaces." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
