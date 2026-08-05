---
layout: post
title: 论文笔记系列--MnasNet：Platform-Aware NAS for Mobile
date: '2020-01-22'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/103802311
related_posts: false
toc:
  sidebar: left
---

> 本文介绍一篇针对移动端自动设计网络的文章《MnasNet：Platform-Aware Neural Architecture Search for Mobile》，由Google提出，很多后续工作都是基于这个工作改进的，因此很有必要学习了解。

## **Related work**

MnasNet的目的很简单就是设计出表现又好，效率又高的网络。在介绍之前简单回顾一下现有的一些提高网络效率的方法：

* **quantization**：就是把模型的权重用更低精度表示，例如之前使用float32来存储权重，那么我们可以试着用8位来存，更极致的思路是0,1来存，这就是Binary Network，也有一些工作研究这个，本文不做细究。
* **pruning**：就是把模型中不重要的参数删掉。常用的一种剪枝方法是对通道数进行剪枝，因为这种方法实现起来方便，得到的模型结构也是规则的，计算起来也方便。
* 人工设计模块 - **ShuffleNet**

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/653a485d.jpg)

上图(a)就是加入Depthwise的ResNet bottleneck结构，而(b)和(c)是加入Group convolution和Channel Shuffle的ShuffleNet的结构。

- **MobileNet**：引入Depthwise Separable Convolution (DWConv)

- **MobileNetv2**：在DWConv基础上引入inverted residuals and linear bottlenecks

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/fbb1458c.jpg)

- **SqueezeNet** 卷积模块设计思路如下图示，首先使用1x1卷积对输入特征图做压缩，所以叫做**Squeeze层**；压缩之后需要经过**Expand层**还原，这里会对压缩后的特征做两路还原，一路用1x1卷积，另一路用3x3卷积，最后对两路的结果做concat。

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/9547b7c0.jpg)

看下图可能会更加有助于理解：

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/459f16a1.jpg)

- **CondenseNet**： 参考文章[CondenseNet算法笔记](https://link.zhihu.com/?target=https%3A//blog.csdn.net/u014380165/article/details/78747711)

## **MnasNet算法介绍**

## **优化目标**

之前的NAS算法（如DARTS，ENAS)考虑更多的是模型最终结果是否是SOTA，MnasNet则是希望搜索出又小又有效的网络结构，因此将多个元素作为优化指标，包括准确率，在真实移动设备上的延迟等，最终定义的优化函数如下：

$$\begin{array}{l}{\quad \underset{m}{\operatorname{maximize} \quad A C C(m) \times\left[\frac{L A T(m)}{T}\right]^{w} \tag{1} \\  {\text { where } w \text { is the weight factor defined as: } \\  {\qquad w=\left\{\begin{array}{ll}{\alpha,} & {\text { if } L A T(m) \leq T} \\ {\beta,} & {\text { otherwise }\end{array}\right.}  \end{array}$$

上式中个符号含义如下：

* $m$表示模型(model)
* $ACC(m)$表示在特定任务上的结果（如准确率）
* $LAT(m)$表示在设备上测得的实际计算延迟时间
* $T$表示目标延迟时间（target latency）
* $w$表示不同场景下对latency的控制因子。当实测延迟时间$LAT(m)$小于目标延迟时间$T$时，$w=α$；反之$w=β$

上面式子其实表示为帕累托最优，因为一般而言延迟越长，代表模型越大，即参数越大，相应地模型结果也会越好；反之延迟越小，模型表现也会有略微下降。

文中提到latency单位提升会带来5%的acc提升。也就是说假如模型A最终延迟为t,准确率为a;模型B延迟为2t，那么它的准确率应该是a(1+5%)。但是这两个模型的reward应该是相等地，套用上面的公式有

$$Reward(A)=a\times(t/T)^\beta \\ \,\, Reward(B)=a(1+5\%)(2t/T)^\beta$$

求解得到$\alpha=\beta=-0.7$

## **搜索空间**

之前的NAS算法都是搜索出一个比较好的cell，然后重复堆叠若干个cell得到最终的网络，这种方式很明显限制了网络的多样性。MnasNet做了一些改进可以让每一层不一样，具体思路是将模型划分成若干个block，每个block可以由不同数量的layer组成，每个layer则由不同的operation来表示，

```
Net
   |__block
          |__layer
               |___operations
```

示意图如下：

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/17d59498.jpg)

可以看到搜索空间包含如下：

* 标准卷积，深度可分离卷积(DWConv), MBConv(即上面提到的MobileNetV2的卷积模块)
* 卷积核大小：3, 5, 7等
* Squeeze-and-excitation ratio (SE-Ratio): 0, 0.25
* Skip-connection
* 输出通道数
* 不同block中的layer数量 $N\_i$

## **搜索算法**

和ENAS一样使用的是强化学习进行搜索，这里不做细究（其实论文里也没怎么说）。

## **实验**

## **实验设置**

之前的算法都是先在CIFAR10上搜索得到网络后，再在ImageNet上训练一个更大的网络。MnasNet则是直接在ImageNet上搜网络，但是只是在训练集上搜了5个epoch。

## **实验结果**

ImageNet实验结果

下图中的结果和预期一样，延迟越高，结果会稍微好一些。

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/00bd6797.jpg)

作者还对比了SE模块的效果，结果如下,可以看到效果还是不错的。

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/0aef21fa.jpg)

有的时候为了适应实际场景需要，我们会对模型的通道数量进行修改，例如都砍掉一半或者增加一倍等，这样就可以达到模型大小减小或增大的作用了，这个可以由`depth multipilier`参数表示。但是有下面的结果可以看出和MobileNetV2相比，基于MnasNet找到的网络对于通道数量变化鲁棒性更强（左图），同样对于输入数据大小也更加具有鲁棒性（右图）。

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/b811b3ef.jpg)

## **消融实验（Ablation Study）**

## **Soft vs. Hard Latency Constraint**

前面介绍过用于控制延迟时间的因子 $\alpha$和$\beta$,实验对比了两组参数设置：

* $\alpha=0，\beta=-1$
* $\alpha=-0.07，\beta=-0.07$。实验结果如下：

设置的目标延迟时间为75ms，可以看到第二个参数配置能够覆盖更加广的模型结构

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/aae99732.jpg)

## **多目标优化和搜索空间**

这一个实验探究的是本文提出的多目标优化和搜索空间的有效性，一共设置了三组实验，其中baseline是NASNet，实验结果如下：

可以看到多目标优化能够找到延迟更小的网络，而Mnas提出的搜索空间对模型表现也有一定提升。

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/646f0005.jpg)

## **MnasNet结构和Layer多样性**

下图给出了搜索得到的MnasNet的结构，可以看到每层结构都不太一样，不像之前的算法是简单地叠加而成。

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/9b981e49.jpg)

最后作者还对比了使用单一操作组成的网络结果对比，实验结果如下，可以看到虽然只使用MBConv5(k5x5)最终accuracy最高，但是他的推理延迟也很高，所以综合来看还是MnasNet-A1表现最好。

### **MARSGGBO♥原创** **如有意合作或学术讨论欢迎私戳联系~** **邮箱:marsggbo@foxmail.com** 微信公众号: 【AutoML机器学习】

![](/assets/img/marsggbo/2020-01-22-论文笔记系列--MnasNetPlatform-Aware-NAS-for-Mobile/228f26c5.jpg)

AutoML机器学习

**2020-01-22 16:47:46**
