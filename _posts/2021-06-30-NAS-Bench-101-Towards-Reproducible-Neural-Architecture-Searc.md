---
layout: post
title: "NAS-Bench-101: Towards Reproducible Neural Architecture Search"
date: 2021-06-30
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/385532786
related_posts: false
toc:
  sidebar: left
---

## **1. 模型整体结构**

![](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/8b3d531f.jpg)

NAS-Bench-101模型结构框架

上图（左）展示了模型的整体结构，由`stem`,`stack`,`downsample`,`global avg pool`和最后的全连接层`dense`组成，每个`stack`由3个`cell`组成，右图给出了某个`cell`的示意图，下面详细介绍`cell`的结构。

## **2. Cell结构**

## **2.1 限制条件**

Cell结构遵循以下设计要求

1. cell由**7**个节点组成，其中第一个和最后一个是固定的，分别是`in`和`out`。
2. cell的最大边数不能超过9
3. 每个节点对应一个操作，候选操作只有三个（卷积操作后面都会加上 `BN+ReLU`）：

* 3x3 conv
* 1x1 conv
* 3x3 max-pool

## **2.2 Cell编码方式**

![](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/83fe729e.jpg)

参考：https://www.cnblogs.com/yzjfree/p/12033840.html

Cell的编码方式用如下表示的上三角矩阵表示，复杂度如下：

1. edges: `1/0`表示对应节点之间是否相连，所以有 ![2^{21}](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/d83192ca.jpg)种连接方式
2. ops: 除去`in & out` 两个节点，每个节点都有3个候选操作,所以总共有 ![3^5](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/79e9bbda.jpg)中组合

所以总共搜索空间有 ![2^{21}\times 3^5 \approx 510M](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/c7f3d022.jpg) 中可能的cell结构。不过因为规定了edge总数不能大于9，另外减去重复的，最后总共有将近 **423K** 个不同的结构图。

下图给出了两个例子用来帮助理解这个编码方式，以左边的cell为例，可以看到矩阵定义了节点之间的联系方式，矩阵下的list `[in,con1x1,conv3x3,mp3x3,out]`定义了5个节点各自的操作。

![](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/4d7be81a.jpg)

## **3. 实验结果和结论**

## **3.1 Metrics**

实验对每一个arch都记录了如下指标：

* **训练**准确率
* **验证**准确率
* **测试**准确率
* 训练时间（单位是 秒）
* 可训练的模型参数

## **3.2 Findings**

## **3.2.1 Statistics数据集统计信息**

由下图（左）可以看到 **1) 模型参数量**, **2) 训练时长** 和 **3) 验证准确率** 三者之间存在**正相关关系。**

下图（右）中给出了基于resnet cell 和 inception cell \*\*\*\*的网络结构的结果， 蓝色的点表示101数据集中的所有网络结构的结果，可以看到人工设计的cell非常靠近Pareto-front，这说明网络拓扑结构和operation非常重要

![](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/d41f214c.jpg)

## **3.2.2 结构设计**

前面介绍了NAS-Bench-101数据集只有三种候选操作，下图展示了不同操作对结果的影响。

![](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/f784936e.jpg)

左图：把3x3卷积替代为1x1卷积或3x3 max-pool，验证精度下降1.16%和1.99%

右图：把3x3卷积替代为1x1卷积或3x3 max-pool，训练时间下降14.11%和9.84%

## **3.2.3 Locality**

有一个直观的假设是 **如果两个架构相似，那么他们的性能应该也是相似的。** 目前的NAS的搜索其实就遵循了这个假设，文中称这个假设叫 `locality`。本文进一步探究了这个假设的准确性。

我们用 \*\*`edit-distance`\*\*来表示相似度,它表示我们将一个模型变换成另一个模型所需要的最少的变化，比如我们把某条边的编码由0变成1，此时distance就是1.

论文中用了 **`random-walk autocorrelation (RWA)`** 来衡量 locality **。**

> “ random-walk（随机游走）可以简单理解成布朗运动，其概念接近于布朗运动，是布朗运动的理想数学状态。一般认为股票就是一种常见的随机游走实例，而autocorrelation就是求时间序列的自相关性。对应到本文，假设我们依次采样了n个模型，它们的准确率表示为( ![x_0,x_1,x_2,...,x_n](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/b4568cdc.jpg) )，那么RWA求的就是 ![(x_0,x_1,x_2,...,x_{n-1})](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/1f0832ff.jpg)和 ![(x_1,x_2,...,x_n)](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/52a0b631.jpg)之间的相关性。参考[[1]](https://zhuanlan.zhihu.com/p/38321845)[[2]](https://link.zhihu.com/?target=https%3A//stats.stackexchange.com/questions/181167/what-is-the-autocorrelation-for-a-random-walk)。  
>  ”

**RWA**能够衡量不同模型性能之间的相似性，1表示相似性最高，0则反之。可以看到当distance增大到6时，RWA的值接近于0，这表示当变化距离超过6后，就无法判断是否存在相关性了

![](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/bcffcfa0.jpg)

由上面的介绍我们知道RWA求的是**一整个时间序列**内采样的模型性能之间的相关性，如果我们只考虑**局部感兴趣的区域**的相关性会得到怎么的结论呢？本文进一步衡量了**全局准确率最高** 的区域（Figure 3中Inception cell那一坨）的locality，衡量指标用的是 [fitness-distance correlation](https://link.zhihu.com/?target=https%3A//www.researchgate.net/publication/216300862_Fitness_Distance_Correlation_as_a_Measure_of_Problem_Difficulty_for_Genetic_Algorithms) (FDC)。

> “ FDC measures the **correlation** between the **fitness values** of a function under investigation and the **distance to the goal** of the search. ”

`FDC`最早被提出是用来衡量遗传算法问题难度的一种手段，这里用来衡量局部区域的相关性。Figure 6左图中显示了在inception cell附近区域的FDC值的变化，可以看到相关性在0.89到0.95之间， 可见相关性较大，而且当distance扩大到6之后趋势发生了明显的变化。

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2021-06-30-NAS-Bench-101-Towards-Reproducible-Neural-Architecture-Searc/029a9211.jpeg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**