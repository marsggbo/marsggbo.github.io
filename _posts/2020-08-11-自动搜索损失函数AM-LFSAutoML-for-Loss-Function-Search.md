---
layout: post
title: 自动搜索损失函数？AM-LFS：AutoML for Loss Function Search
date: '2020-08-11'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/180465704
related_posts: false
toc:
  sidebar: left
---

> 目前AutoML技术非常火，尤其是NAS领域，之前有一篇文章已经对现有的AutoML技术做了总结，可阅读**[【AutoML：Survey of the State-of-the-Art】](https://zhuanlan.zhihu.com/p/158162306)**。  
>  论文：**[AM-LFS：AutoML for Loss Function Search](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1905.07375v1.pdf)**

不过这篇文章将介绍一下如何使用AutoML技术来搜索损失函数。一般来说，损失函数都是需要我们手动设计的，以分类任务而言，我们通常会使用交叉熵。碰到数据集imbalanced的情况，可能会给每个类别加上一个权重。在RetinaNet论文里为目标检测任务提出了FocalLoss。上述都是对交叉熵函数根据特定任务做了修改，可是这样的修改通常需要我们能够洞察到问题的本质，换句话说这需要专业的知识。那我们这种蔡文姬还有设计loss函数的机会吗？商汤科技在这方面做了探索，下面将介绍论论文细节。

## **1. 论文贡献**

论文有两大贡献：

* **设计了损失函数搜索空间**，该搜索空间能够覆盖常用的流行的损失函数设计，其采样的候选损失函数可以调整不同难度级别样本的梯度，并在训练过程中平衡类内距离和类间距离的重要性。
* 提出了一个**bilevel的优化框架**：本文使用强化学习来优化损失函数，其中内层优化是最小化网络参数的损失函数，外层优化是最大化reward。

## **2. 回顾之前的损失函数**

* Softmax Loss

$$L=\frac{1}{N} \sum_{i} L_{i}=\frac{1}{N} \sum_{i}-\log \left(\frac{e^{f_{y_{i}{\sum_{j} e^{f_{j}\right) \tag{1}$$

$N$表示数据集大小，$f$是最后全连接层输出的预测向量（还没有做softmax运算），$f_j$表示向量$f$的第$j$位置上的值，因为真实值是one-hot的向量（即只有一个1，其余全是0），$f_{y_i}$中的$y_i$表示1的索引值。

因为$f$是最后全连接层的输出，所以我们可以将它表示成

$$f_{j}=\left\|\boldsymbol{W}_{j}\right\|\left\|\boldsymbol{x}_{i}\right\| \cos \left(\theta_{j}\right) \tag{2}$$

其中$\theta_{j}\left(0 \leq \theta_{j} \leq \pi\right)$是矢量$\|\boldsymbol{W}_{j}\|$和$x_i$之间的夹角，所以上面公式(1)中的损失函数可以转换成

$$L_{i}=-\log \left(\frac{e^{ \| \boldsymbol{W}_{y_{i}\| \| \boldsymbol{x}_{i} \| \cos \left(\theta_{y_{i}\right)}{\sum_{j} e^{\left\|\boldsymbol{W}_{j}\right\|\left\|\boldsymbol{x}_{i}\right\| \cos \left(\theta_{j}\right)}\right) \tag{3}$$

* Margin-based Softmax Loss

看到公式(3)可以很自然地想到能在$\left\|\boldsymbol{W}_{y_{i}\right\|\left\|\boldsymbol{x}_{i}\right\|$ 和 $\cos \left(\theta_{y_{i}\right)$之间能够插入一个可微变换函数$t( \cdot )$来调节角度，进而得到margin可变的softmax loss：

$$L_{i}^{t}=-\log \left(\frac{e^{\left\|\boldsymbol{W}_{y_{i}\right\|\left\|\boldsymbol{x}_{i}\right\| t\left(\cos \left(\theta_{y_{i}\right)\right)}{e^{\left\|\boldsymbol{W}_{y_{i}\right\|\left\|\boldsymbol{x}_{i}\right\| t\left(\cos \left(\theta_{y_{i}\right)\right)}+\sum_{j \neq y_{i} e^{\left\|\boldsymbol{W}_{j}\right\|\left\|\boldsymbol{x}_{i}\right\| \cos \left(\theta_{j}\right)}\right)\tag{4}$$

不同的$t(\cdot)$可以得到不同的损失函数，原文中总结了如下几种：

![](/assets/img/marsggbo/2020-08-11-自动搜索损失函数AM-LFSAutoML-for-Loss-Function-Search/6c994a52.jpg)

* Focal Loss

除了在概率上做变化外，Focal Loss对softmax loss做了如下变化：

$$\begin{array}{l} L_{i}^{t}=-\tau\left(\log \left(p_{y_{i}\right)\right) \\ \tau(x)=x\left(1-e^{x}\right)^{\alpha} \tag{5} \end{array}$$

## **3. Loss函数分析**

## **3.1 Focal Loss**

focal loss的提出主要是为了解决imbalanced的问题。相对于原始的softmax loss，focal loss在求导之后等于原始的softmax loss求导结果再乘以$\tau^{\prime}\left(\log \left(p_{y_{i}\right)\right.$，换言之$\tau^{\prime}$用来缓解imbalance的问题。

## **3.2 margin-based softmax loss**

为方便说明，我们可以假设所有的矢量是单位矢量，即$\left\|\boldsymbol{W}_{j}\right\|=\left\|\boldsymbol{x}_{i}\right\|=1$ 和 $f_{j}=\cos \left(\theta_{j}\right)$

我们使用公式(4)中的损失函数来分别对$f_{y_i}$（类内，intra-class）和$f_j$（类间，inter-class）求导，得到：

$\left\|\frac{\partial L_{i}^{t}{\partial f_{y_{i}\right\|=\left(1-p_{y_{i}^{t}\right) t^{\prime}\left(f_{y_{i}\right) \tag{6}$$\left\|\frac{\partial L_{i}^{t}{\partial f_{j}\right\|=p_{j}^{t} \tag{7}$

其中

$p_{y_{i}^{t}=\frac{e^{\left\|\boldsymbol{W}_{y_{i}\right\|\left\|\boldsymbol{x}_{i}\right\| t\left(\cos \left(\theta_{y_{i}\right)\right)}{e\left\|\boldsymbol{W}_{y_{i}\right\|\left\|\boldsymbol{x}_{i}\right\| t\left(\cos \left(\theta_{y_{i}\right)\right)+\sum_{j \neq y_{i} e^{\left\|\boldsymbol{W}_{j}\right\|\left\|\boldsymbol{x}_{i}\right\| \cos \left(\theta_{j}\right)}$$p_{j}^{t}=\frac{e^{\left\|\boldsymbol{W}_{j}\right\|\left\|\boldsymbol{x}_{i}\right\| \cos \left(\theta_{j}\right)}{e\left\|\boldsymbol{W}_{y_{i}\right\|\left\|\boldsymbol{x}_{i}\right\| t\left(\cos \left(\theta_{y_{i}\right)\right)+\sum_{j \neq y_{i} e^{\left\|W_{j}\right\|\left\|\boldsymbol{x}_{i}\right\| \cos \left(\theta_{j}\right)}$

文中进一步将**类内距离**与**类间距离**的**相对重要性**定义为$f_{y_i}$和$f_j$的梯度范数相对于margin-based softmax loss的**比率** （$r_{i}^{t}$中的$t$就是表示前面提到的t函数）:

$$r_{i}^{t}=\frac{\left\|\frac{\partial L_{i}^{t}{\partial f_{y_{i}\right\|}{\left\|\frac{\partial L_{i}^{t}{\partial f_{j}\right\|}=\frac{\left(1-p_{y_{i}^{t}\right)}{p_{j}^{t} t^{\prime}\left(f_{y_{i}\right) \tag{8}$$

同理相对于原始的softmax loss（公式1）的重要性比率是：

$$r_{i}^{o}=\frac{\left\|\frac{\partial L_{i}^{o}{\partial f_{y_{i}\right\|}{\left\|\frac{\partial L_{i}^{o}{\partial f_{j}\right\|}=\frac{\left(1-p_{y_{i}^{o}\right)}{p_{j}^{o}\tag{9}$$

进一步可以求得：

$$\begin{array} \frac{r_{i}^{t}{r_{i}^{o}&=\frac{\frac{\left(1-p_{y_{i}^{t}\right)}{p_{j}^{t}{\frac{\left(1-p_{y_{i}^{o}\right)}{p_{j}^{o} t^{\prime}\left(f_{y_{i}\right) = \frac{ \frac{\sum_{t \neq y_{i} e^{\left\|\boldsymbol{W}_{t}\right\|\left\|x_{i}\right\| \cos \left(\theta_{t}\right)}{e^{\left\|\boldsymbol{W}_{j}\right\|\left\|x_{i}\right\| \cos \left(\theta_{j}\right)}{\frac{\sum_{t \neq y_{i} e^{\left\|W_{t}\right\|\left\|x_{i}\right\| \cos \left(\theta_{t}\right)}{e^{\left\|W_{j}\right\|\left\|x_{i}\right\| \cos \left(\theta_{j}\right)}t^{\prime}\left(f_{y_{i}\right)\\ &=t^{\prime}\left(f_{y_{i}\right) \tag{10} \end{array}$$

由公式10可以知道定义的**损失函数表达式（公式4）中的$t(\cdot)$的导函数实际上是具有控制类内距离对于类间距离显著性的作用**

## **4. 搜索空间**

基于第3节的分析，我们可以知道可以在公式（3）作如下两处的变换：

$$L_{i}^{\tau, t}=-\tau\left(\log \left(p_{y_{i}^{t}\right)\right) \tag{11}$$

其中$\tau$和$t$是需要我们进行搜索的任意函数。另外上式中的$\tau$的定义域在$[-\infty，0]$，为了使得其定义域在$[0,1]$进而简化搜索空间，将公式(11)作如下变化 （令公式(12)中的$\tau_{2}(x)=e^{\tau_{1}(\log (x))}$可以使得公式(11)和公式(12)等价）：

$$L_{i}^{\tau, t}=-\log \left(\tau\left(p_{y_{i}^{t}\right)\right) \tag{12}$$

其中（注意看清楚$t$和$\tau$）

* $t(\cdot)=a_{i}^{t} x+b_{i}^{t}, x \in\left[\zeta_{i}^{t}, \zeta_{i+1}^{t}\right]$

+ $\zeta^{t}=\left[\zeta_{0}, \ldots \zeta_{M}\right]$
+ $M$表示间隔数，即$\zeta_{i+1}^{t}-\zeta_{i}^{t}=\left(\zeta_{M}^{t}-\zeta_{0}^{t}\right) / M$
+ 所以$t(\cdot)$函数由三个超参数组成$a_{i}^{t}, b_{i}^{t}$ and $\zeta_{i}^{t}$

* $\tau$同理由三个超参数组成：$a_{i}^{\tau}, b_{i}^{\tau}$ and $\zeta_{i}^{\tau}$

另外$\zeta_{i}^{t}$和$\zeta_{i}^{\tau}$需要实现设定好，那么最后搜索空间定义为

## \boldsymbol{\theta}=\left[\boldsymbol{a}^{t^{T}}, \boldsymbol{b}^{t^{T}}, \boldsymbol{a}^{\tau T}, \boldsymbol{b}^{\tau T}\right]^{T} \tag{13} 5. 优化

双层（Bilevel）优化定义如下： $\begin{array}{l} \max _{\boldsymbol{\theta} R(\boldsymbol{\theta})=r\left(M_{\boldsymbol{\omega}^{*}(\boldsymbol{\theta}), \mathcal{D}_{v}\right) \\ \text { s.t. } \boldsymbol{\omega}^{*}(\boldsymbol{\theta})=\arg \min _{\boldsymbol{\omega} \sum_{(x, y) \in \mathcal{D}_{t} L^{\boldsymbol{\theta}\left(M_{\boldsymbol{\omega}(x), y\right) \tag{14} \end{array}$

可以看到

* **内层优化**是在固定损失函数后，在训练集上更新模型超参数使得损失函数值最小，得到当前最优的模型参数$\omega^*(\theta)$
* **外层优化**则是去找到一组损失函数搜索空间超参数$\theta$使得最优的模型参数$\omega^*(\theta)$在验证集上能取得最大的奖励。

内层优化比较好理解，可是外层优化应该是固定的模型参数，那么最后无论损失函数是什么都不会影响模型输出吧，那怎么最大化奖励呢？文中的做法是这样的（可结合下面的算法流程图来理解）：

1. 在每个epoch采样$B$组损失函数超参数$\left\{\boldsymbol{\theta}_{1}, \ldots \boldsymbol{\theta}_{B}\right\}$，其中超参数$\theta$服从独立的高斯分布，即$\boldsymbol{\theta} \sim \mathcal{N}(\boldsymbol{\mu}, \sigma I)$。
2. 先后用$B$个损失函数来训练当前的模型，得到$\{M_t^0,...,M_t^B\}$，计算每个模型在验证集上的reward，即$\{R^0,...,R^B\}$。
3. 更新：

* **更新模型权重$\omega$**：选择reward最大的模型权重作为下一个epoch模型
* **更新搜索空间$\theta$**：$\mu_{t+1}=\mu_{t}+\eta \frac{1}{B} \sum_{i=1}^{B} R\left(\theta_{i}\right) \nabla_{\theta} \log \left(g\left(\theta_{i} ; \mu_{t}, \sigma\right)\right)$，其中$g(\boldsymbol{\theta} ; \boldsymbol{\mu}, \sigma)$是高斯分布的PDF（概率密度函数），为了简化难度，方差$\sigma$是固定的，因此只需要更新均值$\mu$。

![](/assets/img/marsggbo/2020-08-11-自动搜索损失函数AM-LFSAutoML-for-Loss-Function-Search/9845ea1b.jpg)![](/assets/img/marsggbo/2020-08-11-自动搜索损失函数AM-LFSAutoML-for-Loss-Function-Search/e995938f.jpg)

## **6. 实验结果**

原论文给出了在多个不同类型的数据集的结果：

* **Classification**: Cifar10

![](/assets/img/marsggbo/2020-08-11-自动搜索损失函数AM-LFSAutoML-for-Loss-Function-Search/bfaf0ed4.jpg)

* **Face Recognition**：CASIA-Webface用作训练集，MegaFace用做测试集

![](/assets/img/marsggbo/2020-08-11-自动搜索损失函数AM-LFSAutoML-for-Loss-Function-Search/3520da50.jpg)

* **Person ReID**：Market-1501 and DukeMTMC-reID

![](/assets/img/marsggbo/2020-08-11-自动搜索损失函数AM-LFSAutoML-for-Loss-Function-Search/aa0a9b2f.jpg)![](/assets/img/marsggbo/2020-08-11-自动搜索损失函数AM-LFSAutoML-for-Loss-Function-Search/eaf82ba2.jpg)

## **7. 讨论**

本小节是我自己在读完这篇论文后的一些存疑或者觉得需要讨论的点：

* 文中搜索空间的构造引入了两个需要手动设计的超参数（公式12和13），即$\zeta_{i}^{t}$和$\zeta_{i}^{\tau}$，这或多或少依赖于手动调参技巧。
* 文中的损失函数搜索只是局限在了对softmax loss函数的变体搜索，而且对于多loss组成的任务，论文仅仅对softmax loss部分做了替换，其余部分保持不变。
* 由算法流程图可以看到，每个epoch都需要基于多个损失函数来训练模型，然后再在验证集上得到reward，这需要大量的计算资源，论文中说了，他们使用了64个GPU，这个emm。。。大力出奇迹，赞！

本篇文章是基于自己的理解写的，所以可能会有不正确的地方，欢迎指正！

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-08-11-自动搜索损失函数AM-LFSAutoML-for-Loss-Function-Search/029a9211.jpeg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2020-08-11 17:30:58**
