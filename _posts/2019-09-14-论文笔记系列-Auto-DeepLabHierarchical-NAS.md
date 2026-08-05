---
layout: post
title: 论文笔记系列-Auto-DeepLab:Hierarchical NAS
date: '2019-09-14'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/82491292
related_posts: false
toc:
  sidebar: left
---

> Pytorch实现代码:[https://github.com/MenghaoGuo/AutoDeeplab](https://link.zhihu.com/?target=https%3A//github.com/MenghaoGuo/AutoDeeplab)

## **创新点**

## **cell-level and network-level search**

以往的NAS算法都侧重于搜索cell的结构，即当搜索得到一种cell结构后只是简单地将固定数量的cell按链式结构连接起来组成最终的网络模型。**AutoDeeplab则将如何cell的连接方式也纳入了搜索空间中**，进一步扩大了网络结构的范围。

## **dense image prediction**

之前的大多数NAS算法都是基于image level的分类，例如DARTS，ENAS等都是在CIFAR10和ImageNet上做的实验，AutoDeeplab则是成功地将NAS应用到了目标检测和图像分割任务上。

## **算法**

## **Cell level search space**

cell level的结构搜索方式参考的是DARTS，细节可参阅[论文笔记系列-DARTS: Differentiable Architecture Search](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/marsggbo/p/9594630.html)。

搜索空间主要由如下8个operation组成：

* 3 × 3 max pooling
* 3 × 3 average pooling
* 3 × 3 atrous conv with rate 2
* 5 × 5 atrous conv with rate 2
* 3 × 3 depthwise-separable conv
* 5 × 5 depthwise-separable conv
* skip connection
* no connection (zero)

一个cell的示意图如下(为方便说明每个子节点之间只有三种operation，不同颜色的连线代表不同操作)，0表示第一个子节点，它会接收前两层的cell的输出作为输入；

![](/assets/img/marsggbo/2019-09-14-论文笔记系列-Auto-DeepLabHierarchical-NAS/d083b102.png)

下面我们先以1-2为例看节点之间的计算方式：

1子节点表示为 $H^l_1$ ,1到2子节点之间的操作可以表示为:

$$\overline{O}{1 \rightarrow 2}(H^l_1)=\sum{k=1}^3\alpha^k_{1 \rightarrow 2} O^k(H^l_1)$$

其中 $\alpha^k$ 表示第k个operation的概率，上图中一共有三种操作，所以最终的操作应该是三种操作的加权值，另外三个操作的和应该为1，所以通常需要使用softmax操作来实现。更一般化的表达方式如下：

$$\begin{array}{l}{\qquad \overline{O}{j \rightarrow i}\left(H{j}^{l}\right)=\sum{O^{k} \in \mathcal{O} \alpha{j \rightarrow i}^{k} O^{k}\left(H_{j}^{l}\right)} \  \end{array}$$

其中${\qquad \begin{aligned} \sum{k=1}^{|\mathcal{O}|} \alpha{j \rightarrow i}^{k}=1 & \,\,\,\, \forall i, j \ \alpha_{j \rightarrow i}^{k} \geq 0 & \,\,\,\, \forall i, j, k \end{aligned}$

有了操作的表达式后，那么每个子节点的表达方式也就是对多个输入节点作加权求和，如下：

$$H{i}^{l}=\sum{H{j}^{l} \in \mathcal{I}{i}^{l} O{j \rightarrow i}\left(H{j}^{l}\right)$$

## **Network-level search space**

![](/assets/img/marsggbo/2019-09-14-论文笔记系列-Auto-DeepLabHierarchical-NAS/23d63a5a.jpg)

上图左边画的是network-level，横向表示layer，纵向表示图像分辨率(2表示原图是特征图的2倍，其他同理)。

* **灰色小圆圈表示固定的stem层**，可以理解为固定的预处理层，即原图会首先经过一些列操作后得到缩小4倍的特征图，然后会在该特征图上进行模型结构搜索。
* **蓝色小圆圈表示候选节点**，每个节点都可以是一个cell结构
* **灰色箭头表示每个cell节点数据可能的流动方向，可以看到一个节点最多可能有三种流动方向**，即**分辨率增大一倍，保持不变和减小一倍**。这样做的目的是避免分辨率变化太大而导致信息量丢失过多。例如如果从4直接连接到32，这个画面太美不敢看，所以人为设定了前面的限制（虽然没有实验证明这样不可以，但是凭直觉这样貌似不可以，如果钱和设备像和谷歌一样多也还是可以试一试的）

右边刚开始看的时候还以为就只是介绍了cell结构，但是结合代码后发现有个地方稍微有些不同，这个其实在后面的论文中也有介绍，但是当时没注意看，即每个节点的表达式如下：

$$\begin{aligned}^{s} H^{l}=& \beta{\frac{\varepsilon}{2} \rightarrow s}^{l} \operatorname{Cell}\left(^{\frac{s}{2} H^{l-1},^{s} H^{l-2} ; \alpha\right) \ &+\beta{s \rightarrow s}^{l} \operatorname{Cell}\left(^{s} H^{l-1},^{s} H^{l-2} ; \alpha\right) \ &+\beta_{2 s \rightarrow s}^{l} \operatorname{Cell}\left(^{2 s} H^{l-1},^{s} H^{l-2} ; \alpha\right) \end{aligned}$$

其中$\begin{array}{ll}{\beta{s \rightarrow \frac{s}{2}^{l}+\beta{s \rightarrow s}^{l}+\beta_{s \rightarrow 2 s}^{l}=1} & {\forall s, l} \  {\beta{s \rightarrow \frac{s}{2}^{l} \geq 0 \quad \beta{s \rightarrow s}^{l} \geq 0} & {\beta_{s \rightarrow 2 s}^{l} \geq 0 \quad \forall s, l}\end{array}$

上面的公式乍看会很懵，我们慢慢看：

* 首先 $\beta$ 表示某条路径的概率，例如 $\beta^l_{s \rightarrow s}$ 表示下图中的红色箭头路径的概率，其他同理。
* $\text{Cell}(^{s} H^{l-1},^{s} H^{l-2}; \alpha)$ 表示输入节点为下图中的两个红色节点，α表示cell的内部结构

![](/assets/img/marsggbo/2019-09-14-论文笔记系列-Auto-DeepLabHierarchical-NAS/fdbfaf9f.jpg)
