---
layout: post
title: "论文笔记系列--iCaRL： Incremental Classifier and Learning"
date: '2019-01-25'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/55739596
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/55739596>

## **导言**

传统的神经网络都是基于固定的数据集进行训练学习的，一旦有新的，不同分布的数据进来，一般而言需要重新训练整个网络，这样费时费力，而且在实际应用场景中也不适用，所以增量学习应运而生。

增量学习主要旨在解决**灾难性遗忘(Catastrophic-forgetting)** 问题,本文将要介绍的《iCaRL： Incremental Classifier and Representation Learning》一文中对增量学习算法提出了如下三个要求：

a) 当新的类别在不同时间出现，它都是可训练的

b) 任何时间都在已经学习过的所有类别中有很好的分类效果

c) 计算能力与内存应该随着类别数的增加固定或者缓慢增长

有条件的可以去油管听听原作者对这篇论文的讲座：[Christoph Lampert: iCaRL- incremental Classifier and Representation Learning](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DHCKi41BHDAk)

## **简要概括**

本文提出的方法只需使用**一部分旧数据**而非全部旧数据就能**同时训练得到分类器和数据特征**从而实现增量学习。

大致流程如下：

1.使用特征提取器φ(⋅)对新旧数据的(旧数据只取一部分)提取特征向量，并计算出**各自的平均特征向量**  
2.通过**最近均值分类算法(Nearest-Mean-of-Examplars)** 计算出新旧数据的预测值  
3.在上面得到的预测值代入如下**loss函数**进行优化，最终得到模型。

![](/assets/img/marsggbo/2019-01-25-论文笔记系列--iCaRL-Incremental-Classifier-and-Learning/c69985e2.jpg)

本文的重点在上面三个步骤中用黑体标出，下面对这三个进行具体介绍

## **1.平均特征向量**

这个其实很好理解，就是把某一类的图像的特征向量都计算出来，然后求均值，注意本文对于旧数据，只需要计算一部分的数据的特征向量。

什么意思呢？

假设我们现在已经训练了s−1个类别的数据了，记为 $X^1,...,X^{s−1}$ ，因为通常内存资源有限，所以假设从每个旧数据类中选出一定数量的数据组成examplar sets，记为 $P^1,...,P^{s−1}$ 。

然后现在又得到了t−s个新数据，记为 $X_s,...,X_t$ 。同理我们也需要提取出一部分数据，记为 $P^s,...,P^t$

> 如何选取数据可参见文末[算法示意图](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/marsggbo/p/10321834.html%23%25E9%2580%2589%25E5%258F%2596%25E6%2595%25B0%25E6%258D%25AE%25E7%25AE%2597%25E6%25B3%2595%25E7%25A4%25BA%25E6%2584%258F%25E5%259B%25BE)。

有了新旧数据后，我们可以先将它们合并，记为 $P=\{P^1,...,P^t\}$ ，然后就可以使用特征提取器φ(⋅)φ(·)计算每个类别的平均特征向量了。

![](/assets/img/marsggbo/2019-01-25-论文笔记系列--iCaRL-Incremental-Classifier-and-Learning/2a4ca61c.jpg)

## **2.最近均值分类算法(Nearest-Mean-of-Examplars classification)**

![](/assets/img/marsggbo/2019-01-25-论文笔记系列--iCaRL-Incremental-Classifier-and-Learning/f707bb0d.jpg)

算法第七行在文首给出的讲座中，使用的是 $∥φ(x)−μy∥^2$ 。 emm... anyway，这不是重点，pass。

## **3.优化loss函数**

机器学习归根到底其实就是优化，那么loss函数如何设定才能解决灾难性遗忘的问题呢？

本文的损失函数定义如下，由新数据分类loss和旧数据蒸馏loss组成。下面公式中的 $g_y(x_i)$ 表示分类器，即\_ $g_y(x)=\frac{1}{1+e^{−w^T_yφ(x)}}$ 。

![](/assets/img/marsggbo/2019-01-25-论文笔记系列--iCaRL-Incremental-Classifier-and-Learning/c69985e2.jpg)

其实该想法其实是基于LWF这篇论文，LWF的loss函数如下：

![](/assets/img/marsggbo/2019-01-25-论文笔记系列--iCaRL-Incremental-Classifier-and-Learning/1511709f.jpg)

## **结果**

本文最终结果如下图示，将iCaRL,fixed representation(feature extraction), fine-tuning和LWF进行了比较，可以看到iCaRL表现最好。

![](/assets/img/marsggbo/2019-01-25-论文笔记系列--iCaRL-Incremental-Classifier-and-Learning/1012c239.jpg)

## **讨论**

需要说明的是iCaRL和LWF最大的不同点有如下：

* iCaRL在训练新数据时仍然需要使用到旧数据，而LWF完全不用。所以这也就是为什么LWF表现没有iCaRL好的原因，因为随着新数据的不断加入，LWF逐渐忘记了之前的数据特征。
* iCaRL提取特征的部分是固定的，只需要修改最后分类器的权重矩阵。而LWF是训练整个网络(下图给出了LWF和fine-tuning以及feature extraction的示意图)。

![](/assets/img/marsggbo/2019-01-25-论文笔记系列--iCaRL-Incremental-Classifier-and-Learning/b1128411.jpg)

## **选取数据算法示意图**

![](/assets/img/marsggbo/2019-01-25-论文笔记系列--iCaRL-Incremental-Classifier-and-Learning/ad1837a8.jpg)![](/assets/img/marsggbo/2019-01-25-论文笔记系列--iCaRL-Incremental-Classifier-and-Learning/d594572b.jpg)

**MARSGGBO♥原创**  
  
  
  
  
  
**2019-1-25**