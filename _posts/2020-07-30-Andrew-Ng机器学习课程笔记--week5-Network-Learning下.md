---
layout: post
title: Andrew Ng机器学习课程笔记--week5 Network: Learning(下)
date: '2020-07-30'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/165331366
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/165331366>

Neural Networks: Learning 内容较多，故分成上下两篇文章。

## **一、内容概要**

* **Cost Function and Backpropagation**

+ Cost Function
+ Backpropagation Algorithm
+ Backpropagation Intuition

  
* **Backpropagation in Practice**

+ Implementation Note：Unroll Parameters
+ Gradient Checking
+ Random Initialization
+ Putting it Together

* **Application of Neural Networks**

+ Autonomous Driving

## **二、重点&难点**

## **1. Backpropagation in Practice**

### **1） Implementation Note：Unroll Parameters**

本节主要讲的是利用octave实现神经网络算法的一个小技巧：将多个参数矩阵展开为一个向量。具体可以参考课程视频，此处略。

### **2） Gradient Checking**

神经网络算法是一个很复杂的算法，所以我们很难凭直觉观察出结果是否正确，因此有必要在实现的时候做一些检查，本节给出一个检验梯度的数值化方法。

首先我们可以将损失函数的梯度近似为 $\frac{∂J(θ)}{∂θ}≈\frac{J(θ+ε)-J(θ-ε)}{2ε}$

推广到一般形式是： $\frac{∂J(θ)}{∂θ_j}≈\frac{J(θ_1,θ_2,θ_j+ε……,θ_n)-J(θ_1,θ_2,θ_j-ε……,θ_n)}{2ε}$

一般来说$\epsilon ≈10^{-4}$时就比较接近了

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning下/e0d00b58.jpg)

最后我们的主要目标是检查这个梯度的近似向量与反向传播算法得到的梯度向量是否近似相等。 实现时的注意点：

* 首先实现反向传播算法来计算梯度向量DVec；
* 其次实现梯度的近似gradApprox;
* 确保以上两步计算的值是近似相等的；
* 在实际的神经网络学习时使用反向传播算法，并且关掉梯度检查。

特别重要的是：

* 一定要确保在训练分类器时关闭梯度检查的代码。如果你在梯度下降的每轮迭代中都运行数值化的梯度计算，你的程序将会非常慢。

### **3) Random Initialization**

关于如何学习一个神经网络的细节到目前为止基本说完了，不过还有一点需要注意，就是如何初始化参数向量or矩阵。通常情况下，我们会将参数全部初始化为0，这对于很多问题是足够的，但是对于神经网络算法，会存在一些问题，以下将会详细的介绍。

对于梯度下降和其他优化算法，对于参数向量的初始化是必不可少的。能不能将初始化的参数全部设置为0?

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning下/73f8435f.jpg)

在神经网络中,如果将参数全部初始化为0 会导致一个问题，例如对于上面的神经网络的例子，如果将参数全部初始化为0，在每轮参数更新的时候，与输入单元相关的两个隐藏单元的结果将是相同的，既： $a_1^{(2)} = a_2^{(2)}$ 这个问题又称之为**对称的权重问题**，因此我们需要打破这种对称，这里提供一种随机初始化参数向量的方法： 初始化$θ_{ij}^{(l)}$为一个落在 [-ε,ε]区间内的随机数, 可以很小，但是与上面梯度检验( Gradient Checking)中的ε没有任何关系。

### **4)Putting it together(组合到一起-如何训练一个神经网络)**

> 这个老师说会在后面更加具体的介绍。

关于神经网络的训练，我们已经谈到了很多，现在是时候将它们组合到一起了。那么，如何训练一个神经网络？

* 首先需要确定一个神经网络的结构-神经元的连接模式, 包括：

+ 输入单元的个数：特征 的维数；
+ 输出单元的格式：类的个数
+ 隐藏层的设计：比较合适的是1个隐藏层，如果隐藏层数大于1，确保每个隐藏层的单元个数相同，通常情况下隐藏层单元的个数越多越好。

* 在确定好神经网络的结构后，我们按如下的步骤训练神经网络：

+ 随机初始化权重参数；
+ 实现：对于每一个 通过前向传播得到;
+ 实现：计算代价函数；
+ 实现：反向传播算法用于计算偏导数
+ 使用梯度检查来比较反向传播算法计算的和数值估计的的梯度，如果没有问题，在实际训练时关闭这部分代码；
+ 在反向传播的基础上使用梯度下降或其他优化算法来最小化;

## **Application of Neural Networks**

主要介绍了老师的一个大佬朋友利用神经网络设计的自动驾驶汽车的视频，感兴趣的可以看看。**[自动驾驶汽车](https://link.zhihu.com/?target=https%3A//www.coursera.org/learn/machine-learning/lecture/zYS8T/autonomous-driving)**

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning下/f2e7d76c.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**