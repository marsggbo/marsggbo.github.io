---
layout: post
title: "Andrew Ng机器学习课程笔记--week4(神经网络)"
date: 2020-07-30
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/165326460
related_posts: false
toc:
  sidebar: left
---

Neural Networks: Representation

## **一、 内容概要**

* **Neural Network**

+ Model Representation 1
+ Model Representation 2

  
* **Applications**

+ Examples and Intuitions 1
+ Examples and Intuitions 2
+ Multiclass Classification

## **二、重点&难点**

## **1. Neural Network**

### **1）Model Representation 1**

首先需要明确一些符号的意思，以方便后面的阅读。

> **αi(j)**：表示第j层的第i个激活单元（activation） **θ(j)** ：表示第j层映射到第j+1层的控制函数的权重矩阵。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/08f3e31b.jpg)

如图是一个三层结构的神经网络（输入层，隐藏层、输出层），每一层的激活单元的计算表达式图中也已经写出来了。 还需要注意的是：

> **若神经网络在第j层有sj个单元，在j+1层有sj+1个单元，则θ(j)矩阵的维度是（sj+1，sj）,之所以要加1是因为输入层和隐藏层都需要加一个bias。**

如下图，θ(1)的维度是(4, 3)

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/a507d5f0.jpg)

### **2） Model Representation 2**

在上面内容的基础上我们继续抽象化，向量化，使得神经网络计算表达式看起来更加简洁（但是更加抽象了。。。）

* 神经网络结构示例

![\begin{align*} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align*} \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/159c112b.jpg)

* 向量化

![z_k^{(j)}](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/c551935b.jpg) 来向量化g()函数内的值，例如 ![z_k^{(2)} = \Theta_{k,0}^{(1)}x_0 + \Theta_{k,1}^{(1)}x_1 + \cdots + \Theta_{k,n}^{(1)}x_n](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/9c6a55e5.jpg)

![\begin{align*}a_1^{(2)} = g(z_1^{(2)}) \newline a_2^{(2)} = g(z_2^{(2)}) \newline a_3^{(2)} = g(z_3^{(2)}) \newline \end{align*} \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/ef60d099.jpg)

* z与α的关系

![z^{(j)} = \Theta^{(j-1)}a^{(j-1)} \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/ce7f573d.jpg)![a^{(j)} = g(z^{(j)}) \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/524c9c44.jpg)![h_\Theta(x) = a^{(j+1)} = g(z^{(j+1)}) \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/f978130d.jpg)

## **2. Applications**

### **1）神经网络实现简单的与或非**

这里只简单记录一下**或(or)**&**非(not)**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/398a51de.jpg)

or

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/7173e223.jpg)

not

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week4神经网络/f2e7d76c.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**