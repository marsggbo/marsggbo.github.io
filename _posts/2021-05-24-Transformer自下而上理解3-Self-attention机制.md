---
layout: post
title: "Transformer自下而上理解(3) Self-attention机制"
date: 2021-05-24
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/374868669
related_posts: false
toc:
  sidebar: left
---

> “ 本文笔记参考Wang Shusen老师的课程：[https://www.youtube.com/watch?v=Vr4UNt7X6Gw&list=PLvOO0btloRnuTUGN4XqO85eKPeFSZsEqK&index=9](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DVr4UNt7X6Gw%26list%3DPLvOO0btloRnuTUGN4XqO85eKPeFSZsEqK%26index%3D9)  
>  ”

## **1. 前言**

2015年，在文献[1]中首次提出attention。到了2016年，在文献[2]中提出了**self-attention**方法。作者将self-attention和LSTM结合用在了机器阅读任务上。为了好理解，下文将LSTM表示成SimpleRNN。

在阅读以下内容之前，强烈建议先看看之前关于attention机制的文章介绍

[marsggbo：Transformer自下而上(2) 注意力（Attention）机制](https://zhuanlan.zhihu.com/p/374841046)

## **2. SimpleRNN (LSTM)**

由下图可以看到传统的LSTM的第一个输出![h_1](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/30069570.jpg)只依赖于两个输入![x_1](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/1d118709.jpg)和![h_0](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/34e3bc9b.jpg)

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/62383e5f.jpg)

## **3. SimpleRNN + Attention**

下面我们会逐项介绍计算过程。

## **3.1 计算h_1和c_1**

下图给出了加入Attention机制后的示意图，可以看到和Fig 1. 的区别在于我们把![h_0](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/34e3bc9b.jpg)替换成了![c_0](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/52bfdd2a.jpg)。由于![h_0](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/34e3bc9b.jpg)和![c_0](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/52bfdd2a.jpg)是已经初始化好了的，所以根据下图中的公式我们能直接计算出![h_1](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/30069570.jpg)

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/07b6f364.jpg)

接下来我们需要计算![c_1](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/fd2998fe.jpg)。Attention的目的是为了避免遗忘，所以一种很自然的思路就是![c_i](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/2ddddce6.jpg)是所有之前状态![\{h0,..,h_{i-1}\}](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/413b7765.jpg)的加权求和，他们的权重分别是![\{\alpha_0,...,\alpha_{i-1}\}](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/02659c69.jpg)。由于通常![h_0](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/34e3bc9b.jpg)初始化为0向量，所以![c_1=h_1](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/e247d9e3.jpg)

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/94c1a813.jpg)

## **3.2 计算h_i和c_i**

看完![h_1](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/30069570.jpg)和![c_1](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/fd2998fe.jpg)的计算是不是还有点懵，没关系，下面我们加大学习力度，重复多看几次计算过程。

计算![h](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/aaaa0c45.jpg)的方法千篇一律，都是那当前的输入![x_i](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/55abd6ee.jpg)和前一时刻的context vector ![c_{i-1}](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/4c0897cf.jpg)拼接成一个向量后参与计算，即

![\mathbf{h}_{i}=\tanh \left(\mathbf{A} \cdot\left[\begin{array}{l} \mathbf{x}_{i} \\ \mathrm{c}_{i-1} \end{array}\right]+\mathbf{b}\right) \\](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bh%7D_%7Bi%7D%3D%5Ctanh+%5Cleft%28%5Cmathbf%7BA%7D+%5Ccdot%5Cleft%5B%5Cbegin%7Barray%7D%7Bl%7D+%5Cmathbf%7Bx%7D_%7Bi%7D+%5C%5C+%5Cmathrm%7Bc%7D_%7Bi-1%7D+%5Cend%7Barray%7D%5Cright%5D%2B%5Cmathbf%7Bb%7D%5Cright%29+%5C%5C)

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/3399dc42.jpg)

下一步是计算![c_2](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/3e04538d.jpg)。![c](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/b67fcc1b.jpg)的通用计算公式可以写成 ![c_i=\alpha_1 h_1+...\alpha_{i-1} h_{i-1}](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/36bf29b4.jpg)

权重![\alpha_i](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/68a970d5.jpg)的计算公式为

![\alpha_{i}=\operatorname{align}\left(\mathbf{h}_{i}, \mathbf{h}_{2}\right) \\](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/372f7d9d.jpg)

上面的![align](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/888111b3.jpg)可以有不同的实现方法([3])，你只需要知道![\alpha_i](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/68a970d5.jpg)表示![h_i](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/f73d80f7.jpg)和![h_2](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/a7d9ff7d.jpg)之间的权重（或者是相似度），计算出所有的![\alpha_i](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/68a970d5.jpg)之后我们就能计算出![c_i](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/2ddddce6.jpg)了，这里![c_2=\alpha_1h_1+\alpha_2h_2](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/97bf2cd8.jpg)

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/b527ce63.jpg)

## **3.3 再计算一次**

**计算** ![h_4](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/f43de9dc.jpg)

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/a4d9ccda.jpg)

同理，要计算![c_4](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/67ce5ee6.jpg)，我们仍然要通过使用![align](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/888111b3.jpg)计算符计算出不同的![\alpha](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/b5edc039.jpg)。

注意，Fig 7里的![\alpha_1,\alpha_2,...](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/a9cf4935.jpg)和Fig 5里的![\alpha](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/b5edc039.jpg)是不一样的，这里只是为了方便讲解。也就是说每计算新的![c](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/b67fcc1b.jpg)都要计算一遍不同的![\alpha](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/b5edc039.jpg)。

为了计算这些权重，我们每次都会遍历一遍之前的数据，所以这样可以有效解决SimpleRNN遗忘的问题。

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/34220502.jpg)

## **参考文献**

[1] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate. In ICLR, 2015

[2] Cheng J, Dong L, Lapata M. Long short-term memory-networks for machine reading. In EMNLP, 2016

[3] Transformer自下而上(2) 注意力（Attention）机制 (<https://zhuanlan.zhihu.com/p/374841046>)

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2021-05-24-Transformer自下而上理解3-Self-attention机制/029a9211.jpeg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2021-05-21 16:39:50**