---
layout: post
title: "Transformer自下而上(2) 注意力（Attention）机制"
date: '2021-05-24'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/374841046
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/374841046>

目录

* 1. 早期Seq2Seq缺点
* 2. 注意力机制
* 2.1 注意力机制
* 2.2 注意力计算
* 2.3 注意力应用
* 2.4 Attention计算复杂度

## **1. 早期Seq2Seq的缺点**

在介绍注意力机制之前，我们首先回顾一下Seq2Seq模型并思考一下它有哪些缺点。

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上2-注意力Attention机制/743d59a7.jpg)

可以看到Decoder做预测非常依赖于Encoder传入$s_0=(h_m,c_m)$，所以encoder对句子的编码能力是至关重要的。

不过又实验结果表明，但句子长度很长之后，Encoder的编码能力会明显下降，导致最终的模型性能变差，如下图示。可以看到未加入attention机制时，当句子长度超过20个单词且不断增加的时候，模型性能指标BLEU也会不断下降，这是因为句子过长，Encoder最后的输出状态会忘记前面的句子内容，而加入了attention后，效果得到明显改善。

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上2-注意力Attention机制/7ccf877b.jpg)

## **2. Seq2Seq Model with Attention**

## **2.1 注意力机制**

Attention机制最早是在[[1]](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1409.0473)中提出的。通过该算法Seq2Seq不会忘记原始输入句子信息，而且Decoder也能够知道句子中哪些词比较重要。只不过，attention机制也会引入额外很多计算，下面进行详细介绍。

之前的Seq2Seq输出的$s_0$其实就是$h_m$，而前面所有的隐状态信息都被丢掉了，所以attention做的事情就是把前面信息$H=\{h_1,h_2,...h_m\}$都利用起来。

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上2-注意力Attention机制/e5a886fa.jpg)

具体来说就是将$s_0$和前面的做运算得到对应的权重$\{\alpha_1,...,\alpha_m\}$，计算符号表示为$\alpha_i=align(h_i,s_0)$

## **2.2 注意力计算**

注意力计算可以有不同的实现方法，比如论文[1]中的方法如下：

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上2-注意力Attention机制/955b16f3.jpg)

上图中的计算方法简单理解就是矩阵乘法操作。

* 1.我们先看上图最右边，首先将$h_i$和$s_0$拼接成一个向量
* 2.然后对拼接得到的向量左乘一个矩阵$W$，该矩阵是一个可学习的参数矩阵。另外可以看到矩阵$W$也有不同颜色的矩阵组成，其实也可以理解成$W$是由两个不同的参数矩阵拼凑而成的，左右分别是$h_i$和$s_0$的参数矩阵
* 3.$W$和![[h_i,s_0]^T](https://www.zhihu.com/equation?tex=%5Bh_i%2Cs_0%5D%5ET)的相乘结构取tanh后，会再左乘一个向量$V^T$，这样就得到了$\alpha_i$。
* 4.计算出所有$\alpha_i$后会再使用softmax做处理。

写到这我突然发现，上述过程其实和卷积网络中的**Separable Depthwise Convolution**操作很像，比如上面的第二步的$h_i,s_0$其实就可以理解成两个不同的通道，$W$的作用就是对每一层做特征提取。 第二步完了之后，我们只是得到了每一层通道的特征而已，但是通道之间的关系还没有得到，所以还需要通过一个向量来将通道之间的信息关联起来，这和卷积网络中的1\*1的卷积作用非常类似。

另一种更常用的计算注意力的方式是类似于Transformer的那种方式，如下图示，可以看到计算思路比较类似，主要三个步骤：

* 1. 线性映射
* 对每个$h_i$都有一个对应的参数可学习的矩阵$W_K$,计算得到$m$个矩阵$k_i=W_k\cdot h_i$
* 对$s_0$也会做线性计算得到$q_0=W_Q\cdot s_0$
* 2. $h_i,s_0$的特征计算后，再用内积计算就求得了注意力权重$\tilde{\alpha_i}=k_i^Tq_0$
* 3. 使用softmax对前面计算得到的$\tilde{\alpha_i}$做归一化处理

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上2-注意力Attention机制/aad90eb8.jpg)

## **2.3 注意力应用**

下图给出了Attention整体的计算方法

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上2-注意力Attention机制/0973fd15.jpg)

可以看到在使用Attention之前，Decoder的第一个输出$s_1$为

![\mathrm{s}_{1}=\tanh \left(\mathbf{A}^{\prime} \cdot\left[\begin{array}{l} \mathbf{x}_{1}^{\prime} \\ \mathrm{s}_{0} \end{array}\right]+\mathbf{b}\right) \tag{1} ](https://www.zhihu.com/equation?tex=%5Cmathrm%7Bs%7D_%7B1%7D%3D%5Ctanh+%5Cleft%28%5Cmathbf%7BA%7D%5E%7B%5Cprime%7D+%5Ccdot%5Cleft%5B%5Cbegin%7Barray%7D%7Bl%7D+%5Cmathbf%7Bx%7D_%7B1%7D%5E%7B%5Cprime%7D+%5C%5C+%5Cmathrm%7Bs%7D_%7B0%7D+%5Cend%7Barray%7D%5Cright%5D%2B%5Cmathbf%7Bb%7D%5Cright%29+%5Ctag%7B1%7D+)

计算出各个隐状态$h_i$对应的权重后，会进一步计算出$c_0$，即

$$
c_0=\alpha_1h_1+...\alpha_mh_m  \tag{2}
$$

之后Decoder的会将$c_0$也考虑进去，此时$s_1$的计算公式为：

![\mathbf{s}_{1}=\tanh \left(\mathbf{A}^{\prime} \cdot\left[\begin{array}{l} \mathbf{x}_{1}^{\prime} \\ \mathbf{s}_{0} \\ \mathbf{c}_{0} \end{array}\right]+\mathbf{b}\right)  \tag{3} ](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bs%7D_%7B1%7D%3D%5Ctanh+%5Cleft%28%5Cmathbf%7BA%7D%5E%7B%5Cprime%7D+%5Ccdot%5Cleft%5B%5Cbegin%7Barray%7D%7Bl%7D+%5Cmathbf%7Bx%7D_%7B1%7D%5E%7B%5Cprime%7D+%5C%5C+%5Cmathbf%7Bs%7D_%7B0%7D+%5C%5C+%5Cmathbf%7Bc%7D_%7B0%7D+%5Cend%7Barray%7D%5Cright%5D%2B%5Cmathbf%7Bb%7D%5Cright%29++%5Ctag%7B3%7D+)

计算出第一个预测值$s_1$后，之后的预测值怎么计算呢？如下图示，其实和第一个预测值的计算方法是类似的，差别就在于输入数据状态变成了$s_1$和$c_1$。这个$c_1$的计算和$c_0$是类似的，差别就在于$c_0$的基于$h_i$和$s_0$计算得到的，而$c_1$是基于$h_i$和$s_1$计算得到的。

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上2-注意力Attention机制/00046226.jpg)

## **2.4 Attention计算复杂度**

下图给出了基于Attention的Seq2Seq完整流程图，可以看到Decoder每个输出都会对应一个$c_i$，它记录着上一个输出$s_{i-1}$和Encoder中所有隐状态之间的关系。为了计算$c_i$，我们每次都需要计算得到$m$个权重，即$\alpha_1,...,\alpha_m$。因为Decoder总共有$t$个状态，所以总共需要额外计算$mt$个权重。

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上2-注意力Attention机制/201e3430.jpg)

## **参考文献**

* [1] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.

### **微信公众号：AutoML机器学习**

![](/assets/img/marsggbo/2021-05-24-Transformer自下而上2-注意力Attention机制/029a9211.jpeg)

**MARSGGBO♥原创**

**如有意合作或学术讨论欢迎私戳联系~**

**邮箱:marsggbo@foxmail.com**

**2021-04-19 10:27:18**