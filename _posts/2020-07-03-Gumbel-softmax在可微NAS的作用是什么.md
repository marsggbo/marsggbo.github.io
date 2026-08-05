---
layout: post
title: "Gumbel softmax在可微NAS的作用是什么？"
date: '2020-07-03'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/153562583
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/153562583>

## **一、简单回顾DARTS**

在介绍gumbel softmax之前，我们需要首先介绍一下什么是可微NAS。

可微NAS(**Differentiable Neural Architecture Search, DNAS**)是指以可微的方式搜索网络结构，比较经典的算法是[DARTS](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1806.09055)，其算法示意图如下：

![](/assets/img/marsggbo/2020-07-03-Gumbel-softmax在可微NAS的作用是什么/6ad9b876.jpg)

上图表示的是一个cell的结构。一个cell由若干个节点（node）组成，每组节点之间通过若干条边（edge）连接起来，每条edge表示不同的操作（用$o$表示），比如卷积或者池化操作等。DARTS的想法是每条edge都有一个权重（用$\alpha$表示），而且权重是可以通过梯度更新的，最后会根据权重来选择节点之间的操作，计算公式如下：

$$
\bar{o}^{(i, j)}(x)=\sum_{o \in \mathcal{O}} \frac{\exp \left(\alpha_{o}^{(i, j)}\right)}{\sum_{o^{\prime} \in \mathcal{O}} \exp \left(\alpha_{o^{\prime}}^{(i, j)}\right)} o(x) \\
$$

乍看起来好像挺好的，但是有一个问题。为方便讨论，我们仅讨论两个节点的情况，我们假设一共有3个候选操作，且三个操作的权重随机初始化为[0.2,0.3,0.5]。在经过一波训练后，权重得到了更新变成了[0.1,0.2,0.7]，这表示第三个操作的可能效果更好，所以应该以更大的概率选择第三个操作。

## **二、DARTS缺点**

可是DARTS算法在更新权重的过程中是并不是根据概率选择操作的，而是向上面的公式一样把所有操作乘上对应的权重得到mixed的结果，在权重更新结束后会简单地只保留每组节点之间权重最大的那个操作。这样一来有两个问题： 1）每次更新都是对所有操作进行更新，这导致内存消耗更大； 2）最后只是简单地选择权重最大的操作，那么[0.2,0.3,0.5]和[0.1,0.2,0.7]并没有本质的区别了，而且这样一来可能第一个和第二个操作根本就没有机会得到更新，但是从概率上来说这两个权重分布差别是巨大的。

所以一个很自然的想法就是我们希望以0.1的概率选择第一个操作，0.2的概率选择第二个操作，0.7的概率选择第三个操作。实现起来其实也挺简单的，直接用`np.random.choice`就可以按照一定概率随机选取操作。可是这样一来又产生了一个新的问题，即这种**随机采样的方式没法计算梯度。**

为什么没法计算梯度呢？我们考虑如下简单情况写一下表达式：

![](/assets/img/marsggbo/2020-07-03-Gumbel-softmax在可微NAS的作用是什么/51805cf7.jpg)

* DARTS的计算表达式，可以看到是可以顺利求导的

$$
\begin{array}{cl} y &= z_1 + z_2+z_3 \\ &=w_1o_1(x)+w_2o_2(x)+w_3o_3(x) \\ \Rightarrow & \frac{\partial y}{\partial w_1}=o_1(x),\frac{\partial y}{\partial w_2}=o_2(x),\frac{\partial y}{\partial w_3}=o_3(x) \end{array} \\
$$

* 以一定概率随机采样的表达式（右边表示概率），可以看到这种随机采样无法求出梯度。

$$
y=\left\{\begin{array}{l} o_{1}(x),  \,\,\,(p=w_1) \\ o_{2}(x),  \,\,\,(p=w_2)  \\ o_{3}(x),  \,\,\,(p=w_2)  \end{array}\right. \\
$$

## **三、Gumbel softmax登场**

为了解决上面无法求导的问题，Gumbel softmax登场。它主要是使用了重参数技巧(Re-parameterization Trick)。

举个简单的栗子来帮助理解重参数技巧（gumbel softmax比这要稍微复杂一点，不过原理是一样的）：

假设现在求得的权重分布是![W=[0.1,0.2,0.7]](https://www.zhihu.com/equation?tex=W%3D%5B0.1%2C0.2%2C0.7%5D)。

然后再假设我们可以根据某种分布对每个权重采样一个随机值，比如三个权重对应的采样的随机值分别是![\epsilon=[0.5,0.6,0.05]](https://www.zhihu.com/equation?tex=%5Cepsilon%3D%5B0.5%2C0.6%2C0.05%5D),我们把这些随机值和权重相加之后得到![\hat{W}=[0.1+0.5,0.2+0.6,0.7+0.05]=[0.6,0.8,0.75]](https://www.zhihu.com/equation?tex=%5Chat%7BW%7D%3D%5B0.1%2B0.5%2C0.2%2B0.6%2C0.7%2B0.05%5D%3D%5B0.6%2C0.8%2C0.75%5D)。所以$\hat{W}=W+\epsilon, \epsilon \thicksim P(某种分布)$，一般这个分布可以是0到1之间的均匀分布，即$\epsilon \thicksim U(0,1)$。

之后我们对采样随机值后的权重分布取$argmax(\hat{W})$的话应该是选择第二个操作，当然这种概率是比较小的，这个也叫Gumbel-Max trick。可是argmax也有无法求导的问题，因此可以使用softmax来代替，也就是Gumbel-Softmax trick，那么有如下计算公式（$\tau$表示温度系数，类似于知识蒸馏里的温度系数，也是用来控制分布的平滑度）

$$
\begin{array} \hat{w}_1&=\frac{e^{\hat{w}_1/\tau}}{\sum_{i=1}^3e^{\hat{w}_i/\tau}} \\ &=\frac{e^{({w}_1+\epsilon_1)/\tau}}{\sum_{i=1}^3e^{({w}_i+\epsilon_i)/\tau}} \\ s.t. & \epsilon \thicksim U(0,1) \end{array} \\
$$

我们现在再来看看使用gumbel softmax后的求导表达式：

$$
\begin{array} .y &= \hat{w}_1o_1(x)+\hat{w}_2o_2(x)+\hat{w}_3o_3(x)\\ &\Rightarrow \frac{\partial y}{\partial w_1}=\frac{\partial y}{\partial \hat{w}_1}\frac{\partial \hat{w}_1}{\partial w_1}, ... \end{array} \\
$$

所以**gumbel softmax**成功地引入了随机性，使得每个操作都能以一定的概率被选中，不过貌似也并没有减少内存的消耗，因为还是和DARTS一样计算的mixed值。所以在[GDAS](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1910.04465)这篇论文里作者在选择操作的时候使用的是argmax，而在更新权重的时候采用的是softmax的梯度值。

pytorch中实现的方式如下(感谢评论区补充） :

ret = y\_hard - y\_soft.detach() + y\_soft

总结起来Gumbel-softmax在具体实践上和上面的例子有一丢丢不一样，总结起来步骤如下：

* 对于网络输出的一个n维向量$v$,生成n个服从均匀分布$U(0,1)$的独立样本$\epsilon_1,...,\epsilon_n$
* 通过$G_i=−log(−log(\epsilon_i))$计算得到$G_i$
* 对应相加得到新的值向量![v′=[v_1+G_1,v_2+G_2,...,v_n+G_n]](https://www.zhihu.com/equation?tex=v%E2%80%B2%3D%5Bv_1%2BG_1%2Cv_2%2BG_2%2C...%2Cv_n%2BG_n%5D)
* 计算softmax函数

$$
\sigma_{\tau}\left(v_{i}^{\prime}\right)=\frac{e^{v_{i}^{\prime} / \tau}}{\sum_{j=1}^{n} e^{v_{j}^{\prime} / \tau}} \\
$$

## 代码实现

* pytorch版本

其实pytorch已经内置了gumbel softmax

```python
import torch.nn.functional as F
F.gumbel_softmax(logits, tau, hard, eps, dim=-1)
```

* tensorflow版本

参考：[https://github.com/Baichenjia/Gumbel-softmax/blob/master/vae\_gambel.py](https://link.zhihu.com/?target=https%3A//github.com/Baichenjia/Gumbel-softmax/blob/master/vae_gambel.py)

```python
def sample_gumbel(shape, eps=1e-20): 
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    # logits: [batch_size, n_class] unnormalized log-probs
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)   # 每行之和为1

def gumbel_softmax(logits, temperature, hard=False):
    """
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    """
    # 返回值y.shape=(batchsize, n_class), 每行之和为1，每个数代表概率
    y = gumbel_softmax_sample(logits, temperature)
    if hard: 
        # 将 y 转成one-hot向量，每一行最大值处为1，其余地方为0
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y       # y_hard = y
    return y
```

## **参考：**

为什么gumbel-softmax技巧有效的证明可以参考如下文章

* [Gumbel-Softmax Trick和Gumbel分布](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/initial-h/p/9468974.html)
* [漫谈重参数：从正态分布到Gumbel Softmax](https://link.zhihu.com/?target=https%3A//kexue.fm/archives/6705)
* [automl a survey of the state-of-the-art](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1908.00709)

### **微信公众号：AutoML机器学习**

![](/assets/img/marsggbo/2020-07-03-Gumbel-softmax在可微NAS的作用是什么/029a9211.jpeg)

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2020-07-02 21:18:36**