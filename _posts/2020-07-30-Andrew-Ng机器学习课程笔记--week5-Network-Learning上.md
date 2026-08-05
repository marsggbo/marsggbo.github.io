---
layout: post
title: Andrew Ng机器学习课程笔记--week5 Network: Learning(上)
date: '2020-07-30'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/165328360
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/165328360>

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

* ***Application of Neural Networks***

+ Autonomous Driving

## **二、重点&难点**

## **1.Cost Function and Backpropagation**

### **1） Cost Function**

首先定义一下后面会提到的变量

> **L**: 神经网络总层数  
>  $S_l$ ：l层单元个数（不包括bias unit）  
>  **k**:输出层个数

回顾正则化逻辑回归中的损失函数：

![J(\theta) = - \frac{1}{m} \sum_{i=1}^m [ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2 \\](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29+%3D+-+%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Em+%5B+y%5E%7B%28i%29%7D%5C+%5Clog+%28h_%5Ctheta+%28x%5E%7B%28i%29%7D%29%29+%2B+%281+-+y%5E%7B%28i%29%7D%29%5C+%5Clog+%281+-+h_%5Ctheta%28x%5E%7B%28i%29%7D%29%29%5D+%2B+%5Cfrac%7B%5Clambda%7D%7B2m%7D%5Csum_%7Bj%3D1%7D%5En+%5Ctheta_j%5E2+%5C%5C)

在神经网络中损失函数略微复杂了些,但是也比较好理解，就是把所有层都算进去了。

![\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*} \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Bgather%2A%7D+J%28%5CTheta%29+%3D+-+%5Cfrac%7B1%7D%7Bm%7D+%5Csum_%7Bi%3D1%7D%5Em+%5Csum_%7Bk%3D1%7D%5EK+%5Cleft%5By%5E%7B%28i%29%7D_k+%5Clog+%28%28h_%5CTheta+%28x%5E%7B%28i%29%7D%29%29_k%29+%2B+%281+-+y%5E%7B%28i%29%7D_k%29%5Clog+%281+-+%28h_%5CTheta%28x%5E%7B%28i%29%7D%29%29_k%29%5Cright%5D+%2B+%5Cfrac%7B%5Clambda%7D%7B2m%7D%5Csum_%7Bl%3D1%7D%5E%7BL-1%7D+%5Csum_%7Bi%3D1%7D%5E%7Bs_l%7D+%5Csum_%7Bj%3D1%7D%5E%7Bs_%7Bl%2B1%7D%7D+%28+%5CTheta_%7Bj%2Ci%7D%5E%7B%28l%29%7D%29%5E2%5Cend%7Bgather%2A%7D+%5C%5C)

### **2）BackPropagation反向传播**

> 更详细的公式推导可以参考**[http://ufldl.stanford.edu--反向传导算法](https://link.zhihu.com/?target=https%3A//ask.qcloudimg.com/http-save/yehe-1215004/f7eqbi213j.png%3FimageView2/2/w/1620)**

BP算法示意图如下：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning上/fd4a60cc.jpg)

假设**神经网络结构**如下

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning上/594b078b.jpg)

### **- 1. FP**

1. 利用前向传导公式(FP)计算$2,3……$ 直到 ${n_l}$层（**输出层**）的激活值。 计算过程如下：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning上/1c76dd39.jpg)

### **- 2. BP**

* **权值更新**

首先需要知道的是BP算法是干嘛的？它是用来让神经网络自动更新权重$W$的。 这里权重$W$与之前线性回归权值更新形式上是一样：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning上/a8e46e47.jpg)

那现在要做的工作就是求出后面的偏导，在求之前进一步变形：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning上/ae30cd37.jpg)
> 注意$J(W,b;x^{(i)},y^{(i)})$表示的是单个样例的代价函数，而$J(W,b)$表示的是整体的代价函数。

所以接下来的工作就是求出$\frac{∂J(W,b;x,y)}{∂W_{ij^{(l)}}}$，求解这个需要用到微积分中的链式法则，即

$$
\begin{align*} \frac{∂J(W,b;x,y)}{∂W_{ij^{(l)}}} = \frac{∂J(W,b;x,y)}{∂a_{i^{(l)}}} \frac{∂a_{i^{(l)}}}{∂z_{i^{(l)}}} \frac{∂z_{i^{(l)}}}{∂w_{ij^{(l)}}}  = a_j^{(l)}δ_i^{(l+1)} \end{align*} \\
$$

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning上/92d6181c.jpg)![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning上/ca812c42.jpg)![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning上/e6240a39.jpg)
> 更加详细运算过程可以参考**[一文弄懂神经网络中的反向传播法——BackPropagation](https://link.zhihu.com/?target=http%3A//www.cnblogs.com/charlotte77/p/5629865.html)**,这篇文章详细的介绍了BP算法的每一步骤。

上面的公式中出现了$δ$（误差error），所以后续的目的就是求出每层每个node的$δ$，具体过程如下：

* **计算δ**

对于第 $n_l$层（输出层）的每个输出单元$i$，我们根据以下公式计算残差：

$$
\delta_{i}^{(l)}=\left(\sum_{j=1}^{s_{l+1}} W_{j i}^{(l)} \delta_{j}^{(l+1)}\right) f^{\prime}\left(z_{i}^{(l)}\right)
$$

对 $l = n_l-1, n_l-2, ……,3,2$的各个层，第 $l$ 层的第 $i$ 个节点的残差计算方法如下：

$$
\begin{aligned} \delta_{i}^{\left(n_{l}-1\right)} &=\frac{\partial}{\partial z_{i}^{n_{l}-1}} J(W, b ; x, y)=\frac{\partial}{\partial z_{i}^{n_{l}-1}} \frac{1}{2}\left\|y-h_{W, b}(x)\right\|^{2}=\frac{\partial}{\partial z_{i}^{n_{l}-1}} \frac{1}{2} \sum_{j=1}^{\infty_{l}}\left(y_{j}-a_{j}^{\left(n_{l}\right)}\right)^{2} \\ &=\frac{1}{2} \sum_{j=1}^{S_{n_{l}}} \frac{\partial}{\partial z_{i}^{n_{l}-1}}\left(y_{j}-a_{j}^{\left(n_{l}\right)}\right)^{2}=\frac{1}{2} \sum_{j=1}^{S_{n_{l}}} \frac{\partial}{\partial z_{i}^{n_{l}-1}}\left(y_{j}-f\left(z_{j}^{\left(n_{l}\right)}\right)\right)^{2} \\ &=\sum_{j=1}^{S_{n_{l}}}-\left(y_{j}-f\left(z_{j}^{\left(n_{l}\right)}\right)\right) \cdot \frac{\partial}{\partial z_{i}^{\left(n_{l}-1\right)}} f\left(z_{j}^{\left(n_{l}\right)}\right)=\sum_{j=1}^{S_{n_{l}}}-\left(y_{j}-f\left(z_{j}^{\left(n_{l}\right)}\right)\right) \cdot f^{\prime}\left(z_{j}^{\left(n_{l}\right)}\right) \cdot \frac{\partial z_{j}^{\left(n_{l}\right)}}{\partial z_{i}^{\left(n_{l}-1\right)}} \\ &=\sum_{j=1}^{S_{n_{l}}} \delta_{j}^{\left(n_{l}\right)} \cdot \frac{\partial z_{j}^{\left(n_{l}\right)}}{\partial z_{i}^{n_{l}-1}}=\sum_{j=1}^{S_{n_{l}}}\left(\delta_{j}^{\left(n_{l}\right)} \cdot \frac{\partial}{\partial z_{i}^{n_{l}-1}} \sum_{k=1}^{S_{n_{l}-1}} f\left(z_{k}^{n_{l}-1}\right) \cdot W_{j k}^{n_{l}-1}\right) \\ &=\sum_{j=1}^{S_{n_{l}}} \delta_{j}^{\left(n_{l}\right)} \cdot W_{j i}^{n_{l}-1} \cdot f^{\prime}\left(z_{i}^{n_{l}-1}\right)=\left(\sum_{j=1}^{S_{n_{l}}} W_{j i}^{n_{l}-1} \delta_{j}^{\left(n_{l}\right)}\right) f^{\prime}\left(z_{i}^{n_{l}-1}\right) \end{aligned}
$$

将上式中的 $n_l-1$ 与 $n_l$ 的关系替换为 $l与l+1$ 的关系即可得到前面的残差。

将上面的结果带入**权值更新**的表达式中便可顺利的执行BackPropagation啦~~~

> 但是！！！需要注意的是上面式子中反复出现的 $f '(z_i^{(l)})$ ，表示激活函数的导数。这个在刚开始的确困惑到我了，因为视频里老师在演示计算$δ$的时候根本就乘以这一项，难道老师错了？其实不是的，解释如下： 常用的激活函数有好几种，但使用是分情况的：

* 在**线性**情况下：f(z) = z
* 在**非线性**情况下：(只举一些我知道的例子)

+ sigmoid
+ tanh
+ relu

所以这就是为什么老师在视频中没有乘以 $f '(z_i^{(l)})$ 的原因了，就是因为是线性的，求导后为1，直接省略了。

另外sigmoid函数表达式为$f(z)=\frac{1}{1+e^{-z}}$,很容易知道$f'(z)=\frac{-e^{-z}}{  (1+e^{-z}) ^2  }   = f(z)·(1-f(z))$这也就解释了Coursera网站上讲义的公式是这样的了：

$$
\text { 4. Compute } \delta^{(L-1)}, \delta^{(L-2)}, \ldots, \delta^{(2)} \text { using } \delta^{(l)}=\left(\left(\Theta^{(l)}\right)^{T} \delta^{(l+1)}\right) \text { . * } a^{(l)} . *\left(1-a^{(l)}\right)
$$

所以现在总结一下**BP算法步骤：**

1. 进行前馈传导计算，利用前向传导公式，得到$L_2, L_3, \ldots$直到输出层 $\textstyle L_{n_l}$的激活值。
2. 对输出层（第 $\textstyle n_l$层），计算： $\delta^{(n_l)}= - (y - a^{(n_l)}) \bullet f'(z^{(n_l)})$
3. 对于 $\textstyle l = n_l-1, n_l-2, n_l-3, \ldots, 2$ 的各层，计算： $\delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)}\right) \bullet f'(z^{(l)})$
4. 计算最终需要的偏导数值：

$$
\begin{align} \nabla_{W^{(l)}} J(W,b;x,y) &= \delta^{(l+1)} (a^{(l)})^T, \\ \nabla_{b^{(l)}} J(W,b;x,y) &= \delta^{(l+1)}. \end{align} \\
$$

使用**批量梯度下降一次迭**代过程：

1. 对于所有$\textstyle l$，令 $\textstyle \Delta W^{(l)} := 0 ,  \textstyle \Delta b^{(l)} := 0$ （设置为全零矩阵或全零向量）
2. 对于$\textstyle i = 1$ 到$\textstyle m$ ， 使用反向传播算法计算$\textstyle \nabla_{W^{(l)}} J(W,b;x,y)$ 和$\textstyle \nabla_{b^{(l)}} J(W,b;x,y)$ 。 计算$\textstyle \Delta W^{(l)} := \Delta W^{(l)} + \nabla_{W^{(l)}} J(W,b;x,y)$ 。 计算$\textstyle \Delta b^{(l)} := \Delta b^{(l)} + \nabla_{b^{(l)}} J(W,b;x,y)$ 。
3. 更新权重参数：

![ \begin{align} W^{(l)} &= W^{(l)} - \alpha \left[ \left(\frac{1}{m} \Delta W^{(l)} \right) + \lambda W^{(l)}\right] \\ b^{(l)} &= b^{(l)} - \alpha \left[\frac{1}{m} \Delta b^{(l)}\right] \end{align} \\](https://www.zhihu.com/equation?tex=+%5Cbegin%7Balign%7D+W%5E%7B%28l%29%7D+%26%3D+W%5E%7B%28l%29%7D+-+%5Calpha+%5Cleft%5B+%5Cleft%28%5Cfrac%7B1%7D%7Bm%7D+%5CDelta+W%5E%7B%28l%29%7D+%5Cright%29+%2B+%5Clambda+W%5E%7B%28l%29%7D%5Cright%5D+%5C%5C+b%5E%7B%28l%29%7D+%26%3D+b%5E%7B%28l%29%7D+-+%5Calpha+%5Cleft%5B%5Cfrac%7B1%7D%7Bm%7D+%5CDelta+b%5E%7B%28l%29%7D%5Cright%5D+%5Cend%7Balign%7D+%5C%5C)

### **3) Backpropagation Intuition**

本小节演示了具体如何操作BP，不再赘述。

> 具体可参考**[Coursera讲义](https://link.zhihu.com/?target=https%3A//www.coursera.org/learn/machine-learning/supplement/v5Bu8/backpropagation-intuition)**。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning上/dd9319a2.jpg)

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week5-Network-Learning上/f2e7d76c.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**