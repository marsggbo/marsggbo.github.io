---
layout: post
title: Andrew Ng机器学习课程笔记--week7(SVM)
date: '2020-07-30'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/165335537
related_posts: false
toc:
  sidebar: left
---

本周主要学习**SVM**

## **一、 内容概要**

* **Large Margin Classification**

+ Optimization Objective（优化Objective（损失函数））
+ Large Margin Intuition（大边距的直观理解）
+ Mathematics Behind Large Magin Classification（最大间距分类器背后的数学推导）

* **Kernels**

+ Kernels 1
+ Kernels 2

* **SVMs in Practice**

+ Using An SVM

## **二、重点&难点**

## **1. Large Margin Classification**

### **1) Optimization Objective（优化Objective（损失函数））**

* **回顾一下逻辑回归模型**

$h_θ(x) = \frac{1}{1+e^{-θ^Tx}$$J(θ)=\frac{1}{m} [\sum_{i=1}^{m} y^{(i)}( -logh_θ(x^{(i)})) + (1-y^{(i)})(-log(1-h_θ(x^{(i)})  )  ] + \frac{λ}{2m}\sum_{j=1}^{n}θ_j^2$

在SVM中对cost function作如下等效变化（即将log函数替换成折线）

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/1100bc69.jpg)

* **折线化变形** 替换后cost function变为

$$J(θ)=\frac{1}{m} [\sum_{i=1}^{m} y^{(i)}Cost_1(θ^Tx^{(i)}) + (1-y^{(i)})Cost_0(θ^Tx^{(i)})  ] + \frac{λ}{2m}\sum_{j=1}^{n}θ_j^2$$

> Cost的下标分别表示y所对应的值。

* **去m变形** 另外我们知道为了得到最优化的一组θ，我们需要通过求$min J(θ)$进而得出一组解，所以上式中的m可以约掉，因为m是常数，所以对于求最小值没有影响，所以cost function可以进一步变形为：

$$J(θ)= [\sum_{i=1}^{m} y^{(i)}Cost_1(θ^Tx^{(i)}) + (1-y^{(i)})Cost_0(θ^Tx^{(i)})  ] + \frac{λ}{2}\sum_{j=1}^{n}θ_j^2$$

* **乘以C变形** 继续变形前我们可以假设上式左边为训练数据集项(Training data set term),记为A，右侧为正则项，记为λB，所以有$J(θ) = A+λB$。 按照上面所说，此时在等式两侧乘以一个数不会影响最终的结果，假设乘以一个C（$C=\frac{1}{λ}$），所以有$J(θ)=CA+B$ 此时有

$$J(θ)= C[\sum_{i=1}^{m} y^{(i)}Cost_1(θ^Tx^{(i)}) + (1-y^{(i)})Cost_0(θ^Tx^{(i)})  ] + \frac{1}{2}\sum_{j=1}^{n}θ_j^2$$

### **2) Large Margin Intuition（大边距的直观理解）**

上面将普通逻辑回归中的log函数变形后得到的曲线如下：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/b429e64e.jpg)

区别：

> If y=1, we want $θ^Tx≥1$ (not just ≥0) If y=0, we want $θ^Tx≤-1$ (not just ≤0)

和引入正则项同理，当C取非常大的值时，我们希望如下蓝色圈住的部分接近于0，即使得A=0

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/bc339b6d.jpg)

但是要如何使A=0呢？参考上面的折线图，我们可以知道要使得A=0，则需要满足：

* 当y=1,则$θ^Tx≥1$
* 当y=0,则$θ^Tx≤-1$

此时即等价于

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/cc2e9d66.jpg)

### **3) Mathematics Behind Large Magin Classification**

在推导公式之前需要回顾一下向量内积的概念。 已知SVM的优化目标是：

$min\frac{1}{2}\sum_{j=1}^nθ_j^2 且满足$$当y=1时,theta^Tx^{(i)}≥1$$当y=0时,theta^Tx^{(i)}≤-1$

为了方便理解，令$θ_0=0$，特征数n=2，则有

$$min\frac{1}{2}\sum_{j=1}^nθ_j^2  = \frac{1}{2}(θ_1^2+θ_2^2)=\frac{1}{2}\sqrt{(θ_1^2+θ_2^2)}^2=\frac{1}{2}||θ||^2$$

其中，||θ||为向量θ的长度或称为θ的范数。 如果将$θ^Tx(i)$看成是**经过原点(因为θ0=0)** 的两个向量相乘，如下图：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/f794c300.jpg)

则$θ^Tx^{(i)}$等价于向量$x^{(i)}$在向量θ上的投影$p^{(i)}$与θ的范数||θ||相乘，即

$$θ^Tx^{(i)} = p^{(i)}||θ|| = θ_1x_1^{(1)}+θ_2x_1^{(2)}$$

故SVM优化目标变为

$min\frac{1}{2}||θ||^2 且满足$$当y=1时,p^{(i)}||θ||≥1$$当y=0时,p^{(i)}||θ||≤-1$

直观的理解$p^{(i)}||θ||$的意义。 假设theta0=0,下面展示了一个小间距决策边界的例子。（绿色为决策边界）

* **首先解释一下为什么θ向量会垂直于决策边界**。 因为θ的斜率是$\frac{θ_2}{θ_1}$,决策边界表达式为$θ^Tx=0$,即$θ_1x_1+θ_2x_2=0$,斜率为$\frac{θ_2}{θ_1}$，所以θ向量会垂直于决策边界。
* **看第一个例子(x1)** 假如决策边界最开始如下图

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/eae1cd3c.jpg)

将$x^1$投影到θ向量，得到$p^1$，可以看到$p^1$值很小。SVM的优化目标是$min\frac{1}{2}||θ||^2$，但是还需要满足$|p^{(i)}||θ|||≥1$，而又因为$p^1$值很小，所以||θ||值就需要较大才行，显然这与优化目标背道而驰，所以还有优化的空间。

> x2 同理，不再赘述。

* **优化后的例子**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/c602f1ef.jpg)

此时$p^1$值明显增大，||θ||变小，达到优化目的。

## **2. Kernels**

### **1) Kernels 1**

之前课程中已经提到过通过使用多项式来解决非线性拟合问题，如下图所示

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/fa15bdbf.jpg)

* **引入核函数** 在SVM中我们引入核函数来解决这个问题。 假设$h_θ(x) = θ_0+θ_1f_1+θ_2f_2+……$，然后随机人为的选取几个向量$l^{(i)}$作为标记（landmarks），为方便说明选取三个：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/97153603.jpg)

同时定义核函数（核函数很多种，这里使用的是高斯核函数Gaussion Kernels）为

$$f_i = similarity(x^{(i)}, l^{(i)}) = e^{(-\frac{||x^{(i)}-l^{(i)}||^2}{2δ^2})}$$

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/fb6aa21c.jpg)
> 这里的核函数$f_i$可以理解成**相似度**，即点x与标记点l如果很相近则预判为1，反之为0. 由高斯核函数的表达式也可以很好的理解：

$若x^{(i)}≈l^{(i)},则f_i≈1$$若x^{(i)}与l^{(i)}相距较远,则f_i≈0$

另外高斯核函数中有一个参数$δ^2$，它对于结果的影响如下面几个图所示

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/1080061c.jpg)

可知$δ^2$越小，图像越窄，下降的速度也就越快。

* **核函数计算示例**

首先还是假设选取三个landmarks，并且分类的方法是：

$$若θ_0+θ_1f_1+θ_2f_2+……≥0，预测为1，反之为0$$

假设θ向量已知为$θ_0=-0.5,θ_1=1,θ_2=1,θ_3=0$

下面看第一个点的分类情况：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/6875ba43.jpg)

此时$x≈l^{(1)}，故f_1=1$，同理因为远离其余两个landmarks，所以$f_2=0,f_3=0$。 所以带入计算公式有

$$h_θ(x) = θ_0+θ_1f_1+θ_2f_2+θ_3f_3=-0.5+1*1+1*0+0*0=0.5≥0$$

故该点y=1

继续看下图新添的两个x坐标点

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/d38365ca.jpg)

和如上同样的分析后可以知道，绿色的点有y=1，青色的点是y=0

按照上面的计算方法，在计算了大量点后可以得到如下的边界

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/78232cc5.jpg)

### **2) Kernels 2**

* **优化目标** 上面的landmarks都是人工选取的几个点而已，但是真实计算时会计算很多点。另外因为引入了核函数，所以SVM优化目标变为：

$$minJ(θ)=min C[\sum_{i=1}^{m} y^{(i)}Cost_1(θ^Tf^{(i)}) + (1-y^{(i)})Cost_0(θ^Tf^{(i)})  ] + \frac{1}{2}\sum_{j=1}^{n}θ_j^2$$

> 注意原来cost函数中的x变成了f。

另外上式中右边的正则项可以变成$\sum_{j=1}^{n}θ_j^2=θ^Tθ$，还可以继续变形 $\sum_{j=1}^{n}θ_j^2=θ^Tθ=θ^TMθ$，其中矩阵M取决于你所使用的核函数。 需要注意，上述那些SVM的计算技巧应用到别的算法，如逻辑回归中，会变得非常慢，所以一般不将核函数以及标记点等方法用在逻辑回归中。

* **参数影响**

1.**C**

前面提到过的$C=\frac{1}{λ}$,C对bias和variance的影响如下： C太大，相当于λ太小，会产生高方差，低偏差； C太小，相当于λ太大，会产生高偏差，低方差。

2.**$δ^2$**

$δ^2$大，则特征$f_i$变化较缓慢，可能会产生高偏差，低方差； $δ^2$小，则特征$f_i$变化不平滑，可能会产生高方差，低偏差。

## **3. SVMs in Practice**

### **1) Using An SVM**

**SVM和逻辑回归的选择问题** 什么时候该用逻辑回归？什么时候该用SVM？ ①如果**n相对于m来说很大**，则应该使用逻辑回归或者线性核函数（无核）的SVM。 m较小时，使用线性分类器效果就挺不错了，并且也没有足够的数据去拟合出复杂的非线性分类器。 ②如果**n很小，m中等大小**，则应该使用高斯核函数SVM。 ③如果**n很小，m很大**，则高斯核函数的SVM运行会很慢。这时候应该创建更多的特征变量，然后再使用逻辑回归或者线性核函数（无核）的SVM。

对于以上这些情况，神经网络很可能做得很好，但是训练会比较慢。实际上SVM的优化问题是一种凸优化问题，好的SVM优化软件包总是能找到全局最小值或者是接近全局最小的值。

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week7SVM/f2e7d76c.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**
