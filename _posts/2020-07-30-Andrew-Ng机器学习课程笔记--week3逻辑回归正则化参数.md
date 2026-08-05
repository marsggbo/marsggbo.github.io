---
layout: post
title: Andrew Ng机器学习课程笔记--week3（逻辑回归&正则化参数）
date: '2020-07-30'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/165324105
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/165324105>

Logistic Regression

## **一、内容概要**

* **Classification and Representation**

+ Classification
+ Hypothesis Representation
+ Decision Boundary

  
* **Logistic Regression Model**

+ 损失函数（cost function）
+ 简化损失函数和梯度下降算法
+ Advanced Optimization（高级优化方法）

* **Solving the problem of Overfitting**

+ 什么是过拟合？
+ 正则化损失函数（cost function）
+ 正则化线性回归（Regularized Linear Regression）
+ 正则化逻辑回归（Regularized Logistic Regression）

## **二、重点&难点**

## **1. Classification and Representation**

### **1） Hypothesis Representation**

这里需要使用到**sigmoid函数--g(z)**：

$\begin{equation} h_θ(x) = g(θ^Tx)  \end{equation} \\$$\begin{equation} z = θ^Tx  \end{equation} \\$$\begin{equation} g(z) = \frac{1}{1+e^{-z}} \end{equation} \\$

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week3逻辑回归正则化参数/f88c53a4.jpg)

### **2) Decision Boundary**

决策边界：

$h_θ(x) ≥ 0.5  → y=1  \\$$h_θ(x) < 0.5 →  y=0  \\$

等价于

$g(z) ≥ 0.5  →  y=1  \\$$g(z) < 0.5 →  y=0  \\$

等价于

$z ≥0 →  y=1  \\$$z < 0 →  y=0  \\$

## **2. Logistic Regression Model**

### **1） 逻辑回归的损失函数**

这里之所以再次提到损失函数，是因为线性回归中的损失函数会使得输出呈现起伏，造成许多局部最优值，也就是说**线性回归中的cost function在运用到逻辑回归时，将可能不再是凸函数。**

逻辑回归的cost function如下：

$J_θ = \frac{1}{m} \sum {Cost}( h_θ(x^{(i)}, y^{(i)} ) ) \\$${Cost}(h_θ(x), y) ) = - log(h_θ(x))   \quad  \quad if \quad y=1 \\$${Cost}(h_θ(x), y) ) = - log(1 - h_θ(x))   \quad  if \quad y=0 \\$

结合图来理解：

* **y=1**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week3逻辑回归正则化参数/71005d88.jpg)

由上图可知，y=1，hθ(x)是预测值， - 当其值为1时，表示预测正确，损失函数为0； - 当其值为0时，表示错的一塌糊涂，需要大大的惩罚，所以损失函数趋近于∞。

* **y=0**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week3逻辑回归正则化参数/58d4b4fd.jpg)

上图同理

## **2) Simplified Cost Function and Gradient Descent**

* 损失函数 cost function

$$
Cost(h_θ(x), y) = -ylog(h_θ(x)) - (1-y)log(1-h_θ(x)) \\
$$

Jθ

$J_θ=-\frac{1}{m} \sum Cost(h_θ(x), y)  \\$![\quad  =-\frac{1}{m} \sum  [-y^{i}log(h_θ(x^{(i)})) - (1-y^i)log(1-h_θ(x^{(i)}))]  \\](https://www.zhihu.com/equation?tex=%5Cquad++%3D-%5Cfrac%7B1%7D%7Bm%7D+%5Csum++%5B-y%5E%7Bi%7Dlog%28h_%CE%B8%28x%5E%7B%28i%29%7D%29%29+-+%281-y%5Ei%29log%281-h_%CE%B8%28x%5E%7B%28i%29%7D%29%29%5D++%5C%5C)

* 梯度函数

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week3逻辑回归正则化参数/c44ab5de.jpg)

## **3）高级优化方法**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week3逻辑回归正则化参数/f6e4ca4b.jpg)

如图左边显示的是优化方法，其中后三种是更加高级的算法，其优缺点由图邮编所示： **优点**

* 不需要手动选择α
* 比梯度下降更快

**缺点**

* 更加复杂

后面三种方法只需了解即可，老师建议如果你不是专业的数学专家，没必要自己使用这些方法。。。。。。当然了解一下原理也是好的。

## **3. Solving the problem of Overfitting**

### **1) 过拟合**

主要说一下过拟合的解决办法： 1）减少特征数量

* 手动选择一些需要保留的特征
* 使用模型选择算法（model selection algorithm） 2）正则化
* 保留所有特征，但是参数θ的数量级（大小）要减小
* 当我们有很多特征，而且这些特征对于预测多多少少会由影响，此时正则化怎能起到很大的作用。

### **2） 正则化损失函数**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week3逻辑回归正则化参数/ebe865fb.jpg)

图示右边很明显是过拟合，因此为了纠正加入了正则化项：1000·θ32，为了使得J(θ)最小化，所以算法会使得θ3趋近于0，θ4也趋近于0。

**正则化损失函数表达式：**

![J(θ)=\frac{1}{2m} [\sum_{i=1}^m( h_θ(x^{(i)}) - y^{(i)})^2 + λ\sum_{j=1}^n θ_j^2] \\](https://www.zhihu.com/equation?tex=J%28%CE%B8%29%3D%5Cfrac%7B1%7D%7B2m%7D+%5B%5Csum_%7Bi%3D1%7D%5Em%28+h_%CE%B8%28x%5E%7B%28i%29%7D%29+-+y%5E%7B%28i%29%7D%29%5E2+%2B+%CE%BB%5Csum_%7Bj%3D1%7D%5En+%CE%B8_j%5E2%5D+%5C%5C)![min_θ [\frac{1}{2m} (\sum_{i=1}^m( h_θ(x^{(i)}) - y^{(i)})^2 + λ\sum_{j=1}^n θ_j^2)] \\](https://www.zhihu.com/equation?tex=min_%CE%B8+%5B%5Cfrac%7B1%7D%7B2m%7D+%28%5Csum_%7Bi%3D1%7D%5Em%28+h_%CE%B8%28x%5E%7B%28i%29%7D%29+-+y%5E%7B%28i%29%7D%29%5E2+%2B+%CE%BB%5Csum_%7Bj%3D1%7D%5En+%CE%B8_j%5E2%29%5D+%5C%5C)

### **3) 正则化线性回归**

* **正则化梯度下降：**

![J(θ)=\frac{1}{2m} [\sum_{i=1}^m( h_θ(x^{(i)}) - y^{(i)})^2 + λ\sum_{j=1}^n θ_j^2] \\](https://www.zhihu.com/equation?tex=J%28%CE%B8%29%3D%5Cfrac%7B1%7D%7B2m%7D+%5B%5Csum_%7Bi%3D1%7D%5Em%28+h_%CE%B8%28x%5E%7B%28i%29%7D%29+-+y%5E%7B%28i%29%7D%29%5E2+%2B+%CE%BB%5Csum_%7Bj%3D1%7D%5En+%CE%B8_j%5E2%5D+%5C%5C)$\frac{∂J_θ}{∂θ_j} = \frac{1}{m} \sum_{i=1}^m( h_θ(x^{(i)} )  - y^{(i)} )x_j^{(i)} + \frac{λ}{m}θ_j   \\$

Repeat{

$θ_0 := θ_0  - α\frac{1}{m}\sum_{i=1}{m}( h_θ(x^{(i)} )  - y^{(i)} )x_0^{(i)} \\$![θ_j := θ_j  - α[(\frac{1}{m}\sum_{i=1}{m}( h_θ(x^{(i)} )  - y^{(i)} )x_0^{(i)} ) + \frac{λ}{m}θ_j   ] \quad j∈\{1,2,3……n\} \\](https://www.zhihu.com/equation?tex=%CE%B8_j+%3A%3D+%CE%B8_j++-+%CE%B1%5B%28%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%7Bm%7D%28+h_%CE%B8%28x%5E%7B%28i%29%7D+%29++-+y%5E%7B%28i%29%7D+%29x_0%5E%7B%28i%29%7D+%29+%2B+%5Cfrac%7B%CE%BB%7D%7Bm%7D%CE%B8_j+++%5D+%5Cquad+j%E2%88%88%5C%7B1%2C2%2C3%E2%80%A6%E2%80%A6n%5C%7D+%5C%5C)

}

* **正则化正规方程**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week3逻辑回归正则化参数/31a60440.jpg)

前面提到过，若m< n,那么XTX是不可逆的，但是加上λ·L后则变为可逆的了。

## **4) 正则化逻辑回归**

![J(θ)=-\frac{1}{m} \{\sum_{i=1}^m[  y^{(i)} log(h_θ(x^{(i)}))+(1-y^{(i)})log(1-h_θ(x^{(i)}))]\} + \frac{λ}{2m}\sum_{j=1}^n θ_j^2 \\](https://www.zhihu.com/equation?tex=J%28%CE%B8%29%3D-%5Cfrac%7B1%7D%7Bm%7D+%5C%7B%5Csum_%7Bi%3D1%7D%5Em%5B++y%5E%7B%28i%29%7D+log%28h_%CE%B8%28x%5E%7B%28i%29%7D%29%29%2B%281-y%5E%7B%28i%29%7D%29log%281-h_%CE%B8%28x%5E%7B%28i%29%7D%29%29%5D%5C%7D+%2B+%5Cfrac%7B%CE%BB%7D%7B2m%7D%5Csum_%7Bj%3D1%7D%5En+%CE%B8_j%5E2+%5C%5C)

梯度下降过程

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week3逻辑回归正则化参数/639d56cd.jpg)

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week3逻辑回归正则化参数/f2e7d76c.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**