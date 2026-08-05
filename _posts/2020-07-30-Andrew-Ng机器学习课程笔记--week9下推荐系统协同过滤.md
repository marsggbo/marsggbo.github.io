---
layout: post
title: Andrew Ng机器学习课程笔记--week9(下)（推荐系统&协同过滤）
date: '2020-07-30'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/165343529
related_posts: false
toc:
  sidebar: left
---

本周内容较多，故分为上下两篇文章。 本文为下篇。

## **一、内容概要**

## **1. Anomaly Detection**

* **Density Estimation**

+ Problem Motivation
+ Gaussian Distribution
+ Algorithm

* **Building an Anomaly Detection System（创建异常检测系统）**

+ Developing and Evaluating an Anomaly Detection System
+ Anomaly Detection vs. Supervised Learning
+ Choosing What Features to Use

* **Multivariate Gaussion Distribution（多元高斯分布）**

+ Multivariate Gaussion Distribution
+ Anomaly Detection using the Multivariate Gaussion Distribution

## **2. Recommender System**

* **Predicting Movie**

+ Problem Formulation
+ Content Based Recommendations

* **Collaborative Filtering(协同过滤)**

+ Collaborative Filtering
+ Collaborative Filtering Algorithm

* **Low Rank Matrix Factorization（低秩矩阵分解）**

+ Vectorization（向量化）： Low Rank Matrix Factorization
+ Implementational Detail：Mean Normalization

## **二、重点&难点**

## **Recommender System(推荐系统)**

### **1.Predicting Movie**

### **1）Problem Formulation**

下面将以推荐电影为例来介绍推荐系统的实现。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9下推荐系统协同过滤/b7d138af.jpg)

上面的分数表示用户对该电影的评分（0~5分，？表示未获得评分数据） 为方便下面叙述，对如下符号进行说明：

* $n_u$：表示用户数量
* $n_m$：表示电影数量
* r(i,j)：如果等于1则表示用户j对电影i进行了评分
* $y^{(i,j)}$：表示用户j对电影i的评分

上面例子中可以知道 $n_u=4 \quad n_m=5 \quad y^{(1,1)}=5$

### **2）Content Based Recommendations(基于内容的推荐)**

* **1.获取特征向量** 为了实现推荐，我们为每部电影提取出了两个特征值，即x1（浪漫指数）和x2（动作指数）

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9下推荐系统协同过滤/d90ef5f7.jpg)

由上表可知每部电影都可以用一组特征向量表示：

* 每一步电影都加上一个额外的特征，即 $x_0=1$
* 每部电影都有一个(3,1)的特征向量，例如第一部电影（Love at last）：$x^{(1)}=[1,0.9,0.1]^T$
* 对于所有数据我们有数据特征向量组为$\{x^{(1)},x^{(2)},x^{(3)},x^{(4)},x^{(5)}\}$
* **2.特征权重θ** 用户j对电影i的评分预测可以表示为$(θ^j)^Tx^i=stars$
* **3. 线性回归预测**

和线性回归一样，可以得到如下优化目标函数:

* 对单个用户而言

$$\min_{θ^{(j)}\frac{1}{2}\sum_{i;r(i,j)=1}((θ^{(j)})^Tx^{(i)}-y^{(i,j)})^2 + \frac{λ}{2}\sum_{k=1}^n (θ_k^{(j)})^2$$

* 对所有用户而言

$$\min_{θ^{(1)},...,θ^{(n_u)}\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}((θ^{(j)})^Tx^{(i)}-y^{(i,j)})^2 + \frac{λ}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n (θ_k^{(j)})^2$$

应用梯度下降：

$当k=0，θ_k^{(j)}:=θ_k^{(j)}-α\sum_{i:r(i,j)=1}( (θ^{(j)})^Tx^{(i)}-y^{(i,j)} )x_k^{(i)}$$当k≠0，θ_k^{(j)}:=θ_k^{(j)}-α\sum_{i:r(i,j)=1}( (θ^{(j)})^Tx^{(i)}-y^{(i,j)} )x_k^{(i)}+λθ_k^{(j)}$

### **2.Collaborative Filtering(协同过滤)**

### **1）Collaborative Filtering**

在之前的基于内容的推荐系统中，对于每一部电影，我们都掌握了可用的特征，使用这些特征训练出了每一个用户的参数。相反地，如果我们拥有用户的参数，我们可以学习得出电影的特征。即由θ求出x。

$$\min_{θ^{(1)},...,θ^{(n_m)}\frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}((θ^{(j)})^Tx^{(i)}-y^{(i,j)})^2 + \frac{λ}{2}\sum_{j=1}^{n_m}\sum_{k=1}^n (θ_k^{(j)})^2$$

> 注意累计符号的上限由$n_u$变成了$n_m$

但是如果我们既没有用户的参数也没有电影的特征该怎么办？这时协同过滤就可以起作用了，只需要对优化目标函数进行改进，如下：

$$J(x^{(1)},...,x^{(n_m)},θ^{(1)},...,θ^{(n_u)}) = \frac{1}{2}\sum_{(i,j):r(i,j)=1}((θ^{(j)})^Tx^{(i)}-y^{(i,j)})^2 \\ \quad\quad\quad\quad\quad\quad\quad +\frac{λ}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n (θ_k^{(j)})^2 \\ \quad\quad\quad\quad\quad\quad\quad+ \frac{λ}{2}\sum_{i=1}^{n_m}\sum_{k=1}^n (x_k^{(i)})^2$$

对代价函数求偏导结果如下：

$x_k^{(i)} := x_k^{(i)} - α(\sum_{j:r(i,j)=1}(   (θ^{(j)})^Tx^{(i)}-y^{(i,j)} )θ_k^{(j)} +λx_k^{(i)}   )$$θ_k^{(j)} := θ_k^{(j)} - α(\sum_{i:r(i,j)=1}(   (θ^{(j)})^Tx^{(i)}-y^{(i,j)} )x_k^{(i)} +λθ_k^{(j)}   )$

协同过滤算法使用步骤如下：

1. 初始 x (1) ,x (2) ,...,x ($n_m$) ，θ (1) ,θ (2) ,...,θ ($n_u$) 为一些随机小值
2. 使用梯度下降算法最小化代价函数
3. 在训练完算法后，我们预测$(θ ^{(j)} )^ T x^{ (i)}$ 为用户 j 给电影 i 的评分

### **3. Low Rank Matrix Factorization（低秩矩阵分解）**

### **1）Vectorization（向量化）： Low Rank Matrix Factorizationv**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9下推荐系统协同过滤/90bb1edf.jpg)

（同样的例子）很显然我们可以得到评分矩阵Y

$$Y= \left[     \begin{array}{cccc}       5&5&0&0 \\       5&?&?&0 \\       ?&4&0&? \\       0&0&5&4 \\       0&0&5&0 \\     \end{array} \right]$$

推出评分

$$\begin{pmatrix}      (θ^{(1)})^T(x^{(1)}) &(θ^{(2)})^T(x^{(1)})& \cdots & (θ^{(n_u)})^T(x^{(1)}) \\      (θ^{(1)})^T(x^{(2)}) &(θ^{(2)})^T(x^{(2)})& \cdots & (θ^{(n_u)})^T(x^{(2)}) \\      \vdots  & \vdots& \ddots & \vdots \\      (θ^{(1)})^T(x^{(n_m)}) &(θ^{(2)})^T(x^{(n_m)})& \cdots & (θ^{(n_u)})^T(x^{(n_m)}) \\      \end{pmatrix}$$

如何寻找与电影i相关的电影j呢？满足$||x^{(i)}-x^{(j)}||$较小的前几部影片即可。

### **2）Implementational Detail：Mean Normalization**

假如增加了一个用户**marsggbo**，他很单纯，这5部电影都还没看过，所以没有评分数据，这是可以通过均值正则化来初始化数据，具体实现如下：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9下推荐系统协同过滤/05b4cf5d.jpg)

此时的评分矩阵为

$$Y= \left[     \begin{array}{cccc}       5&5&0&0&? \\       5&?&?&0&? \\       ?&4&0&?&? \\       0&0&5&4&? \\       0&0&5&0&? \\     \end{array} \right]$$

首先求出每行的均值（未评分不用计算）

$$\mu=\left[\begin{array}{l} .5 \\ 2.5 \\ 2 \\ 2.25 \\ 1.25 \end{array}\right] \rightarrow Y=\left[\begin{array}{ccccc} 2.5 & 2.5 & -2.5 & -2.5 & ? \\ 2.5 & ? & ? & -2.5 & ? \\ ? & 2 & -2 & ? & ? \\ -2.25 & -2.25 & 2.75 & 1.75 & ? \\ -1.25 & -1.25 & 3.75 & -1.25 & ? \end{array}\right]$$

预测值为$(θ^{(j)})^T(x^{(i)})+μ_i$，因为优没有评分。所以化目的函数只需要$min\frac{λ}{2}\sum_{j=1}^{n_u}\sum_{k=1}^n (θ_k^{(j)})^2$,很显然$θ=\vec0$，所以新增用户评分数据可初始化为均值，即

$$Y= \left[     \begin{array}{cccc}       5&5&0&0&2.5 \\       5&?&?&0&2.5 \\       ?&4&0&?&2 \\       0&0&5&4&2.25  \\       0&0&5&0&1.25  \\     \end{array} \right]$$

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9下推荐系统协同过滤/f2e7d76c.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**
