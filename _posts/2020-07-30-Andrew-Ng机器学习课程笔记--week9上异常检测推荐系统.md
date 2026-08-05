---
layout: post
title: Andrew Ng机器学习课程笔记--week9(上)(异常检测&推荐系统)
date: '2020-07-30'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/165342318
related_posts: false
toc:
  sidebar: left
---

本周内容较多，故分为上下两篇文章。

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

## **二、 重点&难点**

## **Anomaly Detection**

### **1. Density Estimation**

### **1） Problem Motivation**

假设我们生产了若干产品，现在通过两个特征来衡量产品是否合格，下面表示的是合格产品的分布图。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9上异常检测推荐系统/7e611331.jpg)

现在有两个新生产的产品，分布如下（**绿色×**）

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9上异常检测推荐系统/31edeb38.jpg)

上面我们构建的模型（即蓝色同心圆）能根据测试数据告诉我们其属于一组数据的可能性p(x).上图中，在蓝色圈内的数据属于该组数据的可能性较高，而越是偏远的数据，其属于该组数据的可能性就越低。 这种方法称为密度估计，表达如下：

$if \quad p(x)≤ε，则为anomaly（异常）$$if \quad p(x)>ε，则为normal（正常）$

### **2） Gaussian Distribution**

略

### **3） Algorithm**

还是以上面产品检测为例，我们先得到如下图左边的样品特征分布，然后根据分布图分别画出x1和x2的高斯分布图

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9上异常检测推荐系统/5c74e454.jpg)

下面的三维图表表示的是密度估计函数，z 轴为根据两个特征的值所估计 p(x)值：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9上异常检测推荐系统/3e960e7b.jpg)

我们选择一个 ε，将 p(x)=ε 作为我们的判定边界，当 p(x)>ε 时预测数据为正常数据，否则则为异常。

### **2. Building an Anomaly Detection System（创建异常检测系统）**

### **1） Developing and Evaluating an Anomaly Detection System**

异常检测算法是一个非监督学习算法，意味着我们无法根据结果变量 y 的值来告诉我 们数据是否真的是异常的。我们需要另一种方法来帮助检验算法是否有效。当我们开发一个 异常检测系统时，我们从带标记（异常或正常）的数据着手，我们从其中选择一部分正常数 据用于构建训练集，然后用剩下的正常数据和异常数据混合的数据构成交叉检验集和测试 集。

例如：我们有 10000 台正常引擎的数据，有 20 台异常引擎的数据。 我们这样分配数 据： 6000 台正常引擎的数据作为训练集 2000 台正常引擎和 10 台异常引擎的数据作为交叉检验集 2000 台正常引擎和 10 台异常引擎的数据作为测试集

具体的评价方法如下：

1. 根据测试集数据，我们估计特征的平均值和方差并构建 p(x)函数
2. 对交叉检验集，我们尝试使用不同的 ε 值作为阀值，并预测数据是否异常，根据 F 1 值或者查准率与查全率的比例来选择 ε
3. 选出 ε 后，针对测试集进行预测，计算异常检验系统的 F 1 值，或者查准率与查全 率之比

### **2） Anomaly Detection vs. Supervised Learning**

异常检测和监督学习的确有几分相似，但是还是有区别的，整理如下：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9上异常检测推荐系统/12966c53.jpg)

### **3） Choosing What Features to Use**

异常检测假设特征符合高斯分布，如果数据的分布不是高斯分布，异常检测算法也能够 工作，但是最好还是将数据转换成高斯分布，例如使用对数函数：x = log(x+c)，其中 c 为非 负常数； 或者 x=xc ，c 为 0-1 之间的一个分数，等方法。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9上异常检测推荐系统/7bd2e3d2.jpg)

### **3. Multivariate Gaussion Distribution（多元高斯分布）**

### **1） Multivariate Gaussion Distribution**

假使我们有两个相关的特征，而且这两个特征的值域范围比较宽，这种情况下，一般的 高斯分布模型可能不能很好地识别异常数据。其原因在于，一般的高斯分布模型尝试的是去 同时抓住两个特征的偏差，因此创造出一个比较大的判定边界。 下图中是两个相关特征，洋红色的线（根据 ε 的不同其范围可大可小）是一般的高斯分 布模型获得的判定边界，很明显绿色的 X 所代表的数据点很可能是异常值，但是其 p(x)值却 仍然在正常范围内。多元高斯分布将创建像图中蓝色曲线所示的判定边界

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9上异常检测推荐系统/1cdae3ef.jpg)

* **一般的高斯分布模型** 我们计算 p(x)的方法是： 通过分别计算每个特征对应的几 率然后将其累乘起来，在多元高斯分布模型中，我们将构建特征的协方差矩阵，用所有的特 征一起来计算 p(x)。

$p(x)=∏_{j=1}^{n}p(x_j;μ_j,σ^2_j)=∏_{j=1}^{n}\frac{1}{\sqrt{2π}σ_j}exp(-\frac{(x_j-μ_j)^2}{2σ_j^2})$$μ=\frac{1}{m}\sum_{i=1}^{m}x^{(i)}$

* **多元高斯分布**

$p(x)=\frac{1}{(2π)^{\frac{n}{2} |Σ|^{\frac{1}{2}exp(-\frac{1}{2}(x-μ)^TΣ^{-1}(x-μ))$$Σ=\frac{1}{m}(X-μ)^T(X-μ)$

μ和Σ对模型的影响

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9上异常检测推荐系统/0ba5232c.jpg)

上图是 5 个不同的模型，从左往右依次分析：

1. 是一个一般的高斯分布模型
2. 通过协方差矩阵，令特征 1 拥有较小的偏差，同时保持特征 2 的偏差
3. 通过协方差矩阵，令特征 2 拥有较大的偏差，同时保持特征 1 的偏差
4. 通过协方差矩阵，在不改变两个特征的原有偏差的基础上，增加两者之间的正相关 性
5. 通过协方差矩阵，在不改变两个特征的原有偏差的基础上，增加两者之间的负相关 性

### **2） Anomaly Detection using the Multivariate Gaussion Distribution**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9上异常检测推荐系统/6a2d5a84.jpg)

通过使用多元高斯分布异常检测，可以更好的拟合数据，不再是画同心圆了，2333\~\~。

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week9上异常检测推荐系统/f2e7d76c.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**
