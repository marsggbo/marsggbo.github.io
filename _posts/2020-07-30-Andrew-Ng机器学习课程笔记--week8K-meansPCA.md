---
layout: post
title: "Andrew Ng机器学习课程笔记--week8(K-means&PCA)"
date: 2020-07-30
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/165338777
related_posts: false
toc:
  sidebar: left
---

**Unsupervised Learning** 本周我们讲学习非监督学习算法，会学习到如下概念

* 聚类（clustering）
* PCA(Principal Componets Analysis主成分分析)，用于加速学习算法，有时在可视化和帮助我们理解数据的时候会有难以置信的作用。

## **一、内容概要**

* **Clustering**

+ K-Means Algorithm
+ Optimization Objective
+ Random Initialization
+ Choosing The Number of Clusters

  
* **Dimensionality Reduction（降维）**

+ Motivation
+ PCA（主成分分析）
+ Applying PCA

## **二、重点&难点**

## **1. Clustering**

### **1) K-Means Algorithm**

首先需要知道的是无监督学习下，数据是没有标签的，所以可视化数据后是下面这样的效果（只有一种颜色）

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/8666c4a6.jpg)

**K-Means算法步骤如下：** 1.**随机分配聚类中心（cluster centroid）** 假设我们知道数据可以分为两类（这样做为了方便讨论），所以我们随机分配两个聚类中心(如下图**一个红色，一个蓝色**)。 2.**聚类分配** 遍历每一个数据x计算出其离哪个中心点更近，更近的标上和那个中心点相同的颜色。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/f7cc1796.jpg)

3.**移动聚类中心** 完成步骤2后，计算每个聚类所有数据点的中心，该点即为新的聚类中心。

> 一般来说，求聚类中心点的算法你可以很简的使用**各个点的(X,Y)坐标的平均值**。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/d69c7be4.jpg)

不过，另有三个求中心点的的公式：

> 参考**[深入浅出K-Means算法](https://link.zhihu.com/?target=http%3A//www.csdn.net/article/2012-07-03/2807073-k-means)**

1）**Minkowski Distance**公式——λ可以随意取值，可以是负数，也可以是正数，或是无穷大。

![d_{ij}=\sqrt{ \sum_{k=1}^{n}|x_{ik} - y_{jk}|^λ  } \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/892cc98b.jpg)

2）**Euclidean Distance**公式——也就是第一个公式λ=2的情况

![d_{ij}=\sqrt{ \sum_{k=1}^{n}|x_{ik} - y_{jk}|^2 } \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/7d80c722.jpg)

3）**CityBlock Distance**公式——也就是第一个公式λ=1的情况

![d_{ij}=\sum_{k=1}^{n}|x_{ik} - y_{jk}|   \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/83249e65.jpg)

这三个公式的求中心点有一些不一样的地方，我们看下图（对于第一个λ在0-1之间）。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/0a363b33.jpg)

（1）Minkowski Distance （2）Euclidean Distance （3） CityBlock Distance

上面这几个图的大意是他们是怎么个逼近中心的，第一个图以星形的方式，第二个图以同心圆的方式，第三个图以菱形的方式。

4.**重复2,3步骤，直到收敛，即中心不再变化或变化范围达到设定阈值**

总结起来就是：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/2ba83610.jpg)

m：样本数据集的大小 ![c^{(i)}](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/3a72d0f5.jpg):第i个数据![x^{(i)}](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/1259e80c.jpg)所属聚类的下标 ![μ_k](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/8ea0f592.jpg)：第k个聚类中心点

### **2) Optimization Objective**

是的，k-means也有优化目标函数，如下：

![minJ_(c^{(1)},……c^{(m)},μ_1,……μ_k)=\frac{1}{m}\sum_{i=1}^{m}{||x^{(i)}-μ_{c^{(i)}}||^2} \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/2b8d5438.jpg)

### **3) Random Initialization**

前面的步骤中都提到了随机初始化聚类中心，但是这样可能会得到局部最优点而不是全局最优，如下图所示：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/2ce43550.jpg)

所以为了解决这个问题，我们先需要重复多次的随机初始化，然后看最后得到的结果中是否有很多结果是相同的，如果有那么很可能就是全局最优解。 算法如下

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/e1b8d610.jpg)

### **4) Choosing The Number of Clusters**

本小节将讨论聚类个数K的如何选取。

* **Elbow Method(肘部原理)**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/9da35b77.jpg)

如上图所示，我们可以通过计算不同k值所对应的损失函数的值，然后绘制成曲线，上面的曲线看上去就像是人的手臂，拐点（**k=3**）就是肘部，所以选择k=3是比较好的选择。

但是并不是所有时候都能得到上面那种比较理想的曲线，例如下面的曲线就不太好选择k值了。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/bf89bef6.jpg)

* **根据需求规定k** 上图中的光滑曲线不太适用于肘部原理，所以此时更好的办法是根据当前的需求来选择k值。以下面的数据为例，该数据记录了身高体重与衬衫大小的关系。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/1c52fa5f.jpg)

上图左边按照\*\*‘S,M,L’**划分，右边按照**'XS,S,M,L,XL'\*\*划分，这也不是为办法中的办法2333.

## **2. Dimensionality Reduction（降维）**

### **1) Motivation**

### **- 数据压缩**

在面对数据冗余的时候或者数据维度过大的时候可以通过降维来实现对数据的压缩从而提高计算效率。 例如 2D→1D

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/07d0055f.jpg)

3D→2D

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/24d07e8e.jpg)![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/b949b419.jpg)

### **- 数据可视化**

例如我们描述一个国家可以有50多种特征，但是想要可视化是不可能的，所以通过数据压缩后可以实现50D→2D，这样就能很好的看出各个国家之间的差距关系。

### **2) PCA**

### **PCA Problem Formulation（提法、构想）**

如下图是一些二维的点，现在需要将这些数据转化为一维数据点

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/cfaf0577.jpg)

PCA的方法是

* 寻找一条拟合的曲线（或平面）**U**
* 然后得到每个原始数据点到**U**使映射面对应的映射点**z**
* 计算各个点到该曲线（或平面）距离的总和(这里即是所有紫色线段长度总和).
* 将距离总和优化到最短。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/3368c818.jpg)

乍一看感觉这个线性回归很像啊？但是还是有很大的区别的，见下图

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/0fec2b02.jpg)

左边是线性回归，右边是PCA。

> 区别如下：

* **PCA**优化的目的是使得所有点到降维后的平面的距离之后最小，所以是**垂直**的距离。
* **线性回归**关注的是**实际值y**与**预测值 y\_** 大小之间的差距，优化的目的是使得预测值与实际值尽可能地接近或相等，所以是**竖直**的距离

### **PCA Algorithm**

**1. 数据预处理** 在使用PCA算法之前需要对数据机型预处理，方法有两种：

* **Mean normalization**
* **Feature scaling**

**2. PCA算法描述**

* **计算协方差矩阵Σ**

![Σ=\frac{1}{m}\sum_{i=1}^{m}(x^{(i)})(x^{(i)})^T \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/e12be175.jpg)

> 左边的Σ是希腊大写的σ，右边的∑是求和符号

注意![x^{(i)}](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/1259e80c.jpg)是(n,1)的向量，所以Σ是(n,n)的矩阵。

* **计算矩阵Σ的特征向量U** 视频里介绍的是octave的用于计算特征值的函数有svd和eig，但是svd比eig更加稳定。

```
// Σ=sigma
sigma = 1\m * (X' * X)
[U,S,V] = svd(sigma)
```

* **提取特征向量(![U_{reduce}](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/040a4ea5.jpg)) nD→kD** U也是(n,n)的矩阵，它就是我们需要的特征向量矩阵

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/e7dd6c6d.jpg)

假设原始特征向量是n维的，我们想转化成k维，只需要**取U矩阵的前k列**即可，我们记这前k列向量为![U_{reduce}^{n×k}∈R^{n×n}](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/cf29a650.jpg)。

* **将x向量转化成z向量**

![z = (U_{reduce})^T*x](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/acabbf63.jpg)

维度表示： ![(R^{k×n}*R^{n×1}) = R^{k×1}](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/9c5f9b51.jpg)

所以z是(k,1)向量。

> 上一步骤得到的![U](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/38190287.jpg)可以理解成一个映射面，这里的z就是各个原始数据点**x**对应映射面的映射点**z**。

**总结** ：

1.数据预处理

2.计算协方差Σ: ![Σ=\frac{1}{m}\sum_{i=1}^{m}(x^{(i)})(x^{(i)})^T](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/244d67da.jpg)

3.计算特征向量U: ![[U,S,V] = svd(sigma)](https://www.zhihu.com/equation?tex=%5BU%2CS%2CV%5D+%3D+svd%28sigma%29)

4.获取k维的![U_{reduce}^{n×k}](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/ddc5677e.jpg)

5.计算z: ![z= (U_{reduce})^T*x](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/b821591c.jpg)

### **3) Applying PCA**

### **1. Reconstruction from Compressed Representation(还原数据维度)**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/75ee44db.jpg)

即已知降维后的向量z，如何还原成x？方法如下：

![x = U_{reduce} * z \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/14d74341.jpg)

注意这里的还原并不是真正的还原成原始数据，因为这个公式得到的x是映射面U上的点，记为![x_{approx}](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/f0835c63.jpg),虽然有些误差，但是误差一般很小。

### **2. Choosing the number of Principle Components（选择k值大小）**

* **方法一** 前面已经提到过![x_{approx}](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/f0835c63.jpg)表示U映射面的点，而PCA优化目标就是最小化**投影误差(projection error)**:

![minE_p = min\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}-x_{approx}^{(i)}||^2 \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/6105d087.jpg)

我们记原始数据离原点距离的平方的均值为

![E_{total}=\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}||^2 \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/b27e5f80.jpg)

选择k值的标准就是满足下面的条件

![\frac{E_p }{E_{total}}=\frac{\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}-x_{approx}^{(i)}||^2}{\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}||^2}≤0.01 \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/81873673.jpg)

所以算法描述如下：

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/058f5658.jpg)

即k从1开始不断计算，知道满足小于等于0.01为止（也不一定非得是0.01，具体情况具体分析）。

* **方法二** 这个方法要比上面一个方法**更加简单**。 前面提到过这个方法![[U,S,V] = svd(sigma)](https://www.zhihu.com/equation?tex=%5BU%2CS%2CV%5D+%3D+svd%28sigma%29)，其中的s也是(n,n)的矩阵，如下图所示，是一个对角矩阵。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/e9c52789.jpg)

之所以说这个方法比上一个简单，是因为下面两个式子可以等价计算。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/ad1b0a5f.jpg)

即

![\frac{\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}-x_{approx}^{(i)}||^2}{\frac{1}{m}\sum_{i=1}^{m}||x^{(i)}||^2}=1-\frac{\sum_{i=1}^{k}s_{ii}}{\sum_{i=1}^{n}s_{ii}}≤0.01 \\](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/e714f3cd.jpg)

S矩阵只需要计算一次即可，所以只需要将k从1递增，知道满足小于等于0.01即可求出k值。

### **3. Advice for Applying PCA**

下面是使用PCA的一些误区

* **为了防止过拟合而盲目使用PCA** PCA的确能够压缩数据，提高计算速率，但是要知道的是什么是过拟合？ 过拟合的对象是y值，而PCA算法计算的对象不是y，而是x与![x_{approx}](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/f0835c63.jpg)，所以为了防止过拟合，更好的办法是使用正则化方法。
* **认为使用PCA优化数据准没错** 很多时候想都不想就先直接优化数据，然后再进行计算。视频中老师建议可以先用原始数据计算，看一下效果如何，然后再根据实际情况看是否需要使用PCA算法来压缩数据。

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week8K-meansPCA/f2e7d76c.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**