---
layout: post
title: Andrew Ng机器学习课程笔记--week2（多元线性回归&正规公式）
date: '2020-07-30'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/165322806
related_posts: false
toc:
  sidebar: left
---

## **1. 内容概要**

* **Multivariate Linear Regression(多元线性回归)**

+ 多元特征
+ 多元变量的梯度下降
+ 特征缩放

* **Computing Parameters Analytically**

+ 正规公式（Normal Equation )
+ 正规公式非可逆性（Normal Equation Noninvertibility）

## **2. 重点&难点**

## **1）多元变量的梯度下降**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week2多元线性回归正规公式/68697b93.jpg)

## **2） 特征缩放**

### **为什么要特征缩放**

首先要清楚为什么使用特征缩放。见下面的例子

* 特征缩放前

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week2多元线性回归正规公式/2a49fd25.jpg)

由图可以知道特征缩放前，表示面积的x1变量的值远大于x2，因此J(θ)图像表示就是椭圆的，导致在梯度下降的过程中，收敛速度非常慢。

* 特征缩放后

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week2多元线性回归正规公式/515ddf3f.jpg)

对各变量特征缩放后绘制出来的损失函数J(θ)明显收敛更快，这也是为什么需要特征缩放的原因了。

### **实现方法**

* **feature scaling**

$$\begin{equation} x_i := \frac{x_i}{x_\max - x_\min} \end{equation}$$

每个输入值除以(max - min)

* **mean normalization**

$$\begin{equation} x_i := \frac{x_i - μ_i}{s_i} \end{equation}$$

μi: 均值 si: max - min

## **3) Normal Equation 正规方程式**

**Normal Equation**

$$\begin{equation} θ = （X^T·X）^{﹣1}·X·Y \end{equation}$$

> 具体推理过程详见**[掰开揉碎推导Normal Equation](https://zhuanlan.zhihu.com/p/22757336)**

与梯度下降方法进行比较

**梯度下降正规方程式**需要选择步长α**不需要选择步长α**需要迭代训练很多次**一次都不需要迭代训练O(kn2)**O(n3,计算(XT·X)-1需要花费较长时间**即使数据特征n很大，也可以正常工作**n如果过大，计算会消耗大量时间

## **4） 正规方程不可逆**

当XT·X不可逆时，很显然此时正规方程将不能正常计算，常见原因如下：

* 冗余特征，在两个特点紧密相关(即它们呈线性关系，例如面积和（长，宽)这两个特征线性相关）
* 太多的特征(例如：m≤n)。 在这种情况下，可以删除一些特征或使用"regularization"。

**补充：**

* A是可逆矩阵的充分必要条件是 |A|≠0

### **微信公众号：AutoML机器学习**

http://weixin.qq.com/r/HD8gOHzEmiHlrThb92oO (二维码自动识别)

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2017-8-2**
