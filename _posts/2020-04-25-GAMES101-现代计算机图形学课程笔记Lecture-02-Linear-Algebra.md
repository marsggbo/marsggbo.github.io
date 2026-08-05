---
layout: post
title: "【GAMES101-现代计算机图形学课程笔记】Lecture 02 Linear Algebra"
date: 2020-04-25
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/136293369
related_posts: false
toc:
  sidebar: left
---

## **1. Vector （向量 / 矢量）**

## **1.1 基础回顾**

* 向量表示方式为 ![\vec{a}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/9b24d44e.jpg) 或者 ![\boldsymbol{a}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/b641a73f.jpg)
* 向量长度 ![\|\vec{a}\|](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/18e056a1.jpg)
* 单位向量表示方式为：![\hat{a}=\vec{a} /\|\vec{a}\|](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/73f0aa6d.jpg)
* 向量表示采用**笛卡尔坐标(Cartesian Coordinates)**，例如

![\mathbf{A}=\left(\begin{array}{l}x \\ y\end{array}\right) \quad \mathbf{A}^{T}=(x, y) \quad\|\mathbf{A}\|=\sqrt{x^{2}+y^{2}}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/18af0d51.jpg)

> 注意，一般默认向量为列向量。

## **1.2 向量相乘**

### **1.2.1 点乘**

* 定义：

![\vec{a} \cdot \vec{b}=\|\vec{a}\|\|\vec{b}\| \cos \theta](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/961d959b.jpg) ![\cos \theta=\frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\|\|\vec{b}\|}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/ec018296.jpg)

* 性质

![\vec{a} \cdot \vec{b}=\vec{b} \cdot \vec{a}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/0d885a03.jpg) ![\vec{a} \cdot(\vec{b}+\vec{c})=\vec{a} \cdot \vec{b}+\vec{a} \cdot \vec{c}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/d2dbbf13.jpg) ![(k \vec{a}) \cdot \vec{b}=\vec{a} \cdot(k \vec{b})=k(\vec{a} \cdot \vec{b})](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/6fcdbe7e.jpg)

* 计算示例

![\vec{a} \cdot \vec{b}=\left(\begin{array}{l}x_{a} \\ y_{a}\end{array}\right) \cdot\left(\begin{array}{l}x_{b} \\ y_{b}\end{array}\right)=x_{a} x_{b}+y_{a} y_{b}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/7cc951ac.jpg)

* 用途

1） **计算投影**

![](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/5a9db938.jpg)

2） 判断两个向量是否同向

点乘结果>0就表示基本同向，=1表示方向完全一致。

### **1.2.2 叉乘**

* 定义

![a \times b=-b \times a](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/f4deb34d.jpg) ![\|a \times b\|=\|a\|\|b\| \sin \phi](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/f411c71a.jpg)

使用右手法则。

![](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/c92245e3.jpg)

叉乘不满足交换律。

* 用途

1）生成坐标轴

![\vec{x} \times \vec{y}=+\vec{z}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/19e8ba55.jpg)

![\vec{y} \times \vec{x}=-\vec{z}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/99fee52d.jpg)

![\vec{y} \times \vec{z}=+\vec{x}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/cb63fcf5.jpg)

![\vec{z} \times \vec{y}=-\vec{x}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/f82d5714.jpg)

![\vec{z} \times \vec{x}=+\vec{y}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/d45a79d4.jpg)

![\vec{x} \times \vec{z}=-\vec{y}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/b58d2a2b.jpg)

2）判定左 / 右 或者 内 / 外

比如一直坐标系由XYZ组成，然后现在想判断向量b是在a的左边还是右边，之需要求出![\vec{x} \times \vec{y}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/f8857af2.jpg)可以知道与![\vec{z}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/3383c9fc.jpg)同向，所以b在a左边。

![\vec{AP}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/3b78206c.jpg)始终在三条有向边![\vec{AB},\vec{BC},\vec{CA}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/735768c6.jpg)的同一侧(左侧)，所以p点在三角形内侧。

> 注意：三角形三条边的向量必须首尾相连，所以如果我们把下面三角形的三条边向量换一个方向，但是因为最后可以算出AP都在三条边的右侧，即同一侧，所以P点在三角形内侧。

![](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/c1dd061d.jpg)

## **2. Matrix (矩阵)**

矩阵在图形学里常用于表示变换(Transformations),比如 translation,rotation,shear,scale等。

矩阵相乘运算

![\left(\begin{array}{ll}1 & 3 \\ 5 & 2 \\ 0 & 4\end{array}\right)\left(\begin{array}{llll}3 & 6 & 9 & 4 \\ 2 & 7 & 8 & 3\end{array}\right)=\left(\begin{array}{cccc}9 & ? & 33 & 13 \\ 19 & 44 & 61 & 26 \\ 8 & 28 & 32 & ?\end{array}\right)](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/66940a6c.jpg)

以右边那个8为例，可以看到它是第三行第一列，所以直接找到左边矩阵的第三行，即 ![[0\,\,4]](https://www.zhihu.com/equation?tex=%5B0%5C%2C%5C%2C4%5D)，和右边矩阵第一列 ![[3\,\,2]^T](https://www.zhihu.com/equation?tex=%5B3%5C%2C%5C%2C2%5D%5ET),然后做点积即可求得为8.

* 性质
* ![(\mathrm{AB}) \mathrm{C}=\mathrm{A}(\mathrm{BC})](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/118883b0.jpg)
* ![A(B+C)=A B+A C](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/942841aa.jpg)
* ![(\mathrm{A}+\mathrm{B}) \mathrm{C}=\mathrm{AC}+\mathrm{BC}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/cbfa1f07.jpg)
* 矩阵转置：![(A B)^{T}=B^{T} A^{T}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/6fcec3a7.jpg)
* 对角矩阵：只有对角线上有非零元素
* 单位矩阵：对角线上全为1的对角矩阵
* 矩阵的逆：

+ ![A A^{-1}=A^{-1} A=I](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/fd348b03.jpg)
+ ![(A B)^{-1}=B^{-1} A^{-1}](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/1c8cfd74.jpg)

矩阵乘法转化成矩阵形式

* 点积

![\begin{aligned} & \vec{a} \cdot \vec{b}=\vec{a}^{T} \vec{b} \\=\left(\begin{array}{lll}x_{a} & y_{a} & z_{a}\end{array}\right)\left(\begin{array}{l}x_{b} \\ y_{b} \\ z_{b}\end{array}\right)=\left(x_{a} x_{b}+y_{a} y_{b}+z_{a} z_{b}\right) \end{aligned} \\](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/1c3a77b9.jpg)

* 叉乘

![\vec{a} \times \vec{b}=A^{*} b=\left(\begin{array}{ccc}0 & -z_{a} & y_{a} \\ z_{a} & 0 & -x_{a} \\ -y_{a} & x_{a} & 0\end{array}\right)\left(\begin{array}{l}x_{b} \\ y_{b} \\ z_{b}\end{array}\right)](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/916c70f9.jpg)

注意 ： A\*b的\*不表示乘法

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-04-25-GAMES101-现代计算机图形学课程笔记Lecture-02-Linear-Algebra/228f26c5.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
  
  
**2020-04-24 23:32:12**