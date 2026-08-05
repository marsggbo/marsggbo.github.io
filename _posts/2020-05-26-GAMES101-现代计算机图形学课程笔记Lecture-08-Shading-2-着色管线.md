---
layout: post
title: "【GAMES101-现代计算机图形学课程笔记】Lecture 08 Shading 2 （着色管线）"
date: '2020-05-26'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/143658280
related_posts: false
toc:
  sidebar: left
---

本节内容概要：

* Blinn-Phong reflectance model

+ Specular and ambient terms

* Shading frequencies
* Graphics pipeline

## **1. Blinn-Phong reflectance model**

## **1.1 漫反射项**

在介绍本节内容之前首先回顾一下上一节的内容。

前面提到了光可以分成三种： 漫反射光、镜面反射光和环境光。

上一节主要介绍了漫反射，由下图我们知道着色点（shading point）的明暗程度与相机（观测）角度无关。具体的光线强度计算公式：

$$L_{d}=k_{d}\left(I / r^{2}\right) \max (0, \mathbf{n} \cdot 1)$$

上面公式中的$k_d$表示漫反射系数，中间$I/r^2$表示理论上每个着色点对应的光强度，最后一项$\max (0, \mathbf{n} \cdot 1)$表示吸收的能量比例，可以看到只与法向和光的方向夹角有关。

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/050d20dc.jpg)

## **1.2 高光项(Specular Term)**

下面介绍一下高光（又称 镜面反射）项。根据日常生活经验我们可以发现这样一种规律，就是当我们去看一面镜子的时候，当我们的观察角度越接近光线的镜面反射方向，就越容易看到高光（就是那种闪瞎狗眼的情况）。以下图为例，就是当我们的观察方向$V$越接近镜面放射方向$R$ (**r**eflection),高光就越明显。

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/82ca68c2.jpg)

上面例子中提到了判断高光的方法就是看$V$和$R$两个矢量是否接近，越接近则高光越明显。

但是反射方向在实际计算时并不太好计算，所以**Blinn-Phong模型**对此作了改进，简化了计算，具体方法见下图：

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/48591487.jpg)

可以看到首先需要定义一个新的矢量，叫做**半程矢量(bisector)**,其实表示的就是光入射方向$l$和观测方向$v$的中间矢量。

$$\begin{aligned} \mathbf{h} &=\operatorname{bisector}(\mathbf{v}, \mathbf{l}) \\ &=\frac{\mathbf{v}+\mathbf{l}{\|\mathbf{v}+\mathbf{l}\|} \end{aligned}$$

很显然，当$v$越接近$R$,那么$h$也就会越接近$n$，所以前面判断$v$和$R$的相似问题就转化成了$n$和$h$的相似问题，很显然$h$的计算要简单很多，只需要把两个矢量相加即可求得。

另外，类似于前面介绍的漫反射计算公式，镜面反射公式如下：

$$\begin{aligned} L_{s} &=k_{s}\left(I / r^{2}\right) \max (0, \cos \alpha)^{p} \\ &=k_{s}\left(I / r^{2}\right) \max (0, \mathbf{n} \cdot \mathbf{h})^{p} \end{aligned}$$

* $k_s$表示镜面反射系数，一般默认高光就是白色的，也就是说该系数通常设置为1。
* $I/r^2$: 同最上
* 角度变为$n$和$h$之间的夹角

上面公式中有一个需要特别注意的地方是最后一项有一个系数$p$,这个在漫反射里是没有的，那这个系数是干嘛用的呢？要解释这个我们首先需要看看cosine函数的特点，如下图：

可以看到最左边对应$p=1$，由于衰减比较缓慢，比如当角度等于45°时，最终求得的镜面反射光的亮度还是很大，然后一般现实中，高光分布的范围比较小（想想有时候我们去看高光要很小心去调整角度，否则高光还没那么容易看到），所以我们需要对cosine函数做调整，进而减少高光范围，这就是系数$p$的作用。一般而言$p$常设置的比较大，例如200。

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/47a3420a.jpg)

下图给出了高光项的示意图(其实也包含了漫反射，否则就只剩下一小坨亮光了)，每一行表示镜面反射系数$k_s$保持不变，但是系数$p$不断增加;同理，每一列表示只增加镜面反射系数。可以看到$k_s$越大，越白；$p$越大，则高光范围越小。

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/41695c8d.jpg)

## **1.3 环境项(Ambient Term)**

环境光是由其他类型的光的多次反射或者漫反射而产生的，为了方便研究，一般假设环境光强度永远都是相同的，用$I_a$表示，

那么某个点的环境光计算公式为：

$$L_{a}=k_{a} I_{a}$$

可以看到环境光其实就是一个常数，其与法向、观测方向和光的入射方向都没有关系。（当然，以上都是很强假设和简化）

## **1.4 总的反射效果**

介绍完了漫反射、镜面反射和环境光后，总的反射效果则是三者之和，计算公式如下：

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/50e91c4d.jpg)

$$\begin{aligned} L &=L_{a}+L_{d}+L_{s} \\ &=k_{a} I_{a}+k_{d}\left(I / r^{2}\right) \max (0, \mathbf{n} \cdot \mathbf{l})+k_{s}\left(I / r^{2}\right) \max (0, \mathbf{n} \cdot \mathbf{h})^{p} \end{aligned}$$

## **2. Shading frequencies**

下面三个球的几何形状是一模一样的，但是为什么看起来不一样呢？三者的差别就在于着色的频率。

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/b54f9979.jpg)

## **2.1 Flat shading（平面着色）**

一般默认是对一个三角形着色，根据上面的着色计算公式可知我们首先需要计算出三角形法线方向（三角形的两条线做叉乘即可），之后结合$h,l$直接套用公式即可求得某三角形平面的着色值。 这个方法对应上图最左边的球。

## **2.2 Gouraud shading**

该着色方法的改进在于首先求出每个三角形的顶点的法向（如何求在后面会介绍），之后三角形内部的着色则通过插值（也会在后面介绍）的方法实现。

这个方法对应上图中间的球，对比Flat shading， 该着色方法屏挂了不少。

## **2.3 Phong shading**

Phone shading的大致思路是首先计算出每个三角形顶点的法向方向，之后通过插值的方法计算出三角形内部每个像素的法线方向，这样就可以精确地对每个像素着色。

## **2.4 shading方法对比**

上面介绍了三种着色方法，flat shading是以面（face）来着色，Gouraud则是以顶点（vertex）来着色，Phong就是以像素（pixel）来着色。

感觉上来说phong shading应该是效果最好的，但是也是最麻烦的。可是phong shading真的就一定是最好的吗？其实不然，以下图为例，让我们把面(face)定义地尽可能小的时候，其实Flat shading也能产生很不错的效果。

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/9d763b70.jpg)

## **2.5 Defining Per-Vertex Normal Vectors**

在前面的介绍中留下了两个问题，即**如何求法向**和**如何插值**。

### **2.5.1 顶点法向**

我们知道一个在计算机中一个复杂物体其实是由一些基础的结合图像表示的，常用的是三角形。如果我们要用三角形来表示一个球，那么此时每个三角形的顶点的法向很容易计算，只要知道球心坐标和顶点坐标即可求出顶点法向。但是实际上我们要渲染的物体没这么简单，所以我们要找到一种通用的计算办法。

下图中的中间那个顶点被四个三角形共用，那么该顶点的法向计算公式很简单其实就是四个三角形平面法向相加求平均，这种计算方法在实践中也被证明是有效的。

$$N_{v}=\frac{\sum_{i} N_{i}{\left\|\sum_{i} N_{i}\right\|}$$

当然这样计算会有一定误差，所以一种改进的计算方式是加权平均，权重即为每个三角形的面积。

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/41dfad5a.jpg)

### **2.5.2 barycentric interpolation**

前面的Phong shading方法提到了在三角形内部对每个像素点的法向可以通过插值求得，示意图如下：

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/ab753163.jpg)

具体方法在下一节中介绍。

## **3. Graphics (Real-time Rendering) Pipeline**

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/d141b94f.jpg)

上图给出了之前所有课程中合在一起和的一个流程图，也叫作管线。

可以看到首先我们在3D空间中有若干的点，那么

* 第一步就是需要将3D中的点映射到**屏幕空间(screen space)**,这一步在管线里叫**Vertex Processing**。

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/7ecd7ebe.jpg)

* 得到了屏幕上的各个顶点后，我们会将这些点连接起来得到很多的三角形，即**Triangle Processing**。
* 上一步骤中得到的三角形是连续的，而计算机是离散的，那么该怎么处理呢？这里就需要用到前面提到的**光栅化(Rasterization)技术**,即求解出那些像素对应到三角形内部，哪些对应到三角形外部。通过光栅化就可以生成许多**fragments**（这个是借用OpenGL里的概念），简单理解fragment就代表一个像素。

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/15e7d7d9.jpg)

* 既然知道了需要渲染哪些fragment，很自然下一步就是对这些fragment渲染出来就好了，而渲染涉及到两个问题：

+ 一个是**遮挡问题**，这就需要用到深度信息

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/345ca84e.jpg)



+ 另一个是**着色**

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/2aea1db4.jpg)

上述这些步骤其实在显卡里都已经写好了的，只不过计算过程放在GPU里计算而已。

## **4. Text Mapping（纹理映射）**

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/6bfb42d4.jpg)

由上图我们知道不同物体的纹理是不一样的，比如球上面有一个五角星（龙珠？），还有桌面的木头纹理等。

由前面提到的漫反射计算公式

$$L_d=k_{d}\left(I / r^{2}\right) \max (0, \mathbf{n} \cdot \mathbf{l})$$

可以知道物体表面纹理是由漫反射系数$k_d$控制的,换言之每个像素的漫反射系数应该都可以设置成不同的值从而显示出不同的效果，那么这个怎么做呢？

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/207f3bb3.jpg)

我们看上图中间表示的是一个真实世界空间，真实世界中所有点的纹理拼凑起来其实可以由一个2D纹理平面表示（下面最右边），换言之当我们将3D空间物体、2D屏幕空间以及2D文理空间所有点有一个一一映射的关系，这样就可以很容易地渲染每个像素有不同的纹理。

下图给出了纹理映射的一个例子（略微有点恶心），最左边是没有文理签的模型示意图，由左下角的图可以看到物体是由很多三角形拼凑而成的，那么纹理映射要做的是就是给这些三角形填色，而填哪些色这种事需要美工艺术家们来做，码农们要做的就事其实就是搬运工，即把美工设计好的纹理搬运到指定位置即可。中间是最后渲染的结果，最右边是2D纹理图。

![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/d3bb9673.jpg)

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-05-26-GAMES101-现代计算机图形学课程笔记Lecture-08-Shading-2-着色管线/228f26c5.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2020-05-25 22:03:53**
