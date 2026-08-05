---
layout: post
title: "【Math for ML】线性代数之——向量空间"
date: '2018-12-23'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/53088633
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/53088633>

**I. Groups**

在介绍向量空间之前有必要介绍一下什么**Group**，其定义如下：

![](/assets/img/marsggbo/2018-12-23-Math-for-ML线性代数之向量空间/4f5855a3.jpg)
> 注意定义中的 $\bigotimes$ 不是乘法，而是一种运算符号的统一标识，可以是乘法也可以是加法等。

此外，如果 $\forall{x,y}∈\mathcal{G}:x⊗y=y⊗x$ ,那么此时 $G=(\mathcal{G,⊗})$ 是**Abelian Group(阿尔贝群)**。

举个栗子： - $(Z,+)$ 是group - $(N_0,+)$ 不是group，因为他没有inverse elements，即不满足定义中的第4个条件。

## **II. Vector Spaces**

向量空间定义：

![](/assets/img/marsggbo/2018-12-23-Math-for-ML线性代数之向量空间/118323fe.jpg)
> Group和Vector Space的区别在于前者只是在针对$\mathcal{G}$和在$\mathcal{G}$上的inner operation进行研究和讨论，而后者在其基础上还考虑了outer operation，由定义可以直观地看出差别和关系。

## **II. Vector Subspaces**

向量子空间定义：

![](/assets/img/marsggbo/2018-12-23-Math-for-ML线性代数之向量空间/71d3d324.jpg)

那么我们如何证明一个向量空间是另一个向量空间的子空间呢？我们需要做如下证明：

![](/assets/img/marsggbo/2018-12-23-Math-for-ML线性代数之向量空间/dba93217.jpg)

有一个比较特殊的向量子空间是**Trivial Subspace(平凡子空间)**，其性质为**任意空间的平凡子空间是它本身和** ${0}$ 。

## **MARSGGBO♥原创** **2018-12-16**