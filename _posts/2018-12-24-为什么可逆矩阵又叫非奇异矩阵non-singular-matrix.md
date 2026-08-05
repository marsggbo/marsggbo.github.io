---
layout: post
title: 为什么可逆矩阵又叫“非奇异矩阵(non-singular matrix)”?
date: '2018-12-24'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/53137965
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/53137965>

最近在捡回之前的线性代数知识，在复习可逆矩阵的时候，发现有的书上把可逆矩阵又称为非奇异矩阵，乍一看名字完全不知所云，仔细一分析，还是不明白。要想弄明白，还是得从英文入手，下面的解释主要从这里得来的[Why are invertible matrices called 'non-singular'?](https://link.zhihu.com/?target=https%3A//math.stackexchange.com/questions/42649/why-are-invertible-matrices-called-non-singular)。

先把原回答搬过来:

> If you take an n×n matrix "at random" (you have to make this very precise, but it can be done sensibly), then it will almost certainly be invertible. That is, the generic case is that of an invertible matrix, the special case is that of a matrix that is not invertible. For example, a 1×1 matrix (with real coefficients) is invertible if and only if it is not the 0 matrix; for 2×2 matrices, it is invertible if and only if the two rows do not lie in the same line through the origin; for 3×3, if and only if the three rows do not lie in the same plane through the origin; etc. So here, "singular" is not being taken in the sense of "single", but rather in the sense of "special", "not common". See the [dictionary definition](https://link.zhihu.com/?target=https%3A//www.merriam-webster.com/dictionary/singular): it includes "odd", "exceptional", "unusual", "peculiar". The noninvertible case is the "special", "uncommon" case for matrices. It is also "singular" in the sense of being the "troublesome" case (you probably know by now that when you are working with matrices, the invertible case is usually the easy one).

主要说了个什么事呢，意思就是假设随机生成一个 $(n×n)$ 的矩阵，绝大多数情况这个矩阵都是可逆的，也可以理解为它的行列式不为0。换句话说，不可逆的情况是**少见**的，所以不可逆矩阵就称为**Singular matrix**，这里的singular就是special, not common的意思啊。同理，可逆矩阵很常见，所以就是非奇异矩阵了。

举个例子就更好明白了，现假设一个 $(1×1)$ 的矩阵，我们知道只有这个矩阵等于0的时候才是不可逆的，其余情况都是可逆的；再看 $(2×2)$ 的矩阵，这个可以理解成是一个平面上的两条线，只要当这两条线位于经过零点的同一条线上，那么这个矩阵才是不可逆的，显然这种情况是特殊的； $(3×3)$ 矩阵同理不加赘述。

**MARSGGBO♥原创**

### 微信公众号: 【AutoML机器学习】

![](/assets/img/marsggbo/2018-12-24-为什么可逆矩阵又叫非奇异矩阵non-singular-matrix/228f26c5.jpg)

AutoML机器学习

  
[为什么可逆矩阵又叫“非奇异矩阵(non-singular matrix)”?](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/marsggbo/p/10034629.html)