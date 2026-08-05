---
layout: post
title: Andrew Ng机器学习课程笔记--week11（图像识别&总结划重点）
date: '2020-07-30'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/165346262
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/165346262>

## **一、内容概要**

* **Photo OCR**

+ Problem Decription and pipeline(问题描述和流程图)
+ Sliding Windows(滑动窗口)
+ Getting Lots of Data and Artificial Data
+ Ceiling Analysis（上限分析）:What part of the pipline to Work on Next

## **二、重点&难点**

## **1. Problem Decription and pipeline**

为了实现图像文字识别通常按如下流程图进行操作：

* 文字侦测（Text detection）——将图片上的文字与其他环境对象分离开来
* 字符切分（Character segmentation）——将文字分割成一个个单一的字符
* 字符分类（Character recognition）——文字识别

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week11图像识别总结划重点/cb9b5941.jpg)

## **2. Sliding Windows(滑动窗口)**

滑动窗口是一项用来从图像中抽取对象的技术。 假使我们需要在一张图片中识别行人，**首先**要做的是用许多固定尺寸的图片来训练一个能够准确识别行人的模型。**然后**我们用之前训练识别行人的模型时所采用的图片尺寸在我们要进行行 人识别的图片上进行剪裁，然后将剪裁得到的切片交给模型，让模型判断是否为行人，**然后**在图片上滑动剪裁区域重新进行剪裁，将新剪裁的切片也交给模型进行判断，如此循环直至将图片全部检测完。一旦完成后，我们按比例放大剪裁的区域，再以新的尺寸对图片进行剪裁，将新剪裁的切片按比例缩小至模型所采纳的尺寸，交给模型进行判断，如此循环。

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week11图像识别总结划重点/c53993c8.jpg)

## **3. Getting Lots of Data and Artificial Data**

机器学习要获得更好的效果就需要大量的数据来训练，但是有的数据并不是很方便的获得，所以可以在原有数据的基础上通过人工合成的方式来扩大数据。例如将已有的字符图片进行一些扭曲、旋转、模糊处理。

## **4. Ceiling Analysis:What part of the pipline to Work on Next**

下面以图像文字识别流程图为例来解释**上限分析**的思想。

**Text detection -> Character segmentation -> Character recognition**

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week11图像识别总结划重点/bd74b10d.jpg)

首先按照最开始的模型得出最终的系统识别准确率为72%。

之后我们人为的提高上面三个环节的准确率接近100%，然后观察系统准确率的变化。

例如我们在Text Detection这一步骤中人为的指定出文字所在位置，使得文字检测准确率达到100%，然后其他步骤不变，最后观察到系统准确率为89%，提高了17%。 其他同理，可以看到提高 **文字识别(Character recognition)** 这一步骤的准确率可以使得系统准确率达到100%，所以接下来的工作则是尽量提高文字识别这一步骤的准确率，而不是另外两个步骤。

> 最后一节课了，超级感谢吴大大~~~~~~~~~~~~~~~~~~~~~~ 附上整个课程所学的知识点，撒花✿✿ヽ(°▽°)ノ✿！！！！

![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week11图像识别总结划重点/00d2cc7a.jpg)

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-07-30-Andrew-Ng机器学习课程笔记--week11图像识别总结划重点/f2e7d76c.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**