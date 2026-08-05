---
layout: post
title: "torchline：让Pytorch使用的更加顺滑"
date: 2019-12-13
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/97244535
related_posts: false
toc:
  sidebar: left
---

> torchline地址：[https://github.com/marsggbo/torchline](https://link.zhihu.com/?target=https%3A//github.com/marsggbo/torchline)

相信大家平时在使用Pytorch搭建网络时，多少还是会觉得繁琐，因为我们需要搭建数据读取，模型，训练，checkpoints保存等等一系列模块。每当切换到新的任务后很多情况下之前的代码不能复用，或者说要复用就需要做很多地方的修改，到最后还不如重新写一遍。所幸，pytorch\_lightning让这一过程简化了很多，相信如果你用过这个库你也会体验到它的方便性。但是torchline的存在是让你使用Pytorch更加的顺滑舒畅。

torchline基于[pytorch\_lightning (PL)](https://link.zhihu.com/?target=https%3A//github.com/williamFalcon/pytorch-lightning)开发，整个库的结构设计借鉴了[detectron2](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/detectron2)，具体可以阅读下面几篇文章进行了解：

[marsggbo：Detectron2源码阅读笔记-(一)Config&Trainer](https://zhuanlan.zhihu.com/p/86749823)[marsggbo：Detectron2源码阅读笔记-(二)Registry&build\_\*方法](https://zhuanlan.zhihu.com/p/86808911)[marsggbo：Detectron2源码阅读笔记-（三）Dataset pipeline](https://zhuanlan.zhihu.com/p/88149772)

如下图所示，灰色部分 **PL** 可以自动完成。我们需要做的，差不多也就加载数据、定义模型、确定训练和验证过程

![](/assets/img/marsggbo/2019-12-13-torchline让Pytorch使用的更加顺滑/ad12f40d.jpg)

**torchline**则进一步简化，而且可以让你的模型复用性更高。

以构建模型为例进行大致的介绍(细节可以去github查看)，假如你之前创建了一个`MyModel`的模型，之后如果你想使用这个模型，你只需要在config文件中将`MODEL.NAME`修改成`MyModel`（即只是修改字符串的值，之后`torchline`会自动切换模型）。

总的来说，pytorch\_lightning有的torchline肯定都有哈哈哈，但是使用起来代码复用性和易用性更高，欢迎去github品尝，觉得好用麻烦star，也欢迎issue讨论。

### **MARSGGBO♥原创** 如有意合作，欢迎私戳 邮箱:marsggbo@foxmail.com 微信公众号: 【AutoML机器学习】

![](/assets/img/marsggbo/2019-12-13-torchline让Pytorch使用的更加顺滑/228f26c5.jpg)

AutoML机器学习

**2019-12-13 21:31:15**