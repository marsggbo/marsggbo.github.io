---
layout: post
title: 华为诺亚实验室AutoML框架-Vega：(1) 介绍
date: '2020-09-23'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/258362976
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/258362976>

> 本文主要简单地介绍Vega的特点以及它与其他AutoML框架的区别，让你对Vega有一个直观的理解。  
>  文章太长不看可以直接降落至最后的总结部分。

[huawei-noah/vega](https://link.zhihu.com/?target=https%3A//github.com/huawei-noah/vega)

## **1. 现有AutoML框架总结**

目前已经出现了很多AutoML开源框架，可以大致分成两类：

* 一类是基于传统机器学习算法，例如 **[TPOT](https://link.zhihu.com/?target=https%3A//github.com/EpistasisLab/tpot)**、**[Auto-Sklearn](https://link.zhihu.com/?target=https%3A//github.com/automl/auto-sklearn)**、 **[Hyperopt](https://link.zhihu.com/?target=https%3A//github.com/hyperopt/hyperopt)**，**[H2O](https://link.zhihu.com/?target=https%3A//docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html)** 如果你不需要神经网络的话则可以考虑使用这几个框架。
* 另一类则是基于神经网络的AutoML框架，比较流行的有 **[NNI](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nni)**、 **[Auto-kears](https://link.zhihu.com/?target=https%3A//github.com/keras-team/autokeras)** 以及本篇文章要介绍的华为诺亚方舟实验室开源的 **[Vega](https://link.zhihu.com/?target=https%3A//github.com/huawei-noah/vega)** 框架。

## **2. Auto-Keras & NNI**

在介绍Vega之前，首先简单介绍一下Auto-Keras和NNI。

* Auto-Keras，看名字也知道它是基于Keras实现的，换句话说它只支持TensorFlow，目前最新版本的要求是**Python >= 3.5 and TensorFlow >= 2.3.0**，因此如果你常用的框架是Pytorch，这个可能不适合你。
* NNI是微软开发的**轻量型**AutoML工具包，如下图示，其提供的功能非常丰富，包括自动特征工程、NAS（神经网络架构搜索）、模型压缩、超参数搜索等等，而且还提供了可视化界面方便管理。我本人也用过NNI的NAS模块，易用性非常高。NNI将搜索空间和搜索算法解耦，而且设计了一种统一的接口，可以很方便地实现你的NAS算法，具体可看官方文档介绍：[https://nni.readthedocs.io/zh/latest/nas.html](https://link.zhihu.com/?target=https%3A//nni.readthedocs.io/zh/latest/nas.html)

![](/assets/img/marsggbo/2020-09-23-华为诺亚实验室AutoML框架-Vega1-介绍/823e50c2.jpg)

## **3. Vega**

Vega是华为诺亚方舟实验室自研的AutoML算法工具链([https://github.com/huawei-noah/vega](https://link.zhihu.com/?target=https%3A//github.com/huawei-noah/vega))，有如下几个主要特点。

## **3.1 完备的AutoML能力**

Vega涵盖HPO(超参优化, HyperParameter Optimization)、Data-Augmentation、NAS(网络架构搜索， Network Architecture Search)、Model Compression、Fully Train等关键功能，同时这些功能自身都是高度解耦的，可以根据需要进行配置，构造完整的pipeline。

乍看起来，你也许会觉得这个NNI很相似，但是正如前面介绍的，NNI其实是一个**轻量型**的工具包，也就是说你如果只需要实现某一个具体的功能，例如NAS或者模型压缩，那NNI是一个很好的选择，但是如果你希望将多个功能组合成一个完整的pipeline（如下图示），那么你需要的是Vega。

![](/assets/img/marsggbo/2020-09-23-华为诺亚实验室AutoML框架-Vega1-介绍/1831ebc7.jpg)

Vega解决的思路是将每一个功能（如模型压缩和NAS等）都抽象成一个具有统一接口的PipeStep类，通过遍历每个Step实现端到端的Pipeline，具体的实现原理在后续文章中会介绍。

## **3.2 业界标杆的自研算法**

Vega提供了诺亚方舟实验室自研的 业界标杆 算法，并提供 Model Zoo 下载SOTA(State-of-the-art)模型。

![](/assets/img/marsggbo/2020-09-23-华为诺亚实验室AutoML框架-Vega1-介绍/1f766294.jpg)

## **3.3 高并发模型训练能力**

Vega提供高性能Trainer，加速模型训练和评估。Vega设计了Scheduler模块，可以很方便的管理本地集群部署等任务。

![](/assets/img/marsggbo/2020-09-23-华为诺亚实验室AutoML框架-Vega1-介绍/b8d2db14.jpg)

## **3.4 多Backend支持**

Vega支持PyTorch，TensorFlow，MindSpore。虽然NNI也支持多种深度学习框架，但是Vega对不同框架的算子设计了统一接口。而且你可以只需要指定要运行的Backend即可切换至不同的深度学习框架。

## **3.5 可插拔式**

Vega采用可插拔式的配置方法，你可以只需要修改yml文件中的参数就能灵活地切换不同数据集、算法等组件，换句话说**你可以基于Vega搭建属于你自己的工具库**，因此在处理一个新项目的时候，你可以很方便地复用你之前的工作成果。

## **3.6 提供多个特色模块**

Vega提供了丰富的搜索空间，你可以很方便地自定义你的搜索空间快速验证你的idea。另外Vega还提供了**Report**模块，该模块可以很方便地管理训练过程中的各种日志等信息，使用起来更加简单顺手。

## **4. 总结**

通过上面介绍我们知道Vega和NNI都是提供了非常丰富的功能的AutoML框架，区别在于NNI定位于**轻量型**的工具包，所以如果你只想实现NAS算法或者模型压缩，那么你可以使用NNI来验证你的想法。Vega则定位于从**Pipeline**的角度来解决AutoML问题，因此使用Vega有一定门槛，需要你学习它的设计理念和代码逻辑，但是如果你掌握之后，你会发现Vega的易用性是不输NNI的，本系列文章主要是一起来学习Vega，然后搭建属于你自己的工具库。

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-09-23-华为诺亚实验室AutoML框架-Vega1-介绍/029a9211.jpeg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**