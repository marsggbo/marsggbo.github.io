---
layout: post
title: "【论文笔记系列】AutoML：A Survey of State-of-the-art （上）"
date: '2020-02-11'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/106205363
related_posts: false
toc:
  sidebar: left
---

> 之前已经发过一篇文章来介绍我写的**[AutoML综述](https://zhuanlan.zhihu.com/p/77417817)**，最近把文章内容做了更新，所以这篇稍微细致地介绍一下。由于篇幅有限，下面介绍的方法中涉及到的细节感兴趣的可以移步到论文中查看。 论文地址：[https://arxiv.org/abs/1908.00709](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1908.00709)  
>    
> 【论文笔记系列】AutoML：A Survey of State-of-the-art （下）<https://zhuanlan.zhihu.com/p/106224268>

## **1. Introduction**

以往的模型都是靠大佬们不断试错和调参炼丹炼出来的，而且不同场景或者不同类型的数据集又得设计不同的网络模型，而我等穷&菜鸡在**设计模型的天赋**和**计算资源**上都比不过大佬们。幸运的是终于有体恤民意的大佬提出了**Neural Architecture Search with Reinforcement Learning**这篇论文,它旨在让算法自己针对不同的数据找到合适的模型，实验结果表明算法找到的模型可以和大佬们设计的模型媲美，这篇论文也让NAS技术成为当今的一个热点研究问题。而AutoML则是进一步希望机器学习的pipeline（如下图）完全实现自动化。

![](/assets/img/marsggbo/2020-02-11-论文笔记系列AutoMLA-Survey-of-State-of-the-art-上/c4a69a5d.jpg)

可以看到一个完整的pipeline包括了数据准备（data preparation），特征工程（feature engineering），模型生成（model generation），模型评估（model estimation）。以上pipeline中的每个流程其实都可以写一篇综述，但是该论文是希望让大家能对AutoML整个流程有一个全面的了解，所以会对选择各个流程中会比较经典和具有代表性的算法进行介绍。

相信你也看到了已经有好几篇关于AutoML的综述，本篇综述与它们的区别在于本篇论文是第一篇从pipeline角度来进行介绍的，这样能让大家从一个全局的角度来了解和认识AutoML，而不会陷入某个技术脱离了轨道。另外我们还特意将**数据准备**纳入了AutoML的pipeline中，目的也是希望我们以后能够从繁琐的数据收集、整理过程中摆脱出来。

另外，由于NAS是AutoML中最热门的一个子话题，所以我们还单独写了一节来总结了目前的各种NAS算法比较，同时也基于CIFAR10和PTB两个数据集，将NAS生成的模型和人类设计的模型做了比较。

最后我们提出了几个开放性问题以及未来的可行的研究方向。

## **2. Data Preparation**

数据准备可以细分为两个步骤：**Data Collection**和**Data Cleaning**。

## **2.1 Data Collection**

最简单的数据收集其实就是用目前开源的数据集，比如说分类中常用的就是CIFAR10，ImageNet这种，当然我们也可以用搜索引擎来找我们需要的数据集：

最简单的数据收集其实就是用目前开源的数据集，比如说分类中常用的就是CIFAR10，ImageNet这种，当然我们也可以用搜索引擎来找我们需要的数据集：

[Kaggle: Your Machine Learning and Data Science Community](https://link.zhihu.com/?target=https%3A//www.kaggle.com/)[Google Dataset Search](https://link.zhihu.com/?target=https%3A//toolbox.google.com/datasetsearch)[Elsevier Data Search](https://link.zhihu.com/?target=https%3A//www.datasearch.elsevier.com/)

### **2.1.1 Data Synthesis**

当然其实很多时候你想要的并不一定搜得到，而你看着你手上那少得可怜的数据集暗自呻吟。此时**Data Synthesis**神仙下凡来到你身边，告诉你如下几种办法来扩充你现有的数据集：

第一种方法是**数据增强(data augmentation)**，这个在处理图像的时候用的特别多，比如**torchvision**，**Augmentor**这些python库都提供了各种数据增强操作。当然数据增强操作有很多种，通过组合又可以得到更多种，完全靠人工设计数据增强操作会发疯的，所以谷歌，没错有时谷歌率先提出了AutoAugment【1】，他们在CIFAR10， ImageNet等数据集上找到了各种数据增强策略（即一组增强操作的组合），而后又有人提出了FastAugment【2】，来进一步提高搜索效率。NLP也有数据增强，具体的可以阅读论文。

第二种方法是使用**数据模拟器**，这种是用于像自动驾驶这种场景，因为我们不可能真的在真实世界中做测试。比较流行的模拟器有OpenAI Gym，Unreal Engine。我们实验室的大佬们就用Unreal Engine生成了一个用于视差和表面法向估计的大型合成室内机器人立体数据集（IRS dataset）【3】。

第三种就是使用现在很火的**GAN**技术来生成数据。

### **2.1.2 Data Searching**

前面提到了几个搜索引擎可以搜索别人开源的数据，但是有时候我们也可以用通过Google或者爬虫的方式来爬取数据，但是这样会遇到若干种问题。第一种问题是搜索结果和搜索关键字不匹配，例如你想搜索水果-Apple，但是搜出来的结果可能包含伟大的苹果公司的logo以及他们的产品。还有一个问题就是网络数据的分布和你想构造的数据集分布可能差别很大，比如说A疾病的图像随便一搜到处都是，而B疾病的图像可能寥寥无几，这样搜索出来的数据集会非常的imbalanced。而如何解决这些问题欢迎阅读论文，这里不做展开。

## **2.2 Data Cleaning**

这里主要介绍了需要做数据清洗的情况，另外也介绍了一些自动化清洗的算法，同样不作展开，具体详见论文。

## **3. Feature Engineering**

这一章主要介绍了特征工程的三个方法：feature selection，feature construction和feature extraction。这里更多的是基于传统机器学习方法介绍的，这里不作展开，具体详见论文。

## **4. Model Generation**

> 为方便说明把最开始的pipeline图再搬过来。

![](/assets/img/marsggbo/2020-02-11-论文笔记系列AutoMLA-Survey-of-State-of-the-art-上/c4a69a5d.jpg)

模型生成（model generation）可以分为模型选择和超参数优化两部分。如上图示模型选择又可以划分为两种，一种是对传统的模型做选择，比如是选择SVM啊还是选择KNN等。我们会更加关注第二种，即NAS技术。可以看到NAS技术横跨三个流程，文本会从model structure和hyperparameters optimization（HPO）以及model estimation三个方面进行介绍，其中model estimation会单独做一章进行介绍。

另外这里的HPO和以往我们说的超参数优化本质上是一样的，即都是优化问题，但是不同的是这里还包括了对模型结构进行优化。

## **4.1 Model Structure**

这一节会以一种非常直观和好理解的方式进行介绍，不同模型结构之间的差异和优劣涉及到的比较多的细节，所以感兴趣的还是请移步到论文。

### **4.1.1 Entire structure**

Entire structure简单理解就是我们假设模型深度是$L$层,然后每一层被指定一个操作，上层可以连接到其后所有层，如下图示：左边是最简单设计方法，右边则加入了skip-connection

![](/assets/img/marsggbo/2020-02-11-论文笔记系列AutoMLA-Survey-of-State-of-the-art-上/886d56a1.jpg)

### **4.1.2 Cell-based Structure**

这一结构的提出是为了解决Entire structure的缺点：搜索非常耗时。

示意图如下，可以看到最终的模型是由两种cell组成的：normal 和 reduction cell。两者其实是差不多的，只不过前者的输入和输出大小，维度都保持一致，而reduction会把特征图大小减半，通道数翻倍。

![](/assets/img/marsggbo/2020-02-11-论文笔记系列AutoMLA-Survey-of-State-of-the-art-上/89612d4c.jpg)

cell内部结构如上图右边所示，很多论文后面都是参考了这种设计方式：

一个cell由$B$个block组成，一个block又由2个node组成，可以看到：

* 每个node会被分配一个operation（如conv 3x3等），这个operation是从预定义的search space中选择的，常见的主要就是卷积操作和池化操作，或者一个类似于ResBlock的自定义的模块。
* Node还被分配一个输入数据，**可以看到block是有序号的**，也就是说block1可以接收block 0的输出，但是不能接收block 2.另外block 0只能接收前面两个cell的输出作为输入。

上面介绍的normal-reduction方式有一个问题就是特征图的大小和通道数变化是固定的，所以在Auto-Deeplab【4】中将这个也纳入了搜索过程，使得模型更加灵活。

![](/assets/img/marsggbo/2020-02-11-论文笔记系列AutoMLA-Survey-of-State-of-the-art-上/a9e85a5f.jpg)

normal-reduction方式的另一个问题是每一层的cell结构是固定的，ProxylessNAS【5】通过使用BinaryConnection方法使得每层的cell结构可以不一样。

### **4.1.3 Progressive Structure**

前面的Cell-based structure中cell内部的block数量需要预先设定好，而且在找到一个合适的cell结构后，通过堆叠固定数量的cell构造最后的网络模型。所以有一些论文则通过渐进式的方法来构造网络结构。

这篇论文【6】提出了Hierarchical structure，示意图如下：

![](/assets/img/marsggbo/2020-02-11-论文笔记系列AutoMLA-Survey-of-State-of-the-art-上/8858cd8a.jpg)

可以看到这里的cell是有不同的level的，level-1就是最基本的操作，例如3x3卷积等，第二层就是用第一层的cell连接而成，更高层的cell类似。

P-DARTS则是从layer level的角度来实现渐进式，如下图示。可以看到之前的方法DARTS先是在训练集上搜索一个网络，这个网络由8个cell组成，搜索完毕后，直接堆叠更多的cell，然后直接训练这个更大的网络。而P-DARTS则是不断增加cell的数量。这里面还涉及到不少细节，更多的可以参阅论文。

![](/assets/img/marsggbo/2020-02-11-论文笔记系列AutoMLA-Survey-of-State-of-the-art-上/b4ed609a.jpg)

### **4.1.4 Network Morphism based Structure (NM)**

牛顿曾经说过：“if I have seen further it is by standing on the shoulders of Giants"。Net2Net【7】则是使用了这个思想，它可以直接基于现有的模型进行修改，使得模型变得更宽更深从而提高模型的性能。示意图如下,使得模型更深的方法其实就是插入一个等幂操作，也就是蓝色的结构，用数学公式表示就等价于$f(x)=x$，乍看起来加了等于没加，但是后面参数会不断更新的呀，而且选择等幂操作也是为了保证模型至少不会比原来的差。

![](/assets/img/marsggbo/2020-02-11-论文笔记系列AutoMLA-Survey-of-State-of-the-art-上/f57d152d.jpg)

让模型变得更宽其实就是在某一层插入新的节点，这个节点其实是前一个节点的复制，所以通道数会被减半，这样也是为了等幂。

而后微软团队正式提出了**Network Morphism (NM)** 这一概念，相比于Net2Net，它的改进的地方有两点：1）可以插入non-identity的层，而且可以处理任意的非线性激活函数；2）Net2Net只能单独地执行**加深**或者**加宽**操作，而NM可以同时对深度，宽度，和卷积核大小做修改。

谷歌团队后面提出的EfficientNet【8】发现如果能很好地平衡模型深度，宽度以及分辨率的话可以有效提高模型性能。

## **参考文献**

* 【1】Autoaugment: Learning augmentation strategies from data
* 【2】Fast autoaugment
* 【3】Q. Wang, S. Zheng, Q. Yan, F. Deng, K. Zhao, and X. Chu, “Irs: A large synthetic indoor robotics stereo dataset for disparity and surface normal estimation,” arXiv preprint arXiv:1912.09678, 2019.
* 【4】Auto-deeplab: Hierarchical neural architecture search for semantic image segmentation
* 【5】Proxylessnas: Direct neural architecture search on target task and hardware
* 【6】Hierarchical representations for efficient architecture search
* 【7】Net2net: Accelerating learning via knowledge transfer
* 【8】Efficientnet: Rethinking model scaling for convolutional neural networks

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-02-11-论文笔记系列AutoMLA-Survey-of-State-of-the-art-上/ba7127e7.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
  
  
**2020-02-10 21:44:34**
