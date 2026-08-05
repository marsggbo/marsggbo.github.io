---
layout: post
title: "论文笔记系列-NAS With Reinforcement Learning"
date: 2019-01-08
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/54361495
related_posts: false
toc:
  sidebar: left
---

## **摘要**

神经网络在多个领域都取得了不错的成绩，但是神经网络的合理设计却是比较困难的。在本篇论文中，作者使用 **递归网络**去省城神经网络的模型描述，并且使用 **增强学习**训练RNN，以使得生成得到的模型在验证集上取得最大的准确率。

在 **CIFAR-10**数据集上，基于本文提出的方法生成的模型在测试集上得到结果优于目前人类设计的所有模型。测试集误差率为3.65%，比之前使用相似结构的最先进的模型结构还有低0.09%，速度快1.05倍。

在 **Penn Treebank**数据集上，根据本文算法得到的模型能够生成一个新颖的 **recurrent cell**,其要比广泛使用的 **LSTM cell**或者其他基线方法表现更好。在 **Penn Treebank**测试集上取得62.4的perplexity，比之前的最好方法还有优秀3.6perplexity。这个 **recurrent cell**也可以转移到PTB的字符语言建模任务中，实现1.214的perplexity。

## **1.介绍**

深度神经网络在许多具有挑战性的任务重都取得了不俗的成绩。在这成绩背后涉及到的技术则是从特征设计迁移到的结构设计，例如从SIFT、HOG(特征设计)到AlexNet、VGGNet、GoogleNet、ResNet等(结构设计)。

各种优秀的网络结构使得多种任务处理起来简单不少，但是设计网络结构仍然需要大量的专业知识并且需要耗费大量时间。

为了解决上述问题，本文提出 **Neural Architecture Search**，以期望找到合适的网络结构。大致原理图如下：

**RNN**作为一个 **controller**去生成模型的描述符，然后根据描述符得到模型，进而得到该模型在数据集上的准确度。接着将该准确度作为 \*\*奖励信号(reward signal)\*\*对controller进行更新。如此不断迭代找到合适的网络结构。

![](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/3ec91228.jpg)

## **2.相关工作**

**超参数优化**在机器学习中是个重要的研究话题，也被广泛使用。但是，该方法很难去生成一个长度可变的参数配置，即灵活性不高。虽然 **贝叶斯优化**可以搜索得到非固定长度的结构，但是与本文提出方法相比在通用性和可变性上都稍逊一筹。

**现代神经进化算法**虽然可以很灵活的生成模型，但是在大规模数据上实用性不高。

**program synthesis and inductive programming**的思想是searching a program from examples，**Neural Architecture Search**与其有一些相似的地方。

与本文方法相关的方法还有 **meta-learning**、**使用一个神经网络去学习用于其他网络的梯度下降更新(Andrychowicz et al., 2016)**、以及 **使用增强学习去找到用于其他网络的更新策略(Li & Malik, 2016)**

## **3.方法**

本节将从下面3个方面介绍所提出的方法： 1.介绍递归网络如何通过使用policy gradient method最大化生成框架的准确率 2.介绍几个改善方法，如skip connection(增加复杂度)、parameter server(加速训练)等 3.介绍如何生成递归

## **3.1 Generate Model Descriptions With A Controller Recurrent Neural Network**

用于生成模型描述的RNN结构如下，所生成的超参数是一系列的 **token**。

![](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/da78d899.jpg)

在实验中，如果层数超过一定数量，生成模型就会被停止。这种情况下，或者在收敛时，所生成模型在测试集上得到的准确率会被记录下来。

## **3.2 Training With Reinforcement**

RNN的参数用![θ_c](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/86925552.jpg)表示。controller所预测的一系列tokens记为一系列的actions，即![a_{1:T}](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/1453c855.jpg)，这些tokens是为了子网络(Child network)设计结构。子网络在验证集上得到的准确率用![R](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/cb0492f4.jpg)表示，该准确率作为 **reward signal**，并且会用到增强学习来训练controller。

通过求解最大化reward找到最优的结构，reward表达式如下：

![J(θ_c)=E_{P(a_{1:T;θ_c})}[R] \\](https://www.zhihu.com/equation?tex=J%28%CE%B8_c%29%3DE_%7BP%28a_%7B1%3AT%3B%CE%B8_c%7D%29%7D%5BR%5D+%5C%5C)

因为奖励信号![R](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/cb0492f4.jpg)是不可微分的，所以我们需要一个策略梯度方法来迭代更新![θ_c](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/86925552.jpg)。在本文中，使用到来自 **Williams (1992)** 的增强学习规则：

![\nabla_{θ_c}J(θ_c)=\sum_{t=1}^{T}E_{P(a_{1:T;θ_c})}[\nabla_{θ_c}logP(a_t|a_{(t-1):1};θ_c)R] \\](https://www.zhihu.com/equation?tex=%5Cnabla_%7B%CE%B8_c%7DJ%28%CE%B8_c%29%3D%5Csum_%7Bt%3D1%7D%5E%7BT%7DE_%7BP%28a_%7B1%3AT%3B%CE%B8_c%7D%29%7D%5B%5Cnabla_%7B%CE%B8_c%7DlogP%28a_t%7Ca_%7B%28t-1%29%3A1%7D%3B%CE%B8_c%29R%5D+%5C%5C)

根据经验上式约等于：

![\frac{1}{m} \sum_{k=1}^{m} \sum_{t=1}^{T} \nabla_{θ_c}logP(a_t|a_{(t-1):1};θ_c)R_k \\](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/73ca60b9.jpg)

其中![m](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/f79ab5c4.jpg)是controller在一个batch中采样得到的结构的数量，![T](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/51d9020f.jpg)是controller用于预测和设计神经网络结构的超参数的数量。

![R_k](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/097cd56d.jpg)表示第k个网络结构在验证集上的准确度。

上述的更新算法是对梯度的无偏估计，但是有很高的方差。为了降低方差，文中使用如下基线函数：

![\frac{1}{m} \sum_{k=1}^{m} \sum_{t=1}^{T} \nabla_{θ_c}logP(a_t|a_{(t-1):1};θ_c)(R_k-b) \\](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/04135eae.jpg)

只要![b](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/111cf5c2.jpg)不依懒于当前的action，那么其仍是无偏梯度估计，且![b](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/111cf5c2.jpg)是前面的结构准确率的 **指数平均数指标(Exponential Moving Average, EMA)**

> “ EMA（Exponential Moving Average）是指数平均数指标，它也是一种趋向类指标，指数平均数指标是以指数式递减加权的移动平均。  
>  其公式为： EMA\_{today}=α \* Price\_{today} + ( 1 - α ) \* EMA\_{yesterday};  
>  其中，α为平滑指数，一般取作2/(N+1)。

**Accelerate Training with Parallelism and Asynchronous Updates 使用并行算法和异步更新来加速训练**

每一次用于更新controller的参数![θ_c](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/86925552.jpg)的梯度都对应于一个子网络训练达到收敛。但是因为子网络众多，且每次训练收敛耗时长，所以使用 **分布式训练和异步参数更新**的方法来加速controller的学习速度。

![](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/b3005f7a.jpg)

训练模型如上图所示，一共有![S](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/e6217fb4.jpg)个 **Parameter Server**用于存储 ![K](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/645746d8.jpg)个 **Controller Replica**的共享参数。然后每个 **Controller Replica** 生成![m](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/f79ab5c4.jpg)个并行训练的自网络。

controller会根据![m](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/f79ab5c4.jpg)个子网络结构在收敛时得到的结果收集得到梯度值，然后为了更新所有 **Controller Replica**，会把梯度值传递给 **Parameter Server**。

在本文中，当训练迭代次数超过一定次数则认为子网络收敛。

## **3.3 Increase Architecture Complexity With Skip Connections And Other Layer Types**

3.1节中的示意图为了方便说明，所以其中的网络结构较为简单。本节则会介绍一种方法能够使得controller生成的网络结构假如 **skip connections(如ResNet结构)** 或者 **branching layers(层分叉，如GoogleNet结构)**。

为实现准确预测connections，本文采用了 **(Neelakantan et al., 2015)** 中的基于注意力机制的**set-selection type attention**方法。

在![N](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/39082460.jpg)层，根据sigmoid函数判断与其前面的![N-1](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/8cd7c442.jpg)个层是否相连。sigmoid函数如下：

![P(Layer\,j\,is\,an\,input\,to\,layer\,i)=sigmoid(v^T tanh(W_{prev}*h_j+W_{curr}*h_i)) \\](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/b7b44a2d.jpg)

上式中![h_j](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/5a32481a.jpg)表示controller在第![j](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/ebe2c52d.jpg)层的隐藏状态(![j](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/ebe2c52d.jpg)的大小是从0到![N-1](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/8cd7c442.jpg))。

![](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/5ed24d47.jpg)

下面介绍如何应对有的层可能没有输入或输出的情况：

1.如果没有输入，那么原始图像作为输入 2.在最后一层，将所有还没有connected层的输出concatenate起来作为输入。 3.如果需要concatenated的输入层有不同的size，那么小一点的层通过补0来保证一样大小

## **3.4 GENERATE RECURRENT CELL ARCHITECTURES**

下图展示了生成递归单元结构的具体细节。

由图可知采用了树结构来描述网络结构，这样也便于遍历每个节点。

每棵树由两个叶子节点(用0，1表示)和一个中间节点(用2表示)组成。

![](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/383bfeca.jpg)

## **4. 实验与结果**

具体的实验结果可查阅原论文 **NEURAL ARCHITECTURE SEARCH WITH REINFORCEMENT LEARNING**。

## **5.读后感**

**[【The First Step-by-Step Guide for Implementing Neural Architecture Search with Reinforcement Learning Using TensorFlow】](https://link.zhihu.com/?target=https%3A//lab.wallarm.com/the-first-step-by-step-guide-for-implementing-neural-architecture-search-with-reinforcement-99ade71b3d28)**这篇文章很详细的给出了如何实现NASnet的方法以及源代码，通过阅读代码能更好地理解本论文的思路。

NAS在生成网络的时候之前需要固定网络的结构，或者是说需要固定网络的层数。

以生成CNN网络为例，代码中默认最大层数参数**max\_layers=2**,当然也可以人为修改。

而controller其实就是一个RNN网络，其输出数据表示某一层中各个节点的参数，各个参数是按顺序输出的。例如代码中是按照 **[cnn\_filter\_size,cnn\_num\_filters,max\_pool\_ksize,cnn\_dropout\_rates]** 输出的(貌似并没有实现skip-connection)。

伪代码：

```
state = np.array([[10.0, 128.0, 1.0, 1.0]*max_layers], dtype=np.float32) # 初始化state
for episode in range(MAX_EPISODES):
	action = RLnet.get_action(state)  # 增强学习网络根据当前状态获取下一步的动作，其中是使用原论文所给的NAScell来对动作进行预测的。
	reward, pre_accuracy = net_manager.get_reward(action) # 根据生成的动作得到对应的网络，然后将该网络在训练集上训练至收敛，再将收敛后的网络在验证集上运行得到准确度，根据一定的准则将准确度转化为reward。
	reward = update(reward) # 更新reward
	state = update(action) # 根据action更新state，在例子中是state=action[0]
```

从上面的伪代码可以看出每次采样得到的模型都需要在训练集上训练到收敛，然后再根据在验证集上得到的reward更新。所以NAS其本质是在离散搜索空间进行搜索，而且网络拓扑结构是固定的，并且训练时间较长，不过思路比较简单好懂。

![](/assets/img/marsggbo/2019-01-08-论文笔记系列-NAS-With-Reinforcement-Learning/6a8988a9.jpg)

### **MARSGGBO♥原创** **2018-7-21**