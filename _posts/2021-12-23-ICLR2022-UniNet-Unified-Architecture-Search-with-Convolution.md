---
layout: post
title: ICLR2022 | UniNet: Unified Architecture Search with Convolution, Transformer, and MLP
  and MLP'
date: '2021-12-23'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/449177989
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/449177989>

## **1. 主要创新**

《UniNet: Unified Architecture Search with Convolution, Transformer, and MLP》是ICLR2022的一篇投稿论文，目前还没出结果。这里总结一下该工作的主要创新和贡献点：

1. UniNet是第一个将卷积，Transformer和MLP统一起来进行搜索的工作
2. 论文发现在融合上述三种不同模块的时候，传统的下采样模块（如stride=2的卷积）会成为模型性能的瓶颈。所以作者提出了 context-aware down-sampling modules (DSM), 包含 local-global-DSM (LG\_DSM) 和 global-DSM (G-DSM)
3. UniNet性能超过了efficientnet和SwinTransformer

## **2. 相关工作**

* Transformer：[ViT](https://link.zhihu.com/?target=https%3A//www.notion.so/Vision-Transformer-An-Image-is-Worth-16x16-Words-Transformers-for-Image-Recognition-at-Scale-55fee99eb3304874bc47c73b70e0aa77), [DeiT](https://link.zhihu.com/?target=https%3A//www.notion.so/Training-data-efficient-image-transformers-distillation-through-attention-DeiT-4c7d897e346245bc9ae6a256ad8a1374)
* MLP: [MLP-Mixer](https://link.zhihu.com/?target=https%3A//www.notion.so/MLP-Mixer-An-all-MLP-Architecture-for-Vision-bee3beaa4e01485e844355345899a3ce), [ResMLP](https://link.zhihu.com/?target=https%3A//www.notion.so/ResMLP-Feedforward-networks-for-image-classification-with-data-efficient-training-39be756bc46648198296460f9ace745c)
* 混合模型:

+ ConViT (ICML2021) [1] ：unify convolution and self-attention with gated positional self-attention (GPSA) and is more sample-efficient than self-attention.

![](/assets/img/marsggbo/2021-12-23-ICLR2022-UniNet-Unified-Architecture-Search-with-Convolution/7462847c.jpg)



+ CvT (ICCV2021) [2]：incorporate self-attention and convolution by generating Q, K, and V in self-attention with convolution.

![](/assets/img/marsggbo/2021-12-23-ICLR2022-UniNet-Unified-Architecture-Search-with-Convolution/2192751d.jpg)



+ CeiT [3]：replace the original patchy stem with convolutional stem and add depth-wise convolution to FFN layer, which obtains fast convergence and better performance.

![](/assets/img/marsggbo/2021-12-23-ICLR2022-UniNet-Unified-Architecture-Search-with-Convolution/b95006b3.jpg)

## **3. 方法**

## **3.1 Unified Search Space**

搜索空间如下图所示。

![](/assets/img/marsggbo/2021-12-23-ICLR2022-UniNet-Unified-Architecture-Search-with-Convolution/2b724157.jpg)

* GOP （General Operations）：包含 Convolution, transformer,和 MLP。三种操作都之采用了类似inverted residual的设计方式，即先把原来的通道数c 通过映射扩大ec，然后在通过映射还原为 c，各操作公式如下：

+ conv: $y=x+op(x),$ 其中 $op(x) = Proj_{ec\rightarrow{c}}(Conv(Proj_{c\rightarrow{ec}}(x)))$
+ transformer: $y=y'+FFN(y')$，其中 $y'=x+SelfAttention(x)$, $FFN(y')=Proj_{ec\rightarrow{c}}(Conv(Proj_{c\rightarrow{ec}}(y')))$
+ MLP: $y=y'+FFN(y')$，其中 $y'=x+MLP(x)$, $FFN(y')=Proj_{ec\rightarrow{c}}(Conv(Proj_{c\rightarrow{ec}}(y')))$

* DSM （Down-sampling Modules）

+ L-DSM: 这个就是常规的下采样模块，比如stride=2的卷积操作或者 max-pooling。这些是对local context做下采样
+ LG-DSM: 是局部和全局信息都会考虑的下采样模块。可以看到下采样是通过一个stride=2的卷积操作实现的，这里应该就是提取的local context。而attention机制呢就是一个全局的信息了。因为是2d卷积，所以需要先把input reshape成spatial grid，卷积计算完之后再flatten成原来的形状。
+ G-DSM：和LG-DSM的区别就是用了1d的卷积操作，但是论文里并没有解释为什么这个时候就不会保留local context了。

![](/assets/img/marsggbo/2021-12-23-ICLR2022-UniNet-Unified-Architecture-Search-with-Convolution/4e7484e2.jpg)

完整的搜索空间参数设置如下：

* GOP: { SA (self-attention), LSA (local SA), Conv, Depth-Wise Conv, MLP }
* e (通道数expansion比例)： { 2,3,4,5,6 }
* 模型是基于efficientnet搜索的，

+ 通道 channel缩放比例搜索空间： { 0.5, 0.75, 1.0, 1.25, 1.5 }
+ 堆叠个数 repeats： { -2, -1, 0, 1, 2 }
+ 总共有 K=5 个stages，每个stage的搜索空间大小是 （#channels \* #repeats） 5\*5=125

## **3.2 搜索算法**

搜索算法基于强化学习 PPO算法，所有候选操作借鉴 [Fnas: Uncertainty-aware fast neural architecture search](https://link.zhihu.com/?target=https%3A//www.notion.so/Fnas-Uncertainty-aware-fast-neural-architecture-search-8200bb38374b447bb7d676a667a23648) [4] 的做法把搜索空间映射成了一组tokens。论文也没有给很多细节，感兴趣的可以看看FNAS那篇论文。

## **4. 实验结果**

## **4.1 模型结构**

搜素到的模型结构如下表，可以看到每个stage的GOP都是固定的，这样一来搜索空间其实要少很多了，本来还以为是每一层的操作都要搜索。不过可以看到的是shallow stage采用的都是卷积操作，DSM也是传统的L-DSM（即stride=2的卷积层），可以理解成，浅层还是需要卷积来提取特征。

到了深层，GOP就选了Transformer和LG-DSM，可以理解成此时模型倾向于去提取全局的信息，这个也比较符合直觉。

不过有意思的是MLP貌似被遗忘在角落了。。。

![](/assets/img/marsggbo/2021-12-23-ICLR2022-UniNet-Unified-Architecture-Search-with-Convolution/9d894a03.jpg)

## **4.2 ImageNet上的结果**

结果一个字：好。就完事了hhh

![](/assets/img/marsggbo/2021-12-23-ICLR2022-UniNet-Unified-Architecture-Search-with-Convolution/efe68121.jpg)

## **5. Ablation Study**

作者做了实验去证明他们提出的 GOP和 DSM模块的有效性。

## **5.1 GOP vs. 纯Conv**

可以看到用GOP比纯Conv高了1.4个点。

![](/assets/img/marsggbo/2021-12-23-ICLR2022-UniNet-Unified-Architecture-Search-with-Convolution/2b18a228.jpg)

## **5.2 不同DSM模块的有效性验证和迁移性实验**

Table 6 对比了将 DSM全都替换成某一种下采样模块后模型的性能变化，可以看到如果替换成 G-DSM后性能掉的最多，而LG-DSM性能保持的还不错，但是FLOPs和参数量都有一定的增加。L-DSM效果也还不错的亚子

![](/assets/img/marsggbo/2021-12-23-ICLR2022-UniNet-Unified-Architecture-Search-with-Convolution/c030d167.jpg)

作者还将DSM模块放到了Swin-Transformer (ST)和PVT-Tiny 这些模型上去，这两个模型都是总共由4个stage组成，所以作者把前两个stage的下采样模块替换成了L-DSM，最后两个stage替换成了 LG-DSM，结果如Table 7所示，可以看到都有一定的性能提升。

![](/assets/img/marsggbo/2021-12-23-ICLR2022-UniNet-Unified-Architecture-Search-with-Convolution/4fe38b55.jpg)

## 个人看法

这篇论文标题是设计了一个把Convolution,Transformer和MLP一统江湖的搜索空间，但是仔细看了论文后感觉就是把每个stage的op换成了对应的操作而已，而且每个stage只是使用某一种操作，感觉标题略微优点标题党，不够也可以理解哈哈​。当然如何更加高效细腻度地基于三种OP进行搜索​也是一个值得探索的问题。

## **References**

* [1] St´ephane d’Ascoli, Hugo Touvron, Matthew Leavitt, Ari Morcos, Giulio Biroli, and Levent Sagun. [ConViT: Improving vision transformers with soft convolutional inductive biases](https://link.zhihu.com/?target=https%3A//www.notion.so/ConViT-Improving-vision-transformers-with-soft-convolutional-inductive-biases-d10425c9db5a479eafb0c5fedb6f339e). arXiv preprint arXiv:2103.10697, 2021.
* [2] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. [CvT: Introducing convolutions to vision transformers.](https://link.zhihu.com/?target=https%3A//www.notion.so/CvT-Introducing-convolutions-to-vision-transformers-fe917e96f329448f9d8522b6e4b6622d) arXiv preprint arXiv:2103.15808, 2021.
* [3] Kun Yuan, Shaopeng Guo, Ziwei Liu, Aojun Zhou, Fengwei Yu, and Wei Wu. Incorporating convolution designs into visual transformers. arXiv preprint arXiv:2103.11816, 2021.
* [4] Jihao Liu, Ming Zhang, Yangting Sun, Boxiao Liu, Guanglu Song, Yu Liu, and Hongsheng Li. Fnas: Uncertainty-aware fast neural architecture search. arXiv preprint arXiv:2105.11694, 2021a.

### **微信公众号：AutoML机器学习**

http://weixin.qq.com/r/HD8gOHzEmiHlrThb92oO (二维码自动识别)

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**