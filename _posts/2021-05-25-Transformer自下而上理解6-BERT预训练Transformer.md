---
layout: post
title: "Transformer自下而上理解(6) BERT：预训练Transformer"
date: 2021-05-25
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/375151601
related_posts: false
toc:
  sidebar: left
---

> “ 本文参考Wang Shusen老师的教学视频：[https://www.youtube.com/watch?v=UlC6AjQWao8&list=PLvOO0btloRntpSWSxFbwPIjIum3Ub4GSC&index=3](https://link.zhihu.com/?target=https%3A//www.youtube.com/watch%3Fv%3DUlC6AjQWao8%26list%3DPLvOO0btloRntpSWSxFbwPIjIum3Ub4GSC%26index%3D3)  
>  ”

## **1. BERT**

上一篇文章留了一个引子，就是说Transformer的Encoder编码能力非常重要，而如何提高它的编码能力是一个值得研究的问题。

BERT，全称是**B**idirectional **E**ncoder **R**epresentations from **T**ransformers,就是用来提高Encoder编码能力的预训练方法。

BERT定义了如下两个任务来提高Encoder的编码能力：

1. 预测masked单词：具体来说就是我把任意一个句子中的某个单词隐藏后，Encoder应当能预测出这个单词原来是什么
2. 预测两个句子是否相邻：Encoder应当能够判断两个句子是不是相邻或相关的，比如前一句“我今天去上学”，后一句是“上官婉儿是最厉害的英雄”，显然如果Encoder编码能力强的话，应该能判断出这两个句子不相关。

下面我们看看BERT是如何实现这两个任务的。

## **2. Predict Masked Words**

假如原句子是"the cat sat on the mat",Encoder通过一系列计算会得到每个单词的映射结果，即![u_i](/assets/img/marsggbo/2021-05-25-Transformer自下而上理解6-BERT预训练Transformer/ca222c44.jpg)。

![](/assets/img/marsggbo/2021-05-25-Transformer自下而上理解6-BERT预训练Transformer/c966ab47.jpg)

BERT会随机mask掉某个单词，比如**cat**就被mask了，然后被替换成了特别的符号 **[MASK]**，这个符号有对应的embedding向量，同样地，它也有一个对应输出向量![u_M](/assets/img/marsggbo/2021-05-25-Transformer自下而上理解6-BERT预训练Transformer/0280c380.jpg)，之后这个向量会喂给一个分类器，之后分类器会输出一个分布向量![p](/assets/img/marsggbo/2021-05-25-Transformer自下而上理解6-BERT预训练Transformer/1dcb592d.jpg),真实的分布应该是**cat**的one-hot向量。我们可以用交叉熵来更新分类器和Encoder。

![](/assets/img/marsggbo/2021-05-25-Transformer自下而上理解6-BERT预训练Transformer/a0b33884.jpg)

## **3. Predict the Next Sentence**

和预测masked word一样，预测句子相关性不需要人工标注数据。原文章中的任意两句相邻句子就是正样本，随机采样的两个句子就是负样本。

整个流程如下图示，第一个句子之前可以插入一个特殊符号 **[CLS]**,第二个句子之前会插入另一个特殊符号 **[SEP]**。两个句子同时作为输入喂给Encoder。最后会再接一个二分类器用来判断两个句子是否相邻。

![](/assets/img/marsggbo/2021-05-25-Transformer自下而上理解6-BERT预训练Transformer/6bf9b5d9.jpg)

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2021-05-25-Transformer自下而上理解6-BERT预训练Transformer/029a9211.jpeg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2021-05-25 15:22:30**