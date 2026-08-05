---
layout: post
title: "强化学习1-基础概念(state,action,reward,policy)"
date: '2021-03-21'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/358787399
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/358787399>

> 该系列笔记基于 Wang Shusen老师的强化学习课程,课程内容深入浅出，建议有需要的同学学习。  
> 课程视频原链接为[https://www.youtube.com/playlist?list=PLvOO0btloRnsiqM72G4Uid0UWljikENlU](https://link.zhihu.com/?target=https%3A//www.youtube.com/playlist%3Flist%3DPLvOO0btloRnsiqM72G4Uid0UWljikENlU) （或知乎<https://www.zhihu.com/zvideo/1356919090918076416>）  
> 课程资源：[https://github.com/wangshusen/DRL](https://link.zhihu.com/?target=https%3A//github.com/wangshusen/DRL)

本节主要介绍常见而且**常忘记**的强化学习术语和概念。

## **State action policy reward trajectory**

下图展示了马里奥的游戏的某一时刻的截图，通过这个截图我们来学校以下几个术语定义：

* **state**：这个就表示$t$时刻马里奥的状态，用$s_t$表示。状态的定义方式也是可以由我们自己定义，比如可以是马里奥所在的位置，前后是否有怪兽等
* **action**：知道了状态后，我们需要做出决策或者说动作，即**action**,用$a_t$表示。在这个游戏里，动作有三个： up,left,right
* **policy**：这个表示决策策略$\pi(a_t|s_t)$，它会决定在$s_t$时刻，应该采取什么样的动作$a_t$。如果是均匀随机策略，那么就表示每个动作被选择的概率相等，即0.33。这当然不行，所以我们需要设计更好的策略。
* **reward**: 强化学习里另外一个很重要的概念就是**奖励（reward）**，用$r_t$表示。对应到马里奥游戏就是迟到的金币越多奖励越多。这个reward需要预先设置或者用其他方法定义，在这里你只需要知道在$s_t$，如果根据$\pi$采取$a_t$动作，那么得到的奖励就是$r_t$。也就是说不同的动作会对应不同的奖励

![](/assets/img/marsggbo/2021-03-21-强化学习1-基础概念stateactionrewardpolicy/624577c0.jpg)

总结起来，如果要用强化学习来玩马里奥游戏，其步骤就是首先得到$s_1$时刻的状态，然后根据策略$\pi$采取$a_t$动作，并且得到奖励$r1$，状态转移到$s_2$，继续重复上述步骤。最后我们得到了**轨迹（trajectory）**,即$s_1,a_1,r_1,s_2,a_2,r_2,...,s_T,a_T,r_T$

![](/assets/img/marsggbo/2021-03-21-强化学习1-基础概念stateactionrewardpolicy/2c34eab4.jpg)

## **Reward和Return**

* Reward表示每个时刻采取动作后得到的是即时奖励
* Return表示在$t$时刻采取某个动作$a_t$后到游戏结束可以得到的总的奖励，即$U_t=R_t+R_{t+1}+R_{t+2}...$

> “ 注意上面$U_t$的公式中的reward采用的是大写字母，因为它们表示的是随机变量，小写字母$r_t$表示的是在$s_t$时刻采取某个具体动作$a_t$后得到的具体的奖励值。  
>  ”

上面给的$U_t$计算公式是从$t$时刻开始未来每个时刻的奖励的累加，可以看到所有时刻的reward都是相同权重的。但是这样设计有一个问题，就是假如我现在给你100 和 1年后才给你100，这两个100显然不应该赋予相同权重，所以你经常可以看到return计算时会有一个参数$\gamma$,得到的是**discounted return**，即$U_t=R_t+\gamma R_{t+1}+\gamma ^2R_{t+2}...$

## **价值函数**

价值函数分为两种，一种是动作价值函数，另一种是状态价值函数。

## **动作价值函数 Q(s,a)**

前面说过了$U_t$只是一个随机变量，另外$R_t$由$(S_t,A_t)$决定，所以$U_t$依赖于从$t$时刻开始所有时间的动作($A_t,A_{t+1},A_{t+2},..$)和状态($S_t,S_{t+1},S_{t+2},...$)决定。

既然是一个随机变量，那么我们就可以求出该随机变量的期望。$U_t$的期望就是我们常说的动作价值函数 Q(s,a),有

![Q_\pi(s_t,a_t)=E[U_t|S_t=s_t,A_t=a_t] \\](https://www.zhihu.com/equation?tex=Q_%5Cpi%28s_t%2Ca_t%29%3DE%5BU_t%7CS_t%3Ds_t%2CA_t%3Da_t%5D+%5C%5C)

虽然$U_t$依赖未来所有时刻的$(A_i,S_i),i\in\{t,t+1,...\}$，但是在求具体的$S_t=s_t$时刻下，采取$A_t=a_t$的期望时，未来时刻的因子会通过积分（连续）或者求和（离散）等计算约掉，所以最后的$Q_\pi(s_t,a_t)$只依赖于当前时刻$s_t$和采取的具体动作$a_t$。

动作价值函数的意义就在于可以评估在$s_t$时刻采取$a_t$动作，从长远来看总的价值是多少。就假如我们计算出上图中，如果马里奥采取向上跳的动作得到的$Q(s_t,up)=10$，而向右移动的$Q(s_t,right)=-10$，那么我们就可以选择Q值最大所对应的动作，这样就可以找到Q值最大的策略了

$$
Q^{\star}\left(s_{t}, a_{t}\right)=\max _{\pi} Q_{\pi}\left(s_{t}, a_{t}\right) \\
$$

## **状态价值函数 V(s)**

前面介绍的$Q(s_t,a_t)$是在$s_t$时刻，**某一个**动作$a_t$的价值，那如果我们计算所有动作的价值的期望呢？即

![V_{\pi}\left(s_{t}\right)=\mathbb{E}_{A}\left[Q_{\pi}\left(s_{t}, A\right)\right] \\](https://www.zhihu.com/equation?tex=V_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%5Cright%29%3D%5Cmathbb%7BE%7D_%7BA%7D%5Cleft%5BQ_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%2C+A%5Cright%29%5Cright%5D+%5C%5C)

考虑动作是连续和离散两种情况计算公式如下：

![\begin{aligned} &V_{\pi}\left(s_{t}\right)=\mathbb{E}_{A}\left[Q_{\pi}\left(s_{t}, A\right)\right]=\sum_{a} \pi\left(a \mid s_{t}\right) \cdot Q_{\pi}\left(s_{t}, a\right) . \quad \text { (Actions are discrete.) }\\ &V_{\pi}\left(s_{t}\right)=\mathbb{E}_{A}\left[Q_{\pi}\left(s_{t}, A\right)\right]=\int \pi\left(a \mid s_{t}\right) \cdot Q_{\pi}\left(s_{t}, a\right) d a \cdot \text { (Actions are continuous.) } \end{aligned} \\](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%26V_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%5Cright%29%3D%5Cmathbb%7BE%7D_%7BA%7D%5Cleft%5BQ_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%2C+A%5Cright%29%5Cright%5D%3D%5Csum_%7Ba%7D+%5Cpi%5Cleft%28a+%5Cmid+s_%7Bt%7D%5Cright%29+%5Ccdot+Q_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%2C+a%5Cright%29+.+%5Cquad+%5Ctext+%7B+%28Actions+are+discrete.%29+%7D%5C%5C+%26V_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%5Cright%29%3D%5Cmathbb%7BE%7D_%7BA%7D%5Cleft%5BQ_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%2C+A%5Cright%29%5Cright%5D%3D%5Cint+%5Cpi%5Cleft%28a+%5Cmid+s_%7Bt%7D%5Cright%29+%5Ccdot+Q_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%2C+a%5Cright%29+d+a+%5Ccdot+%5Ctext+%7B+%28Actions+are+continuous.%29+%7D+%5Cend%7Baligned%7D+%5C%5C)

可以看到$V(s_t)$就表示在$s_t$时刻的价值，它把所有的动作都考虑了一遍，所以最后只与状态有关，所以它评估的是某一个状态的好坏。

直观地理解的话我们看上一张图片，此状态下马里奥可以选择向上和向左移动，假设价值是10； 而如果下一时刻调到了两个怪兽中间，因为这个时候除了向上移动，左右都不是太好，因为很可能会碰到怪兽，所以此时的价值可能只有3。（此处的价值大小只是举个例子帮助理解）。

## **小结**

价值函数小结：

![](/assets/img/marsggbo/2021-03-21-强化学习1-基础概念stateactionrewardpolicy/3973b4e2.jpg)

术语小结:

![](/assets/img/marsggbo/2021-03-21-强化学习1-基础概念stateactionrewardpolicy/878c3224.jpg)

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2021-03-21-强化学习1-基础概念stateactionrewardpolicy/029a9211.jpeg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2021-03-20 17:09:04**