---
layout: post
title: "强化学习3-Policy gradient"
date: 2021-03-23
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/359282662
related_posts: false
toc:
  sidebar: left
---

## **1. Policy function**

策略函数 ![\pi(a|s)](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/3e99cc91.jpg)是一个概率密度函数(probability density function, PDF)，状态![s](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/4aa2212c.jpg)作为输入，输出是不同动作的概率。比如前面文章里的马里奥游戏中，一共有三个可选动作，left,right,up，这三个动作的概率之和为1，即 ![\sum_{a\in A}\pi(a|s), A=\{left,right,up\}](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/96620627.jpg)。

![](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/f7269da8.jpg)

## **2. Policy Network**

## **2.1 State-Value Function**

![](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/5677b844.jpg)

在介绍Policy Network之前，我们先回顾一下状态价值函数的定义，如上图示。可以知道状态价值函数只与状态![s_t](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/6f1b42b4.jpg)有关，因为它是所有动作价值的的加权求和，权重就是每个动作的概率![\pi(a|s_t)](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/f51d2512.jpg)。

![V_{\pi}\left(s_{t}\right)=\mathbb{E}_{A}\left[Q_{\pi}\left(s_{t}, A\right)\right]=\sum_{a} \pi\left(a \mid s_{t}\right) \cdot Q_{\pi}\left(s_{t}, a\right) \\](https://www.zhihu.com/equation?tex=V_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%5Cright%29%3D%5Cmathbb%7BE%7D_%7BA%7D%5Cleft%5BQ_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%2C+A%5Cright%29%5Cright%5D%3D%5Csum_%7Ba%7D+%5Cpi%5Cleft%28a+%5Cmid+s_%7Bt%7D%5Cright%29+%5Ccdot+Q_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%2C+a%5Cright%29+%5C%5C)

## **2.2 Policy Gradient**

但是每个动作的概率我们通常是不知道的，所以我们可以用policy network![\pi(a|s_t;\theta)](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/ee4506ce.jpg)去预测每个动作的概率。那我们该如何更新这个网络呢？

很显然我们知道最终的目的是想要最大化所有状态的价值，即最大化![J(\theta)](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/1547a1c9.jpg)

![J(\boldsymbol{\theta})=\mathbb{E}_{S}[V(S ; \boldsymbol{\theta})] \\](https://www.zhihu.com/equation?tex=J%28%5Cboldsymbol%7B%5Ctheta%7D%29%3D%5Cmathbb%7BE%7D_%7BS%7D%5BV%28S+%3B+%5Cboldsymbol%7B%5Ctheta%7D%29%5D+%5C%5C)

所以只要求出![J(\theta)](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/1547a1c9.jpg)对![\theta](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/679f2e1d.jpg)的梯度后，用**梯度上升**对![\theta](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/679f2e1d.jpg)进行更新,即

![\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \frac{\partial J(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \\](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/d7022597.jpg)

但是通常我们是无法知道所有时刻的状态价值的，所以一种近似的更新方法就是用每一个时刻的状态价值进行更新，即

![\boldsymbol{\theta} \leftarrow \boldsymbol{\theta}+\beta \cdot \frac{\partial V(s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \\](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/db300cd3.jpg)

![\frac{\partial V(s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}}](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/46abfae1.jpg)的推导如下

![\begin{aligned} \frac{\partial V(s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} &=\frac{\partial \sum_{a} \pi(a \mid s ; \boldsymbol{\theta}) \cdot Q_{\pi}(s, a)}{\partial \boldsymbol{\theta}}  \\ &=\sum_{a} \frac{\partial \pi(a \mid s ; \boldsymbol{\theta}) \cdot Q_{\pi}(s, a)}{\partial \boldsymbol{\theta}} \\ &=\sum_{a} \frac{\partial \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(s, a) \\ &=\sum_{a} \pi\left(\left.a\right|_{S} ; \boldsymbol{\theta}\right) \cdot \frac{\partial \log \pi(a \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(s, a)\\ &=\mathbb{E}_{A}\left[\left(\frac{\partial \log \pi(A \mid s ; \boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \cdot Q_{\pi}(s, A)\right)\right] \end{aligned}\tag{1} ](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Cfrac%7B%5Cpartial+V%28s+%3B+%5Cboldsymbol%7B%5Ctheta%7D%29%7D%7B%5Cpartial+%5Cboldsymbol%7B%5Ctheta%7D%7D+%26%3D%5Cfrac%7B%5Cpartial+%5Csum_%7Ba%7D+%5Cpi%28a+%5Cmid+s+%3B+%5Cboldsymbol%7B%5Ctheta%7D%29+%5Ccdot+Q_%7B%5Cpi%7D%28s%2C+a%29%7D%7B%5Cpartial+%5Cboldsymbol%7B%5Ctheta%7D%7D++%5C%5C+%26%3D%5Csum_%7Ba%7D+%5Cfrac%7B%5Cpartial+%5Cpi%28a+%5Cmid+s+%3B+%5Cboldsymbol%7B%5Ctheta%7D%29+%5Ccdot+Q_%7B%5Cpi%7D%28s%2C+a%29%7D%7B%5Cpartial+%5Cboldsymbol%7B%5Ctheta%7D%7D+%5C%5C+%26%3D%5Csum_%7Ba%7D+%5Cfrac%7B%5Cpartial+%5Cpi%28a+%5Cmid+s+%3B+%5Cboldsymbol%7B%5Ctheta%7D%29%7D%7B%5Cpartial+%5Cboldsymbol%7B%5Ctheta%7D%7D+%5Ccdot+Q_%7B%5Cpi%7D%28s%2C+a%29+%5C%5C+%26%3D%5Csum_%7Ba%7D+%5Cpi%5Cleft%28%5Cleft.a%5Cright%7C_%7BS%7D+%3B+%5Cboldsymbol%7B%5Ctheta%7D%5Cright%29+%5Ccdot+%5Cfrac%7B%5Cpartial+%5Clog+%5Cpi%28a+%5Cmid+s+%3B+%5Cboldsymbol%7B%5Ctheta%7D%29%7D%7B%5Cpartial+%5Cboldsymbol%7B%5Ctheta%7D%7D+%5Ccdot+Q_%7B%5Cpi%7D%28s%2C+a%29%5C%5C+%26%3D%5Cmathbb%7BE%7D_%7BA%7D%5Cleft%5B%5Cleft%28%5Cfrac%7B%5Cpartial+%5Clog+%5Cpi%28A+%5Cmid+s+%3B+%5Cboldsymbol%7B%5Ctheta%7D%29%7D%7B%5Cpartial+%5Cboldsymbol%7B%5Ctheta%7D%7D+%5Ccdot+Q_%7B%5Cpi%7D%28s%2C+A%29%5Cright%29%5Cright%5D+%5Cend%7Baligned%7D%5Ctag%7B1%7D+)

> “ 注意：为了方便理解和简化推导，我们假设公式(1)中![Q_{\pi}](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/3ec4abd8.jpg)与网络权重![\theta](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/679f2e1d.jpg)无关。但是这个假设是不对的，因为![Q](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/7eb07b99.jpg)与![\pi(\theta)](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/b4dd5bb3.jpg)有关。不过这个假设不影响最终的结论。  
>  ”

公式（1）推导解释：

* 第一行到第二行就是把偏微分移到了求和符号内
* 第三行：因为我们假设了![Q_{\pi}](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/3ec4abd8.jpg)与网络权重![\theta](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/679f2e1d.jpg)无关，所以![Q_{\pi}](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/3ec4abd8.jpg)可以单独提出来
* 第四行：我们把![\pi(a|s;\theta)](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/7802d48e.jpg)看成几个整体，记为![z](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/524eb39a.jpg)，另外需要用到这个推导技巧，即![z\cdot  \frac{\partial{log z}}{\partial{x}}=z \cdot \frac{1}{z} \frac{\partial{z}}{\partial{x}}=\frac{\partial{z}}{\partial{x}}](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/fa3cc995.jpg)
* 第五行：因为![\pi(a|s;\theta)](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/7802d48e.jpg)是动作的概率分布，简单理解就是每个动作的概率，那么第四行就等价于求期望

总结起来，policy gradient的计算有两种形式，如下：

![](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/091cdc60.jpg)

* 离散

如果动作是离散的就可以使用第一种计算形式

![](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/ce990ec0.jpg)

* 连续

当动作是连续的，我们则需要通过Monte Carlo采样来近似计算，即每次从策略中随机抽取某个动作，并计算该动作对应的梯度，然后用这个梯度对网络进行更新

![](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/f7fe93ec.jpg)

## **2.3 使用policy gradient更新policy network**

![](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/aa8a888c.jpg)

更新policy network的完整步骤如上图所示：

1. 首先我们观察得到某一时刻的状态![s_t](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/6f1b42b4.jpg)
2. 根据policy network ![\pi(\cdot|s_t;\theta_t)](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/229e6ea1.jpg)随机采样某一个动作![a_t](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/b303552b.jpg)
3. 计算该采样动作的价值![q_t \approx Q_\pi(s_t,a_t)](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/9b1a197f.jpg)
4. 根据上一小节计算policy gradient
5. 更新policy network

第三步骤中计算动作的价值有两种方法：

* **Reinforce**

以马里奥游戏为例，我们把每一局最开始到结束（即马里奥撕掉，比如碰到怪兽，或者掉进陷阱里）的状态，动作和奖励都记录下来，这样就得到了每一局的trajectory，即

![trajectory=\{s_{1}, a_{1}, r_{1}, s_{2}, a_{2}, r_{2}, \cdots, s_{T}, a_{T}, r_{T}\} \\](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/b2b6bf0e.jpg)

有了这个之后我们就可以求出这一局的![u_t=\sum_{k=t}^T\gamma^{k-t}r_k](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/bb6f7ebd.jpg)

另外由于![Q_{\pi}\left(s_{t}, a_{t}\right)=\mathbb{E}\left[U_{t}\right]](https://www.zhihu.com/equation?tex=Q_%7B%5Cpi%7D%5Cleft%28s_%7Bt%7D%2C+a_%7Bt%7D%5Cright%29%3D%5Cmathbb%7BE%7D%5Cleft%5BU_%7Bt%7D%5Cright%5D)，所以我们用![u_t](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/6a70c3ac.jpg)来近似![Q_{\pi}\left(s_{t}, a_{t}\right)](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/a6a0f8da.jpg)，即![q_t=u_t](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/3195afc9.jpg)

* **Actor-Critic**

另一种方法就是actor-critic方法，即用一个神经网络（比如![DQN](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/7cf08367.jpg)）来预测某个动作的价值，而本节介绍的policy network就用来在不同状态下选择不同的动作。

简单类比一下，policy network就像是一个运动员，它会在不同时刻执行不同的动作，而DQN网络就会对运动员的动作进行打分。具体的方法会在下一篇文章中介绍。

## **总结**

![](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/a092c1e2.jpg)

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2021-03-23-强化学习3-Policy-gradient/029a9211.jpeg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2021-03-23 11:37:19**