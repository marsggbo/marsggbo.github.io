---
layout: post
title: "Pytorch autograd,backward详解"
date: 2019-09-19
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/83172023
related_posts: false
toc:
  sidebar: left
---

平常都是无脑使用backward，每次看到别人的代码里使用诸如autograd.grad这种方法的时候就有点抵触，今天花了点时间了解了一下原理，写下笔记以供以后参考。**以下笔记基于Pytorch1.0**

## **Tensor**

Pytorch中所有的计算其实都可以回归到Tensor上，所以有必要重新认识一下Tensor。如果我们需要计算某个Tensor的导数，那么我们需要设置其`.requires_grad`属性为`True`。为方便说明，在本文中对于这种我们自己定义的变量，我们称之为**叶子节点(leaf nodes)**，而基于叶子节点得到的中间或最终变量则可称之为**结果节点**。例如下面例子中的`x`则是叶子节点，`y`则是结果节点。

```
x = torch.rand(3, requires_grad=True)
y = x**2
z = x + x
```

另外一个Tensor中通常会记录如下图中所示的属性：

* `data`: 即存储的数据信息
* `requires_grad`: 设置为`True`则表示该Tensor需要求导
* `grad`: 该Tensor的梯度值，每次在计算backward时都需要将前一时刻的梯度归零，否则梯度值会一直累加，这个会在后面讲到。
* `grad_fn`: 叶子节点通常为None，只有结果节点的grad\_fn才有效，用于指示梯度函数是哪种类型。例如上面示例代码中的`y.grad_fn=<PowBackward0 at 0x213550af048>, z.grad_fn=<AddBackward0 at 0x2135df11be0>`
* `is_leaf`: 用来指示该Tensor是否是叶子节点。

![](/assets/img/marsggbo/2019-09-19-Pytorch-autogradbackward详解/7121a6db.jpg)

*\*图片出处：[PyTorch Autograd]([https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95](https://link.zhihu.com/?target=https%3A//towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95))\**

*上图中的z.is\_leaf应该是False。原图作者应该是画错了。*

## **torch.autograd.backward**

有如下代码：

```
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
z.backward()
print(z, x.grad, y.grad)

>>> tensor(3., grad_fn=<AddBackward0>) tensor(2.) tensor(1.)
```

可以z是一个标量，当调用它的backward方法后会根据链式法则自动计算出叶子节点的梯度值。

但是如果遇到z是一个向量或者是一个矩阵的情况，这个时候又该怎么计算梯度呢？这种情况我们需要定义`grad_tensor`来计算矩阵的梯度。在介绍为什么使用之前我们先看一下源代码中backward的接口是如何定义的：

```
torch.autograd.backward(
		tensors, 
		grad_tensors=None, 
		retain_graph=None, 
		create_graph=False, 
		grad_variables=None)
```

* `tensor`: 用于计算梯度的tensor。也就是说这两种方式是等价的：`torch.autograd.backward(z) == z.backward()`
* `grad_tensors`: 在计算矩阵的梯度时会用到。他其实也是一个tensor，shape一般需要和前面的`tensor`保持一致。
* `retain_graph`: 通常在调用一次backward后，pytorch会自动把计算图销毁，所以要想对某个变量重复调用backward，则需要将该参数设置为`True`
* `create_graph`: 当设置为`True`的时候可以用来计算更高阶的梯度
* `grad_variables`: 这个官方说法是grad\_variables' is deprecated. Use 'grad\_tensors' instead.也就是说这个参数后面版本中应该会丢弃，直接使用`grad_tensors`就好了。

好了，参数大致作用都介绍了，下面我们看看pytorch为什么设计了`grad_tensors`这么一个参数，以及它有什么用呢？

还是用代码做个示例

```
x = torch.ones(2,requires_grad=True)
z = x + 2
z.backward()

>>> ...
RuntimeError: grad can be implicitly created only for scalar outputs
```

当我们运行上面的代码的话会报错，报错信息为**RuntimeError: grad can be implicitly created only for scalar outputs**。

上面的报错信息意思是只有对标量输出它才会计算梯度，而求一个矩阵对另一矩阵的导数束手无策。

![X = \left[\begin{array}{cc} x_0 & x_1 \ \end{array}\right] \,\,\,\,\,\,\,\,\,\ Z=X+2=\left[\begin{array}{cc} x_0+2 & x_1+2 \ \end{array}\right] \Rightarrow \frac{\partial{Z}}{\partial{X}}=?](https://www.zhihu.com/equation?tex=X+%3D+%5Cleft%5B%5Cbegin%7Barray%7D%7Bcc%7D+x_0+%26+x_1+%5C+%5Cend%7Barray%7D%5Cright%5D+%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C%2C%5C+Z%3DX%2B2%3D%5Cleft%5B%5Cbegin%7Barray%7D%7Bcc%7D+x_0%2B2+%26+x_1%2B2+%5C+%5Cend%7Barray%7D%5Cright%5D+%5CRightarrow+%5Cfrac%7B%5Cpartial%7BZ%7D%7D%7B%5Cpartial%7BX%7D%7D%3D%3F)

那么我们只要想办法把矩阵转变成一个标量不就好了？比如我们可以对z求和，然后用求和得到的标量在对x求导，这样不会对结果有影响，例如：

![\begin{align} &Z_{sum}=\sum{z_i}=x_0+x_1+4 \notag \ &\text{then} \,\,\,\,\,  \frac{\partial{Z{sum}}}{\partial{x_0}}=\frac{\partial{Z{sum}}}{\partial{x_1}}=1 \notag \end{align}](/assets/img/marsggbo/2019-09-19-Pytorch-autogradbackward详解/09b31bfe.jpg)

我们可以看到对z求和后再计算梯度没有报错，结果也与预期一样：

```
x = torch.ones(2,requires_grad=True)
z = x + 2
z.sum().backward()
print(x.grad)

>>> tensor([1., 1.])
```

我们再仔细想想，对z求和不就是等价于z 点乘一个相同维度的全为1的矩阵吗？即 ![sum(Z)=dot(Z,I)](/assets/img/marsggbo/2019-09-19-Pytorch-autogradbackward详解/99f9c502.jpg) ,而这个I也就是我们需要传入的`grad_tensors`参数。(点乘只是相对于一维向量而言的，对于矩阵或更高为的张量，可以看做是对每一个维度做点乘)

代码如下：

```
x = torch.ones(2,requires_grad=True)
z = x + 2
z.backward(torch.ones_like(z)) # grad_tensors需要与输入tensor大小一致
print(x.grad)

>>> tensor([1., 1.])
```

弄个再复杂一点的：

```
x = torch.tensor([2., 1.], requires_grad=True)
y = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)

z = torch.mm(x.view(1, 2), y)
print(f"z:{z}")
z.backward(torch.Tensor([[1., 0]]), retain_graph=True)
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")

>>> z:tensor([[5., 8.]], grad_fn=<MmBackward>)
x.grad: tensor([[1., 3.]])
y.grad: tensor([[2., 0.],
        [1., 0.]])
```

结果解释如下：

![](/assets/img/marsggbo/2019-09-19-Pytorch-autogradbackward详解/fb1ee269.jpg)

总结：

说了这么多，`grad_tensors`的作用其实可以简单地理解成在求梯度时的权重，因为可能不同值的梯度对结果影响程度不同，所以pytorch弄了个这种接口，而没有固定为全是1。引用自[知乎上的一个评论](https://zhuanlan.zhihu.com/p/29923090)：如果从最后一个节点(总loss)来backward，这种实现(**torch.sum(y\*w)**)的意义就具体化为 multiple loss term with difference weights 这种需求了吧。

## **torch.autograd.grad**

```
torch.autograd.grad(
		outputs, 
		inputs, 
		grad_outputs=None, 
		retain_graph=None, 
		create_graph=False, 
		only_inputs=True, 
		allow_unused=False)
```

看了前面的内容后在看这个函数就很好理解了，各参数作用如下：

* `outputs`: 结果节点，即被求导数
* `inputs`: 叶子节点
* `grad_outputs`: 类似于`backward`方法中的`grad_tensors`
* `retain_graph`: 同上
* `create_graph`: 同上
* `only_inputs`: 默认为`True`, 如果为`True`, 则只会返回指定`input`的梯度值。 若为`False`，则会计算所有叶子节点的梯度，并且将计算得到的梯度累加到各自的`.grad`属性上去。
* `allow_unused`: 默认为`False`, 即必须要指定`input`,如果没有指定的话则报错。

## **参考**

* [PyTorch 中 backward() 详解](https://link.zhihu.com/?target=https%3A//www.pytorchtutorial.com/pytorch-backward/)
* [PyTorch 的backward 为什么有一个grad\_variables 参数?](https://zhuanlan.zhihu.com/p/29923090)
* [AUTOMATIC DIFFERENTIATION PACKAGE - TORCH.AUTOGRAD](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/autograd.html%23)

### **MARSGGBO♥原创**

### 微信公众号: 【AutoML机器学习】

http://weixin.qq.com/r/HD8gOHzEmiHlrThb92oO (二维码自动识别)