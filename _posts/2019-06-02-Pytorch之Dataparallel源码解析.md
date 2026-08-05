---
layout: post
title: "Pytorch之Dataparallel源码解析"
date: 2019-06-02
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/67801020
related_posts: false
toc:
  sidebar: left
---

之前对Pytorch 1.0 的Dataparallel的使用方法一直似懂非懂，总是会碰到各种莫名其妙的问题，今天就好好从源头梳理一下，更好地理解它的原理或者说说下步骤。

> 源码地址: [https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/data\_parallel.py](https://link.zhihu.com/?target=https%3A//github.com/pytorch/pytorch/blob/master/torch/nn/parallel/data_parallel.py)

## 初始化

首先我们一行一行地来看一下Dataparallel是如何初始化的。

* `super`就是继承torch.nn.Module父类,这里不做解释
* 第一个if判断语句：检查是否有可用GPU
* 第二个if判断语句：如果没有指定GPU，则默认使用所有可用的GPU
* 第三个if判断语句：`output_device`表示输出到哪一个GPU上，默认是第一个GPU，注意这个**第一个**是**device\_ids**列表上的第一个，所以如果你有三个GPU，而你在将model复制到cuda上时写的代码是`model.cuda(1)`或者`model.cuda(2)`，则会报错,因为`device_ids`是[0,1,2].其第一个元素是0。这一点可以在后面的`forward`函数中看到。
* emm，后面每行代码的作用很清楚，就不再一一解释了。

```
def __init__(self, module, device_ids=None, output_device=None, dim=0):
    super(DataParallel, self).__init__()

    if not torch.cuda.is_available():
        self.module = module
        self.device_ids = []
        return

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    if output_device is None:
        output_device = device_ids[0]

    self.dim = dim
    self.module = module
    self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    self.output_device = _get_device_index(output_device, True)
    self.src_device_obj = torch.device("cuda:{}".format(self.device_ids[0]))

    _check_balance(self.device_ids)

    if len(self.device_ids) == 1:
        self.module.cuda(device_ids[0])
```

## 前向传播

下面进入到重头戏：Dataparallel的forward函数。

```
def forward(self, *inputs, **kwargs):
    if not self.device_ids:
        return self.module(*inputs, **kwargs)

    for t in chain(self.module.parameters(), self.module.buffers()):
        if t.device != self.src_device_obj:
            raise RuntimeError("module must have its parameters and buffers "
                               "on device {} (device_ids[0]) but found one of "
                               "them on device: {}".format(self.src_device_obj, t.device))

    inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
    if len(self.device_ids) == 1:
        return self.module(*inputs[0], **kwargs[0])
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(replicas, inputs, kwargs)
    return self.gather(outputs, self.output_device)
```

* 第一个if判断语句：如果没有可用的GPU设备，则使用原来的module进行计算。
* for循环就是对应了前面提到的问题，用于检查model和input是不是放在第一个GPU上
* 之后下一步就是将将input平均划分到每个GPU上,用到的是下面的`scatter`函数

```
def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res
```

* 数据划分之后呢，再判断一下有几个可用的GPU（前面是判断有没有，这里是判断有几个），如果只有一个GPU，那就不用进入到下一步了。
* 如果有多个GPU，那么就需要用到`replica`函数，这个函数比较复杂，就不解释了，感兴趣的可以阅读一下源码:[https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/replicate.py](https://link.zhihu.com/?target=https%3A//github.com/pytorch/pytorch/blob/master/torch/nn/parallel/replicate.py) 。不过它的主要作用就是**将模型复制到多个GPU上。**
* 下一步中的`parallel_apply`作用就是并行地在多个GPU上计算模型，每个模型是一样的，只不过输入数据是不一样的，因为前面将数据平均划分了。例如你有两个GPU，一个batch大小是64，那么两个GPU分别处理batch大小为32的数据。
* 最后就是将输出值`gather`到一起，传送到`output_device`，即第一个GPU设备上。

### **MARSGGBO♥原创** 微信公众号: 【AutoML机器学习】

![](/assets/img/marsggbo/2019-06-02-Pytorch之Dataparallel源码解析/228f26c5.jpg)

AutoML机器学习

  
  
  
  

**2019-6-2**