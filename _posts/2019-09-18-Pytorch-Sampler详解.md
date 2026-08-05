---
layout: post
title: "Pytorch Sampler详解"
date: '2019-09-18'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/82985227
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/82985227>

> 关于为什么要用Sampler可以阅读[一文弄懂Pytorch的DataLoader, DataSet, Sampler之间的关系](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/marsggbo/p/11308889.html)。

本文我们会从源代码的角度了解Sampler。

## **Sampler**

首先需要知道的是所有的采样器都继承自`Sampler`这个类，如下：

可以看到主要有三种方法：分别是：

* `__init__`: 这个很好理解，就是初始化
* `__iter__`: 这个是用来产生迭代索引值的，也就是指定每个step需要读取哪些数据
* `__len__`: 这个是用来返回每次迭代器的长度

```python
class Sampler(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """
    # 一个 迭代器 基类
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
```

## **子类Sampler**

介绍完父类后我们看看Pytorch给我们提供了哪些采样器

## **SequentialSampler**

这个看名字就很好理解，其实就是按顺序对数据集采样。

其原理是首先在初始化的时候拿到数据集`data_source`，之后在`__iter__`方法中首先得到一个和`data_source`一样长度的`range`可迭代器。**每次只会返回一个索引值**。

```python
class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """
   # 产生顺序 迭代器
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)
```

使用示例：

```python
a = [1,5,78,9,68]
b = torch.utils.data.SequentialSampler(a)
for x in b:
    print(x)
    
>>> 0
    1
    2
    3
    4
```

## **RandomSampler**

参数作用：

* data\_source: 同上
* num\_samples: 指定采样的数量，默认是所有。
* replacement: 若为True，则表示可以重复采样，即同一个样本可以重复采样，这样可能导致有的样本采样不到。所以此时我们可以设置num\_samples来增加采样数量使得每个样本都可能被采样到。

```python
class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples

        if self.num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return len(self.data_source)
```

## **SubsetRandomSampler**

```python
class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)
```

这个采样器常见的使用场景是将训练集划分成训练集和验证集，示例如下：

```python
n_train = len(train_dataset)
split = n_train // 3
indices = random.shuffle(list(range(n_train)))
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
train_loader = DataLoader(..., sampler=train_sampler, ...)
valid_loader = DataLoader(..., sampler=valid_sampler, ...)
```

## **WeightedRandomSampler**

参数作用同上面的RandomSampler，不再赘述。

```python
class WeightedRandomSampler(Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).
    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, weights, num_samples, replacement=True):
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples  ## 指的是一次一共采样的样本的数量
```

## **BatchSampler**

前面的采样器每次都只返回一个索引，但是我们在训练时是对批量的数据进行训练，而这个工作就需要BatchSampler来做。也就是说BatchSampler的作用就是将前面的Sampler采样得到的索引值进行合并，当数量等于一个batch大小后就将这一批的索引值返回。

```python
class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
# 批次采样
    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
```

### **MARSGGBO♥原创** **2019-9-18**