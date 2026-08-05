---
layout: post
title: detectron2安装出现Kernel not compiled with GPU support
date: '2019-11-23'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/93278639
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/93278639>

在安装使用detectron2的时候碰到**Kernel not compiled with GPU support** 问题，前后拖了好久都没解决，现总结一下以备以后查阅。

不想看心路历程的可以直接跳到最后一小节，哈哈哈。

## environment

因为我使用的是实验室的服务器，所以很多东西没法改，我的cuda环境如下：

* ubuntu
* `nvcc`默认版本是9.2
* `nvidia-smi`版本又是10.0的

> 我之前一直没搞清楚这`nvcc`和`nvidia-smi`版本为什么可以不一样，想了解原因的可以看一下我之前的文章[显卡，显卡驱动,nvcc, cuda driver,cudatoolkit,cudnn到底是什么？](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/marsggbo/p/11838823.html%23%25E5%25A4%259A%25E7%2589%2588%25E6%259C%25ACcuda%25E5%2588%2587%25E6%258D%25A2)。

## reproduce

我一般都用Anaconda来安装pytorch，第一次安装的时候使用的如下命令安装的：

```bash
conda create -n myenv python=3.7
conda activate myenv
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

按理说这个命令会给`myenv`环境安装cuda编译器和驱动等，但是在运行代码的时候还是会出现标题中的报错信息。我猜可能是因为detectron2在build的时候使用的是`/usr/local`路径下的cuda compiler（即nvcc），而不是我的虚拟幻境下的compiler。所以我重新安装了cuda-9.2版本的pytorch，

```bash
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

但是运行的时候还是出现同样的错误，这更加说明detectron2的编译使用的不是虚拟环境路径下的编译器，所以我在想是不是我没有正确设置系统路径下的CUDA，于是我用官方提供的检验代码查看CUDA路径：

```python
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'
```

输出的结果是

```
true /usr/local
```

可以看到`CUDA_HOME`对应的输出结果有问题，照理来说输出结果应该是`/usr/local/cuda`或者`/usr/local/cuda-9.2`之类的，于是我又查看了`~/.bashrc`文件，找到与CUDA有关的代码部分,发现我并没有设置`CUDA_HOME`这个环境变量，于是我做了如下修改：

```bash
# vim ~/.bashrc
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
```

之后souce一下

```bash
source ~/.bashrc
```

再运行`python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; print(torch.cuda.is_available(), CUDA_HOME)'`输出的结果变为`true /usr/local/cuda`

此时`Kernel not compiled with GPU support`的问题就解决了，代码正常运行了。

## 总结

## 步骤总结

1. 安装相关库

```bash
conda create -n myenv python=3.7
conda activate myenv
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
pip install opencv-python
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

1. 编译detectron2

```bash
python setup.py build develop
```

## 可能出现的问题即解决办法

出现标题中的错误的原因主要是因为你的cuda版本或者路径除了问题，你可以按照如下几个步骤排查可能是那个地方出了问题：

1. 运行`nvcc --version`查看你的cuda编译器版本，那么你的pytorch-gpu也建议安装对应版本。当然如果你`nvcc`都没安装。。。那你就先找教程安装。
2. 如果安装的pytorch版本和`nvcc`版本一致，你可以看一下你的CUDA路径是否在`~/.bashrc`中设置正确，参考的配置路径如下：

```bash
# vim ~/.bashrc
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
```

设置好后`source ~/.bashrc`,然后删除`detectron2/build`文件夹（如果你之前已经编译过一遍了），然后重新编译`dete2tron2`。

### **MARSGGBO♥原创** 如有意合作或学术讨论欢迎私戳联系~ 微信:marsggbo 邮箱:marsggbo@foxmail.com **2019-11-23 10:52:25**