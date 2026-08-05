---
layout: post
title: Detectron2源码阅读笔记-(一)Config&Trainer
date: '2019-10-15'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/86749823
related_posts: false
toc:
  sidebar: left
---

## 一、代码结构概览

## 1.核心部分

* configs：储存各种网络的yaml配置文件
* datasets：存放数据集的地方
* detectron2：运行代码的核心组件
* tools：提供了运行代码的入口以及一切可视化的代码文件。

## 2.Tutorial部分

* demo：显而易见就是demo
* docs: 同样显而易见。。
* tests：提供了一些测试代码
* projects：提供了真实的项目代码示例，之后自己的代码结构可参照这个结构写。

## 二、代码逻辑分析

## 1.超参数配置

进入`tools/train_net.py`的**main**函数，第一行`cfg = setup(args)`是配置参数。Detectron2中的参数配置使用了[yacs](https://link.zhihu.com/?target=https%3A//github.com/rbgirshick/yacs)这个库，这个库能够很好地重用和拼接超参数文件配置。

我们先看一下`detrctron2/config/`的文件结构：

* `compat.py`: 应该是对之前的Detectron库的兼容吧，可忽略。
* `config.py`: 定义了一个`CfgNode`类，这个类继承自`fvcore`库(fb写的一个共公共库，提供一些共享的函数，方便各种不同项目使用)中定义的`CfgNode`,总之就是不断继承。。。继承关系是这样的： `detrctron2.config.CfgNode->fcvore.common.config.CfgNode->yacs.config.CfgNode->dict` 另外该文件还提供了`get_cfg()`方法，该方法会返回一个含有默认配置的`CfgNode`,而这些默认的配置值在下面的`default.py`中定义了，之所以这样做是因为要配置的默认值太多了，所以为了文档清晰才写到了一个新的文件中去，不过，`yacs`库的作者也建议这样做。
* `default.py`: 如上面所说，该文件定义了各种参数的默认值。

了解配置函数的方法后我们再回到`tools/train_net.py`，我们一行一行的来理解。

* `tools/train_net.py`

```
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
...

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() 
    cfg.merge_from_file(args.config_file) 
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg
```

* cfg = get\_cfg()： 获取已经配置好默认参数的cfg
* cfg.merge\_from\_file(args.config\_file)：config\_file是指定的yaml配置文件，通过`merge_from_file`这个函数会将yaml文件中指定的超参数对默认值进行覆盖。
* cfg.merge\_from\_list(args.opts)：`merge_from_list`作用同上面的类似，只不过是通过命令行的方式覆盖。 例如

```
opts = ["SYSTEM.NUM_GPUS", 8, "TRAIN.SCALES", "(1, 2, 3, 4)"]
cfg.merge_from_list(opts)
print("cfg\n",cfg)
```

那么最后会有

```
cfg
... (一些默认值超参数)
SYSTEM:
    NUM_GPUS: 8
TRAIN:
    SCALES: (1,2,3,4)
```

* cfg.freeze(): freeze函数的作用是将超参数值冻结，避免被程序不小心修改。
* default\_setup(cfg, args)：`default_setup`是`detectron2/engine/default.py`中提供的一个默认配置函数，具体是怎么配置的这里不详细说明了。不过需要知道的值这个文件中还提供了很多其他的配置函数，例如还提供了两个类：`DefaultPredictor`和`DefaultTrainer`。

## 2.Trainer

既然上面提到了`DefaultTrainer`，那么我们就从这个类入手了解一下`detectron2.engine`,其代码结构如下：

* `train_loop.py`: 这个函数主要作用是提供了三个重要的类：

+ `HookBase`: 这是一个Hook的基类，用于指定在训练前后或者每一个step前后需要做什么事情，所以根据特定的需求需要对如下四种方法做不同的定义：`before_train`,`after_train`,`before_step`,`after_step`。以`before_step`。
+ `TrainerBase`: 该类中定义的函数可以归纳成三种：

- `register_hooks`:这个很好理解，就是将用户定义的一些hooks进行注册，**说大白话就是把若干个Hook放在一个list里面去**。之后只需要遍历这个list依次执行就可以了。
- 第二类其实就是上面提到的遍历hook list并执行hook，不过这个遍历有四种，分别是`before_train`,`after_train`,`before_step`,`after_step`。还有一个就是`run_step`,这个函数其实就是平常我们在编写训练过程的代码，例如读数据，训练模型，获取损失值，求导数，反向梯度更新等,只不过在这个类里面没有定义。
- 第三类就是`train`函数，它有两个参数，分别是开始的迭代数和最大的迭代数。之后就是重复依次执行第二类中的函数指定迭代次数。

+ `SimpleTrainer`:其实就是继承自`TrainerBase`,然后定义了`run_step`等方法。我们后面也可以继承这个类做进一步的自定义。

* `defaults.py`: 上面已介绍，提供了两个类：`DefaultPredictor`和`DefaultTrainer`，这个`DefaultTrainer`就继承自`SimpleTrainer`,所以存在如下继承关系： `detectron2.engine.default.DefaultTrainer->detectron2.engine.train_loop.SimpleTrainer->detectron2.engine.train_loop.TrainerBase`
* `hooks.py`:定义了很多继承自`train_loop.HookBase`的Hook。
* `launch.py`: 前面提到过，可以理解成代码启动器，可以根据命令决定是否采用分布式训练（或者单机多卡）或者单机单卡训练。

好了，我们继续回到`tools/train_net.py`的main函数,代码如下所示。

```
def main(args):
    cfg = setup(args)

    if args.eval_only:
        ...
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()
```

可以看到下面定义了一个`Trainer`,它继承自`detectron2.engine.default.DefaultTrainer`，这个父类会自动解析cfg。之后只需要调用`trainer.train()`就可以开始训练了。

## 三、小结

![](/assets/img/marsggbo/2019-10-15-Detectron2源码阅读笔记-一ConfigTrainer/88ad2f79.jpg)

至此我们对detectron2的逻辑有了大致的了解了，那么接下来我们来了解一下`detectron2.engine.default.DefaultTrainer`是如何解析cfg的，这部分内容请参见

[Detectron2代码阅读笔记-(二) - marsggbo - 博客园](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/marsggbo/p/11678371.html)

### **MARSGGBO♥原创** 微信公众号: 【AutoML机器学习】

![](/assets/img/marsggbo/2019-10-15-Detectron2源码阅读笔记-一ConfigTrainer/228f26c5.jpg)

AutoML机器学习

  
  
  

**2019-10-15 10:37:50**