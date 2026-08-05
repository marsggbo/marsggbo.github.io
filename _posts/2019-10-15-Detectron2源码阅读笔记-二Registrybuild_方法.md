---
layout: post
title: Detectron2源码阅读笔记-(二)Registry&build_*方法
date: '2019-10-15'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/86808911
related_posts: false
toc:
  sidebar: left
---

## Trainer解析

我们继续[Detectron2代码阅读笔记-(一)](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/marsggbo/p/11677086.html)中的内容。

![](/assets/img/marsggbo/2019-10-15-Detectron2源码阅读笔记-二Registrybuild_方法/88ad2f79.jpg)

上图画出了`detectron2`文件夹中的三个子文件夹(**tools,config,engine**)之间的关系。那么剩下的文件夹又是如何起作用的呢？

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

## build\_\*方法

我们从`trainer = Trainer(cfg)`开始进一步了解。

[Detectron2代码阅读笔记-(一)](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/marsggbo/p/11677086.html)中已经提到过一连串的Trainer的继承关系如下： `tools.train_net.Trainer->detectron2.engine.default.DefaultTrainer->detectron2.engine.train_loop.SimpleTrainer->detectron2.engine.train_loop.TrainerBase`，而`detectron2.engine.default.DefaultTrainer`在其`__init__(self, cfg)`函数中定义了解析cfg。如下面代码所示，cfg会作为参数倍若干个`build_*`方法解析，得到解析后的model,optimizer,data\_loader等。

```
from detectron2.modeling import build_model
class DefaultTrainer(SimpleTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        ... 

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model
```

下面我们以`DefaultTrainer.build_model`为例来介绍注册机制,该方法调用了`detectron2/modeling/meta_arch/build_model.py`的`build_model`函数,其源代码如下：

```
from detectron2.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")
META_ARCH_REGISTRY.__doc__ = """
def build_model(cfg):
    """
    Built the whole model, defined by `cfg.MODEL.META_ARCHITECTURE`.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    return META_ARCH_REGISTRY.get(meta_arch)(cfg)
```

* meta\_arch = cfg.MODEL.META\_ARCHITECTURE： 根据超参数获得网络结构的名字
* return META\_ARCH\_REGISTRY.get(meta\_arch)(cfg)：META\_ARCH\_REGISTRY是一个`Registry`类(这个在后面会详细介绍)，可以将这一行代码拆成如下几个步骤：

```
model = META_ARCH_REGISTRY.get(meta_arch)
return model(cfg)
```

## 注册机制Registry

那么`Registry`到底是什么呢？在分析源代码之前我们先了解一下如何使用它，假如你想自己实现一个新的backbone网络，那么你可以这样做：

首先在detectron2中定义好如下（实际上已经定义了）：

```
# detectron2/modeling/backbone/build.py
BACKBONE_REGISTRY = Registry('BACKBONE')
```

之后在你创建的新的文件下按如下方式创建你的backbone

```
# detectron2/modeling/backbone/your_backbone.py
from .build import BACKBONE_REGISTRY

# 方式1
@BACKBONE_REGISTRY.register()
class MyBackbone():
    ...

# 方式2
class MyBackbone():
    ...
BACKBONE_REGISTRY.register(MyBackbone)
```

`Registry`源代码如下（有删减）：

```
class Registry(object):
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj):
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(name, self._name)
        self._obj_map[name] = obj

    def register(self, obj=None):
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object named '{}' found in '{}' registry!".format(name, self._name))
        return ret
```

* 首先是`__init__`部分：

+ `self._name`则是你要注册的名字，例如对于完整的模型而言，name一般取`META_ARCH`。当然如果你需要自定义backbone网络，你也可以定义一个`Registry('BACKBONE')`
+ `self._obj_map`:其实就是一个字典。以模型为例，key就是你的模型名字，而value就是对应的模型类。这样你在传参时只需要修改一下模型名字就能使用不同的模型了。具体实现方法就是后面这几个函数。

* `register`: 可以看到该方法定义了注册的两种方式，一种是当`obj==None`的时候，使用装饰器的方式注册，另外一种就是直接将obj作为参数调用`_do_register`进行注册。
* `_do_register`:真正注册的函数，可以看到它首先会判断name是否已经存在于`self._obj_map`了。什么意思呢？还是以backbone为例，我们定义了一个`BACKBONE_REGISTRY = Registry('BACKBONE')`,然后又定义了很多种backbone，而这些backbone都使用`@BACKBONE_REGISTRY.register()`的方式注册到了`BACKBONE_REGISTRY._obj_map`中了,所以才取名为`Registry`,还是蛮形象的吼。
* `get`: 这个其实就是根据key值对字典进行取值。

## Detectron2 整体代码架构

虽然Detectron2还有很多部分没有介绍到，但是源代码分析到这应该对整体架构有了一定的理解了，具体的一些细节会在后续的文章中进行分析。现对Detectron2 整体代码架构总结一下：

![](/assets/img/marsggbo/2019-10-15-Detectron2源码阅读笔记-二Registrybuild_方法/c95a747d.jpg)

### **MARSGGBO♥原创** 微信公众号: 【AutoML机器学习】

![](/assets/img/marsggbo/2019-10-15-Detectron2源码阅读笔记-二Registrybuild_方法/228f26c5.jpg)

AutoML机器学习

  
  
  
  

**2019-10-15 13:16:32**