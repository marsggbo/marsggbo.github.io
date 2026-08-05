---
layout: post
title: "Detectron2源码阅读笔记-（三）Dataset pipeline"
date: 2019-10-23
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/88149772
related_posts: false
toc:
  sidebar: left
---

## 构建data\_loader原理步骤

```
# engine/default.py
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
class DefaultTrainer(SimpleTrainer):
    def __init__(self, cfg):
        # Assume these objects must be constructed in this order.
        data_loader = self.build_train_loader(cfg)
        ...    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        """
        return build_detection_train_loader(cfg)
```

函数调用关系如下图：

![](/assets/img/marsggbo/2019-10-23-Detectron2源码阅读笔记-三Dataset-pipeline/abe72a43.jpg)

结合前面两篇文章的内容可以看到detectron2在构建model,optimizer和data\_loader的时候都是在对应的`build.py`文件里实现的。我们看一下`build_detection_train_loader`是如何定义的(对应上图中**紫色方框内**的部分(**自下往上**的顺序)）：

```
def build_detection_train_loader(cfg, mapper=None):
    """
    A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Start workers to work on the dicts. Each worker will:
      * Map each metadata dict into another format to be consumed by the model.
      * Batch them by simply putting dicts into a list.
    The batched ``list[mapped_dict]`` is what this dataloader will return.

    Args:
        cfg (CfgNode): the config
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            By default it will be `DatasetMapper(cfg, True)`.

    Returns:
        a torch DataLoader object
    """
    # 获得dataset_dicts
    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=True,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )
    
    # 将dataset_dicts转化成torch.utils.data.Dataset
    dataset = DatasetFromList(dataset_dicts, copy=False)

    # 进一步转化成MapDataset，每次读取数据时都会调用mapper来对dict进行解析
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)
    
    # 采样器
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    if sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(dataset))
        ...
    batch_sampler = build_batch_data_sampler(
        sampler, images_per_worker, group_bin_edges, aspect_ratios
    )
    
    # 数据迭代器 data_loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )
    return data_loader
```

由上面的源代码可以看出总共是五个步骤，我们只对前面三个部分进行详细介绍，后面的采样器和data\_loader可以参阅[一文弄懂Pytorch的DataLoader, DataSet, Sampler之间的关系](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/marsggbo/p/11308889.html)。

## ***1*获得dataset\_dicts**

`get_detection_dataset_dicts(dataset_names)`函数需要传递的一个重要参数是`dataset_names`,这个参数其实就是一个字符串，用来指定数据集的名称。通过这个字符串，该函数会调用`data/catalog.py`的`DatasetCatalog`类来进行解析得到一个包含数据信息的字典。

解析的原理是：`DatasetCatalog`有一个字典`_REGISTERED`，默认已经注册好了例如`coco,voc`这些数据集的信息。如果你想要使用你自己的数据集，那么你需要在最开始前你需要定义你的数据集名字以及定义一个函数(这个函数不需要传参，而且最后会返回一个dict，该dict包含你的数据集信息)，举个栗子：

```
from detectron2.data import DatasetCatalog
my_dataset_name = 'apple'
def get_dicts():
    ...
    return dict

DatasetCatalog.register(my_dataset_name, get_dicts)
```

当然，如果你的数据集已经是COCO的格式了，那么你也可以使用如下方法进行注册：

```
from detectron2.data.datasets import register_coco_instances
my_dataset_name = 'apple'
register_coco_instances(my_dataset_name, {}, "json_annotation.json", "path/to/image/dir")
```

另外需要注意的是一个数据集其实是可以由两个类来定义的，一个是前面介绍了的`DatasetCatalog`,另一个是`MetadataCatalog`。

**`MetadataCatalog`的作用是记录数据集的一些特征**，这样我们就可以很方便的在整个代码中获取数据集的特征信息。在注册`DatasetCatalog`后，我们可以按如下栗子对`MetadataCatalog`进行注册并定义我们后面可能会用到的属性特征：

```
from detectron2.data import MetadataCatalog
MetadataCatalog.get("my_dataset").thing_classes = ["person", "dog"]

# 也可以这样
MetadataCatalog.get("my_dataset").set("thing_classes",["person", "dog"])
```

注意：如果你的数据集名字未注册过，`MetadataCatalog.get`会自动进行注册，然后会自动设置你所设定的属性值。

其实`MetadataCatalog`还有其他的特征属性可以设置，如`stuff_classes`,`stuff_colors`等等。你可能会好奇`thing_classes`和`stuff_classes`有什么区别，区别如下：

* 抽象解释：`thing_classes`用于指定instance-level任务，`stuff_classes`用于semantic segmentation任务。
* 具体解释：像椅子，书这种可数的东西，就可以理解成`thing`,所以用于instance-level;而雪、天空这种不可数的就理解成`stuff`,所以用于semantic segmentation。参考[On Seeing Stuff: The Perception of Materials by Humans and Machines](https://link.zhihu.com/?target=http%3A//persci.mit.edu/pub_pdfs/adelson_spie_01.pdf)

最后，`get_detection_dataset_dicts`会返回一个包含若干个dict的list，之所以是list是因为参数`dataset_names`也是一个list，这样我们就可以制定多个names来同时对数据进行读取。

## ***2*解析成DatasetFromList**

`DatasetFromList(dataset_dict)`函数定义在`detectron2/data/common.py`中，它其实就是一个`torch.utils.data.Dataset`类，其源码如下

```
class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, lst: list, copy: bool = True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
        """
        self._lst = lst
        self._copy = copy

    def __len__(self):
        return len(self._lst)

    def __getitem__(self, idx):
        if self._copy:
            return copy.deepcopy(self._lst[idx])
        else:
            return self._lst[idx]
```

这个很简单就不加赘述了

## ***3*将DatsetFromList转化成MapDataset**

其实`DatsetFromList`和`MapDataset`都是`torch.utils.data.Dataset`的子类，那他们的区别是什么呢？很简单，区别就是后者使用了`mapper`。

在解释`mapper`是什么之前我们首先要知道的是，在detectron2中，一张图片对应的是一个dict，那么整个数据集就是list[dict]。之后我们再看`DatsetFromList`，它的`__getitem__`函数非常简单，它只是简单粗暴地就返回了指定idx的元素。显然这样是不行的，因为在把数据扔给模型训练之前我们肯定还要对数据做一定的处理，而这个工作就是由`mapper`来做的，默认情况下使用的是`detectron2/data/dataset_mapper.py`中定义的`DatasetMapper`，如果你需要自定义一个`mapper`也可以参考这个写。

### **`DatasetMapper(cfg, is_train=True)`**

我们继续了解一下`DatasetMapper`的实现原理，首先看一下官方给的定义：

> A **callable** which takes a dataset dict in Detectron2 Dataset format, and map it into a format used by the model.

简单概括就是**这个类是可调用的(callable)**,所以在下面的源码中可以看到定义了`__call__`方法。

该类主要做了这三件事：

1. Read the image from "file\_name"
2. Applies cropping/geometric transforms to the image and annotations
3. Prepare data and annotations to Tensor and :class:`Instances`

其源码如下（有删减）：

```
class DatasetMapper:
    def __init__(self, cfg, is_train=True):
        # 读取cfg的参数
        ...

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        
        # 1. 读取图像数据
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        
        # 2. 对image和box等做Transformation
        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            ...
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w
        
        # 3.将数据转化成tensor格式
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        ...

        return dataset_dict
```

### **`MapDataset`**

```
class MapDataset(data.Dataset):
    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )
```

* `self._fallback_candidates`是一个`set`,它的特点是其中的元素是独一无二的，定义这个的作用是记录可正常读取的数据索引，因为有的数据可能无法正常读取，所以这个时候我们就可以把这个坏数据的索引从`_fallback_candidates`中剔除，并随机采样一个索引来读取数据。
* `__getitem__`中的逻辑就是首先读取指定索引的数据，如果正常读取就把该所索引值加入到`_fallback_candidates`中去；反之，如果数据无法读取，则将对应索引值删除，并随机采样一个数据，并且尝试3次，若3次后都无法正常读取数据，则报错，但是好像也没有退出程序，而是继续读数据，可能是以为总有能正常读取的数据吧hhh。

### **MARSGGBO♥原创** **如有意合作，欢迎私戳** **邮箱:marsggbo@foxmail.com** 微信公众号: 【AutoML机器学习】

![](/assets/img/marsggbo/2019-10-23-Detectron2源码阅读笔记-三Dataset-pipeline/228f26c5.jpg)

AutoML机器学习

  

**2019-10-23 13:37:13**