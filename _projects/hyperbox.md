---
layout: page
title: "Hyperbox | Open-Source AutoML Toolbox"
description: >
  A clean and scalable AutoML framework built on PyTorch + PyTorch Lightning.
  Supports NAS, HPO, and Auto Data Augmentation with a unified, modular API.
importance: 2
category: fun
github: https://github.com/marsggbo/hyperbox
github_stars: marsggbo/hyperbox
---

## English

**GitHub:** [marsggbo/hyperbox](https://github.com/marsggbo/hyperbox) · **Docs:** [hyperbox-doc.readthedocs.io](https://hyperbox-doc.readthedocs.io/en/latest/)

Hyperbox is an open-source AutoML framework I developed during my PhD, built on top of PyTorch and PyTorch Lightning. It provides a clean, modular API covering Neural Architecture Search (NAS), Hyperparameter Optimization (HPO), and Auto Data Augmentation (ADA).

---

### Core Modules

**`hyperbox.mutables` — Searchable building blocks**

Three types of searchable modules cover the vast majority of NAS search spaces:

| Module | Purpose |
|---|---|
| `OperationSpace` | Select one operation from a candidate list (e.g. conv 3×3 vs 5×5 vs 7×7) |
| `InputSpace` | Select which previous layer outputs to use as skip connections |
| `ValueSpace` | Search fine-grained parameters (kernel size, channel width, etc.) inside a single op |

```python
from hyperbox.mutables.spaces import OperationSpace, InputSpace, ValueSpace
from hyperbox.mutables.ops import Conv2d

# Choose one conv kernel size
ks = ValueSpace([3, 5, 7], key='kernelSize')
cout = ValueSpace([16, 32, 64], key='channelOut')
conv = Conv2d(3, cout, ks, stride=1, padding=2, bias=False)
```

**`hyperbox.mutator` — Search algorithms**

Plug-and-play search algorithms that work with any `mutables`-based search space:

| Algorithm | File |
|---|---|
| Random Search | `random_mutator.py` |
| DARTS (gradient-based) | `darts_mutator.py` |
| ENAS (RL-based) | `enas_mutator.py` |
| Evolution | `evolution_mutator.py` |
| FairNAS | `fairnas_mutator.py` |
| Few-shot NAS | `fewshot_mutator.py` |
| ProxylessNAS | `proxyless_mutator.py` |
| RepNAS | `repnas_mutator.py` |

```python
from hyperbox.mutator import RandomMutator
from hyperbox.networks.ofa import OFAMobileNetV3

net = OFAMobileNetV3()          # supernet (mask=None → searchable)
mutator = RandomMutator(net)
mutator.reset()                 # sample one subnet
arch = mutator._cache           # arch as dict

subnet = net.build_subnet(arch) # inherit supernet weights
```

**`hyperbox.networks` — Pre-built search spaces**

Implementations of 15+ published NAS search spaces, including:
`DARTS`, `ENAS`, `OFA (MobileNetV3)`, `ProxylessNAS`, `SPOS`, `NAS-Bench-201`, `NAS-Bench-301`, `ViT`, `GPT`, `ResNet`, `RepNAS`, …

---

### Applications

Hyperbox is the experimental backbone of several published papers:

| Paper | Venue | Description |
|---|---|---|
| [NAS-LID](https://arxiv.org/abs/2211.12759) | AAAI 2023 | NAS via local intrinsic dimension |
| [EMARS](https://dl.acm.org/doi/abs/10.1007/978-3-031-16431-6_53) | MICCAI 2022 | Multi-objective evolutionary NAS for medical imaging |
| [MedPipe](https://github.com/marsggbo/hyperbox_app/tree/medmnistv1) | IEEE MedAI 2023 | Joint search of augmentation policy and architecture |

---

### Install

```bash
pip install hyperbox
# or from source
git clone https://github.com/marsggbo/hyperbox && cd hyperbox
pip install -r requirements.txt && python setup.py develop
```

---

## 中文版本

**GitHub：** [marsggbo/hyperbox](https://github.com/marsggbo/hyperbox) · **文档：** [hyperbox-doc.readthedocs.io](https://hyperbox-doc.readthedocs.io/en/latest/)

Hyperbox 是我在读博期间开发的开源 AutoML 框架，基于 PyTorch 和 PyTorch Lightning 构建，提供一套模块化、可扩展的 API，覆盖神经架构搜索（NAS）、超参优化（HPO）和自动数据增强（ADA）三大方向。

---

### 核心模块

**`hyperbox.mutables` — 可搜索模块**

三种可搜索模块基本覆盖了现有 NAS 搜索空间的所有主流设计：

| 模块 | 用途 |
|---|---|
| `OperationSpace` | 从候选操作列表中选一个（如 3×3 / 5×5 / 7×7 卷积） |
| `InputSpace` | 从前面若干层输出中选一个或多个作为当前层输入（跳跃连接） |
| `ValueSpace` | 在单个操作内部搜索细粒度参数（卷积核大小、通道数等） |

每个模块都有 `key` 和 `mask` 两个关键参数：`key` 是唯一标识符，`mask` 指定当前选择（`None` 表示待搜索）。

**`hyperbox.mutator` — 搜索算法**

Mutator 与搜索空间完全解耦，可以自由替换搜索算法而无需改动模型定义。已内置 Random、DARTS、ENAS、进化算法、FairNAS、Few-shot NAS、ProxylessNAS、RepNAS 等多种经典算法。

**`hyperbox.networks` — 内置搜索空间**

重新实现了 15+ 篇经典 NAS 论文的搜索空间，包括 DARTS、ENAS、OFA（MobileNetV3）、ProxylessNAS、SPOS、NAS-Bench-201/301、ViT、GPT 等。

---

### 应用论文

| 论文 | 会议 | 说明 |
|---|---|---|
| [NAS-LID](https://arxiv.org/abs/2211.12759) | AAAI 2023 | 基于局部本征维度的神经架构搜索 |
| [EMARS](https://dl.acm.org/doi/abs/10.1007/978-3-031-16431-6_53) | MICCAI 2022 | 多目标进化 NAS 用于医疗图像分类 |
| [MedPipe](https://github.com/marsggbo/hyperbox_app/tree/medmnistv1) | IEEE MedAI 2023 | 联合搜索数据增强策略与网络结构 |
