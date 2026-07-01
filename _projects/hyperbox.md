---
layout: page
title: "Hyperbox | Open-Source AutoML Toolbox"
description: >
  A lightweight, extensible AutoML framework built on PyTorch.
  Covers NAS, HPO, and model compression in a unified API.
importance: 2
category: fun
github: https://github.com/marsggbo/hyperbox
github_stars: marsggbo/hyperbox
---

## Hyperbox — 让 AutoML 像搭积木一样简单

**[GitHub](https://github.com/marsggbo/hyperbox)** · **[文档 Docs](https://hyperbox-doc.readthedocs.io/en/latest/)**

Hyperbox 是我在读博期间开发的 AutoML 开源框架，基于 PyTorch 和 PyTorch Lightning 构建，目标是让研究者用最少的代码跑通 NAS、HPO、剪枝等实验。

Hyperbox is an open-source AutoML toolbox I built during my PhD, powered by PyTorch and PyTorch Lightning. The goal: let researchers run NAS, HPO, and model compression experiments with minimal boilerplate.

---

## 核心设计 / Design Philosophy

标准 AutoML 框架往往把搜索算法和模型结构耦合得很死，换一个搜索空间就要大改代码。Hyperbox 的核心设计是**解耦**：

Most AutoML frameworks tightly couple search algorithms with model architecture. Hyperbox's core design is **decoupling**:

```
搜索空间 (SearchSpace)  ×  搜索策略 (Searcher)  ×  评估器 (Evaluator)
     ↕                           ↕                        ↕
  自由替换                     自由替换                  自由替换
```

三个模块独立，任意组合——换 NAS 算法不需要动模型定义，换搜索空间不需要动训练代码。

Three independent modules, freely composable — swap the NAS algorithm without touching the model definition.

---

## 主要功能 / Features

| 模块 Module | 说明 Description |
|---|---|
| **NAS** | One-shot NAS, 支持 DARTS / SPOS / Random Search |
| **HPO** | 贝叶斯优化、进化算法、随机搜索 / Bayesian opt, evolutionary, random |
| **Model Compression** | 剪枝、量化接口 / Pruning & quantization interfaces |
| **Benchmark** | 内置 NAS-Bench-201 / NAS-Bench-201 built-in |
| **Multi-objective** | Pareto front 搜索，兼顾精度与延迟 / Pareto-front search for accuracy-latency trade-off |

---

## 应用案例 / Applications

Hyperbox 是多篇顶会论文的实验基础：

Hyperbox underpins experiments in several top-venue papers:

- **[NAS-LID (AAAI 2023)](https://arxiv.org/abs/2211.12759)** — 基于局部本征维度的神经架构搜索 / NAS via local intrinsic dimension
- **[EMARS (MICCAI 2022)](https://dl.acm.org/doi/abs/10.1007/978-3-031-16431-6_53)** — 多目标进化 NAS 用于医疗图像分类 / Multi-objective evolutionary NAS for medical imaging
- **[MedPipe (IEEE MedAI 2023)](https://github.com/marsggbo/hyperbox_app/tree/medmnistv1)** — 联合搜索数据增强策略与网络结构 / Joint search of augmentation policy and architecture

---

## 安装 / Install

```bash
pip install hyperbox
```

或从源码安装 / or from source:

```bash
git clone https://github.com/marsggbo/hyperbox
cd hyperbox && pip install -e .
```

---

## 快速上手 / Quick Start

```python
from hyperbox.networks.ofa import OFAMobileNetV3Space
from hyperbox.searcher import EvolutionSearcher
from hyperbox.mutator import RandomMutator

# 定义搜索空间
space = OFAMobileNetV3Space()

# 随机采样一个子网
mutator = RandomMutator(space)
mutator.reset()  # 采样一次

# 用进化算法搜索
searcher = EvolutionSearcher(space, population_size=50, n_gen=20)
best = searcher.search(eval_fn=lambda net: validate(net))
```

---

## 链接 / Links

- 📦 **GitHub**: [github.com/marsggbo/hyperbox](https://github.com/marsggbo/hyperbox)
- 📖 **文档 Docs**: [hyperbox-doc.readthedocs.io](https://hyperbox-doc.readthedocs.io/en/latest/)
- ⭐ **Stars**: 27 · **Forks**: 4
