---
layout: post
title: "RouteMark: 基于路由行为指纹的模型合并知识产权归属 | A Fingerprint for IP Attribution in Routing-based Model Merging"
date: 2026-03-12 15:50:00
category: research
tags: Model-Merging MoE Intellectual-Property Fingerprint Security
related_posts: false
toc:
  sidebar: left
---

**论文链接 / Paper:** [arXiv:2508.01784](https://arxiv.org/abs/2508.01784)

**作者 / Authors:** Xin He, Junxi Shen, Zhenheng Tang, Xiaowen Chu, Bo Li, Ivor W. Tsang, Yew-Soon Ong

---

# 中文版

## 研究动机

基于路由的模型合并（如 MoE 架构）正在成为复用和组合多个微调模型的流行方式。然而，当多个任务特定的专家被合并到一个统一的 MoE 模型中时，**如何验证各个专家的来源和知识产权归属**成为一个重要但尚未解决的问题。

现有的基于权重或激活的检测方法难以应对 MoE 中动态路由带来的复杂性。

## 核心方法

RouteMark 提出了一种基于**专家路由行为指纹**的知识产权归属框架，核心洞察是：**任务特定的专家在探测输入下展现出稳定且独特的路由模式**。

### 两种互补指纹

1. **路由分数指纹（RSF, Routing Score Fingerprint）**
   - 通过固定的探测数据集，测量每个专家在不同任务和混合层上的平均路由 logit 值
   - 形成专家激活强度的特征矩阵

2. **路由偏好指纹（RPF, Routing Preference Fingerprint）**
   - 捕获偏好激活每个专家的输入分布
   - 补充 RSF，突出专家的选择倾向和任务专业化特征

### 相似度匹配

基于相似度的匹配算法将可疑模型与参考（受害者）模型的指纹进行比对，综合分数和偏好两种度量产生最终的归属得分，能够准确检测专家复用并区分无关专家。

## 实验结果

- 在多种任务和基于 CLIP 的 MoE 模型上验证了高相似度检测能力
- 对多种篡改手段具有鲁棒性：
  - 专家替换、添加/删除
  - 微调、剪枝
  - 排列（permutation）
- 优于基于权重或激活的现有方法
- 无需访问模型权重即可完成指纹提取

---

# English Version

## Motivation

Routing-based model merging (e.g., MoE architectures) is becoming a popular approach for reusing and combining multiple fine-tuned models. However, when task-specific experts are merged into a unified MoE model, **verifying the origin and intellectual property attribution of individual experts** becomes an important yet unsolved problem.

Existing weight-based or activation-based detection methods struggle with the complexity of dynamic routing in MoE models.

## Key Methods

RouteMark proposes an **IP attribution framework based on expert routing behavior fingerprints**. The core insight: **task-specific experts exhibit stable, distinctive routing patterns under probing inputs**.

### Two Complementary Fingerprints

1. **Routing Score Fingerprint (RSF)**
   - Measures each expert's average routing logit values across tasks and mixture layers on a fixed probe dataset
   - Forms a characteristic matrix of expert activation intensity

2. **Routing Preference Fingerprint (RPF)**
   - Captures input distributions that preferentially activate each expert
   - Complements RSF by highlighting expert selection tendencies and task specialization

### Similarity Matching

A similarity-based matching algorithm compares fingerprints between a suspect model and a reference (victim) model, combining score-based and preference-based metrics for final attribution scores. It accurately detects expert reuse while distinguishing unrelated experts.

## Results

- High similarity detection validated across diverse tasks and CLIP-based MoE models
- Robust against various tampering methods:
  - Expert replacement, addition/deletion
  - Fine-tuning, pruning
  - Permutation
- Outperforms weight-based and activation-based methods
- Lightweight fingerprint extraction without requiring access to model weights
