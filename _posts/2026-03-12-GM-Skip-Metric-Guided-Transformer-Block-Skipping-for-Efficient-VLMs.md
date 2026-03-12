---
layout: post
title: "GM-Skip: 基于度量引导的 Transformer 块跳过策略加速视觉语言模型 | Metric-Guided Transformer Block Skipping for Efficient VLMs"
date: 2026-03-12 15:30:00
category: research
tags: VLM Transformer Efficiency Inference-Acceleration
related_posts: false
toc:
  sidebar: left
---

**论文链接 / Paper:** [arXiv:2508.18227](https://arxiv.org/abs/2508.18227)

**作者 / Authors:** Lianming Huang, Haibo Hu, Qiao Li, Xin He, Nan Guan, Chun Jason Xue

---

# 中文版

## 研究动机

视觉语言模型（Vision-Language Models, VLMs）在多模态理解任务中表现出色，但其巨大的 Transformer 参数量导致推理延迟高、部署成本大。如何在保持模型性能的前提下加速推理，是一个关键问题。

## 核心方法

GM-Skip 提出了一种**基于度量引导的 Transformer 块跳过框架**，通过策略性地跳过冗余的 Transformer 块来加速 VLM 推理：

### 1. 度量引导的块选择
使用贪心算法评估每个 Transformer 块的重要性，通过衡量移除某个块对任务特定指标（如准确率、CIDEr 分数）的影响，逐步识别并移除影响最小的块。

### 2. 逆序删除策略
优先从网络后层开始删除块，而非从前层开始。这是因为早期层对视觉-语言对齐至关重要，移除它们会导致性能灾难性下降。

### 3. 可调节的稀疏性-性能权衡
引入分数-稀疏度平衡机制，允许在计算效率和精度保持之间灵活调节，适应不同的部署场景。

## 实验结果

- 在 COCO 数据集上，跳过超过 **40% 的 Transformer 块**的同时，单目标分类准确率从 19.1% 提升到 87.3%
- 在自动驾驶平台 Autoware.Universe 上实现了高达 **45.4% 的延迟降低**
- 为延迟敏感的应用场景（如自动驾驶）提供了切实可行的加速方案

---

# English Version

## Motivation

Vision-Language Models (VLMs) excel at multimodal understanding but suffer from high inference latency and deployment costs due to their massive Transformer parameters. Accelerating inference while preserving model performance is a critical challenge.

## Key Methods

GM-Skip proposes a **metric-guided Transformer block skipping framework** that strategically skips redundant blocks to accelerate VLM inference:

### 1. Metric-Guided Block Selection
A greedy algorithm evaluates each block's importance by measuring the impact of its removal on task-specific metrics (accuracy, CIDEr, etc.), progressively identifying and removing the least impactful blocks.

### 2. Reverse-Order Deletion Strategy
Blocks are deleted starting from later layers rather than early ones. Early foundational blocks are critical for vision-language alignment, and their removal causes catastrophic performance collapse.

### 3. Tunable Sparsity-Performance Trade-off
A score-sparsity balance mechanism enables flexible control between computational efficiency and accuracy, accommodating diverse deployment scenarios.

## Results

- On COCO, single-object classification accuracy improved from 19.1% to 87.3% while skipping over **40% of Transformer blocks**
- Up to **45.4% latency reduction** on Autoware.Universe for autonomous driving
- Practical acceleration for latency-sensitive applications
