---
layout: post
title: "ExpertFlow: 基于预测性专家缓存与令牌调度的高效MoE推理 | Efficient MoE Inference via Predictive Expert Caching and Token Scheduling"
date: 2026-03-12 15:00:00
category: research
tags: MoE Inference Expert-Caching Token-Scheduling DAC LLM
related_posts: false
toc:
  sidebar: left
---

> 🎉 本文已被 **DAC 2026** 录用！
>
> 🎉 This paper has been accepted by **DAC 2026**!

**论文链接 / Paper:** [arXiv:2410.17954](https://arxiv.org/abs/2410.17954)

**作者 / Authors:** Xin He, Shunkang Zhang, Kaijie Tang, Shaohuai Shi, Yuxin Wang, Zihao Zeng, Zhenheng Tang, Xiaowen Chu, Haiyan Yin, Ivor Tsang, Ong Yew Soon

---

# 中文版

## 研究动机

稀疏混合专家模型（Mixture-of-Experts, MoE）通过仅激活部分专家来实现高效推理，但在实际部署中面临两大核心挑战：

1. **GPU 显存占用巨大**：MoE 模型包含大量专家参数，远超单张 GPU 的显存容量
2. **专家切换开销高**：推理过程中不同 token 需要激活不同的专家，专家在 GPU 和 CPU 之间的频繁迁移导致严重的 I/O 瓶颈

## 核心方法

ExpertFlow 提出了两项关键技术来解决上述问题：

### 1. 预测性专家缓存（Predictive Expert Caching）

- 设计了一个轻量级预测器，在实际计算之前预测下一步需要激活的专家路由路径
- 利用预测结果提前将所需专家加载到 GPU，实现主动式预取（proactive prefetching）
- 引入实时纠错机制，在预测出错时快速修正，保证缓存命中率

### 2. 动态令牌调度（Dynamic Token Scheduling）

- 对输入 token 进行跨批次重新编排，使得每个批次内激活的专家数量最少
- 减少每步推理的专家切换次数，提升计算效率

## 实验结果

- **GPU 显存节省高达 93.72%**
- **推理速度提升 2-10 倍**
- 适用于资源受限的部署场景，为大规模 MoE 模型的实际应用提供了可行方案

---

# English Version

## Motivation

Sparse Mixture-of-Experts (MoE) models achieve efficient inference by activating only a subset of experts per token. However, real-world deployment faces two critical challenges:

1. **Massive GPU memory footprint**: MoE models contain a huge number of expert parameters that far exceed the memory capacity of a single GPU.
2. **High expert swapping overhead**: Different tokens activate different experts during inference, causing frequent expert transfers between GPU and CPU memory—a severe I/O bottleneck.

## Key Methods

ExpertFlow introduces two core techniques:

### 1. Predictive Expert Caching

- A lightweight predictor forecasts expert routing paths before actual computation begins.
- Predicted results enable proactive expert prefetching onto the GPU, eliminating reactive loading delays.
- A real-time error correction mechanism quickly adjusts when predictions are incorrect, maintaining high cache hit ratios.

### 2. Dynamic Token Scheduling

- Input tokens are rearranged across batches to minimize the number of activated experts per batch.
- This reduces the number of expert swaps per inference step, improving computational throughput.

## Results

- **Up to 93.72% GPU memory savings**
- **2-10× inference speedup** compared to baselines
- Practical and deployable solution for resource-constrained environments running large-scale MoE models
