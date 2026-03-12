---
layout: post
title: "Ghost in the Cloud: 地理分布式大模型训练的安全隐患 | Your Geo-distributed LLM Training is Easily Manipulated"
date: 2026-03-12 15:20:00
category: research
tags: LLM Security Federated-Learning Distributed-Training ICLR
related_posts: false
toc:
  sidebar: left
---

> 🎉 本文已被 **ICLR 2026** 录用！
>
> 🎉 This paper has been accepted by **ICLR 2026**!

**论文链接 / Paper:** [OpenReview](https://openreview.net/forum?id=FwnmQnVc7g)

**作者 / Authors:** Zichen Tang, Zhenheng Tang, Gaoning Pan, Buhua Liu, Xin He, Kunfeng Lai, Xiaowen Chu, Bo Li

---

# 中文版

## 研究动机

地理分布式大语言模型训练和联邦学习（Federated Learning）正在成为主流训练范式——多个参与方保留本地原始数据，仅交换模型更新（如梯度）。这种方式在保护数据隐私的同时实现了协作训练。

然而，本文揭示了一个严重的安全隐患：**单个恶意参与方就能通过注入越狱触发器（jailbreak triggers）来破坏整个模型的安全对齐**。

## 核心方法：CloudGhost 攻击

本文提出了 **CloudGhost** 攻击方法，核心机制包括：

### 攻击原理
- 恶意客户端在地理分布式训练环境中（如 Local-SGD 或 INTELLECT-1 范式）提交被污染的模型更新
- 利用 LoRA 微调等技术嵌入隐蔽的越狱触发器
- 通过散度最小化技术绕过良性更新的中和效应

### 防御绕过
论文分析了两种主流服务器端防御机制，并展示了 CloudGhost 如何绕过它们：

- **MOS（恶意输出审查）**：检测不安全的生成结果 → CloudGhost 通过隐蔽的触发器植入成功规避
- **TPC（任务性能检查）**：过滤降低下游性能的更新 → CloudGhost 采用"下游任务保持型恶意训练"，在注入触发器的同时维持正常任务性能

## 实验结果

- **攻击成功率（ASR）**：74-93%
- **检测真阳性率（DTR）**：低于 5%（即极难被检测）
- 在 LLaMA、Qwen、Mistral 等主流模型上验证有效
- 揭示了当前分布式训练安全机制的严重不足

---

# English Version

## Motivation

Geo-distributed LLM training and Federated Learning (FL) are becoming mainstream paradigms—multiple participants keep raw data local while exchanging only model updates (e.g., gradients). This enables collaborative training while preserving data privacy.

However, this paper reveals a critical security vulnerability: **a single malicious client can compromise the safety alignment of the entire model by injecting jailbreak triggers**.

## Core Method: CloudGhost Attack

The paper proposes the **CloudGhost** attack with the following mechanisms:

### Attack Principle
- A malicious client submits poisoned model updates in geo-distributed setups (e.g., Local-SGD or INTELLECT-1 paradigms)
- Jailbreak triggers are embedded via techniques like LoRA fine-tuning
- Divergence minimization techniques bypass neutralization by benign updates

### Defense Bypass
The paper analyzes two mainstream server-side defenses and demonstrates how CloudGhost circumvents them:

- **MOS (Malicious Output Scrutiny)**: Detects unsafe generations → CloudGhost evades through subtle trigger implantation
- **TPC (Task Performance Check)**: Filters updates that degrade downstream performance → CloudGhost uses "downstream-preserved malicious training" to maintain task performance while injecting triggers

## Results

- **Attack Success Rate (ASR)**: 74–93%
- **Detection True Rate (DTR)**: Below 5% (extremely difficult to detect)
- Validated on LLaMA, Qwen, and Mistral models
- Exposes critical gaps in current distributed training security mechanisms
