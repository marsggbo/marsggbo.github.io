---
layout: post
title: "Lang-PINN: 从自然语言到物理信息神经网络的多智能体框架 | From Language to PINNs via a Multi-Agent Framework"
date: 2026-03-12 15:10:00
category: research
tags: PINN LLM Multi-Agent Physics-Informed ICLR
related_posts: false
toc:
  sidebar: left
---

> 🔥 本文被 **ICLR 2026 Workshop on AI with Recursive Self-Improvement** 录用为 **Spotlight**！
>
> 🔥 This paper has been accepted as **Spotlight** at the **ICLR 2026 Workshop on AI with Recursive Self-Improvement**!

**论文链接 / Paper:** [arXiv:2510.05158](https://arxiv.org/abs/2510.05158)

**作者 / Authors:** Xin He, Liangliang You, Hongduan Tian, Bo Han, Ivor Tsang, Yew-Soon Ong

---

# 中文版

## 研究动机

物理信息神经网络（Physics-Informed Neural Networks, PINNs）是求解偏微分方程（PDEs）的强大工具，但传统构建流程繁琐且易出错：

- 科学家需要手动将实际问题转化为 PDE 形式
- 需要精心设计网络架构和损失函数
- 需要实现稳定的训练流程

这一过程对专业知识要求极高，限制了 PINNs 的广泛应用。

## 核心方法

Lang-PINN 提出了一个 **LLM 驱动的多智能体系统**，能够从自然语言任务描述自动构建可执行的 PINN 求解器。系统由四个协作智能体组成：

### 1. PDE Agent（PDE 智能体）
解析自然语言任务描述，提取偏微分方程中的算子、系数以及边界/初始条件，将其转化为符号化的 PDE 表示。

### 2. PINN Agent（PINN 智能体）
根据 PDE 的特征（周期性、几何复杂度、多尺度动态等），自动选择合适的神经网络架构和归纳偏置。

### 3. Code Agent（代码智能体）
生成模块化的、可执行的 PINN 训练代码实现。

### 4. Feedback Agent（反馈智能体）
执行代码，诊断错误，并向前面的阶段提供迭代式的修正反馈，确保最终输出的科学有效性。

## 实验结果

- **误差降低**：均方误差（MSE）降低了 3-5 个数量级
- **执行成功率**：端到端执行成功率提升超过 50%
- **计算效率**：时间开销减少高达 74%
- **可靠性**：在 1D 和 2D 场景下成功率超过 80%，而基线方法通常低于 35%

---

# English Version

## Motivation

Physics-Informed Neural Networks (PINNs) are powerful tools for solving partial differential equations (PDEs), but the traditional workflow for constructing PINNs is labor-intensive and error-prone:

- Scientists must manually formulate real-world problems as PDEs
- Careful design of network architectures and loss functions is required
- Implementing stable training pipelines demands deep domain expertise

These barriers significantly limit the broader adoption of PINNs.

## Key Methods

Lang-PINN introduces an **LLM-driven multi-agent system** that automatically constructs executable PINN solvers from natural language task descriptions. The system consists of four collaborative agents:

### 1. PDE Agent
Parses natural language task descriptions to extract PDE operators, coefficients, and boundary/initial conditions, transforming them into symbolic PDE representations.

### 2. PINN Agent
Automatically selects appropriate neural network architectures and inductive biases based on PDE characteristics (periodicity, geometric complexity, multiscale dynamics, etc.).

### 3. Code Agent
Generates modular, executable PINN training code.

### 4. Feedback Agent
Executes the generated code, diagnoses errors, and provides iterative corrections to earlier stages, ensuring scientifically valid outputs.

## Results

- **Error reduction**: MSE reduced by 3–5 orders of magnitude
- **Execution success**: End-to-end success rate improved by over 50%
- **Computational efficiency**: Time overhead reduced by up to 74%
- **Reliability**: Success rates exceed 80% in 1D and 2D regimes, vs. below 35% for baselines
