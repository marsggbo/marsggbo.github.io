---
layout: page
title: "BatteryGate | macOS Charge Limit Control"
description: >
  A macOS menu-bar app that pauses charging at an upper threshold and resumes
  at a lower one — for long-term battery health. Reverse-engineers the SMC charging
  interface, with pluggable backends and zero telemetry.
importance: 3
category: tools
img: assets/img/project_preview/batterygate.png
github: https://github.com/marsggbo/BatteryGate
github_stars: marsggbo/BatteryGate
---

## English

**GitHub:** [marsggbo/BatteryGate](https://github.com/marsggbo/BatteryGate) · Requires macOS 14+

macOS has no public API for controlling charging. BatteryGate works around this by talking to the SMC (System Management Controller) directly via the private `CH0B`/`CH0C` keys — a privileged LaunchDaemon helper is the only component that runs as root, communicating with the GUI over a UNIX socket that exposes exactly five verbs: `status` / `pause` / `resume` / `reset` / `limit`.

### Highlights

- 🔋 Menu-bar battery percentage plus live telemetry: voltage, current, temperature, cycle count, health
- 🎚️ Upper-limit slider (50–100%) stops charging, lower-limit slider (5–95%) resumes it
- 🖥️ Manual "pause / resume charging now" buttons
- 🧩 Three pluggable backends: Dry Run (read-only demo), built-in SMC helper, or your own external command
- 🚀 Optional launch-at-login
- 📊 Detail panel with raw health, design capacity, and real-time power draw
- 🔒 No App Store, no telemetry, no cloud — everything stays on your Mac

The clean GUI/daemon separation and the tiny five-verb socket protocol were designed to make the privileged surface easy to audit.

---

## 中文版本

**GitHub：** [marsggbo/BatteryGate](https://github.com/marsggbo/BatteryGate) · 需要 macOS 14+

macOS 没有公开的充电控制 API。BatteryGate 通过私有 SMC（系统管理控制器）密钥 `CH0B`/`CH0C` 直接与硬件通信——只有一个 LaunchDaemon 守护进程以 root 运行，通过 UNIX socket 与界面通信，且只暴露五个动词：`status` / `pause` / `resume` / `reset` / `limit`。

### 核心功能

- 🔋 菜单栏实时电量，附电压、电流、温度、循环次数、健康度等遥测数据
- 🎚️ 上限滑杆（50–100%）控制停止充电，下限滑杆（5–95%）控制恢复充电
- 🖥️ 手动「立即暂停 / 恢复充电」按钮
- 🧩 三种可插拔后端：Dry Run（只读演示）、内置 SMC 守护进程、自定义外部命令
- 🚀 可选开机自启
- 📊 详情面板展示原始健康度、设计容量、实时功率
- 🔒 无 App Store、无遥测、无云端，一切数据留在本机

界面与守护进程的分离设计、只暴露五个动词的极小 socket 协议，都是为了让特权攻击面足够小、便于审计。
