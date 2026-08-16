---
layout: page
title: "MyPaperTrend | Google Scholar Citation Trends"
description: >
  A zero-config, local-first Chrome extension that tracks Google Scholar
  citations over time and overlays the trend chart right on the Scholar profile page.
  No scraping API, no server, nothing leaves your browser.
importance: 2
category: tools
img: assets/img/project_preview/mypapertrend.png
github: https://github.com/marsggbo/MyPaperTrend
github_stars: marsggbo/MyPaperTrend
---

## English

**GitHub:** [marsggbo/MyPaperTrend](https://github.com/marsggbo/MyPaperTrend)

Google Scholar shows your citation count, but not *how it evolves*. Server-side scrapers get CAPTCHA'd fast — MyPaperTrend instead runs a **content script that reads the page you already opened** in your own logged-in session, making no extra network request, so there is nothing for anti-bot systems to detect.

### Highlights

- 📈 **Trend panel on any Scholar profile** — total-citations chart with hover crosshair (date · total · daily new), styled to match Scholar
- 🔍 **Per-paper history** — a 📈 icon under each paper expands its own citation trend inline
- 👥 **Follow any scholar** — flip the *Track* switch on a profile; each scholar is stored separately, nothing is recorded until you opt in
- ⏰ **Daily auto-update** — refreshes all tracked scholars in a hidden tab, no API key needed
- ☁️ **Optional cloud sync** — sign in with GitHub or Google Drive to sync one JSON file per scholar to *your own* account
- 🗂 **Full dashboard** — stat cards, all-time trend, milestones, per-paper charts, search/sort, NEW badges, profile switcher
- 🌐 **English & 中文** interface

### Privacy

History lives in your browser by default; export/import JSON anytime. Optional metadata lookups send only a paper title to the free Semantic Scholar API.

---

## 中文版本

**GitHub：** [marsggbo/MyPaperTrend](https://github.com/marsggbo/MyPaperTrend)

Google Scholar 只显示引用数，却看不到**引用是如何增长的**。服务器端爬虫很快会被弹验证码——MyPaperTrend 改用 **content script 读取你自己打开的、登录状态下的页面**，不发起任何额外请求，反爬系统无从检测。

### 核心功能

- 📈 **趋势面板**：任意学者主页顶部显示总引用曲线，悬停显示「日期 · 总引用 · 当日新增」
- 🔍 **每篇论文历史**：论文下方的 📈 图标点击就地展开该论文的引用趋势
- 👥 **追踪任意学者**：打开「追踪」开关才开始记录，每位学者分开存储
- ⏰ **每日自动更新**：隐藏标签页自动刷新所有已追踪学者，无需任何 API
- ☁️ **可选云同步**：用 GitHub 或 Google Drive 登录，把每位学者同步为 JSON 文件存到你自己的账号
- 🗂 **完整仪表盘**：统计卡片、总趋势、里程碑、单篇图表、搜索排序、NEW 徽标
- 🌐 中英双语界面

### 隐私

数据默认只存在本地浏览器，随时导出/导入 JSON；可选的元数据查询只把论文标题发给免费的 Semantic Scholar API。
