---
layout: page
title: "markdown2anything | Write Once, Publish Everywhere"
description: >
  A VS Code extension that turns Markdown into platform-ready content for
  WeChat, Zhihu, Xiaohongshu, Twitter/X threads, PPT, and Word — with LaTeX math,
  14 themes, live preview, and AI copywriting.
importance: 1
category: tools
img: assets/img/project_preview/markdown2anything.png
github: https://github.com/marsggbo/markdown2anything
github_stars: marsggbo/markdown2anything
---

## English

**GitHub:** [marsggbo/markdown2anything](https://github.com/marsggbo/markdown2anything) · **Install:** `ext install marsggbo.markdown2anything`

After writing a tech article, the real work begins: pasting it into WeChat breaks code blocks and LaTeX, Xiaohongshu needs screenshots and captions, Twitter needs a thread under 280 characters per tweet. **markdown2anything** turns all of this into a few clicks inside VS Code.

I built it for my own writing workflow — it powers the publishing pipeline behind my [content creation](../content-creation/) across Zhihu, WeChat, and Xiaohongshu.

### Highlights

| Feature | What it does |
|---|---|
| 🟢 WeChat Official Account | One-click copy of inline-styled HTML — layout, code highlighting, and LaTeX formulas all preserved; optional API upload to drafts |
| 🔵 Zhihu | Auto-opens the Zhihu editor with title, body, code, and formulas filled in |
| 📱 Xiaohongshu | HTML or phone-adaptive (1080×1440) long-image export, plus AI-generated title, body, and tags |
| 🐦 Twitter (X) | AI splits long posts into a thread under 280 chars, attaching images in order |
| 📊 PPT / Word | Slidev / Marp / Pandoc export — editable PPTX and DOCX with formulas converted to OMML |
| 🤖 AI copywriting | Works with any OpenAI-compatible API (DeepSeek, SiliconFlow, OpenRouter, Groq, Ollama…) |

### Themes & Preview

14 built-in themes (WeChat Classic, Claude Pro, Medium, Zhihu, Notion, Academic…), live side-by-side preview with bidirectional scroll sync between editor and preview.

### Privacy

API keys go into VS Code SecretStorage (system keychain), cookies stay local, and the final "Publish" button is always left for you to press.

---

## 中文版本

**GitHub：** [marsggbo/markdown2anything](https://github.com/marsggbo/markdown2anything) · **安装：** `ext install marsggbo.markdown2anything`

写完一篇技术文章，真正的工作才刚开始：粘贴到公众号代码块炸了、公式没了；小红书要截图配文案；Twitter 要拆串推。**markdown2anything** 把这些全部变成 VS Code 里的几次点击。

这个插件源于我自己的写作流程——它就是我在知乎、公众号、小红书等平台[内容创作](../content-creation/)背后的发布工具。

### 核心功能

| 功能 | 说明 |
|---|---|
| 🟢 微信公众号 | 一键复制带内联样式的 HTML，排版、代码高亮、LaTeX 公式全部保留；支持 API 上传草稿箱 |
| 🔵 知乎 | 自动打开知乎编辑器，标题 + 正文 + 代码 + 公式一次填好 |
| 📱 小红书 | HTML 截图或手机自适应（1080×1440）长图导出，AI 自动生成标题、正文、标签 |
| 🐦 Twitter (X) | AI 把长文拆成 280 字以内的串推，配图按顺序自动挂载 |
| 📊 PPT / Word | Slidev / Marp / Pandoc 导出，生成真正可编辑的 PPTX / DOCX，公式转 OMML |
| 🤖 AI 文案 | 兼容任意 OpenAI 接口（DeepSeek、硅基流动、OpenRouter、Groq、本地 Ollama…） |

### 主题与预览

内置 14 套主题（微信经典、Claude Pro、Medium、知乎、Notion、学术论文…），侧边实时预览，编辑区与预览区双向定位跳转。

### 安全

API Key 存入系统钥匙串，Cookie 只存本地，最后一下「发布」永远留给你自己点。
