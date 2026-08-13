---
layout: post
title: Codex 同时兼容 AWS Bedrock 和 OneAPI 的配置教程
date: "2026-08-13"
tags: [Codex, AWS Bedrock, OneAPI, LLM, 工具]
category: techniques
grammar_cjkRuby: true
related_posts: false
toc:
  sidebar: left
---

# Codex 同时兼容 AWS Bedrock 和 OneAPI 的配置教程

最近公司内部同时提供了两种 LLM 接入方式：

- `AWS Bedrock`
- `OneAPI`

而 Codex 默认只会读取一套 `~/.codex/config.toml`。如果直接在同一个文件里来回改 `model_provider`，短期能用，长期一定会乱：

- 今天切到 Bedrock，明天忘了切回来；
- OneAPI 的 key 和 Bedrock 的配置混在一起，不容易排查；
- 以后再加第三种 provider，整个配置会更难维护。

我最后采用的方案是：

> **保留一份基础配置 + 用 Codex 原生 `profile` 机制切换不同 provider。**

这个方案有几个优点：

- 不需要维护多份 `CODEX_HOME`
- 不需要来回覆盖 `config.toml`
- 切换命令简单，`codex --profile xxx` 即可
- Bedrock 和 OneAPI 的配置可以完全分开，后面继续扩展也方便

下面把完整做法记录一下。

---

## 1. 最终目录结构

我最后采用的是下面这套结构：

```bash
~/.codex/
├── .env
├── config.toml
├── bedrock.config.toml
└── oneapi.config.toml
```

含义分别是：

- `config.toml`：基础配置
- `bedrock.config.toml`：AWS Bedrock 专用 profile
- `oneapi.config.toml`：OneAPI 专用 profile
- `.env`：OneAPI 的 API Key

---

## 2. 为什么不用一份 config.toml 硬切

最直接的想法其实是：

```toml
model_provider = "amazon-bedrock"
```

用 Bedrock 时写这个，用 OneAPI 时改成：

```toml
model_provider = "oneapi"
```

但这个方案的问题是：**每次切 provider 都要改配置文件本身**。

一旦你还有下面这些差异项：

- `model`
- `model_reasoning_effort`
- `base_url`
- `API Key`
- `AWS region`

那来回切换就会很烦，而且很容易忘。

Codex 本身原生支持 `profile` 文件，所以更好的思路是：

- `~/.codex/config.toml` 放基础配置
- `~/.codex/bedrock.config.toml` 放 Bedrock 差异
- `~/.codex/oneapi.config.toml` 放 OneAPI 差异

然后使用：

```bash
codex --profile bedrock
codex --profile oneapi
```

来切换。

---

## 3. 基础配置：`~/.codex/config.toml`

我这里保留了当前默认使用 Bedrock 的基础配置：

```toml
model = "openai.gpt-5.6-sol"
model_reasoning_effort = "medium"
model_provider = "amazon-bedrock"

[model_providers.amazon-bedrock.aws]
region = "us-east-2"
```

这里有两个注意点：

1. 这份基础配置可以直接默认走 Bedrock。
2. 如果你本机还有 `trusted project` 之类的个人配置，可以继续保留，但不要写进公开教程。

---

## 4. Bedrock profile：`~/.codex/bedrock.config.toml`

Bedrock 的 profile 单独拆出来如下：

```toml
model = "openai.gpt-5.6-sol"
model_reasoning_effort = "medium"
model_provider = "amazon-bedrock"

[model_providers.amazon-bedrock.aws]
region = "us-east-2"
```

这份配置很薄，只保留跟 Bedrock 强相关的差异项即可。

如果你们公司用的是 AWS SSO 或命名好的 profile，也可以继续走 AWS 标准凭证链，例如：

```bash
export AWS_PROFILE=codex-bedrock
```

或者事先：

```bash
aws sso login --profile codex-bedrock
```

---

## 5. OneAPI profile：`~/.codex/oneapi.config.toml`

OneAPI 这边的 profile 如下：

```toml
model = "gpt-5.5"
model_reasoning_effort = "xhigh"
model_provider = "oneapi"
disable_response_storage = true

[model_providers.oneapi]
name = "oneapi"
base_url = "https://oneapi-comate.baidu-int.com/v1"
wire_api = "responses"
env_key = "ONEAPI_API_KEY"
```

这里我**没有**采用把 key 写进 `auth.json` 的方案，而是改成：

```toml
env_key = "ONEAPI_API_KEY"
```

原因很简单：

- `OneAPI` 在 Codex 里本质上是一个**自定义 provider**
- 对自定义 provider，使用环境变量提供 key 更干净
- 这样不会和 Bedrock 的认证逻辑混在一起

换句话说，**Bedrock 走 AWS 凭证链，OneAPI 走环境变量 key，各自独立，边界清晰。**

---

## 6. OneAPI 的 key 放哪里

我最后把 OneAPI 的 key 放进了：

```bash
~/.codex/.env
```

内容如下：

```bash
ONEAPI_API_KEY=你的_oneapi_key
```

这样做的好处是：

- key 不需要硬编码进 `config.toml`
- 后面如果还有第二个自定义 provider，也可以继续往 `.env` 里加
- `~/.codex` 本身就是 Codex 的本地目录，管理起来更集中

---

## 7. 让 zsh 自动加载 `~/.codex/.env`

如果只是把 key 写进 `~/.codex/.env`，但 shell 没有自动导出这个变量，那么运行 `codex --profile oneapi` 时还是读不到它。

所以我在 `~/.zshrc` 里加了这段：

```bash
if [ -f "$HOME/.codex/.env" ]; then
  set -a
  . "$HOME/.codex/.env"
  set +a
fi
```

这段的作用是：

- 自动读取 `~/.codex/.env`
- 并把里面的变量导出成环境变量

这样 OneAPI 的 `env_key = "ONEAPI_API_KEY"` 才能真正生效。

---

## 8. 再加两个快捷命令

为了避免每次都手敲完整参数，我在 `~/.zshrc` 里顺手加了两个 alias：

```bash
alias codexb='codex --profile bedrock'
alias codexo='codex --profile oneapi'
```

然后重新加载 shell：

```bash
source ~/.zshrc
```

之后就可以直接：

```bash
codexb
codexo
```

来切换。

---

## 9. 实际使用方式

### 9.1 使用 Bedrock

```bash
codexb
```

或者显式写成：

```bash
codex --profile bedrock
```

### 9.2 使用 OneAPI

```bash
codexo
```

或者：

```bash
codex --profile oneapi
```

---

## 10. 如何验证 OneAPI 配置是否真的生效

最简单的办法不是盯着配置文件看，而是直接跑一个最小请求。

例如：

```bash
codex exec --profile oneapi --ephemeral --sandbox read-only "Reply with exactly OK."
```

如果配置正确，Codex 启动时通常会显示类似信息：

```text
model: gpt-5.5
provider: oneapi
reasoning effort: xhigh
```

然后正常返回：

```text
OK
```

我本地实际验证过，这套配置是可以跑通的。

---

## 11. 最后总结

如果你也遇到 Codex 需要同时兼容多种 LLM provider 的场景，我的建议是：

- **不要**在同一个 `config.toml` 里来回手改 `model_provider`
- **优先使用** Codex 原生的 `profile` 机制
- **自定义 provider 的 key** 优先走 `env_key`
- **Bedrock** 继续沿用 AWS 的标准认证链

最终这套方案的核心只有一句话：

> **基础配置放 `config.toml`，provider 差异拆到 `xxx.config.toml`，用 `--profile` 切换。**

这是目前我觉得最简单、最稳、也最好维护的做法。
