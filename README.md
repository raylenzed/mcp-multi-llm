# mcp-multi-llm

**Any AI agent as the host, any others as consultants.**

**任意 AI Agent 做主控，任意其他 Agent 做顾问。**

---

## The Idea / 起源

Modern AI coding agents (Claude Code, Codex CLI, etc.) are powerful individually. But what if they could **discuss with each other**?

现代 AI 编程 Agent 各自都很强大。但如果它们能**互相讨论**呢？

This MCP server enables **any MCP-compatible agent** to consult others as discussion partners — with full conversation context and persistent history. Built-in providers (Claude, Codex) use your CLI subscriptions. Any other model (Gemini, MiniMax, Moonshot, DeepSeek…) is added via a simple config file.

这个 MCP Server 让**任何支持 MCP 的 Agent** 都能将其他 Agent 作为讨论伙伴 — 保留完整的对话上下文。内置的 Claude 和 Codex 走 CLI 订阅，其他所有模型（Gemini、MiniMax、Moonshot、DeepSeek…）通过配置文件接入 API。

### Who can be the host? / 谁可以做主控？

Any agent that supports MCP can be the host:

| Host Agent / 主控 | Consults / 咨询 | How / 方式 |
|---|---|---|
| **Claude Code** | Codex + Gemini + others | Native MCP support |
| **Codex CLI** | Claude + Gemini + others | `codex mcp` config |
| **Any MCP client** | All configured providers | Standard MCP protocol |

## How It Works / 工作原理

```
┌──────────────────────────────────────┐
│  Any MCP-compatible Host Agent       │
│  (Claude Code / Codex / …)          │
└──────────────┬───────────────────────┘
               │ MCP Protocol (stdio)
               ▼
      ┌──────────────────┐
      │  mcp-multi-llm   │
      │  (FastMCP Server) │
      ├──────┬───────────┤
      │      │           │
      ▼      ▼           ▼
   ┌──────┐ ┌────┐  ┌──────────────────────┐
   │Claude│ │Codex│  │  Custom Providers     │
   │ CLI  │ │ CLI │  │ (Gemini, MiniMax,     │
   └──────┘ └────┘  │  Moonshot, GLM, …)    │
                     │  via HTTP API         │
                     └──────────────────────┘
```

1. The host agent calls MCP tools like `discuss_with_claude`, `discuss_with_gemini`, or `group_discuss`
2. Built-in providers (Claude, Codex) use CLI subprocesses; custom providers call HTTP APIs directly
3. Responses are returned with conversation history maintained per topic
4. The host agent synthesizes all perspectives

## Features / 功能

| Feature | Description |
|---------|-------------|
| **Host-agnostic** | Any MCP client can be the host |
| **Multi-LLM Discussion** | Claude (CLI), Codex (CLI), plus any API-based provider |
| **Custom Providers (OpenAI)** | Any `/chat/completions`-compatible model via config file |
| **Custom Providers (Anthropic)** | Any `/v1/messages`-compatible model via config file |
| **Extra Request Body** | Inject extra fields per provider (e.g. disable Kimi thinking mode) |
| **CLI ↔ API Switch** | Claude/Codex can be switched from CLI to direct API via settings |
| **Context Continuity** | Conversations scoped by topic, context carries across rounds |
| **Parallel Consultation** | `group_discuss` queries all available LLMs simultaneously |
| **App Factory Roundtable** | `app_factory_roundtable` runs a structured two-round market review for app-factory skills |
| **Persistent History** | Conversation history saved to disk, survives restarts |
| **Project Isolation** | Each project's history stored separately via `MCP_PROJECT` env var |
| **Language Control** | All providers respond in your language via `MCP_LANGUAGE` env var |
| **Web Console** | Local web UI (`viewer.py`) with standalone chat, history, and App Factory roundtable runner |

| 功能 | 说明 |
|------|------|
| **主控无关** | 任何 MCP 客户端都能做主控 |
| **多 LLM 讨论** | Claude（CLI）、Codex（CLI），加上任意 API 模型 |
| **自定义模型 (OpenAI 协议)** | 通过配置文件接入任意 `/chat/completions` 兼容模型 |
| **自定义模型 (Anthropic 协议)** | 通过配置文件接入任意 `/v1/messages` 兼容模型 |
| **自定义请求体** | 每个 provider 可注入额外字段（如关闭 Kimi 思考模式） |
| **CLI ↔ API 切换** | Claude/Codex 可通过 settings 文件切换到直接 API 模式 |
| **上下文连续** | 按主题维护对话，多轮讨论保持上下文 |
| **并行咨询** | `group_discuss` 同时向所有可用模型提问 |
| **App Factory 圆桌评审** | `app_factory_roundtable` 为 app-factory skills 执行结构化两轮选题评审 |
| **持久化历史** | 对话历史保存到磁盘，重启不丢失 |
| **项目隔离** | 通过 `MCP_PROJECT` 环境变量按项目分开存储历史 |
| **语言控制** | 通过 `MCP_LANGUAGE` 环境变量让所有模型用指定语言回复 |
| **Web 控制台** | 本地 Web UI（`viewer.py`），可发起独立对话、查看历史并运行 App Factory 圆桌评审 |

## Prerequisites / 前提条件

- **Python** >= 3.13
- **[uv](https://docs.astral.sh/uv/)** — Python package manager
- At least one of the following:
  - **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** (CLI) — `npm install -g @anthropic-ai/claude-code`
  - **[Codex CLI](https://github.com/openai/codex)** (CLI) — `npm install -g @openai/codex`
  - Any API-based provider configured in `custom_providers.json` (see below)

## Installation / 安装

```bash
git clone https://github.com/RaylenZed/mcp-multi-llm.git
cd mcp-multi-llm
uv sync
```

## Configuration / 配置

### As Claude Code host / Claude Code 做主控

Add to `~/.claude.json` under `mcpServers`:

在 `~/.claude.json` 的 `mcpServers` 中添加：

```json
{
  "mcpServers": {
    "multi-llm": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-multi-llm", "python", "server.py"],
      "type": "stdio",
      "env": {
        "MCP_HOST_PROVIDER": "claude",
        "MCP_PROJECT": "my-project-name",
        "MCP_LANGUAGE": "zh"
      }
    }
  }
}
```

### As Codex CLI host / Codex CLI 做主控

```bash
codex mcp add multi-llm -- uv run --directory /path/to/mcp-multi-llm python server.py
```

Then edit `~/.codex/config.json` to add env vars under the `multi-llm` entry:

然后在 `~/.codex/config.json` 的 `multi-llm` 条目中添加 env：

```json
{
  "mcpServers": {
    "multi-llm": {
      "command": "...",
      "env": {
        "MCP_HOST_PROVIDER": "codex",
        "MCP_PROJECT": "my-project-name",
        "MCP_LANGUAGE": "zh"
      }
    }
  }
}
```

Restart your agent and the tools will be available.

重启你的 Agent 后即可使用。

### Environment Variables / 环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `MCP_HOST_PROVIDER` | 当前主控 AI 的名称，跳过注册对应的 `discuss_with_*` 工具 | `claude`、`codex` |
| `MCP_PROJECT` | 项目标识，历史记录存入独立子目录（不设则取 CWD 目录名） | `my-web-app` |
| `MCP_LANGUAGE` | 语言偏好，所有 AI 回复都会使用该语言 | `zh`、`en`、`ja` |
| `MCP_DISABLED_PROVIDERS` | 逗号分隔的 provider 名称，完全不加载、不注册对应工具 | `gemini`、`gemini,deepseek` |

支持的语言代码：`zh`（中文）、`en`（英文）、`ja`（日文）、`ko`（韩文）、`fr`（法文）、`de`（德文）、`es`（西班牙文）、`ru`（俄文）、`pt`（葡萄牙文）

---

## Custom Providers / 自定义模型

Add any API-based model by editing `~/.mcp-multi-llm/custom_providers.json`. Each entry gets its own MCP tool automatically.

通过编辑 `~/.mcp-multi-llm/custom_providers.json` 接入任意 API 模型，每个模型自动注册为独立 MCP 工具。

### OpenAI-compatible protocol (default)

```json
[
  {
    "name": "gemini",
    "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "model": "gemini-2.5-flash-preview-04-17",
    "api_key": "your-gemini-api-key"
  },
  {
    "name": "deepseek",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-chat",
    "api_key": "your-key"
  },
  {
    "name": "Moonshot",
    "base_url": "https://api.moonshot.cn/v1",
    "model": "kimi-k2.5",
    "api_key": "your-key",
    "extra_body": {
      "thinking": {"type": "disabled"}
    }
  }
]
```

### Anthropic-compatible protocol

For providers that implement the Anthropic `/v1/messages` API format:

对于实现了 Anthropic `/v1/messages` 格式的 provider：

```json
[
  {
    "name": "MyProvider",
    "base_url": "https://your-provider.com",
    "model": "some-model",
    "api_key": "your-key",
    "protocol": "anthropic"
  }
]
```

### Field reference / 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `name` | ✅ | 工具名称，只能用字母/数字/下划线，不能是 `claude` / `codex` |
| `base_url` | ✅ | API 基础地址 |
| `model` | ✅ | 模型名称 |
| `api_key` | ✅ | API Key |
| `protocol` | — | `"openai"`（默认）或 `"anthropic"` |
| `extra_body` | — | 合并进请求 body 的额外字段（仅 openai 协议有效） |
| `capabilities` | — | 声明 provider 能力，例如 `["web_search"]`；用于 App Factory 第二轮独立自采 |

重启 MCP 服务后，每个模型会自动注册为独立工具：`discuss_with_gemini`、`discuss_with_deepseek`…

如果某个 API provider 本身具备联网搜索能力，可以这样声明，让它参与 `app_factory_roundtable` 的第二轮独立自采：

```json
[
  {
    "name": "searchy",
    "base_url": "https://example.com/v1",
    "model": "search-model",
    "api_key": "your-key",
    "capabilities": ["web_search"]
  }
]
```

Claude/Codex/Gemini 等 CLI provider 默认视为可自采；普通 HTTP API provider 默认只参与评审，不计入独立搜索证据。


### Common providers / 常用 provider 参考

| 模型 | base_url | 常用 model |
|------|----------|-----------|
| **Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-chat` |
| **阿里千问** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus`、`qwen-max` |
| **Moonshot (Kimi)** | `https://api.moonshot.cn/v1` | `kimi-k2.5` |
| **MiniMax** | `https://api.minimaxi.com/v1` | `MiniMax-M2.7` |
| **智谱 GLM** | `https://open.bigmodel.cn/api/paas/v4` | `glm-4` |
| **Groq** | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |

---

## Switching Claude/Codex to API mode / Claude/Codex 切换为 API 模式

Claude 和 Codex 默认走 CLI。如果想切换为直接 API 模式，直接在 `custom_providers.json` 里加上对应条目即可，和其他模型完全一样：

```json
[
  {
    "name": "claude",
    "base_url": "https://api.anthropic.com",
    "model": "claude-opus-4-6",
    "api_key": "sk-ant-your-key",
    "protocol": "anthropic"
  },
  {
    "name": "codex",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "api_key": "sk-your-key"
  }
]
```

有 `claude`/`codex` 条目时自动走 API；没有则 fallback 到 CLI，无需任何改动。

---

## Usage Guide / 使用教程

### 基本概念

| 概念 | 解释 |
|------|------|
| **主控 (Host)** | 你正在对话的那个 AI，比如 Claude Code |
| **顾问 (Consultant)** | 主控通过这个 MCP server 去调用的其他 AI |
| **话题 (topic)** | 一个对话的标签，同一 topic 的多轮对话共享上下文记忆 |

**你不需要手动调用任何工具。** 用自然语言告诉主控 AI "去问问 Gemini" 或 "让所有模型评审一下"，主控会自行决定调用哪个工具。

---

### 场景 1：让所有模型评审你的 PRD

```
我有一个新产品的 PRD，请用 group_discuss 让所有模型同时评审，
topic 用 "prd-review"。

[粘贴你的 PRD 内容]
```

**追问（上下文连续）：**

```
针对 Gemini 提到的"技术可行性问题"，让所有模型继续讨论解决方案，
topic 还是 "prd-review"。
```

---

### 场景 2：只问某一个模型

```
用 discuss_with_gemini 搜索一下"AI 代码审查工具"的竞品现状，
topic 用 "competitor-research"。
```

---

### 场景 3：多轮讨论

**第一轮：**
```
用 discuss_with_gemini 评审这个数据库设计方案，topic 用 "db-design"。
[粘贴方案]
```

**第二轮：**
```
Gemini 说了上面这些，用 discuss_with_codex 让 Codex 评价 Gemini 的意见，
topic 还是 "db-design"。
```

**第三轮：**
```
用 group_discuss 问一下：综合前面的讨论，这个设计最大的风险点是什么？
topic 还是 "db-design"。
```

---

### 场景 4：App Factory 多模型圆桌评审

`app_factory_roundtable` 是给 app-factory skills 使用的结构化工具。它会自动健康检查、选择 2-5 个可用模型、执行两轮评审，并返回 JSON + 中文摘要。

```json
{
  "app_name": "Tiny Invoice",
  "app_mode": "local",
  "pitch": "=== PITCH ===\nApp: Tiny Invoice\n...",
  "data_brief": "=== DATA BRIEF v1 ===\n关键词热度...\n...",
  "keywords": ["invoice tracker", "receipt log"],
  "min_participants": 2,
  "max_participants": 5
}
```

返回核心字段：

| 字段 | 说明 |
|------|------|
| `status` | `completed` / `partial` / `skipped` |
| `participants` | 实际参与模型、健康状态、是否具备独立自采能力 |
| `rounds` | 第一轮本地数据评审、第二轮模型自采评审、各自投票 |
| `comparison` | 本地数据与模型自采数据的一致性检查 |
| `final_gate` | `PROCEED_TO_EVIDENCE_CARD` / `NEEDS_EVIDENCE` / `KILL` / `SKIPPED` |
| `summary_markdown` | 可直接写入 app-factory Phase 报告的中文摘要 |

注意：这个工具永远不会输出 `BUILD_READY`。它最多只允许进入证据卡和最终风控，保留 app-factory skills 的硬闸门语义。

---

### 常见问题

**Q: 某个模型没有响应怎么办？**

运行 `list_available_providers` 工具，它会列出哪些 provider 可用、哪些不可用。不可用的 provider 会被自动跳过。

**Q: 对话历史会一直保留吗？**

会，保存在 `~/.mcp-multi-llm/history/{project}/`（配置了 `MCP_PROJECT` 时）或 `~/.mcp-multi-llm/history/`（未配置时）。用 `clear_discussion` 清除指定 topic，或换一个新的 topic 名字开启全新讨论。

**Q: topic 名字有什么规则吗？**

没有规则，随便起。建议用英文短横线格式，比如 `prd-review`、`api-design`、`db-v2`。同一 topic 下各模型各自维护独立的对话记录。

---

## MCP Tools / 工具列表

| 工具 | 说明 |
|------|------|
| `discuss_with_claude` | 向 Claude 提问（CLI 模式默认） |
| `discuss_with_codex` | 向 Codex 提问（CLI 模式默认） |
| `discuss_with_<name>` | 向任意自定义 provider 提问（自动注册） |
| `group_discuss` | 并行向所有可用模型提问 |
| `app_factory_roundtable` | 为 app-factory 执行两轮多模型选题评审，返回结构化 JSON |
| `list_available_providers` | 列出所有可用 / 不可用的 provider |
| `health_check_providers` | 测试各 provider 的连通性和鉴权状态 |
| `dismiss_provider` | 临时将某 provider 排除在本次会话之外 |
| `list_discussions` | 列出所有进行中的讨论主题 |
| `clear_discussion` | 清除某个主题的对话历史 |

---

## Web Console / Web 控制台

本地 Web 控制台，可发起独立多模型对话、浏览历史，也可直接运行 App Factory 圆桌评审。

A local web console for standalone multi-model chat, conversation history, and App Factory roundtable reviews.

```bash
uv run python viewer.py        # http://localhost:7432
uv run python viewer.py 8080   # 自定义端口
```

### Features / 功能

- **项目选择器** — 顶部下拉菜单切换项目，各项目历史独立展示
- **独立对话** — 选择单个 provider 或 `group_discuss`，用新 topic 发起隔离会话
- **圆桌评审** — 填入 App Name、mode、Pitch、DATA BRIEF v1，调用 `app_factory_roundtable`
- **结构化结果** — 展示 `status`、`final_gate`、参与模型、中文摘要和完整 JSON
- **私聊视图** — 单 provider 对话，气泡+头像布局（类微信）
- **群聊视图** — 多 provider 同一 topic 的消息按时间合并，2×2 头像展示（类群聊）
- **自动刷新** — 每 3 秒拉取最新消息，实时跟进进行中的讨论
- **搜索过滤** — 侧边栏支持按 topic / provider 名搜索

### Provider Logos / 服务商 Logo

在项目根目录创建 `logos/` 文件夹，放入各服务商的 logo 图片，文件名与 `custom_providers.json` 中的 `name` 字段一致：

```
logos/
├── claude.png
├── codex.png
├── gemini.png
├── GLM.png
├── MiniMax.png
└── Moonshot.png
```

支持 `.png`、`.jpg`、`.jpeg`、`.svg`、`.webp` 格式。找不到 logo 时自动 fallback 到彩色首字母头像。

---

## Architecture / 架构

```
mcp-multi-llm/
├── server.py                       # FastMCP server, dynamic tool registration
├── app_factory_roundtable.py       # App Factory two-round evaluation orchestrator
├── viewer.py                       # Local web chat viewer
├── logos/                          # Provider logo images (optional)
│   ├── claude.png
│   └── ...
├── sessions/
│   ├── base.py                     # BaseSession (abstract) + CLISession (subprocess)
│   ├── api_session.py              # APISession (httpx base)
│   ├── claude_session.py           # Claude CLI provider
│   ├── codex_session.py            # Codex CLI provider
│   ├── gemini_session.py           # Gemini CLI provider
│   ├── openai_compat_session.py    # OpenAI-compatible API providers
│   ├── anthropic_compat_session.py # Anthropic-compatible API providers
│   └── provider_config.py          # Load custom_providers.json
├── history/
│   └── store.py                    # Conversation history persistence (project-scoped)
└── pyproject.toml
```

History is stored at `~/.mcp-multi-llm/history/{project}/` when `MCP_PROJECT` is set.

历史记录在设置 `MCP_PROJECT` 后保存于 `~/.mcp-multi-llm/history/{project}/`。

### CLI vs API / 为什么 Claude/Codex 默认走 CLI？

| | CLI | API |
|---|---|---|
| **Cost / 费用** | 使用已有订阅 | 按 token 计费 |
| **Tools / 工具** | 完整工具链（搜索、文件、代码执行） | 纯文本对话 |
| **Capability / 能力** | Agent 级别 | Chat 级别 |

其他模型（Gemini、MiniMax、Moonshot 等）无官方 CLI 工具，直接走 API 是唯一选项。

## License / 许可

MIT
