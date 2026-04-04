#!/usr/bin/env python3
"""Local web viewer for mcp-multi-llm conversation history.

Chat-style UI with project isolation, group chat merging, and provider logos.

Usage:
    python viewer.py          # starts on http://localhost:7432
    python viewer.py 8080     # custom port
"""

import json
import mimetypes
import sys
import webbrowser
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HISTORY_BASE = Path.home() / ".mcp-multi-llm" / "history"
LOGOS_DIR = Path(__file__).parent / "logos"
DEFAULT_PORT = 7432


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def list_projects() -> list[str]:
    """Return project names (subdirectories under history/)."""
    if not HISTORY_BASE.exists():
        return []
    projects = []
    for p in sorted(HISTORY_BASE.iterdir()):
        if p.is_dir() and any(p.glob("*.json")):
            projects.append(p.name)
    # Also include root-level .json files as "default" project (legacy compat)
    if any(HISTORY_BASE.glob("*.json")):
        if "default" not in projects:
            projects.insert(0, "default")
    return projects


def _project_dir(project: str) -> Path:
    d = HISTORY_BASE / project
    if d.is_dir():
        return d
    # Legacy: root-level files for "default"
    if project == "default":
        return HISTORY_BASE
    return d


def load_conversations(project: str) -> list[dict]:
    """Load all conversations for a project, grouped by topic.

    Returns a list of conversation dicts:
      { topic, type("private"|"group"), providers: [...], last_time, preview }
    """
    d = _project_dir(project)
    if not d.exists():
        return []

    topic_map: dict[str, list[dict]] = defaultdict(list)
    for path in d.glob("*.json"):
        stem = path.stem
        parts = stem.split("_", 1)
        provider, topic = (parts[0], parts[1]) if len(parts) == 2 else ("unknown", stem)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        messages = data.get("messages", [])
        last_time = messages[-1]["timestamp"] if messages else data.get("created_at", 0)
        topic_map[topic].append({
            "provider": provider,
            "file": path.name,
            "message_count": len(messages),
            "last_time": last_time,
            "preview": messages[-1]["content"][:60] if messages else "",
        })

    conversations = []
    for topic, providers in topic_map.items():
        is_group = len(providers) > 1
        last_time = max(p["last_time"] for p in providers)
        total_msgs = sum(p["message_count"] for p in providers)
        preview = max(providers, key=lambda p: p["last_time"])["preview"]
        conversations.append({
            "topic": topic,
            "type": "group" if is_group else "private",
            "providers": [p["provider"] for p in providers],
            "files": [p["file"] for p in providers],
            "last_time": last_time,
            "message_count": total_msgs,
            "preview": preview,
        })

    conversations.sort(key=lambda c: c["last_time"], reverse=True)
    return conversations


def load_merged_chat(project: str, topic: str) -> dict:
    """Load and merge all provider messages for a topic, sorted by timestamp."""
    d = _project_dir(project)
    all_messages = []
    providers = []

    for path in d.glob(f"*_{topic}.json"):
        provider = path.stem.split("_", 1)[0]
        providers.append(provider)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for msg in data.get("messages", []):
            all_messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg.get("timestamp", 0),
                "provider": provider,
            })

    all_messages.sort(key=lambda m: m["timestamp"])
    is_group = len(providers) > 1
    return {
        "topic": topic,
        "type": "group" if is_group else "private",
        "providers": sorted(set(providers)),
        "messages": all_messages,
    }


# ---------------------------------------------------------------------------
# HTML Page
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-LLM Chat</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #ededed;
  --sidebar-bg: #fff;
  --sidebar-border: #e5e5e5;
  --sidebar-hover: #f5f5f5;
  --sidebar-active: #c7e0ff;
  --header-bg: #f7f7f7;
  --header-border: #ddd;
  --chat-bg: #f0efe8;
  --bubble-self: #95ec69;
  --bubble-self-text: #000;
  --bubble-other: #fff;
  --bubble-other-text: #000;
  --text-primary: #111;
  --text-secondary: #888;
  --text-muted: #b0b0b0;
  --avatar-size: 40px;
  --green: #07c160;
}
html, body { height: 100%; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB",
               "Microsoft YaHei", "Segoe UI", sans-serif;
  display: flex;
  height: 100vh;
  overflow: hidden;
  background: var(--bg);
  color: var(--text-primary);
}

/* ── Sidebar ── */
#sidebar {
  width: 280px;
  flex-shrink: 0;
  background: var(--sidebar-bg);
  border-right: 1px solid var(--sidebar-border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
#sidebar-top {
  padding: 12px 14px;
  border-bottom: 1px solid var(--sidebar-border);
  display: flex;
  flex-direction: column;
  gap: 8px;
}
#sidebar-top h1 { font-size: 16px; font-weight: 600; }
#project-select {
  width: 100%;
  padding: 6px 8px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 13px;
  background: #fff;
  outline: none;
}
#project-select:focus { border-color: var(--green); }
#search-input {
  width: 100%;
  padding: 7px 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 13px;
  background: #f5f5f5;
  outline: none;
}
#search-input:focus { background: #fff; border-color: var(--green); }

#conv-list {
  flex: 1;
  overflow-y: auto;
}
#conv-list::-webkit-scrollbar { width: 4px; }
#conv-list::-webkit-scrollbar-thumb { background: #ccc; border-radius: 2px; }

.conv-item {
  display: flex;
  align-items: center;
  padding: 12px 14px;
  gap: 10px;
  cursor: pointer;
  border-bottom: 1px solid #f0f0f0;
  transition: background 0.1s;
}
.conv-item:hover { background: var(--sidebar-hover); }
.conv-item.active { background: var(--sidebar-active); }

.conv-avatar {
  width: var(--avatar-size);
  height: var(--avatar-size);
  border-radius: 6px;
  flex-shrink: 0;
  position: relative;
  overflow: hidden;
}
.conv-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 6px;
}
.conv-avatar .fallback {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 15px;
  font-weight: 700;
  color: #fff;
  border-radius: 6px;
}
/* Group avatar: 2x2 grid */
.conv-avatar.group-av {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 1px;
  background: #ddd;
  border-radius: 6px;
}
.conv-avatar.group-av .mini-av {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 9px;
  font-weight: 700;
  color: #fff;
  overflow: hidden;
}
.conv-avatar.group-av .mini-av img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 0;
}

.conv-info { flex: 1; min-width: 0; }
.conv-title-row { display: flex; justify-content: space-between; align-items: center; }
.conv-name {
  font-size: 14px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}
.conv-time { font-size: 11px; color: var(--text-muted); flex-shrink: 0; margin-left: 8px; }
.conv-preview {
  font-size: 12px;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-top: 3px;
}
.conv-badge {
  display: inline-block;
  background: var(--green);
  color: #fff;
  font-size: 10px;
  font-weight: 600;
  padding: 1px 5px;
  border-radius: 8px;
  margin-right: 4px;
}

/* ── Main chat area ── */
#main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--chat-bg);
}
#chat-header {
  background: var(--header-bg);
  border-bottom: 1px solid var(--header-border);
  padding: 12px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}
#chat-header-title { font-size: 15px; font-weight: 600; }
#chat-header-sub { font-size: 12px; color: var(--text-secondary); margin-top: 1px; }

#messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px 60px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
#messages::-webkit-scrollbar { width: 5px; }
#messages::-webkit-scrollbar-thumb { background: #c8c8c8; border-radius: 3px; }

/* Empty state */
#empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  gap: 8px;
}
#empty-state .icon { font-size: 48px; opacity: 0.3; }
#empty-state p { font-size: 14px; }

/* ── Message rows ── */
.msg-row {
  display: flex;
  gap: 8px;
  max-width: 80%;
}
.msg-row.self { align-self: flex-end; flex-direction: row-reverse; }
.msg-row.other { align-self: flex-start; }

.msg-avatar {
  width: var(--avatar-size);
  height: var(--avatar-size);
  border-radius: 6px;
  flex-shrink: 0;
  overflow: hidden;
  margin-top: 2px;
}
.msg-avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 6px;
}
.msg-avatar .fallback {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 700;
  color: #fff;
  border-radius: 6px;
}

.msg-body { display: flex; flex-direction: column; gap: 2px; }
.msg-row.self .msg-body { align-items: flex-end; }

.msg-sender {
  font-size: 11px;
  color: var(--text-secondary);
  padding: 0 4px;
}

.msg-bubble {
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 14px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  position: relative;
  max-width: 100%;
}
.msg-row.self .msg-bubble {
  background: var(--bubble-self);
  color: var(--bubble-self-text);
  border-top-right-radius: 2px;
}
.msg-row.other .msg-bubble {
  background: var(--bubble-other);
  color: var(--bubble-other-text);
  border-top-left-radius: 2px;
}

.msg-time {
  font-size: 10px;
  color: var(--text-muted);
  padding: 0 4px;
}

/* Date divider */
.date-divider {
  text-align: center;
  font-size: 11px;
  color: var(--text-secondary);
  padding: 4px 12px;
  background: rgba(0,0,0,0.05);
  border-radius: 4px;
  align-self: center;
  margin: 4px 0;
}

/* Status bar */
#status-bar {
  padding: 6px 20px;
  background: var(--header-bg);
  border-top: 1px solid var(--header-border);
  font-size: 11px;
  color: var(--text-muted);
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-top">
    <h1>Multi-LLM Chat</h1>
    <select id="project-select"></select>
    <input id="search-input" type="text" placeholder="搜索对话...">
  </div>
  <div id="conv-list"></div>
</div>

<div id="main">
  <div id="chat-header" style="display:none">
    <div>
      <div id="chat-header-title"></div>
      <div id="chat-header-sub"></div>
    </div>
  </div>
  <div id="messages">
    <div id="empty-state">
      <div class="icon">💬</div>
      <p>选择一个对话开始查看</p>
    </div>
  </div>
  <div id="status-bar">
    <span id="status-text">每 3 秒自动刷新</span>
  </div>
</div>

<script>
const PROVIDER_COLORS = {
  claude: '#d97706', codex: '#16a34a', gemini: '#7c3aed',
  GLM: '#2563eb', MiniMax: '#dc2626', Moonshot: '#0891b2',
};
const DEFAULT_COLOR = '#64748b';

let currentProject = '';
let currentTopic = null;
let allConversations = [];
let lastMsgCount = 0;
let logoCache = {};  // provider -> url or false

function getColor(provider) {
  return PROVIDER_COLORS[provider] || DEFAULT_COLOR;
}

function providerInitial(p) {
  return p ? p[0].toUpperCase() : '?';
}

function logoUrl(provider) {
  return `/logos/${encodeURIComponent(provider)}.png`;
}

function avatarHtml(provider, size) {
  const url = logoUrl(provider);
  const color = getColor(provider);
  const initial = providerInitial(provider);
  const sz = size || '';
  return `<img src="${url}" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"${sz ? ` style="width:${sz};height:${sz}"` : ''}>` +
         `<div class="fallback" style="background:${color};display:none;${sz ? `width:${sz};height:${sz};font-size:${parseInt(sz)/2.5}px` : ''}">${initial}</div>`;
}

function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  const now = new Date();
  const diff = now - d;
  if (diff < 86400000 && d.getDate() === now.getDate()) {
    return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
  }
  if (diff < 86400000 * 2) return '昨天';
  if (diff < 86400000 * 7) return ['日','一','二','三','四','五','六'][d.getDay()] ? `周${['日','一','二','三','四','五','六'][d.getDay()]}` : '';
  return `${d.getMonth()+1}/${d.getDate()}`;
}

function formatFullTime(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatDateHeader(ts) {
  const d = new Date(ts * 1000);
  const now = new Date();
  if (d.toDateString() === now.toDateString()) return '今天';
  const yesterday = new Date(now); yesterday.setDate(now.getDate()-1);
  if (d.toDateString() === yesterday.toDateString()) return '昨天';
  return d.toLocaleDateString('zh-CN', { month: 'long', day: 'numeric' });
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Projects ──
async function loadProjects() {
  try {
    const r = await fetch('/api/projects');
    const projects = await r.json();
    const sel = document.getElementById('project-select');
    const prev = sel.value;
    sel.innerHTML = projects.map(p => `<option value="${p}"${p===currentProject?' selected':''}>${p}</option>`).join('');
    if (!currentProject && projects.length) {
      currentProject = projects[0];
      sel.value = currentProject;
    }
    if (prev && prev !== currentProject) {
      currentTopic = null;
      lastMsgCount = 0;
    }
  } catch(e) {}
}

document.getElementById('project-select').addEventListener('change', function() {
  currentProject = this.value;
  currentTopic = null;
  lastMsgCount = 0;
  document.getElementById('chat-header').style.display = 'none';
  document.getElementById('messages').innerHTML = '<div id="empty-state"><div class="icon">💬</div><p>选择一个对话开始查看</p></div>';
  loadConversations();
});

// ── Conversations ──
async function loadConversations() {
  if (!currentProject) return;
  try {
    const r = await fetch(`/api/conversations?project=${encodeURIComponent(currentProject)}`);
    allConversations = await r.json();
    renderConversations();
  } catch(e) {}
}

function renderConversations() {
  const search = document.getElementById('search-input').value.toLowerCase();
  const filtered = search ? allConversations.filter(c => c.topic.toLowerCase().includes(search) || c.providers.some(p => p.toLowerCase().includes(search))) : allConversations;

  const list = document.getElementById('conv-list');
  if (!filtered.length) {
    list.innerHTML = '<div style="padding:24px 14px;color:#aaa;font-size:13px;text-align:center">暂无对话记录</div>';
    return;
  }

  list.innerHTML = filtered.map(c => {
    const isActive = currentTopic === c.topic;
    const isGroup = c.type === 'group';
    let avatarContent;
    if (isGroup) {
      const slots = c.providers.slice(0, 4);
      avatarContent = `<div class="conv-avatar group-av">${slots.map(p => `<div class="mini-av">${avatarHtml(p, '20px')}</div>`).join('')}</div>`;
    } else {
      avatarContent = `<div class="conv-avatar">${avatarHtml(c.providers[0])}</div>`;
    }
    const badge = isGroup ? `<span class="conv-badge">群聊 ${c.providers.length}</span>` : '';
    const name = c.topic;
    const time = formatTime(c.last_time);
    const preview = escapeHtml(c.preview.replace(/\n/g, ' '));
    return `<div class="conv-item${isActive ? ' active' : ''}" onclick="selectConv('${escapeHtml(c.topic)}')">
      ${avatarContent}
      <div class="conv-info">
        <div class="conv-title-row">
          <div class="conv-name">${escapeHtml(name)}</div>
          <div class="conv-time">${time}</div>
        </div>
        <div class="conv-preview">${badge}${preview}</div>
      </div>
    </div>`;
  }).join('');
}

document.getElementById('search-input').addEventListener('input', renderConversations);

// ── Chat view ──
function selectConv(topic) {
  currentTopic = topic;
  lastMsgCount = 0;
  renderConversations(); // update active state
  loadChat(true);
}

async function loadChat(forceScroll) {
  if (!currentTopic || !currentProject) return;
  try {
    const r = await fetch(`/api/chat?project=${encodeURIComponent(currentProject)}&topic=${encodeURIComponent(currentTopic)}`);
    if (!r.ok) return;
    const data = await r.json();
    renderChat(data, forceScroll);
  } catch(e) {}
}

function renderChat(data, forceScroll) {
  const { topic, type: chatType, providers, messages } = data;

  // Header
  const header = document.getElementById('chat-header');
  header.style.display = 'flex';
  document.getElementById('chat-header-title').textContent = topic;
  if (chatType === 'group') {
    document.getElementById('chat-header-sub').textContent = `群聊 · ${providers.join(', ')}`;
  } else {
    document.getElementById('chat-header-sub').textContent = `与 ${providers[0]} 的对话`;
  }

  if (messages.length === lastMsgCount && !forceScroll) return;

  const container = document.getElementById('messages');
  const wasAtBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 60;
  container.innerHTML = '';

  let lastDate = '';
  messages.forEach(msg => {
    const msgDate = formatDateHeader(msg.timestamp);
    if (msgDate !== lastDate) {
      lastDate = msgDate;
      const div = document.createElement('div');
      div.className = 'date-divider';
      div.textContent = msgDate;
      container.appendChild(div);
    }

    const isUser = msg.role === 'user';
    const row = document.createElement('div');
    row.className = `msg-row ${isUser ? 'self' : 'other'}`;

    const provider = msg.provider || providers[0];
    const senderName = isUser ? '主模型' : provider;
    const showSender = chatType === 'group' || !isUser;

    let avHtml;
    if (isUser) {
      avHtml = `<div class="fallback" style="background:#07c160">AI</div>`;
    } else {
      avHtml = avatarHtml(provider);
    }

    row.innerHTML = `
      <div class="msg-avatar">${avHtml}</div>
      <div class="msg-body">
        ${showSender ? `<div class="msg-sender">${escapeHtml(senderName)}</div>` : ''}
        <div class="msg-bubble">${escapeHtml(msg.content)}</div>
        <div class="msg-time">${formatFullTime(msg.timestamp)}</div>
      </div>
    `;
    container.appendChild(row);
  });

  if (forceScroll || wasAtBottom) {
    container.scrollTop = container.scrollHeight;
  }
  lastMsgCount = messages.length;
}

// ── Polling ──
async function tick() {
  await loadProjects();
  await loadConversations();
  await loadChat(false);
  document.getElementById('status-text').textContent = `上次刷新：${new Date().toLocaleTimeString('zh-CN')}`;
}

tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self.send_html(HTML_PAGE)

        elif path == "/api/projects":
            self.send_json(list_projects())

        elif path == "/api/conversations":
            project = qs.get("project", ["default"])[0]
            self.send_json(load_conversations(project))

        elif path == "/api/chat":
            project = qs.get("project", ["default"])[0]
            topic = qs.get("topic", [None])[0]
            if not topic:
                self.send_json({"error": "missing topic"}, 400)
                return
            # Safety: no path traversal
            if "/" in topic or "\\" in topic or ".." in topic:
                self.send_json({"error": "invalid topic"}, 400)
                return
            data = load_merged_chat(project, topic)
            self.send_json(data)

        elif path.startswith("/logos/"):
            filename = path[len("/logos/"):]
            # Safety check
            if "/" in filename or "\\" in filename or ".." in filename:
                self.send_response(404)
                self.end_headers()
                return
            # Try multiple extensions
            for ext in (".png", ".jpg", ".jpeg", ".svg", ".webp"):
                base = filename.rsplit(".", 1)[0] if "." in filename else filename
                logo_path = LOGOS_DIR / f"{base}{ext}"
                if logo_path.exists():
                    ct = mimetypes.guess_type(str(logo_path))[0] or "image/png"
                    self.send_file(logo_path, ct)
                    return
            # Exact filename match
            exact = LOGOS_DIR / filename
            if exact.exists():
                ct = mimetypes.guess_type(str(exact))[0] or "image/png"
                self.send_file(exact, ct)
                return
            self.send_response(404)
            self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
    url = f"http://localhost:{port}"
    print(f"Multi-LLM Chat Viewer  →  {url}")
    print(f"历史目录: {HISTORY_BASE}")
    print(f"Logo 目录: {LOGOS_DIR}")
    print("Ctrl+C 退出\n")
    webbrowser.open(url)
    server = HTTPServer(("localhost", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
