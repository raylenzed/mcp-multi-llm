"""Gemini CLI session provider (OAuth via `gemini` CLI)."""

import json
import re
from sessions.base import CLISession
from history.store import HistoryStore, TopicHistory


class GeminiSession(CLISession):
    provider_name = "gemini"

    _AUTH_STRINGS = (
        "oauth",
        "not authenticated",
        "please log in",
        "authentication",
        "login required",
        "401",
        "unauthorized",
        "credentials",
    )

    def __init__(self, history_store: HistoryStore, model: str | None = None):
        super().__init__(history_store)
        self.model = model
        self.cli_path = self._get_cli_path("gemini")

    def _build_prompt_with_context(self, message: str, history: TopicHistory) -> str:
        if len(history.messages) <= 1:
            return message
        context = history.format_for_prompt(max_messages=10)
        return (
            f"Previous conversation context:\n{context}\n\n"
            f"Current question/request:\n{message}"
        )

    def _build_command(self, message: str, history: TopicHistory) -> list[str]:
        prompt = self._build_prompt_with_context(message, history)
        cmd = [self.cli_path, "-p", prompt, "--output-format", "json", "--yolo"]
        if self.model:
            cmd.extend(["-m", self.model])
        return cmd

    def _health_check_command(self) -> list[str]:
        cmd = [self.cli_path, "-p", "Reply with the single word: ok",
               "--output-format", "json", "--yolo"]
        if self.model:
            cmd.extend(["-m", self.model])
        return cmd

    def _auth_fix_hint(self) -> str:
        return "Run 'gemini auth login' in your terminal."

    def _install_hint(self) -> str:
        return "Run: npm install -g @google/gemini-cli"

    def _parse_output(self, stdout: str, stderr: str) -> str:
        text = stdout.strip()
        if text:
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "response" in data:
                    return data["response"].strip() or "(no output)"
            except json.JSONDecodeError:
                pass

        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', stdout)
        noise_prefixes = (
            "Loaded cached credentials",
            "Registering notification",
            "Server '",
            "Scheduling MCP",
            "Executing MCP",
            "MCP context refresh",
            "Skill ",
            "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
        )
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(p) for p in noise_prefixes):
                continue
            lines.append(line)
        return "\n".join(lines).strip() or "(no output)"
