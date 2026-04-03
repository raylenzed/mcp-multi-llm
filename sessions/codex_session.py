"""Codex CLI session provider."""

import re
from sessions.base import CLISession
from history.store import HistoryStore, TopicHistory


class CodexSession(CLISession):
    provider_name = "codex"

    _AUTH_STRINGS = (
        "not logged in",
        "unauthorized",
        "401",
        "api key",
        "openai_api_key",
        "authentication",
        "login",
    )

    def __init__(self, history_store: HistoryStore, model: str | None = None):
        super().__init__(history_store)
        self.model = model
        self.cli_path = self._get_cli_path("codex")

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
        cmd = [self.cli_path, "exec", "--skip-git-repo-check", "--ephemeral"]
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.append(prompt)
        return cmd

    def _health_check_command(self) -> list[str]:
        cmd = [self.cli_path, "exec", "--skip-git-repo-check", "--ephemeral"]
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.append("Reply with the single word: ok")
        return cmd

    def _auth_fix_hint(self) -> str:
        return "Run 'codex' in your terminal and follow the login prompts."

    def _install_hint(self) -> str:
        return "Run: npm install -g @openai/codex"

    def _parse_output(self, stdout: str, stderr: str) -> str:
        text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', stdout)
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")):
                continue
            lines.append(line)
        return "\n".join(lines).strip() or "(no output)"
