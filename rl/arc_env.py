"""
ArcEnv — zero-skew training/eval environment wrapping sandbox-agent.ts.

One Bun subprocess per environment instance. Variables persist across steps.
The TS subprocess handles code parsing, execution, output formatting, and
feedback formatting identically to agent-runner.ts (the eval source of truth).
"""

import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_AGENT = PROJECT_ROOT / "domains" / "arc-agi-2" / "src" / "sandbox-agent.ts"


class ArcEnv:
    """Environment wrapping sandbox-agent.ts for zero-skew training/eval."""

    def __init__(self, task: dict, timeout: float = 30.0):
        self.timeout = timeout
        self._proc = subprocess.Popen(
            ["bun", "run", str(SANDBOX_AGENT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )
        # Send init
        resp = self._send({"type": "init", "task": task})
        if resp.get("type") != "ready":
            raise RuntimeError(f"sandbox-agent init failed: {resp}")
        self._system_prompt: str = resp["systemPrompt"]
        self._user_message: str = resp["userMessage"]

    def _send(self, msg: dict) -> dict:
        """Send a JSON-line message and read one JSON-line response."""
        if self._proc.poll() is not None:
            raise RuntimeError("sandbox-agent process is no longer running")
        line = json.dumps(msg) + "\n"
        self._proc.stdin.write(line.encode())
        self._proc.stdin.flush()
        resp_line = self._proc.stdout.readline()
        if not resp_line:
            stderr = self._proc.stderr.read().decode() if self._proc.stderr else ""
            raise RuntimeError(f"sandbox-agent process died: {stderr}")
        return json.loads(resp_line)

    def reset(self) -> dict:
        """Return initial observation: systemPrompt + userMessage."""
        return {
            "systemPrompt": self._system_prompt,
            "userMessage": self._user_message,
        }

    def step(self, assistant_text: str) -> tuple[str, float, bool, dict]:
        """Execute one agent turn.

        Returns:
            (feedback, reward, done, info)
        """
        resp = self._send({"type": "step", "assistantText": assistant_text})
        if resp.get("type") == "error":
            raise RuntimeError(f"sandbox-agent error: {resp.get('error')}")
        return (
            resp["feedback"],
            resp["reward"],
            resp["done"],
            resp.get("info", {}),
        )

    def close(self):
        """Kill subprocess."""
        if self._proc.poll() is None:
            try:
                self._send({"type": "close"})
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
