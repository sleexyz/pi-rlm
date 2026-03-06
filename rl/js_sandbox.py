"""
Python wrapper for the TypeScript sandbox server.

Manages long-lived bun subprocesses — one per rollout instance.
Each subprocess runs sandbox-server.ts with EvalRuntime + createArcAdapter.
"""

import json
import os
import subprocess
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_SCRIPT = PROJECT_ROOT / "domains" / "arc-agi-2" / "src" / "sandbox-server.ts"


class SandboxSession:
    """Wraps a single bun subprocess running sandbox-server.ts."""

    def __init__(self, task: dict, timeout: float = 30.0):
        self.timeout = timeout
        self._proc = subprocess.Popen(
            ["bun", "run", str(SANDBOX_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )
        # Send init command
        resp = self._send({"type": "init", "task": task})
        if resp.get("type") != "ready":
            raise RuntimeError(f"Sandbox init failed: {resp}")

    def _send(self, msg: dict) -> dict:
        """Send a JSON-line message and read one JSON-line response."""
        line = json.dumps(msg) + "\n"
        self._proc.stdin.write(line.encode())
        self._proc.stdin.flush()
        resp_line = self._proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("Sandbox process died unexpectedly")
        return json.loads(resp_line)

    def eval(self, code: str) -> dict:
        """Execute code in the sandbox and return the result dict."""
        if self._proc.poll() is not None:
            raise RuntimeError("Sandbox process is no longer running")
        return self._send({"type": "eval", "code": code})

    def close(self):
        """Shut down the sandbox subprocess."""
        if self._proc.poll() is None:
            try:
                self._send({"type": "shutdown"})
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()


class SandboxPool:
    """Manages multiple SandboxSessions by instance_id."""

    def __init__(self):
        self._sessions: dict[str, SandboxSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, instance_id: str, task: dict) -> SandboxSession:
        with self._lock:
            if instance_id not in self._sessions:
                self._sessions[instance_id] = SandboxSession(task)
            return self._sessions[instance_id]

    def close(self, instance_id: str):
        with self._lock:
            session = self._sessions.pop(instance_id, None)
            if session:
                session.close()

    def close_all(self):
        with self._lock:
            for session in self._sessions.values():
                session.close()
            self._sessions.clear()
