"""Trace logger for ARC eval runs.

Writes JSONL files in the format expected by the viewer
(domains/arc-agi-2/src/viewer.ts: convertMessagesToEvents).
"""

import json
import os
import time
from datetime import datetime, timezone


class TraceLogger:
    """Writes agent-0.jsonl for a single session (task attempt)."""

    def __init__(self, run_dir: str, session_id: str, metadata: dict | None = None):
        session_dir = os.path.join(run_dir, "sessions", session_id)
        os.makedirs(session_dir, exist_ok=True)
        self._path = os.path.join(session_dir, "agent-0.jsonl")
        self._f = open(self._path, "w")
        # Session header
        header = {
            "type": "session",
            "version": 1,
            "agentIndex": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self._f.write(json.dumps(header) + "\n")

    def log_message(self, role: str, content: str | list):
        """Log a message. Content is either a string or structured content blocks."""
        entry = {
            "type": "message",
            "ts": int(time.time()),
            "message": {"role": role, "content": content},
        }
        self._f.write(json.dumps(entry) + "\n")
        self._f.flush()

    def close(self, usage: dict | None = None):
        entry = {"type": "session_end", "ts": int(time.time())}
        if usage:
            entry["usage"] = usage
        self._f.write(json.dumps(entry) + "\n")
        self._f.close()


def write_run_json(
    run_dir: str,
    name: str,
    config: dict,
    results: list[dict],
):
    """Write run.json matching the viewer's expected format."""
    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    total_tokens = sum(r.get("tokens", 0) for r in results)
    total_time_ms = sum(r.get("timeMs", 0) for r in results)

    run_json = {
        "name": name,
        "startedAt": config.get("startedAt", datetime.now(timezone.utc).isoformat()),
        "config": config,
        "results": results,
        "summary": {
            "correct": correct,
            "total": total,
            "pct": correct / total if total > 0 else 0,
            "tokens": total_tokens,
            "timeMs": total_time_ms,
        },
    }
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(run_json, f, indent=2)
