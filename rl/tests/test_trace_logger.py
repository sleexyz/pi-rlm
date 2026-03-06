"""Tests for TraceLogger and write_run_json."""

import json
import os
import tempfile

from rl.trace_logger import TraceLogger, write_run_json


def test_trace_logger_creates_valid_jsonl():
    """3-turn conversation: user → assistant → user → assistant → user → session_end."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = TraceLogger(tmp, "task1_a0", metadata={"taskId": "task1", "model": "test"})

        # Turn 1: user prompt
        logger.log_message("user", "Solve this ARC task.")
        # Turn 1: assistant response (code block format)
        logger.log_message("assistant", [{"type": "text", "text": "Let me analyze...\n```js\nconsole.log('hello')\n```"}])
        # Turn 2: user feedback
        logger.log_message("user", "Code executed:\n```js\nconsole.log('hello')\n```\n\nREPL output:\nhello")
        # Turn 2: assistant submit
        logger.log_message("assistant", [{"type": "text", "text": "```js\nsubmit(transform)\n```"}])
        # Turn 3: user feedback (termination)
        logger.log_message("user", "Submitted. Reward: 1.0")
        logger.close(usage={"totalTokens": 5000})

        # Verify file structure
        jsonl_path = os.path.join(tmp, "sessions", "task1_a0", "agent-0.jsonl")
        assert os.path.exists(jsonl_path)

        with open(jsonl_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        # Header
        assert lines[0]["type"] == "session"
        assert lines[0]["version"] == 1
        assert lines[0]["agentIndex"] == 0
        assert lines[0]["metadata"]["taskId"] == "task1"

        # Messages
        assert lines[1]["type"] == "message"
        assert lines[1]["message"]["role"] == "user"
        assert isinstance(lines[1]["message"]["content"], str)

        assert lines[2]["type"] == "message"
        assert lines[2]["message"]["role"] == "assistant"
        assert isinstance(lines[2]["message"]["content"], list)
        assert lines[2]["message"]["content"][0]["type"] == "text"

        assert lines[3]["type"] == "message"
        assert lines[3]["message"]["role"] == "user"

        assert lines[4]["type"] == "message"
        assert lines[4]["message"]["role"] == "assistant"

        assert lines[5]["type"] == "message"
        assert lines[5]["message"]["role"] == "user"

        # Session end
        assert lines[6]["type"] == "session_end"
        assert lines[6]["usage"]["totalTokens"] == 5000

        # All entries have ts
        for line in lines[1:]:
            assert "ts" in line
            assert isinstance(line["ts"], int)


def test_write_run_json():
    with tempfile.TemporaryDirectory() as tmp:
        results = [
            {"taskId": "t1", "correct": True, "tokens": 1000, "timeMs": 5000,
             "attempts": [{"correct": True}]},
            {"taskId": "t2", "correct": False, "tokens": 2000, "timeMs": 8000,
             "attempts": [{"correct": False}]},
        ]
        config = {"model": "test-model", "split": "training", "startedAt": "2026-03-06T00:00:00Z"}

        write_run_json(tmp, "test-run", config, results)

        with open(os.path.join(tmp, "run.json")) as f:
            data = json.load(f)

        assert data["name"] == "test-run"
        assert len(data["results"]) == 2
        assert data["summary"]["correct"] == 1
        assert data["summary"]["total"] == 2
        assert data["summary"]["pct"] == 0.5
        assert data["summary"]["tokens"] == 3000
        assert data["summary"]["timeMs"] == 13000
