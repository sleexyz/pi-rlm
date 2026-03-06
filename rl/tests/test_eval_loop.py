"""Tests for eval_loop — mocks agent-runner.ts subprocess to verify dispatch logic."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


SAMPLE_TASK = {
    "train": [
        {"input": [[1, 0], [0, 1]], "output": [[1, 1], [1, 1]]},
    ],
    "test": [
        {"input": [[0, 1], [1, 0]], "output": [[1, 1], [1, 1]]},
    ],
}


def _mock_subprocess_result(task_id, correct=True, turns=3, tokens=500, time_ms=2000):
    """Create a mock subprocess.CompletedProcess with agent-runner JSON output."""
    result = {
        "taskId": task_id,
        "reward": 1.0 if correct else 0.0,
        "submitted": True,
        "correct": correct,
        "turns": turns,
        "tokens": tokens,
        "timeMs": time_ms,
        "trajectory": [],
        "attempts": [{"correct": correct, "tokens": tokens, "timeMs": time_ms}],
    }
    mock = MagicMock()
    mock.returncode = 0
    mock.stdout = json.dumps(result).encode()
    mock.stderr = b""
    return mock


def test_eval_loop_single_task_correct():
    """Verify the eval loop dispatches a task and collects the result."""
    from rl.eval_loop import evaluate_tasks

    tasks = [{"id": "test001", "task": SAMPLE_TASK}]

    with tempfile.TemporaryDirectory() as tmp, \
         patch("rl.eval_loop.subprocess.run", return_value=_mock_subprocess_result("test001", correct=True)):
        run_dir = os.path.join(tmp, "test-run")
        os.makedirs(run_dir, exist_ok=True)

        config = evaluate_tasks(
            model_name="test-model",
            tasks=tasks,
            run_dir=run_dir,
            run_name="test-run",
            vllm_base_url="http://localhost:8000/v1",
        )

        # Verify run.json was written
        run_json_path = os.path.join(run_dir, "run.json")
        assert os.path.exists(run_json_path)
        with open(run_json_path) as f:
            data = json.load(f)

        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["taskId"] == "test001"
        assert result["correct"] is True

        # Verify config
        assert config["model"] == "test-model"
        assert config["vllmBaseUrl"] == "http://localhost:8000/v1"


def test_eval_loop_subprocess_error():
    """Verify graceful handling of agent-runner.ts subprocess failures."""
    from rl.eval_loop import evaluate_tasks

    tasks = [{"id": "test002", "task": SAMPLE_TASK}]

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = b""
    mock_result.stderr = b"Error: connection refused"

    with tempfile.TemporaryDirectory() as tmp, \
         patch("rl.eval_loop.subprocess.run", return_value=mock_result):
        run_dir = os.path.join(tmp, "test-run")
        os.makedirs(run_dir, exist_ok=True)

        evaluate_tasks(
            model_name="test-model",
            tasks=tasks,
            run_dir=run_dir,
            run_name="test-run",
            vllm_base_url="http://localhost:8000/v1",
        )

        with open(os.path.join(run_dir, "run.json")) as f:
            data = json.load(f)

        result = data["results"][0]
        assert result["correct"] is False
        assert "connection refused" in result.get("error", "")


def test_eval_loop_subprocess_timeout():
    """Verify graceful handling of subprocess timeout."""
    import subprocess as sp
    from rl.eval_loop import evaluate_tasks

    tasks = [{"id": "test003", "task": SAMPLE_TASK}]

    with tempfile.TemporaryDirectory() as tmp, \
         patch("rl.eval_loop.subprocess.run", side_effect=sp.TimeoutExpired(cmd="bun", timeout=300)):
        run_dir = os.path.join(tmp, "test-run")
        os.makedirs(run_dir, exist_ok=True)

        evaluate_tasks(
            model_name="test-model",
            tasks=tasks,
            run_dir=run_dir,
            run_name="test-run",
            vllm_base_url="http://localhost:8000/v1",
        )

        with open(os.path.join(run_dir, "run.json")) as f:
            data = json.load(f)

        result = data["results"][0]
        assert result["correct"] is False
        assert "timeout" in result.get("error", "").lower()


def test_eval_loop_multiple_tasks():
    """Verify the eval loop processes multiple tasks."""
    from rl.eval_loop import evaluate_tasks

    tasks = [
        {"id": "task_a", "task": SAMPLE_TASK},
        {"id": "task_b", "task": SAMPLE_TASK},
        {"id": "task_c", "task": SAMPLE_TASK},
    ]

    call_count = {"n": 0}
    correct_map = {"task_a": True, "task_b": False, "task_c": True}

    def mock_run(cmd, **kwargs):
        # Extract task_id from stdin JSON
        input_data = json.loads(kwargs.get("input", b"{}"))
        # Get task_id from cmd args
        task_id = cmd[cmd.index("--task-id") + 1]
        call_count["n"] += 1
        return _mock_subprocess_result(task_id, correct=correct_map.get(task_id, False))

    with tempfile.TemporaryDirectory() as tmp, \
         patch("rl.eval_loop.subprocess.run", side_effect=mock_run):
        run_dir = os.path.join(tmp, "test-run")
        os.makedirs(run_dir, exist_ok=True)

        evaluate_tasks(
            model_name="test-model",
            tasks=tasks,
            run_dir=run_dir,
            run_name="test-run",
            vllm_base_url="http://localhost:8000/v1",
        )

        with open(os.path.join(run_dir, "run.json")) as f:
            data = json.load(f)

        assert len(data["results"]) == 3
        assert data["summary"]["correct"] == 2
        assert data["summary"]["total"] == 3
        assert call_count["n"] == 3
