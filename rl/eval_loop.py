"""Baseline evaluation loop for ARC-AGI.

Spawns agent-runner.ts (Bun) per task as a subprocess.
Bun owns all agent logic (system prompt, multi-turn loop, code extraction, sandbox).
Python just dispatches tasks and collects results.
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .trace_logger import write_run_json

# Path to agent-runner.ts relative to rl/ package
AGENT_RUNNER_PATH = Path(__file__).resolve().parent.parent / "domains" / "arc-agi-2" / "src" / "agent-runner.ts"


def _run_agent(
    task_id: str,
    task: dict,
    base_url: str,
    model: str,
    run_dir: str | None = None,
    max_turns: int = 15,
    timeout: int = 300,
    top_logprobs: int = 0,
    num_attempts: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> dict:
    """Spawn one agent-runner.ts subprocess and collect JSON result."""
    cmd = [
        "bun", "run", str(AGENT_RUNNER_PATH),
        "--base-url", base_url,
        "--model", model,
        "--task-id", task_id,
        "--task-from-stdin",
        "--max-turns", str(max_turns),
        "--timeout", str(timeout * 1000),
        "--temperature", str(temperature),
        "--top-p", str(top_p),
        "--num-attempts", str(num_attempts),
    ]
    if top_logprobs > 0:
        cmd += ["--top-logprobs", str(top_logprobs)]
    if run_dir:
        cmd += ["--run-dir", run_dir, "--session-id", f"{task_id}_a0"]

    try:
        result = subprocess.run(
            cmd,
            input=json.dumps(task).encode(),
            capture_output=True,
            timeout=timeout + 60,  # extra buffer beyond agent timeout
        )
    except subprocess.TimeoutExpired:
        return {
            "taskId": task_id,
            "correct": False,
            "submitted": False,
            "reward": 0.0,
            "turns": 0,
            "tokens": 0,
            "timeMs": timeout * 1000,
            "error": "subprocess timeout",
            "attempts": [{"correct": False, "failed": True, "error": "subprocess timeout", "tokens": 0, "timeMs": timeout * 1000}],
        }

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[:500]
        return {
            "taskId": task_id,
            "correct": False,
            "submitted": False,
            "reward": 0.0,
            "turns": 0,
            "tokens": 0,
            "timeMs": 0,
            "error": stderr,
            "attempts": [{"correct": False, "failed": True, "error": stderr, "tokens": 0, "timeMs": 0}],
        }

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        stderr = result.stderr.decode(errors="replace")[:500]
        return {
            "taskId": task_id,
            "correct": False,
            "submitted": False,
            "reward": 0.0,
            "turns": 0,
            "tokens": 0,
            "timeMs": 0,
            "error": f"JSON parse error: {e}\nstderr: {stderr}",
            "attempts": [{"correct": False, "failed": True, "error": f"JSON parse: {e}", "tokens": 0, "timeMs": 0}],
        }


def evaluate_tasks(
    model_name: str,
    tasks: list[dict],
    run_dir: str,
    run_name: str,
    vllm_base_url: str,
    max_turns: int = 15,
    temperature: float = 0.6,
    top_p: float = 0.95,
    timeout: int = 300,
    num_attempts: int = 1,
    top_logprobs: int = 0,
) -> dict:
    """Spawn agent-runner.ts per task, collect results.

    Returns config dict suitable for run.json.
    """
    config = {
        "model": model_name,
        "split": "eval",
        "maxTurns": max_turns,
        "temperature": temperature,
        "topP": top_p,
        "vllmBaseUrl": vllm_base_url,
        "startedAt": datetime.now(timezone.utc).isoformat(),
    }
    results = []

    for i, item in enumerate(tasks):
        task_id = item["id"]
        task = item["task"]

        result = _run_agent(
            task_id=task_id,
            task=task,
            base_url=vllm_base_url,
            model=model_name,
            run_dir=run_dir,
            max_turns=max_turns,
            timeout=timeout,
            top_logprobs=top_logprobs,
            num_attempts=num_attempts,
            temperature=temperature,
            top_p=top_p,
        )
        results.append(result)

        # Write incremental run.json
        write_run_json(run_dir, run_name, config, results)

        correct = result.get("correct", False)
        tokens = result.get("tokens", 0)
        turns = result.get("turns", 0)
        time_ms = result.get("timeMs", 0)
        c = sum(1 for r in results if r.get("correct"))
        print(
            f"[{i+1}/{len(tasks)}] {task_id}: {'CORRECT' if correct else 'WRONG'} "
            f"(turns={turns}, tokens={tokens}, {time_ms}ms) "
            f"-- running: {c}/{len(results)}"
        )

    return config
