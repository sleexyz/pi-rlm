"""Baseline evaluation loop for ARC-AGI.

Spawns agent-runner.ts (Bun) per task as a subprocess.
Bun owns all agent logic (system prompt, multi-turn loop, code extraction, sandbox).
Python just dispatches tasks and collects results.
"""

import asyncio
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
    # Use full bun path for Modal compatibility; resolve workspace root for imports
    import shutil
    bun = shutil.which("bun") or "/root/.bun/bin/bun"
    cmd = [
        bun, "run", str(AGENT_RUNNER_PATH),
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

    # cwd must be workspace root so bun resolves pi-rlm workspace dependency
    workspace_root = str(Path(__file__).resolve().parent.parent)
    try:
        result = subprocess.run(
            cmd,
            input=json.dumps(task).encode(),
            capture_output=True,
            timeout=timeout + 60,  # extra buffer beyond agent timeout
            cwd=workspace_root,
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

    # Print stderr for debugging (agent-runner writes progress/errors there)
    stderr_text = result.stderr.decode(errors="replace").strip()
    if stderr_text:
        for line in stderr_text.split("\n"):
            print(f"  {line}")

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


async def _run_agent_async(
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
    """Async version of _run_agent using asyncio subprocess."""
    import shutil
    bun = shutil.which("bun") or "/root/.bun/bin/bun"
    cmd = [
        bun, "run", str(AGENT_RUNNER_PATH),
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

    workspace_root = str(Path(__file__).resolve().parent.parent)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workspace_root,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=json.dumps(task).encode()),
            timeout=timeout + 60,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
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

    if proc.returncode != 0:
        stderr_text = stderr.decode(errors="replace")[:500]
        return {
            "taskId": task_id,
            "correct": False,
            "submitted": False,
            "reward": 0.0,
            "turns": 0,
            "tokens": 0,
            "timeMs": 0,
            "error": stderr_text,
            "attempts": [{"correct": False, "failed": True, "error": stderr_text, "tokens": 0, "timeMs": 0}],
        }

    stderr_text = stderr.decode(errors="replace").strip()
    if stderr_text:
        for line in stderr_text.split("\n"):
            print(f"  {line}")

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        stderr_text = stderr.decode(errors="replace")[:500]
        return {
            "taskId": task_id,
            "correct": False,
            "submitted": False,
            "reward": 0.0,
            "turns": 0,
            "tokens": 0,
            "timeMs": 0,
            "error": f"JSON parse error: {e}\nstderr: {stderr_text}",
            "attempts": [{"correct": False, "failed": True, "error": f"JSON parse: {e}", "tokens": 0, "timeMs": 0}],
        }


async def evaluate_tasks_concurrent(
    model_name: str,
    tasks: list[dict],
    run_dir: str,
    run_name: str,
    vllm_base_url: str,
    max_parallel: int = 4,
    max_turns: int = 15,
    temperature: float = 0.6,
    top_p: float = 0.95,
    timeout: int = 300,
    num_attempts: int = 1,
    top_logprobs: int = 0,
) -> dict:
    """Concurrent version of evaluate_tasks using asyncio subprocesses."""
    config = {
        "model": model_name,
        "split": "eval",
        "maxTurns": max_turns,
        "temperature": temperature,
        "topP": top_p,
        "vllmBaseUrl": vllm_base_url,
        "startedAt": datetime.now(timezone.utc).isoformat(),
    }
    results: list[dict] = []
    sem = asyncio.Semaphore(max_parallel)

    async def run_one(item: dict) -> dict:
        async with sem:
            return await _run_agent_async(
                task_id=item["id"],
                task=item["task"],
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

    all_results = await asyncio.gather(*[run_one(t) for t in tasks])

    for i, result in enumerate(all_results):
        results.append(result)
        write_run_json(run_dir, run_name, config, results)
        correct = result.get("correct", False)
        tokens = result.get("tokens", 0)
        turns = result.get("turns", 0)
        time_ms = result.get("timeMs", 0)
        c = sum(1 for r in results if r.get("correct"))
        print(
            f"[{i+1}/{len(tasks)}] {result.get('taskId', '?')}: "
            f"{'CORRECT' if correct else 'WRONG'} "
            f"(turns={turns}, tokens={tokens}, {time_ms}ms) "
            f"-- running: {c}/{len(results)}"
        )

    return config
