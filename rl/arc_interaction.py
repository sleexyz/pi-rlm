"""
ARC Interaction — bridges verl's rollout engine to the TS execution environment.

Implements verl's BaseInteraction interface. Delegates code execution to
long-lived bun subprocesses running sandbox-server.ts (which reuses
EvalRuntime + createArcAdapter for zero execution skew).

verl calling convention:
  - interaction_kwargs comes from extra_info["interaction_kwargs"] in the data row
  - start_interaction(request_id, **interaction_kwargs) is called once per rollout
  - generate_response(request_id, messages, **interaction_kwargs) is called each turn
"""

from typing import Any, Optional
from uuid import uuid4

from .js_sandbox import SandboxPool
from .parser import parse_tool_call


class ArcInteraction:
    """verl Interaction for ARC-AGI-2 multi-turn code execution.

    Tasks can be provided two ways:
    1. Constructor injection (for testing): ArcInteraction(tasks={"id": task_data})
    2. Via interaction_kwargs (verl pipeline): start_interaction(id, task=task_data)
    """

    def __init__(self, config: dict[str, Any] | None = None, tasks: dict[str, dict] | None = None):
        self.config = config or {}
        self.tasks = tasks or {}
        self.pool = SandboxPool()
        self._instance_meta: dict[str, dict] = {}
        self.name = config.get("name", "arc") if config else "arc"

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        task_id = kwargs.get("task_id", instance_id)

        # Task data from kwargs (verl pipeline) or constructor (testing)
        task = kwargs.get("task") or self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task_id: {task_id}")

        self._instance_meta[instance_id] = {
            "task_id": task_id,
            "task": task,
            "turns": 0,
        }
        # Pre-create the sandbox session
        self.pool.get_or_create(instance_id, task)
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        meta = self._instance_meta.get(instance_id)
        if meta is None:
            # Auto-init if start_interaction wasn't called explicitly
            task_id = kwargs.get("task_id", instance_id)
            task = kwargs.get("task") or self.tasks.get(task_id)
            if task is None:
                return False, "No code blocks found.", 0.0, {}
            meta = {"task_id": task_id, "task": task, "turns": 0}
            self._instance_meta[instance_id] = meta
            self.pool.get_or_create(instance_id, task)

        meta["turns"] += 1

        # Extract latest assistant message
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content", "")
                break

        # Extract code from tool call
        code = parse_tool_call(content)

        if not code:
            return False, "No code found. Use the eval tool to execute JavaScript code.", 0.0, {"submitted": False}

        # Execute code
        session = self.pool.get_or_create(instance_id, meta["task"])
        result = session.eval(code)

        # Format feedback
        output = result.get("stdout", "") or ""
        if result.get("error"):
            output = result["error"] if not output else f"{output}\n\nError: {result['error']}"
        combined_feedback = output or "(no output)"
        submitted = result.get("submitted")

        # Compute reward
        reward = 0.0
        if submitted is not None:
            expected = meta["task"]["test"][0]["output"]
            reward = 1.0 if submitted == expected else 0.0

        should_terminate = submitted is not None
        metrics = {
            "submitted": submitted is not None,
            "accuracy": reward,
            "turns": meta["turns"],
        }

        return should_terminate, combined_feedback, reward, metrics

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        meta = self._instance_meta.get(instance_id, {})
        return meta.get("last_reward", 0.0)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self.pool.close(instance_id)
        self._instance_meta.pop(instance_id, None)
