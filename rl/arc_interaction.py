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
from .parser import parse_code_blocks


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

        # Parse code blocks
        segments = parse_code_blocks(content)
        code_blocks = [s for s in segments if s["type"] == "code"]

        if not code_blocks:
            return False, "No code blocks found. Write a ```js code block to execute code.", 0.0, {"submitted": False}

        # Execute each code block
        session = self.pool.get_or_create(instance_id, meta["task"])
        last_result = None
        feedbacks = []

        for block in code_blocks:
            code = block["content"]
            result = session.eval(code)
            last_result = result

            # Format feedback matching toolResultToUserMessage from pi-rlm
            output = result.get("stdout", "") or ""
            if result.get("error"):
                output = result["error"] if not output else f"{output}\n\nError: {result['error']}"
            feedback = f"Code executed:\n```js\n{code}\n```\n\nREPL output:\n{output or '(no output)'}"
            feedbacks.append(feedback)

            # Stop executing more blocks if submitted
            if result.get("submitted") is not None:
                break

        combined_feedback = "\n\n".join(feedbacks)
        submitted = last_result.get("submitted") if last_result else None

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
