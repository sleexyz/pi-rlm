"""
ARC Interaction — bridges verl's rollout engine to the TS execution environment.

Delegates all code parsing, execution, and feedback formatting to ArcEnv
(which wraps sandbox-agent.ts) for zero skew between training and eval.

verl calling convention:
  - start_interaction(request_id, **interaction_kwargs) is called once per rollout
  - generate_response(request_id, messages, **interaction_kwargs) is called each turn
"""

from typing import Any, Optional
from uuid import uuid4

from .arc_env import ArcEnv


class ArcInteraction:
    """verl Interaction for ARC-AGI-2 multi-turn code execution.

    Tasks can be provided two ways:
    1. Constructor injection (for testing): ArcInteraction(tasks={"id": task_data})
    2. Via interaction_kwargs (verl pipeline): start_interaction(id, task=task_data)
    """

    def __init__(self, config: dict[str, Any] | None = None, tasks: dict[str, dict] | None = None):
        self.config = config or {}
        self.tasks = tasks or {}
        self.envs: dict[str, ArcEnv] = {}
        self.name = config.get("name", "arc") if config else "arc"

    async def start_interaction(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())

        task_id = kwargs.get("task_id", instance_id)
        task = kwargs.get("task") or self.tasks.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.envs[instance_id] = ArcEnv(task)
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        env = self.envs.get(instance_id)
        if env is None:
            # Auto-init if start_interaction wasn't called explicitly
            task_id = kwargs.get("task_id", instance_id)
            task = kwargs.get("task") or self.tasks.get(task_id)
            if task is None:
                return False, "No code found. Write a ```javascript block to execute code.", 0.0, {}
            env = ArcEnv(task)
            self.envs[instance_id] = env

        # Extract latest assistant message content
        content = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content", "")
                break

        feedback, reward, done, info = env.step(content)
        return done, feedback, reward, info

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        env = self.envs.pop(instance_id, None)
        if env:
            env.close()
