"""Tests for ArcInteraction with tool call format."""

import asyncio
import pytest
from rl.arc_interaction import ArcInteraction

TASK = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2, 1, 2], [3, 4, 3, 4]]}],
    "test": [{"input": [[5, 6], [7, 8]], "output": [[5, 6, 5, 6], [7, 8, 7, 8]]}],
}


def _tool_call(code: str) -> str:
    """Format code as a tool call string (what the model would output)."""
    import json
    return f'<tool_call>\n{json.dumps({"name": "eval", "arguments": {"code": code}})}\n</tool_call>'


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_correct_submission():
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": _tool_call("submit([[5,6,5,6],[7,8,7,8]])")}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is True
    assert reward == 1.0
    assert metrics["submitted"] is True
    interaction.pool.close_all()


def test_incorrect_submission():
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": _tool_call("submit([[0,0],[0,0]])")}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is True
    assert reward == 0.0
    interaction.pool.close_all()


def test_no_submission_continues():
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": _tool_call("console.log(trainingExamples[0])")}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is False
    assert reward == 0.0
    interaction.pool.close_all()


def test_no_tool_call():
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": "Let me think about this..."}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is False
    assert reward == 0.0
    interaction.pool.close_all()


def test_variable_persistence_across_turns():
    """Multi-turn: variables from turn 1 available in turn 2."""
    interaction = ArcInteraction(tasks={"task0": TASK})
    # Turn 1: define variable
    msgs1 = [
        {"role": "assistant", "content": _tool_call("const grid = trainingExamples[0].input")}
    ]
    run(interaction.generate_response("task0", msgs1))
    # Turn 2: use variable from turn 1
    msgs2 = [
        {"role": "assistant", "content": _tool_call("console.log(JSON.stringify(grid))")}
    ]
    _, feedback, _, _ = run(interaction.generate_response("task0", msgs2))
    assert "[[1,2],[3,4]]" in feedback
    interaction.pool.close_all()


def test_syntax_error():
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": _tool_call("function(")}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is False
    assert "error" in feedback.lower() or "Error" in feedback
    interaction.pool.close_all()
