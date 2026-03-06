"""Gate 3 — 6 tests for ArcInteraction."""

import asyncio
import pytest
from rl.arc_interaction import ArcInteraction

TASK = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2, 1, 2], [3, 4, 3, 4]]}],
    "test": [{"input": [[5, 6], [7, 8]], "output": [[5, 6, 5, 6], [7, 8, 7, 8]]}],
}


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_correct_submission():
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": "```js\nsubmit([[5,6,5,6],[7,8,7,8]])\n```"}
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
        {"role": "assistant", "content": "```js\nsubmit([[0,0],[0,0]])\n```"}
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
        {"role": "assistant", "content": "```js\nconsole.log(trainingExamples[0])\n```"}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is False
    assert reward == 0.0
    assert "REPL output" in feedback
    interaction.pool.close_all()


def test_no_code_blocks():
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
        {"role": "assistant", "content": "```js\nconst grid = trainingExamples[0].input\n```"}
    ]
    run(interaction.generate_response("task0", msgs1))
    # Turn 2: use variable from turn 1
    msgs2 = [
        {"role": "assistant", "content": "```js\nconsole.log(JSON.stringify(grid))\n```"}
    ]
    _, feedback, _, _ = run(interaction.generate_response("task0", msgs2))
    assert "[[1,2],[3,4]]" in feedback
    interaction.pool.close_all()


def test_syntax_error():
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": "```js\nfunction(\n```"}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is False
    assert "error" in feedback.lower() or "Error" in feedback
    interaction.pool.close_all()
