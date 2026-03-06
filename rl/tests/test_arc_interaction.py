"""Tests for ArcInteraction — code extraction and execution via ArcEnv."""

import asyncio
import pytest
from rl.arc_interaction import ArcInteraction

TASK = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2, 1, 2], [3, 4, 3, 4]]}],
    "test": [{"input": [[5, 6], [7, 8]], "output": [[5, 6, 5, 6], [7, 8, 7, 8]]}],
}


def _code_block(code: str) -> str:
    """Format code as a ```javascript block (what the model outputs)."""
    return f"```javascript\n{code}\n```"


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_correct_submission():
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": _code_block(
            "function transform(grid) { return grid.map(row => [...row, ...row]); }\n"
            "submit(transform);"
        )}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is True
    assert reward == 1.0
    assert metrics["submitted"] is True
    run(interaction.finalize_interaction("task0"))


def test_incorrect_submission():
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": _code_block(
            "function transform(grid) { return grid; }\nsubmit(transform);"
        )}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is True
    assert reward == 0.0
    run(interaction.finalize_interaction("task0"))


def test_no_submission_continues():
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": _code_block("console.log(trainingExamples[0])")}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is False
    assert reward == 0.0
    run(interaction.finalize_interaction("task0"))


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
    assert "No code found" in feedback
    run(interaction.finalize_interaction("task0"))


def test_variable_persistence_across_turns():
    """Multi-turn: variables from turn 1 available in turn 2."""
    interaction = ArcInteraction(tasks={"task0": TASK})
    # Turn 1: define variable
    msgs1 = [
        {"role": "assistant", "content": _code_block("const grid = trainingExamples[0].input")}
    ]
    run(interaction.generate_response("task0", msgs1))
    # Turn 2: use variable from turn 1
    msgs2 = [
        {"role": "assistant", "content": _code_block("console.log(JSON.stringify(grid))")}
    ]
    _, feedback, _, _ = run(interaction.generate_response("task0", msgs2))
    assert "[[1,2],[3,4]]" in feedback
    run(interaction.finalize_interaction("task0"))


def test_code_block_format():
    """Code extracted from ```javascript block."""
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": "Let me check.\n\n```javascript\nconsole.log(42)\n```"}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is False
    assert "42" in feedback
    run(interaction.finalize_interaction("task0"))


def test_syntax_error():
    interaction = ArcInteraction(tasks={"task0": TASK})
    messages = [
        {"role": "assistant", "content": _code_block("function(")}
    ]
    terminated, feedback, reward, metrics = run(
        interaction.generate_response("task0", messages)
    )
    assert terminated is False
    assert "ERROR" in feedback
    run(interaction.finalize_interaction("task0"))
