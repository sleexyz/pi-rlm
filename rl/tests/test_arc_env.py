"""Tests for ArcEnv — feedback parity with eval-tool format."""

import pytest
from arc_env import ArcEnv

TASK = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2, 1, 2], [3, 4, 3, 4]]}],
    "test": [{"input": [[5, 6], [7, 8]], "output": [[5, 6, 5, 6], [7, 8, 7, 8]]}],
}


@pytest.fixture
def env():
    e = ArcEnv(TASK)
    yield e
    e.close()


def test_reset_returns_system_prompt_and_user_message(env):
    obs = env.reset()
    assert "systemPrompt" in obs
    assert "userMessage" in obs
    assert "ARC" in obs["systemPrompt"]
    assert "transform" in obs["userMessage"]


def test_console_log_feedback_format(env):
    """console.log output appears in REPL output section."""
    feedback, reward, done, info = env.step(
        'I will check.\n```javascript\nconsole.log("hello")\n```'
    )
    assert feedback.startswith("Code executed:\n```javascript\n")
    assert "REPL output:\nhello" in feedback
    assert reward == 0.0
    assert done is False


def test_expression_return_value(env):
    """Expression return value formatted as → {value}."""
    feedback, _, _, _ = env.step("```javascript\n42\n```")
    assert "→ 42" in feedback


def test_error_in_code(env):
    """Runtime error formatted as ERROR: ..."""
    feedback, _, done, _ = env.step("```javascript\nthrow new Error('boom')\n```")
    assert "ERROR:" in feedback
    assert done is False


def test_no_code_blocks(env):
    """No code blocks → specific feedback message."""
    feedback, reward, done, _ = env.step("Let me think about this...")
    assert "No code found" in feedback
    assert reward == 0.0
    assert done is False


def test_correct_submission(env):
    """submit(transform) with correct transform → done=True, reward=1.0."""
    feedback, reward, done, info = env.step(
        "```javascript\n"
        "function transform(grid) { return grid.map(row => [...row, ...row]); }\n"
        "submit(transform);\n"
        "```"
    )
    assert done is True
    assert reward == 1.0
    assert info["submitted"] is True


def test_incorrect_submission(env):
    """submit(transform) with wrong transform → done=True, reward=0.0."""
    feedback, reward, done, info = env.step(
        "```javascript\n"
        "function transform(grid) { return grid; }\n"
        "submit(transform);\n"
        "```"
    )
    assert done is True
    assert reward == 0.0
    assert info["submitted"] is True


def test_thinking_blocks_stripped(env):
    """<think>...</think> blocks are stripped before parsing."""
    feedback, _, done, _ = env.step(
        "<think>Let me reason carefully.</think>\n"
        "```javascript\nconsole.log('after think')\n```"
    )
    assert "after think" in feedback
    assert done is False


def test_multiple_code_blocks_all_execute(env):
    """Multiple code blocks: all execute, feedback for last one."""
    feedback, _, _, _ = env.step(
        "```javascript\nconst x = 10\n```\n\n"
        "```javascript\nconsole.log(x * 2)\n```"
    )
    # Feedback should be for the last code block
    assert "console.log(x * 2)" in feedback
    assert "20" in feedback


def test_feedback_starts_with_code_executed(env):
    """Feedback string matches the exact eval-tool format prefix."""
    feedback, _, _, _ = env.step("```javascript\n1 + 1\n```")
    assert feedback.startswith("Code executed:\n```javascript\n")
    assert "\n```\n\nREPL output:\n" in feedback


def test_variable_persistence_across_steps(env):
    """Variables declared in step 1 available in step 2."""
    env.step("```javascript\nconst myVar = 'persisted'\n```")
    feedback, _, _, _ = env.step("```javascript\nconsole.log(myVar)\n```")
    assert "persisted" in feedback


def test_no_output(env):
    """Code with no output → (no output)."""
    feedback, _, _, _ = env.step("```javascript\nlet x = 5\n```")
    assert "(no output)" in feedback
