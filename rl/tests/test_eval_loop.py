"""Tests for eval_loop — mocks vLLM engine to verify loop logic."""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


SAMPLE_TASK = {
    "train": [
        {"input": [[1, 0], [0, 1]], "output": [[1, 1], [1, 1]]},
    ],
    "test": [
        {"input": [[0, 1], [1, 0]], "output": [[1, 1], [1, 1]]},
    ],
}


def _make_mock_llm(tokenizer):
    """Create a mock vLLM LLM that returns output matching tokenizer.decode."""
    mock_llm = MagicMock()
    output = MagicMock()
    output.token_ids = [10, 11, 12]
    completion = MagicMock()
    completion.outputs = [output]
    mock_llm.generate.return_value = [completion]
    return mock_llm


def _make_mock_interaction(terminate_on_turn=2, reward=1.0):
    """Create a mock ArcInteraction that terminates on a given turn."""
    mock_interaction = MagicMock()
    mock_interaction.pool = MagicMock()
    call_count = {"n": 0}

    async def mock_start(instance_id=None, **kwargs):
        return instance_id

    async def mock_generate(instance_id, messages, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= terminate_on_turn:
            return True, "Submitted. Reward: 1.0", reward, {"submitted": True}
        return False, "Code executed:\n```js\nconsole.log('hi')\n```\n\nREPL output:\nhi", 0.0, {"submitted": False}

    async def mock_finalize(instance_id, **kwargs):
        pass

    mock_interaction.start_interaction = AsyncMock(side_effect=mock_start)
    mock_interaction.generate_response = AsyncMock(side_effect=mock_generate)
    mock_interaction.finalize_interaction = AsyncMock(side_effect=mock_finalize)
    return mock_interaction


def test_eval_loop_single_task():
    """Verify the eval loop processes a task and produces correct output."""
    call_count = {"n": 0}

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]

    def decode_side_effect(ids, skip_special_tokens=True):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "Let me analyze...\n```js\nfunction transform(grid) { return grid.map(r => r.map(() => 1)); }\n```"
        else:
            return "```js\nsubmit(transform)\n```"

    mock_tokenizer.decode.side_effect = decode_side_effect

    mock_llm = _make_mock_llm(mock_tokenizer)
    mock_interaction = _make_mock_interaction(terminate_on_turn=2, reward=1.0)

    from rl.eval_loop import evaluate_tasks

    tasks = [{"id": "test001", "task": SAMPLE_TASK}]

    with tempfile.TemporaryDirectory() as tmp, \
         patch("rl.eval_loop.ArcInteraction", return_value=mock_interaction):
        run_dir = os.path.join(tmp, "test-run")
        evaluate_tasks(
            model_name="test-model",
            tasks=tasks,
            system_prompt="You are an ARC solver.",
            run_dir=run_dir,
            run_name="test-run",
            max_turns=5,
            llm=mock_llm,
            tokenizer=mock_tokenizer,
            sampling_params=MagicMock(),
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
        assert result["turns"] == 2
        assert result["reward"] == 1.0

        # Verify trace was written
        trace_path = os.path.join(run_dir, "sessions", "test001_a0", "agent-0.jsonl")
        assert os.path.exists(trace_path)
        with open(trace_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert lines[0]["type"] == "session"
        assert lines[1]["type"] == "message"
        assert lines[1]["message"]["role"] == "user"
        assert lines[2]["type"] == "message"
        assert lines[2]["message"]["role"] == "assistant"
        assert lines[-1]["type"] == "session_end"

        # Verify tokenizer was called with enable_thinking=False
        mock_tokenizer.apply_chat_template.assert_called()
        call_kwargs = mock_tokenizer.apply_chat_template.call_args_list[0]
        assert call_kwargs.kwargs.get("enable_thinking") is False
        assert call_kwargs.kwargs.get("add_generation_prompt") is True

        # Verify decode was called with skip_special_tokens=True
        mock_tokenizer.decode.assert_called()
        decode_kwargs = mock_tokenizer.decode.call_args_list[0]
        assert decode_kwargs.kwargs.get("skip_special_tokens") is True


def test_eval_loop_max_turns_reached():
    """Verify loop terminates when max_turns reached even without submission."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer.decode.return_value = "Let me think more...\n```js\nconsole.log('hi')\n```"

    mock_llm = _make_mock_llm(mock_tokenizer)
    # Never terminates
    mock_interaction = _make_mock_interaction(terminate_on_turn=999, reward=0.0)

    from rl.eval_loop import evaluate_tasks

    tasks = [{"id": "test002", "task": SAMPLE_TASK}]

    with tempfile.TemporaryDirectory() as tmp, \
         patch("rl.eval_loop.ArcInteraction", return_value=mock_interaction):
        run_dir = os.path.join(tmp, "test-run")
        evaluate_tasks(
            model_name="test-model",
            tasks=tasks,
            system_prompt="You are an ARC solver.",
            run_dir=run_dir,
            run_name="test-run",
            max_turns=3,
            llm=mock_llm,
            tokenizer=mock_tokenizer,
            sampling_params=MagicMock(),
        )

        with open(os.path.join(run_dir, "run.json")) as f:
            data = json.load(f)

        result = data["results"][0]
        assert result["turns"] == 3
        assert result["correct"] is False
