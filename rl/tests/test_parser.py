"""Tests for parse_tool_call (parser.py)."""

from rl.parser import parse_tool_call


def test_basic_tool_call():
    text = '<tool_call>\n{"name": "eval", "arguments": {"code": "console.log(42)"}}\n</tool_call>'
    assert parse_tool_call(text) == "console.log(42)"


def test_tool_call_with_preamble():
    text = 'Let me analyze...\n<tool_call>\n{"name": "eval", "arguments": {"code": "x = 1"}}\n</tool_call>'
    assert parse_tool_call(text) == "x = 1"


def test_no_tool_call():
    assert parse_tool_call("Just some text") is None


def test_empty():
    assert parse_tool_call("") is None


def test_wrong_tool_name():
    text = '<tool_call>\n{"name": "other", "arguments": {"code": "x"}}\n</tool_call>'
    assert parse_tool_call(text) is None


def test_malformed_json():
    text = "<tool_call>\nnot json\n</tool_call>"
    assert parse_tool_call(text) is None


def test_multiline_code():
    import json
    code = "function transform(grid) {\n  return grid.map(r => r.map(() => 1));\n}"
    call = json.dumps({"name": "eval", "arguments": {"code": code}})
    text = f"<tool_call>\n{call}\n</tool_call>"
    assert parse_tool_call(text) == code
