"""Gate 2 — 6 tests for parse_code_blocks (parser.py)."""

from rl.parser import parse_code_blocks


def test_single_block():
    segs = parse_code_blocks("Text\n```js\ncode()\n```\nMore")
    assert segs == [
        {"type": "text", "content": "Text"},
        {"type": "code", "content": "code()"},
        {"type": "text", "content": "More"},
    ]


def test_ignores_non_js():
    segs = parse_code_blocks("```python\nno\n```\n```js\nyes\n```")
    code = [s for s in segs if s["type"] == "code"]
    assert len(code) == 1 and code[0]["content"] == "yes"


def test_multiple():
    segs = parse_code_blocks("A\n```js\n1\n```\nB\n```js\n2\n```")
    code = [s for s in segs if s["type"] == "code"]
    assert len(code) == 2


def test_empty():
    assert parse_code_blocks("") == []


def test_no_blocks():
    assert parse_code_blocks("Just text") == [{"type": "text", "content": "Just text"}]


def test_whitespace_stripped():
    segs = parse_code_blocks("\n\n```js\nx = 1\n```\n\n")
    assert segs == [{"type": "code", "content": "x = 1"}]
