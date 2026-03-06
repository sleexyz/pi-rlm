"""
Parser for extracting code from model tool call output.

Parses <tool_call>{"name":"eval","arguments":{"code":"..."}}</tool_call>
from model-generated text (Qwen3 tool call format).
"""

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str) -> str | None:
    """Extract code from a <tool_call> block. Returns the code string or None."""
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        call = json.loads(m.group(1))
        if call.get("name") == "eval":
            return call.get("arguments", {}).get("code")
    except json.JSONDecodeError:
        pass
    return None
