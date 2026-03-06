"""
Code-block parser — Python port of parseReplBlocks from pi-rlm/src/repl-stream.ts.

Same regex: triple-backtick js/repl blocks.
"""

import re

_BLOCK_RE = re.compile(r"```(?:repl|js)\s*\n(.*?)\n```", re.DOTALL)


def parse_code_blocks(text: str) -> list[dict]:
    """Parse text into segments: alternating prose and ```js/```repl code blocks."""
    if not text or not text.strip():
        return []

    segments = []
    last_end = 0

    for m in _BLOCK_RE.finditer(text):
        before = text[last_end : m.start()].strip()
        if before:
            segments.append({"type": "text", "content": before})
        segments.append({"type": "code", "content": m.group(1).strip()})
        last_end = m.end()

    after = text[last_end:].strip()
    if after:
        segments.append({"type": "text", "content": after})

    return segments
