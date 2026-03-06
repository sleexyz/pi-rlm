"""Gate 1 — 9 tests for SandboxSession (js_sandbox.py + sandbox-server.ts)."""

import json
import pytest
from rl.js_sandbox import SandboxSession

TASK = {
    "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2, 1, 2], [3, 4, 3, 4]]}],
    "test": [{"input": [[5, 6], [7, 8]], "output": [[5, 6, 5, 6], [7, 8, 7, 8]]}],
}


def test_console_log():
    s = SandboxSession(TASK)
    r = s.eval('console.log("hello")')
    assert r["stdout"].strip() == "hello"
    assert r["error"] is None
    s.close()


def test_accuracy_match():
    s = SandboxSession(TASK)
    r = s.eval("console.log(accuracy([[1,2],[3,4]], [[1,2],[3,4]]))")
    assert "1" in r["stdout"]
    s.close()


def test_accuracy_mismatch():
    s = SandboxSession(TASK)
    r = s.eval("console.log(accuracy([[1,2],[3,4]], [[9,9],[9,9]]))")
    assert "0" in r["stdout"]
    s.close()


def test_rotate90():
    s = SandboxSession(TASK)
    r = s.eval("console.log(JSON.stringify(rotate90([[1,2],[3,4]])))")
    assert json.loads(r["stdout"].strip()) == [[3, 1], [4, 2]]
    s.close()


def test_connected_components():
    s = SandboxSession(TASK)
    r = s.eval(
        """
    const grid = [[1,0,1],[0,0,0],[1,0,1]];
    const comps = connectedComponents(grid);
    console.log(comps.length);
    """
    )
    assert r["stdout"].strip() == "4"
    s.close()


def test_training_examples_available():
    s = SandboxSession(TASK)
    r = s.eval("console.log(trainingExamples.length)")
    assert r["stdout"].strip() == "1"
    s.close()


def test_submit_detection():
    s = SandboxSession(TASK)
    r = s.eval("submit([[5,6,5,6],[7,8,7,8]])")
    assert r["submitted"] == [[5, 6, 5, 6], [7, 8, 7, 8]]
    s.close()


def test_variable_persistence():
    """Variables must persist across eval calls — same as pi-rlm's EvalRuntime."""
    s = SandboxSession(TASK)
    s.eval("const grid = trainingExamples[0].input")
    r = s.eval("console.log(JSON.stringify(grid))")
    assert json.loads(r["stdout"].strip()) == [[1, 2], [3, 4]]
    s.close()


def test_error_handling():
    s = SandboxSession(TASK)
    r = s.eval('throw new Error("boom")')
    assert r["error"] is not None
    assert "boom" in r["error"]
    s.close()
