import ast
import json
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .tasks import CodeRepairTask, count_test_assertions


RESULT_MARKER = "__CODE_REPAIR_RESULT__"


@dataclass(frozen=True)
class TestRunResult:
    suite: str
    passed: bool
    score: float
    passed_count: int
    total: int
    stdout: str
    stderr: str
    error: str
    timed_out: bool
    returncode: int
    elapsed: float


def _assert_expressions(test_code: str) -> List[str]:
    try:
        module = ast.parse(test_code or "")
    except SyntaxError:
        return []
    expressions = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            for statement in node.body:
                if isinstance(statement, ast.Assert):
                    expressions.append(ast.unparse(statement.test))
            break
    return expressions


def _instrumented_check(test_code: str) -> str:
    expressions = _assert_expressions(test_code)
    if not expressions:
        return "def check(candidate):\n    pass\n"

    lines = [
        "def check(candidate):",
        "    global __code_repair_total, __code_repair_passed, __code_repair_failures",
    ]
    for expr in expressions:
        safe_expr = repr(expr)
        lines.extend(
            [
                "    __code_repair_total += 1",
                "    try:",
                f"        assert {expr}",
                "        __code_repair_passed += 1",
                "    except BaseException as exc:",
                f"        __code_repair_failures.append({safe_expr} + ' -> ' + type(exc).__name__ + ': ' + str(exc))",
            ]
        )
    return "\n".join(lines) + "\n"


def _driver_source(task: CodeRepairTask, solution_code: str, suite_code: str, suite: str) -> str:
    check_code = _instrumented_check(suite_code)
    total = count_test_assertions(suite_code)
    return textwrap.dedent(
        f"""
        import json
        import math
        import functools
        import collections
        import itertools
        import heapq
        import bisect
        import string
        from typing import *
        from functools import *
        from collections import *
        from itertools import *
        from heapq import *
        from bisect import *
        from math import *

        inf = float("inf")

        __code_repair_total = 0
        __code_repair_passed = 0
        __code_repair_failures = []
        __code_repair_expected_total = {int(total)}
        """
    ) + "\n" + task.support_code + "\n" + solution_code + "\n\n" + check_code + textwrap.dedent(
        f"""

        result = {{
            "suite": {suite!r},
            "passed": False,
            "passed_count": 0,
            "total": __code_repair_expected_total,
            "failures": [],
            "error": "",
        }}
        try:
            candidate = {task.entry_point}
            check(candidate)
            result["passed_count"] = int(__code_repair_passed)
            result["total"] = int(__code_repair_total or __code_repair_expected_total)
            result["failures"] = list(__code_repair_failures)
            result["passed"] = result["passed_count"] == result["total"] and not result["failures"]
            if result["failures"]:
                result["error"] = "\\n".join(result["failures"][:5])
        except BaseException as exc:
            result["passed_count"] = int(__code_repair_passed)
            result["total"] = int(__code_repair_total or __code_repair_expected_total)
            result["failures"] = list(__code_repair_failures)
            result["error"] = type(exc).__name__ + ": " + str(exc)
        print({RESULT_MARKER!r} + json.dumps(result, ensure_ascii=False))
        """
    )


def run_code_repair_tests(task: CodeRepairTask, solution_code: str, suite: str = "visible", timeout: int = 8) -> TestRunResult:
    if suite not in {"visible", "full"}:
        raise ValueError(f"Unsupported code repair test suite: {suite}")
    suite_code = task.visible_test_code if suite == "visible" else task.test_code
    expected_total = count_test_assertions(suite_code)
    source = _driver_source(task, solution_code, suite_code, suite)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="code_repair_") as tmpdir:
        runner = Path(tmpdir) / "run_tests.py"
        runner.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(runner)],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=float(timeout),
                env={"PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - start
            stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            return TestRunResult(
                suite=suite,
                passed=False,
                score=0.0,
                passed_count=0,
                total=expected_total,
                stdout=stdout,
                stderr=stderr,
                error=f"Timed out after {timeout} seconds.",
                timed_out=True,
                returncode=-1,
                elapsed=elapsed,
            )

    elapsed = time.monotonic() - start
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    payload = None
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                payload = json.loads(line[len(RESULT_MARKER) :])
            except json.JSONDecodeError:
                payload = None
            break
    if payload is None:
        error = (stderr or stdout or "Test runner did not produce a result.").strip()
        return TestRunResult(
            suite=suite,
            passed=False,
            score=0.0,
            passed_count=0,
            total=expected_total,
            stdout=stdout,
            stderr=stderr,
            error=error,
            timed_out=False,
            returncode=int(completed.returncode),
            elapsed=elapsed,
        )

    total = int(payload.get("total") or expected_total or 0)
    passed_count = int(payload.get("passed_count") or 0)
    score = float(passed_count / total) if total > 0 else (1.0 if payload.get("passed") else 0.0)
    return TestRunResult(
        suite=suite,
        passed=bool(payload.get("passed")),
        score=score,
        passed_count=passed_count,
        total=total,
        stdout=stdout,
        stderr=stderr,
        error=str(payload.get("error") or ""),
        timed_out=False,
        returncode=int(completed.returncode),
        elapsed=elapsed,
    )
