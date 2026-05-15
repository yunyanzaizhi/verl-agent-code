import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class CodeRepairAction:
    tool_name: str
    parameters: Dict[str, str]
    raw_text: str = ""
    valid: bool = True
    error: str = ""
    parse_warning: str = ""
    recovered: bool = False
    recovery_reason: str = ""


_FUNCTION_BLOCK_RE = re.compile(
    r"<function\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*>(.*?)</function>",
    re.DOTALL | re.IGNORECASE,
)
_PARAM_RE = re.compile(r"<parameter\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*>(.*?)</parameter>", re.DOTALL)
_PLAIN_TAG_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)\s*>(.*?)</\1>", re.DOTALL)
_PYTHON_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

MULTIPLE_TOOL_CALLS_WARNING = (
    "multiple XML tool calls were found; only the first tool call was executed, "
    "please output exactly one tool call."
)

_TOOL_SCHEMAS = {
    "view_problem": {
        "required": set(),
        "allowed": {"section"},
    },
    "replace_solution": {
        "required": {"code"},
        "allowed": {"code"},
    },
    "run_tests": {
        "required": set(),
        "allowed": {"suite"},
    },
    "finish": {
        "required": set(),
        "allowed": {"result"},
    },
}
_VIEW_SECTIONS = {"problem", "code", "tests", "history", "all"}
_TEST_SUITES = {"visible", "full"}


def _invalid(error: str, raw_text: str) -> CodeRepairAction:
    return CodeRepairAction(tool_name="", parameters={}, raw_text=str(raw_text or ""), valid=False, error=error)


def _parse_params(block_body: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for match in _PARAM_RE.finditer(block_body):
        params[match.group(1).strip()] = match.group(2).strip()
    return params


def _fallback_merge_plain_tags(block_body: str, tool_name: str, existing_params: Dict[str, str]) -> Dict[str, str]:
    """Recover plain XML tags like <code>...</code> as parameters.

    The strict <parameter=name> form always wins. Plain tags are accepted only
    when their tag names are allowed by the selected tool schema.
    """
    schema = _TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return existing_params

    merged = dict(existing_params)
    for match in _PLAIN_TAG_RE.finditer(block_body):
        tag_name = match.group(1).strip()
        tag_value = match.group(2).strip()
        if tag_name in merged:
            continue
        if tag_name in schema["allowed"]:
            merged[tag_name] = tag_value
    return merged


def _recover_markdown_solution(raw_text: str) -> CodeRepairAction:
    """Recover a Markdown Python solution as a replace_solution tool call."""
    candidates = []
    for match in _PYTHON_FENCE_RE.finditer(raw_text):
        code = textwrap.dedent(match.group(1)).strip()
        if "class Solution" in code:
            candidates.append(code)

    if len(candidates) != 1:
        return _invalid(
            "Action format error: output must contain exactly one <function=...>...</function> XML tool call. "
            "If replacing code, use <function=replace_solution><parameter=code>...</parameter></function>.",
            raw_text,
        )

    return CodeRepairAction(
        tool_name="replace_solution",
        parameters={"code": candidates[0]},
        raw_text=raw_text,
        valid=True,
        parse_warning="Recovered Markdown Python code block as replace_solution.",
        recovered=True,
        recovery_reason="markdown_solution_to_replace_solution",
    )


def parse_code_repair_action(text: str) -> CodeRepairAction:
    raw_text = str(text or "")
    if not raw_text.strip():
        return _invalid("Action format error: empty model output. Use one XML tool call.", raw_text)

    blocks = _FUNCTION_BLOCK_RE.findall(raw_text)
    if not blocks:
        return _recover_markdown_solution(raw_text)

    warning = MULTIPLE_TOOL_CALLS_WARNING if len(blocks) > 1 else ""
    tool_name, body = blocks[0]
    tool_name = tool_name.strip()
    params = _parse_params(body)
    params = _fallback_merge_plain_tags(body, tool_name, params)
    schema = _TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return _invalid(
            f"Action schema error: unknown tool '{tool_name}'. Use view_problem, replace_solution, run_tests, or finish.",
            raw_text,
        )

    keys = set(params)
    missing = sorted(schema["required"] - keys)
    unknown = sorted(keys - schema["allowed"])
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing required parameter(s): {', '.join(missing)}")
        if unknown:
            parts.append(f"unknown parameter(s): {', '.join(unknown)}")
        return _invalid(f"Action schema error for {tool_name}: {'; '.join(parts)}.", raw_text)

    if tool_name == "view_problem" and "section" in params and params["section"] not in _VIEW_SECTIONS:
        allowed = ", ".join(sorted(_VIEW_SECTIONS))
        return _invalid(f"Action schema error: invalid view_problem section '{params['section']}'. Use one of: {allowed}.", raw_text)
    if tool_name == "run_tests" and "suite" in params and params["suite"] not in _TEST_SUITES:
        allowed = ", ".join(sorted(_TEST_SUITES))
        return _invalid(f"Action schema error: invalid run_tests suite '{params['suite']}'. Use one of: {allowed}.", raw_text)

    return CodeRepairAction(
        tool_name=tool_name,
        parameters=params,
        raw_text=raw_text,
        valid=True,
        parse_warning=warning,
    )


def code_repair_projection(actions: List[str]) -> Tuple[List[CodeRepairAction], List[int]]:
    parsed_actions: List[CodeRepairAction] = []
    valids: List[int] = []
    for action_text in actions:
        parsed = parse_code_repair_action(action_text)
        parsed_actions.append(parsed)
        valids.append(1 if parsed.valid else 0)
    return parsed_actions, valids


def format_code_repair_action_for_history(action: Any) -> str:
    if isinstance(action, CodeRepairAction):
        if action.error:
            return action.error
        params = action.parameters or {}
        if action.tool_name == "replace_solution":
            code = params.get("code", "")
            return f"replace_solution(code_chars={len(code)})"
        if params:
            return f"{action.tool_name}({', '.join(f'{key}={value}' for key, value in params.items())})"
        return f"{action.tool_name}()"
    if isinstance(action, dict):
        return str(action.get("error") or action.get("tool_name") or "invalid action")
    return str(action)
