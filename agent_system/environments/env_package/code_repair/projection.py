import re
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
_FUNCTION_OPEN_RE = re.compile(r"<function\s*=", re.IGNORECASE)

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

STRICT_XML_ERROR = (
    "Action format error: output must contain exactly one <function=...>...</function> XML tool call. "
    "If replacing code, use <function=replace_solution><parameter=code>...</parameter></function>."
)


def _invalid(error: str, raw_text: str) -> CodeRepairAction:
    return CodeRepairAction(tool_name="", parameters={}, raw_text=str(raw_text or ""), valid=False, error=error)


def _parse_params(block_body: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for match in _PARAM_RE.finditer(block_body):
        params[match.group(1).strip()] = match.group(2).strip()
    return params


def parse_code_repair_action(text: str) -> CodeRepairAction:
    raw_text = str(text or "")
    stripped_text = raw_text.strip()
    if not stripped_text:
        return _invalid("Action format error: empty model output. Use one XML tool call.", raw_text)

    match = _FUNCTION_BLOCK_RE.fullmatch(stripped_text)
    if match is None:
        return _invalid(STRICT_XML_ERROR, raw_text)

    warning = ""
    tool_name, body = match.groups()
    if _FUNCTION_OPEN_RE.search(body):
        return _invalid(STRICT_XML_ERROR, raw_text)

    tool_name = tool_name.strip()
    params = _parse_params(body)
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
