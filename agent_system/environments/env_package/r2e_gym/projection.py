import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from r2egym.agenthub.action import Action


@dataclass(frozen=True)
class ParsedR2EAction:
    action: Any
    valid: bool
    error: str = ""


_FUNCTION_BLOCK_RE = re.compile(r"<function\s*=\s*[^>]+>.*?</function>", re.DOTALL | re.IGNORECASE)
MULTIPLE_TOOL_CALLS_WARNING = (
    "multiple XML tool calls were found; only the first tool call was executed, "
    "please output exactly one tool call."
)
_FILE_EDITOR_COMMANDS = {"view", "create", "str_replace", "insert", "undo_edit"}
_TOOL_SCHEMAS = {
    "execute_bash": {
        "required": {"cmd"},
        "allowed": {"cmd"},
    },
    "search": {
        "required": {"search_term"},
        "allowed": {"search_term", "path", "python_only"},
    },
    "file_editor": {
        "required": {"command"},
        "allowed": {
            "command",
            "path",
            "file_text",
            "view_range",
            "old_str",
            "new_str",
            "insert_line",
            "enable_linting",
            "concise",
            "python_only",
        },
    },
    "finish": {
        "required": set(),
        "allowed": {"command", "result"},
    },
    "submit": {
        "required": set(),
        "allowed": {"command", "result"},
    },
}

# Regex for fallback: matches <tag_name>value</tag_name> where tag_name is NOT "function" or "parameter..."
_PLAIN_TAG_RE = re.compile(
    r"<(\w+)>(.*?)</\1>", re.DOTALL
)


def _fallback_merge_plain_tags(block: str, function_name: str, existing_params: Dict[str, str]) -> Dict[str, str]:
    """Parse plain XML tags like <command>view</command> as parameters.

    Merges into existing_params without overwriting already-parsed <parameter=key> values.
    Only accepts tag names that are valid parameters for the given tool.
    """
    schema = _TOOL_SCHEMAS.get(function_name)
    if schema is None:
        return existing_params
    allowed = schema["allowed"]
    merged = dict(existing_params)
    for match in _PLAIN_TAG_RE.finditer(block):
        tag_name = match.group(1).strip()
        tag_value = match.group(2).strip()
        # Skip tags that are already parsed via <parameter=key> or not valid param names
        if tag_name in merged:
            continue
        if tag_name in allowed:
            merged[tag_name] = tag_value
    return merged


def _invalid_action(error: str, raw_action: str = "") -> Dict[str, Any]:
    action = {
        "function_name": "",
        "parameters": {},
        "error": error,
    }
    if raw_action:
        action["raw_action"] = raw_action
    return action


def _schema_error(action: Action) -> str:
    function_name = (action.function_name or "").strip()
    params = getattr(action, "parameters", {}) or {}
    schema = _TOOL_SCHEMAS.get(function_name)
    if schema is None:
        return f"Action schema error: unknown tool '{function_name}'. Use execute_bash, file_editor, search, or finish."

    param_keys = set(params)
    missing = sorted(schema["required"] - param_keys)
    unknown = sorted(param_keys - schema["allowed"])
    if unknown or missing:
        parts = []
        if unknown:
            parts.append(f"unknown parameter(s): {', '.join(unknown)}")
        if missing:
            parts.append(f"missing required parameter(s): {', '.join(missing)}")
        message = f"Action schema error for {function_name}: {'; '.join(parts)}."
        if function_name == "file_editor" and "file_path" in unknown:
            message += " Use parameter path, not file_path."
        return message

    if function_name == "file_editor":
        command = (params.get("command") or "").strip()
        if command not in _FILE_EDITOR_COMMANDS:
            allowed = ", ".join(sorted(_FILE_EDITOR_COMMANDS))
            return f"Action schema error: invalid file_editor command '{command}'. Use one of: {allowed}."
        if command in {"view", "create", "str_replace", "insert", "undo_edit"} and not str(params.get("path", "")).strip():
            return "Action schema error: missing required parameter(s) for file_editor: path."
        command_required = {
            "create": {"file_text"},
            "str_replace": {"old_str", "new_str"},
            "insert": {"insert_line", "new_str"},
        }
        missing_for_command = sorted(key for key in command_required.get(command, set()) if key not in params)
        if missing_for_command:
            return (
                f"Action schema error: missing required parameter(s) for file_editor {command}: "
                f"{', '.join(missing_for_command)}."
            )

    return ""


def parse_r2e_gym_action(text: str) -> ParsedR2EAction:
    if text is None or not str(text).strip():
        error = "Action format error: empty model output. Use one <function=...>...</function> XML tool call."
        return ParsedR2EAction(_invalid_action(error, str(text or "")), False, error)

    blocks = _FUNCTION_BLOCK_RE.findall(str(text))
    if not blocks:
        error = "Action format error: output must contain exactly one <function=...>...</function> XML tool call."
        return ParsedR2EAction(_invalid_action(error, str(text)), False, error)

    warning = MULTIPLE_TOOL_CALLS_WARNING if len(blocks) > 1 else ""
    block = blocks[0].strip()
    try:
        action = Action.from_string(block)
    except Exception as exc:
        error = f"Action format error: failed to parse XML tool call: {exc}"
        return ParsedR2EAction(_invalid_action(error, str(text)), False, error)

    if not action.function_name:
        error = "Action format error: missing function name."
        return ParsedR2EAction(_invalid_action(error, str(text)), False, error)

    # --- FALLBACK: merge any plain-XML tags the model used instead of <parameter=key> ---
    existing = getattr(action, "parameters", {}) or {}
    merged = _fallback_merge_plain_tags(block, action.function_name, existing)
    if merged != existing:
        action.parameters = merged

    schema_error = _schema_error(action)
    if schema_error:
        return ParsedR2EAction(_invalid_action(schema_error, str(text)), False, schema_error)
    if warning:
        action.parse_warning = warning
    return ParsedR2EAction(action, True, "")


def r2e_gym_projection(actions: List[str]) -> Tuple[List[Any], List[int]]:
    parsed_actions: List[Any] = []
    valids: List[int] = []
    for action_text in actions:
        parsed = parse_r2e_gym_action(action_text)
        parsed_actions.append(parsed.action)
        valids.append(1 if parsed.valid else 0)
    return parsed_actions, valids
