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


def _invalid_action(error: str) -> Dict[str, Any]:
    return {
        "function_name": "",
        "parameters": {},
        "error": error,
    }


def parse_r2e_gym_action(text: str) -> ParsedR2EAction:
    if text is None or not str(text).strip():
        error = "Action format error: empty model output. Use one <function=...>...</function> XML tool call."
        return ParsedR2EAction(_invalid_action(error), False, error)

    blocks = _FUNCTION_BLOCK_RE.findall(str(text))
    if len(blocks) != 1:
        error = "Action format error: output must contain exactly one <function=...>...</function> XML tool call."
        return ParsedR2EAction(_invalid_action(error), False, error)

    block = blocks[0].strip()
    try:
        action = Action.from_string(block)
    except Exception as exc:
        error = f"Action format error: failed to parse XML tool call: {exc}"
        return ParsedR2EAction(_invalid_action(error), False, error)

    if not action.function_name:
        error = "Action format error: missing function name."
        return ParsedR2EAction(_invalid_action(error), False, error)
    return ParsedR2EAction(action, True, "")


def r2e_gym_projection(actions: List[str]) -> Tuple[List[Any], List[int]]:
    parsed_actions: List[Any] = []
    valids: List[int] = []
    for action_text in actions:
        parsed = parse_r2e_gym_action(action_text)
        parsed_actions.append(parsed.action)
        valids.append(1 if parsed.valid else 0)
    return parsed_actions, valids
