from typing import Iterable, List, Optional

from .tasks import R2EGymTask


R2E_TOOL_SPEC = """Available tools:
- execute_bash: run a shell command inside /testbed. Required parameter: cmd.
- file_editor: view, create, str_replace, insert, or undo_edit files. Required parameters depend on the command.
- search: search for a term in a file or directory. Required parameter: search_term.
- finish: submit the final repository patch. Use <function=finish><parameter=command>submit</parameter></function> when done.
"""

R2E_ACTION_RULES = """Your response must be exactly one XML tool call and nothing else:
<function=tool_name>
  <parameter=name>value</parameter>
</function>
"""


def format_relevant_files(files: Optional[Iterable[str]]) -> str:
    values = [str(path) for path in (files or [])]
    if not values:
        return "No relevant files were provided."
    return "\n".join(f"- {path}" for path in values)


def format_r2e_action_for_history(action) -> str:
    if hasattr(action, "function_name"):
        params = getattr(action, "parameters", {}) or {}
        if params:
            param_text = ", ".join(f"{key}={value}" for key, value in params.items())
            return f"{action.function_name}({param_text})"
        return f"{action.function_name}()"
    if isinstance(action, dict):
        return action.get("error") or action.get("function_name") or "invalid action"
    return str(action)


def format_r2e_history_turn(step: int, action_text: str, observation: str, max_observation_chars: int = 4000) -> str:
    obs = observation or ""
    if len(obs) > max_observation_chars:
        obs = "..." + obs[-max_observation_chars:]
    return f"Step {step} action:\n{action_text}\n\nStep {step} observation:\n{obs}"


def build_r2e_initial_prompt(task: Optional[R2EGymTask], current_observation: str) -> str:
    repo = task.repo_name if task is not None else "unknown"
    files = format_relevant_files(task.relevant_files if task is not None else [])
    issue = current_observation.strip()
    return f"""You are a repository-level software engineering agent running inside an R2E-Gym Docker environment.

Repository: {repo}
Workspace: /testbed

GitHub issue:
{issue}

Relevant files:
{files}

Rules:
- Inspect the repository before editing.
- Make source changes in /testbed to fix the issue.
- Do not edit hidden answers, reward metadata, or test expectations.
- Prefer focused validation commands after editing.
- Finish only when the patch is ready to be scored.

{R2E_TOOL_SPEC}
{R2E_ACTION_RULES}"""


def build_r2e_followup_prompt(
    task: Optional[R2EGymTask],
    current_observation: str,
    history: List[str],
    step_count: int,
) -> str:
    repo = task.repo_name if task is not None else "unknown"
    history_text = "\n\n".join(history).strip() or "No previous tool calls."
    return f"""You are a repository-level software engineering agent running inside an R2E-Gym Docker environment.

Repository: {repo}
Workspace: /testbed
Steps completed: {step_count}

Recent history:
{history_text}

Current observation:
{current_observation}

{R2E_TOOL_SPEC}
{R2E_ACTION_RULES}"""
