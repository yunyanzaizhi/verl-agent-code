from typing import Any, Dict, Iterable, List, Optional

from .tasks import CodeRepairTask


CODE_REPAIR_TOOL_SPEC = """We have access to the following functions:

-- BEGIN FUNCTION #1: view_problem --
Description:
Inspect the current LeetCode repair task, including the problem statement,
current solution, visible tests, and recent repair history.

Parameters:
  1. section (string, optional)
Allowed values: [problem, code, tests, history, all]
Defaults to all.

-- END FUNCTION #1 --

-- BEGIN FUNCTION #2: replace_solution --
Description:
Replace the entire Python solution with a complete class Solution
implementation. This is the only tool that edits the candidate solution.

Parameters:
  1. code (string, required)
The complete Python code. It must include class Solution. Do not provide a
diff. Do not wrap the code in Markdown fences.

-- END FUNCTION #2 --

-- BEGIN FUNCTION #3: run_tests --
Description:
Run tests against the current solution.

Parameters:
  1. suite (string, optional)
Allowed values: [visible, full]
Defaults to visible. Use visible during repair and full before finish when
available.

-- END FUNCTION #3 --

-- BEGIN FUNCTION #4: finish --
Description:
Finish the episode after the solution has been edited and verified.

Parameters:
  1. result (string, optional)
A short final summary.

-- END FUNCTION #4 --"""

CODE_REPAIR_ACTION_RULES = """Each response must contain exactly one XML function call.

## Required format

<function=TOOL_NAME>
<parameter=PARAM_NAME>value</parameter>
</function>

CRITICAL FORMAT RULES:
- Output exactly one <function=...>...</function> block.
- Every parameter tag MUST use <parameter=NAME>value</parameter>.
- Do NOT write <code>...</code>. Write <parameter=code>...</parameter>.
- Do NOT write <suite>visible</suite>. Write <parameter=suite>visible</parameter>.
- Do NOT write Markdown code fences such as ```python.
- Do NOT answer with plain text only.
- No text is allowed after </function>.
- replace_solution must contain the complete class Solution implementation, not a patch or explanation.

## Examples

Example 1 - Inspect the task:
<function=view_problem>
<parameter=section>all</parameter>
</function>

Example 2 - Replace the solution:
<function=replace_solution>
<parameter=code>class Solution:
    def twoSum(self, nums, target):
        return []</parameter>
</function>

Example 3 - Run visible tests:
<function=run_tests>
<parameter=suite>visible</parameter>
</function>

Example 4 - Finish:
<function=finish>
<parameter=result>Verified the repaired solution.</parameter>
</function>

## Repair policy
Before choosing the next action, silently check the immediately previous action in Recent repair history and the
Latest environment observation's "Next required action".

Follow exactly this workflow:
1. Start with view_problem(section=all).
2. After view_problem(section=all), call replace_solution with the complete class Solution implementation.
3. After every replace_solution call, call run_tests with suite=visible.
4. If visible tests fail, call view_problem(section=all) before editing again.
5. After that refreshed view_problem(section=all), call replace_solution again, then run_tests(suite=visible) again.
6. Repeat steps 4 and 5 until visible tests pass.
7. If visible tests pass for the current code, call finish.

Wrong-order actions receive large reward penalties:
- Do not call replace_solution twice in a row.
- Do not call replace_solution immediately after failed visible tests; refresh first with view_problem(section=all).
- Do not call finish before visible tests pass for the current replacement.
- Do not ignore "Next required action"; it names the only action that follows the intended workflow.
"""

CODE_REPAIR_PROTOCOL_REMINDER = """Protocol reminder:
Your next message must end with exactly one XML function call.
Do not output Markdown. Do not output a plain-text answer.

Use this shape for code edits:
<function=replace_solution>
<parameter=code>class Solution:
    ...
</parameter>
</function>
"""


def _format_examples(examples: Iterable[Dict[str, Any]], limit: int = 5) -> str:
    rows = []
    for idx, item in enumerate(list(examples)[:limit], start=1):
        rows.append(f"{idx}. input: {item.get('input', '')}\n   expected: {item.get('output', '')}")
    return "\n".join(rows) if rows else "No visible examples were provided."


def format_code_repair_history_turn(step: int, action_text: str, observation: str, max_chars: int = 3000) -> str:
    obs = observation or ""
    if len(obs) > max_chars:
        obs = "..." + obs[-max_chars:]
    return f"Step {step} action:\n{action_text}\n\nStep {step} observation:\n{obs}"


def build_code_repair_initial_prompt(task: Optional[CodeRepairTask], current_observation: str) -> str:
    if task is None:
        return current_observation
    tags = ", ".join(task.tags) if task.tags else "unknown"
    return f"""You are a Python code repair agent. Repair the current LeetCode solution over multiple turns.

Task id: {task.dataset_task_id}
Difficulty: {task.difficulty or "unknown"}
Tags: {tags}

Problem:
{task.problem_description}

Visible examples:
{_format_examples(task.visible_examples)}

Current solution:
```python
{task.current_starter_code.rstrip()}
```

{CODE_REPAIR_TOOL_SPEC}
{CODE_REPAIR_ACTION_RULES}
{CODE_REPAIR_PROTOCOL_REMINDER}
"""


def build_code_repair_followup_prompt(
    task: Optional[CodeRepairTask],
    current_observation: str,
    history: List[str],
    step_count: int,
) -> str:
    if task is None:
        return current_observation
    history_text = "\n\n".join(history).strip() or "No previous repair steps."
    return f"""You are continuing a multi-turn Python code repair episode.

Task id: {task.dataset_task_id}
Steps completed: {step_count}

Problem:
{task.problem_description}

Latest environment observation:
{current_observation}

Recent repair history:
{history_text}

Action selection reminder:
First inspect the immediately previous action in Recent repair history.
Then obey the Latest environment observation's "Next required action".
Output only the XML function call for that next action.

{CODE_REPAIR_ACTION_RULES}
{CODE_REPAIR_PROTOCOL_REMINDER}
"""
