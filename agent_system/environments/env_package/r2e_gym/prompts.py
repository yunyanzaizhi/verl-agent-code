from typing import Iterable, List, Optional

from .tasks import R2EGymTask


R2E_FOLLOWUP_ISSUE_MAX_CHARS = 2000

R2E_TOOL_SPEC = """We have access to the following functions:

-- BEGIN FUNCTION #1: file_editor --
Description:
Custom editing tool for viewing, creating and editing files.
  - State is persistent across command calls and discussions with the user.
  - If path is a file, view displays the result of applying cat -n. If path is a directory, view lists non-hidden files and directories up to 2 levels deep.
  - The create command cannot be used if the specified path already exists as a file.
  - If a command generates a long output, it will be truncated and marked with <response clipped>.
  - The undo_edit command will revert the last edit made to the file at path.

Notes for using the str_replace command:
  - The old_str parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces.
  - If the old_str parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in old_str to make it unique.
  - The new_str parameter should contain the edited lines that should replace the old_str.

Parameters:
  1. command (string, required)
Allowed values: [view, create, str_replace, insert, undo_edit]
The command to run.
  2. path (string, required)
Absolute path to file or directory, e.g. /testbed/file.py or /testbed.
  3. file_text (string, optional)
Required for the create command. Contains the content of the file to be created.
  4. old_str (string, optional)
Required for the str_replace command. The exact string in path to replace.
  5. new_str (string, optional)
  - Optional for the str_replace command to specify the replacement string.
  - Required for the insert command to specify the string to insert.
  6. insert_line (integer, optional)
Required for the insert command. The new_str will be inserted after the line number specified here.
  7. view_range (array, optional)
  - Optional for the view command when path is a file.
  - If provided, specifies the line range to view, e.g. [11, 12] shows lines 11 and 12.
  - [start_line, -1] will show all lines from start_line to the end of file.
  8. concise (boolean, optional)
  - Optional for the view command.
  - Defaults to True; displays a concise skeletal view of the file. If set to False, displays the full content in the specified view_range.

-- END FUNCTION #1 --

-- BEGIN FUNCTION #2: execute_bash --
Description:
Execute a bash command in the terminal.

Behavior notes:
  - If a command may run indefinitely, consider running it in the background and redirecting output, e.g. python3 app.py > server.log 2>&1 &.
  - If the bash command returns exit code -1, it means the process is still running. The assistant may:
  - Call this function again with cmd as an empty string ("") to retrieve additional logs.
  - Send more input to STDIN of the running process by calling this function again with cmd set to the text input.
  - Send cmd="ctrl+c" to interrupt the currently running process.
  - If the command times out, it will be interrupted with SIGINT. The assistant may then retry or do further steps if needed.

Parameters:
  1. cmd (string, required)
The bash command and optional arguments to execute.
  - Can be empty ("") to retrieve more logs if the process is still running.
  - Can be "ctrl+c" to interrupt the running process.

-- END FUNCTION #2 --

-- BEGIN FUNCTION #3: search --
Description:
Search for a term in a directory or a single file.
  - If path is a directory or unspecified, default is ., it recursively searches all non-hidden files and directories for the search term.
  - If path points to a file, it runs a grep -n in that file to show line numbers matching the search term.
  - If more than 100 files match in a directory search, results are truncated and the tool will inform you to narrow your search.
  - If no matches are found, it will inform you as well.

Parameters:
  1. search_term (string, required)
The term or string to search for in files.
  2. path (string, optional)
The file or directory to search in. Defaults to . if not specified.

-- END FUNCTION #3 --

-- BEGIN FUNCTION #4: finish --
Description:
Finish the interaction once the task is complete or if no further progress can be made.

Behavior notes:
  - The submit command finalizes your output.

Parameters:
  1. command (string, required)
Currently allowed value: [submit]
  2. result (string, optional)
The result text or final message to submit. Defaults to an empty string if not provided.

-- END FUNCTION #4 --"""

R2E_ACTION_RULES = """Each response must include both reasoning (as natural text) and exactly one function call.

## Format

<function=TOOL_NAME>
<parameter=PARAM_NAME>value</parameter>
</function>

CRITICAL FORMAT RULES:
- Every parameter tag MUST use the <parameter=NAME> syntax. The equals sign and parameter name go inside the opening tag.
- Do NOT write <command>view</command>. Write <parameter=command>view</parameter> instead.
- Do NOT write <path>/testbed</path>. Write <parameter=path>/testbed</parameter> instead.
- Do NOT write <cmd>ls</cmd>. Write <parameter=cmd>ls</parameter> instead.

## Examples

Example 1 - View a directory:
<function=file_editor>
<parameter=command>view</parameter>
<parameter=path>/testbed</parameter>
</function>

Example 2 - View a file with line range:
<function=file_editor>
<parameter=command>view</parameter>
<parameter=path>/testbed/src/module.py</parameter>
<parameter=view_range>[1, 50]</parameter>
</function>

Example 3 - Run a bash command:
<function=execute_bash>
<parameter=cmd>grep -rn "def my_function" /testbed/src/</parameter>
</function>

Example 4 - Search for a term:
<function=search>
<parameter=search_term>class MyClass</parameter>
<parameter=path>/testbed/src</parameter>
</function>

Example 5 - Edit a file:
<function=file_editor>
<parameter=command>str_replace</parameter>
<parameter=path>/testbed/src/module.py</parameter>
<parameter=old_str>def my_function(x):
    return x</parameter>
<parameter=new_str>def my_function(x):
    if x is None:
        raise ValueError("x must not be None")
    return x</parameter>
</function>

Example 6 - Submit when done:
<function=finish>
<parameter=command>submit</parameter>
<parameter=result>Fixed the issue by adding input validation.</parameter>
</function>

## Rules
- Only call one function at a time.
- No text after </function>.
- Always start by exploring the repository structure with file_editor view on /testbed, then locate the relevant source files.
- Do NOT submit until you have actually edited files and verified the fix works.
"""

R2E_CODE_REPAIR_WORKFLOW = """Follow these steps to resolve the issue:
1. EXPLORE: Use file_editor to view /testbed directory structure first, then locate the relevant source files.
   - Start with: file_editor view /testbed to see the project layout.
   - Use search or execute_bash with grep to find the relevant code.
   - Do NOT guess file paths. Always explore first.

2. REPRODUCE: Create a script at /testbed/reproduce_issue.py that demonstrates the error, then run it.

3. ANALYZE: Identify the root cause from exploration and reproduction results.

4. IMPLEMENT: Use file_editor str_replace to make targeted changes to fix the issue.

5. VERIFY: Rerun your reproduction script to confirm the fix works.

6. TEST: Run relevant unit tests to ensure no regressions.

7. SUBMIT: Only after verification passes, use finish with command=submit.

IMPORTANT: Do NOT submit until you have actually used file_editor to edit files and verified the fix.
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


def format_r2e_issue_for_followup(task: Optional[R2EGymTask], max_chars: int = R2E_FOLLOWUP_ISSUE_MAX_CHARS) -> str:
    if task is None:
        return "No original Github issue was provided."
    issue = (task.problem_statement or "").strip()
    if not issue:
        return "No original Github issue was provided."
    if len(issue) <= max_chars:
        return issue
    return f"{issue[:max_chars].rstrip()}\n[truncated]"


def build_r2e_initial_prompt(task: Optional[R2EGymTask], current_observation: str) -> str:
    repo = task.repo_name if task is not None else "unknown"
    files = format_relevant_files(task.relevant_files if task is not None else [])
    issue = current_observation.strip()
    return f"""You are a software engineering agent inside a Docker environment at /testbed.
Your task: fix the github issue below by editing source files.

Repository: {repo}
Workspace: /testbed
Relevant files:
{files}

{R2E_TOOL_SPEC}

<github_issue>
{issue}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I have already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way. Your task is to make changes to non-test files in the /testbed directory to ensure the <github_issue> is resolved.

{R2E_CODE_REPAIR_WORKFLOW}
{R2E_ACTION_RULES}"""


def build_r2e_followup_prompt(
    task: Optional[R2EGymTask],
    current_observation: str,
    history: List[str],
    step_count: int,
) -> str:
    repo = task.repo_name if task is not None else "unknown"
    history_text = "\n\n".join(history).strip() or "No previous tool calls."
    issue = format_r2e_issue_for_followup(task)
    return f"""You are a software engineering agent inside a Docker environment at /testbed.
Repository: {repo} | Steps completed: {step_count}

Original issue:
<github_issue>
{issue}
</github_issue>

Recent history:
{history_text}

Current observation:
{current_observation}

Continue working. Remember:
- Do NOT submit until you have actually edited files and verified the fix.
- Start by exploring /testbed if you haven't yet.

{R2E_ACTION_RULES}"""
