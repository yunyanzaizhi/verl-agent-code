from typing import Iterable, List, Optional

from .tasks import R2EGymTask


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

R2E_ACTION_RULES = """Each response must include both reasoning (as natural text) and function call to solve the task.
Always provide concise natural-language reasoning before the function call. Then include exactly one function call in the following format, with no text after </function>:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

Important:
- Function calls MUST follow the specified format, start with <function= and end with </function>.
- Required parameters MUST be specified.
- Use <parameter=actual_argparse_name>value</parameter>. Do not write JSON, markdown tool blocks, unkeyed parameter tags, or parameter tags with a literal name placeholder.
- Only call one function at a time.
- Use cmd for execute_bash, command/path for file_editor, search_term/path for search, and command=submit for finish.
"""

R2E_CODE_REPAIR_WORKFLOW = """Follow these steps to resolve the issue:
1. First, explore the codebase to locate and understand the code relevant to the <github_issue>.
   - Use efficient search commands to identify key files and functions (i.e. use grep instead of search).
   - Err on the side of caution and look at various relevant files and build your understanding of:
     - how the code works
     - what the expected behaviors and edge cases are
     - what the potential root causes for the given issue are

2. Assess whether you can reproduce the issue:
   - Create a script at /testbed/reproduce_issue.py that demonstrates the error.
   - Execute this script to confirm the error behavior.
   - You should reproduce the issue before fixing it.
   - Your reproduction script should also assert the expected behavior for the fixed code.

3. Analyze the root cause:
   - Identify the underlying problem based on your code exploration and reproduction results.
   - Critically analyze different potential approaches to fix the issue.
   - Explicitly reason about multiple approaches, then choose the most elegant and effective solution considering correctness, generality, side effects, and tradeoffs.
   - Reason about execution paths, edge cases, and other potential issues. Look at unit tests to understand expected behavior.

4. Implement your solution:
   - Make targeted changes to the necessary files following idiomatic code patterns once you determine the root cause.
   - Be thorough and methodical.

5. Verify your solution:
   - Rerun your reproduction script to confirm the error is fixed.
   - If verification fails, iterate on your solution until successful. If the reproduction script is buggy, adjust it as needed.

6. Run unit tests:
   - Find and run the relevant unit tests for the performed fix.
   - Run unit tests to ensure your solution is correct and does not cause regressions.
   - If unit tests do not pass, consider whether the tests do not reflect the new expected behavior. If so, write additional edge test cases.
   - Use the existing test runner to run the relevant tests. For example:
     - python -m pytest -xvs sympy/physics/units/tests/test_dimensions_transcendental.py
     - python -m pytest tests/test_domain_py.py::test_pymethod_options
     - ./tests/runtests.py constraints.tests.CheckConstraintTests -v 2
   - RUN ALL relevant unit tests.
   - DO NOT MODIFY any of the existing unit tests. You can add new edge test cases in a separate file if needed, BUT DO NOT MODIFY THE EXISTING TESTS.

7. Test edge cases:
   - Identify potential edge cases that might challenge your solution.
   - Create additional test cases in a separate file /testbed/edge_case_tests.py.
   - Execute these tests to verify your solution's robustness.
   - Run multiple rounds of edge cases. When creating edge cases:
     - Consider complex scenarios beyond the original issue description.
     - Test for regressions to ensure existing functionality remains intact.
     - At each round, write multiple edge test cases in the same file to be efficient.

8. Refine if necessary:
   - If edge case testing reveals issues, refine your solution accordingly.
   - Ensure your final implementation handles all identified scenarios correctly.
   - Document any assumptions or limitations of your solution.

9. Submit your solution:
   - Once you have verified your solution, submit your solution using the finish tool.

A successful resolution means:
- The specific error or issue described no longer occurs.
- Your changes maintain compatibility with existing functionality.
- Edge cases are properly handled.

Additional recommendations:
- Be thorough, methodical, and prioritize quality over speed.
- Think carefully before making each tool call about what should be done. However, each step should only use one tool call.
- Do not use tools inside your thought process. Use thinking for identifying the root cause, making changes, and creating reproduction or edge case tests.
- Each action is somewhat expensive. Wherever possible, combine multiple actions into a single action.
- Grep commands should identify both relevant files and line numbers so you can use the file_editor tool.
- Use grep with -A, -B, and -C flags to quickly identify relevant code blocks during exploration.
- Use targeted search patterns to minimize unnecessary operations.
- When creating edge cases, look at relevant existing tests to understand regression cases. Ensure the fix does not break existing functionality.
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
You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve code repair and editing tasks to resolve the issue.

Repository: {repo}
Workspace: /testbed
Relevant files:
{files}

{R2E_TOOL_SPEC}

I have uploaded a python code repository in the /testbed directory.

Now consider the following Github issue:
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
    return f"""You are a repository-level software engineering agent running inside an R2E-Gym Docker environment.
You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve code repair and editing tasks to resolve the issue.

Repository: {repo}
Workspace: /testbed
Steps completed: {step_count}

Recent history:
{history_text}

Current observation:
{current_observation}

Continue following the R2E code repair workflow: explore, reproduce, analyze, implement, verify, test, check edge cases, refine, and submit only when ready.

{R2E_TOOL_SPEC}
{R2E_ACTION_RULES}"""
