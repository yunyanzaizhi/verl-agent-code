import json
import re
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


VIEW_ALL_ACTION = "<function=view_problem><parameter=section>all</parameter></function>"
RUN_VISIBLE_ACTION = "<function=run_tests><parameter=suite>visible</parameter></function>"
FINISH_READY_ACTION = "<function=finish><parameter=result>ready</parameter></function>"
WRONG_SOLUTION_CODE = """class Solution:
    def addOne(self, x: int) -> int:
        return x"""
CORRECT_SOLUTION_CODE = """class Solution:
    def addOne(self, x: int) -> int:
        return x + 1"""


def sample_code_repair_record(**overrides):
    record = {
        "task_id": "add-one",
        "question_id": 1,
        "difficulty": "Easy",
        "tags": ["Math"],
        "problem_description": "Return x + 1.",
        "prompt": "from typing import *\n\n",
        "starter_code": "class Solution:\n    def addOne(self, x: int) -> int:\n        ",
        "completion": "class Solution:\n    def addOne(self, x: int) -> int:\n        return x + 1\n",
        "entry_point": "Solution().addOne",
        "input_output": [
            {"input": "x = 1", "output": "2"},
            {"input": "x = -1", "output": "0"},
            {"input": "x = 41", "output": "42"},
        ],
        "test": "\n".join(
            [
                "def check(candidate):",
                "    assert candidate(x=1) == 2",
                "    assert candidate(x=-1) == 0",
                "    assert candidate(x=41) == 42",
                "    assert candidate(x=99) == 100",
            ]
        ),
    }
    record.update(overrides)
    return record


def make_replace_solution_action(code: str) -> str:
    return (
        '<function=replace_solution>'
        f'<parameter=code>{code}</parameter>'
        '</function>'
    )


WRONG_SOLUTION_ACTION = make_replace_solution_action(WRONG_SOLUTION_CODE)
CORRECT_SOLUTION_ACTION = make_replace_solution_action(CORRECT_SOLUTION_CODE)


def project_single_action(action_text: str):
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection

    actions, _ = code_repair_projection([action_text])
    return actions


def create_code_repair_task(*, visible_test_count: int = 2):
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    return normalize_code_repair_record(
        sample_code_repair_record(),
        'dataset',
        'train',
        0,
        visible_test_count=visible_test_count,
    )


def create_code_repair_env(*, history_length: int = 5, max_steps: int = 64):
    from agent_system.environments.env_package.code_repair.envs import CodeRepairVectorEnv

    task = create_code_repair_task()
    return CodeRepairVectorEnv(tasks=[task], env_num=1, group_n=1, seed=0, max_steps=max_steps, history_length=history_length)


def step_env(env, action_text: str):
    return env.step(project_single_action(action_text))


def assert_text_contains_all(text: str, fragments: list[str]) -> None:
    for fragment in fragments:
        assert fragment in text, f'Expected text to contain fragment: {fragment!r}'


def assert_prompt_matches_in_order(prompt: str, patterns: list[str]) -> None:
    search_start = 0
    for pattern in patterns:
        match = re.search(pattern, prompt[search_start:], re.DOTALL)
        assert match is not None, f'Expected prompt to match pattern after index {search_start}: {pattern!r}'
        search_start += match.end()


def assert_protocol_rejection_details(info: dict, *, expected_action: str, actual_action: str, violation_reason: str) -> None:
    assert info['is_action_valid'] is False
    assert info['code_repair_protocol_accepted'] is False
    assert info['code_repair_protocol_expected_action'] == expected_action
    assert info['code_repair_protocol_actual_action'] == actual_action
    assert info['code_repair_next_required_action'] == expected_action
    assert info['code_repair_protocol_allowed_actions'] == [expected_action]
    assert info['code_repair_protocol_violation_reason'] == violation_reason
    assert info['code_repair_protocol_side_effect_applied'] is False


def assert_observation_guides_next_action(observation: str, expected_action: str) -> None:
    assert 'Next required action:' in observation
    assert expected_action in observation


def test_code_repair_task_normalization_extracts_visible_tests_and_valid_starter():
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = normalize_code_repair_record(
        sample_code_repair_record(),
        dataset_name="local-leetcode",
        split="train",
        index=7,
        visible_test_count=2,
    )

    assert task.task_id == "local-leetcode:train:7:add-one"
    assert task.dataset_task_id == "add-one"
    assert task.entry_point == "Solution().addOne"
    assert task.current_starter_code.rstrip().endswith("pass")
    assert "assert candidate(x=1) == 2" in task.visible_test_code
    assert "assert candidate(x=-1) == 0" in task.visible_test_code
    assert "assert candidate(x=41) == 42" not in task.visible_test_code
    assert len(task.visible_examples) == 3
    assert task.raw_record["difficulty"] == "Easy"


def test_code_repair_projection_parses_tools_and_rejects_bad_schema():
    from agent_system.environments.env_package.code_repair.projection import (
        CodeRepairAction,
        code_repair_projection,
    )

    actions, valids = code_repair_projection(
        [
            """
            <function=replace_solution>
              <parameter=code>class Solution:
    def addOne(self, x: int) -> int:
        return x + 1</parameter>
            </function>
            """,
            "<function=run_tests><parameter=suite>visible</parameter></function>",
            "<function=replace_solution><parameter=note>missing code</parameter></function>",
            "plain text",
        ]
    )

    assert valids == [1, 1, 0, 0]
    assert isinstance(actions[0], CodeRepairAction)
    assert actions[0].tool_name == "replace_solution"
    assert "return x + 1" in actions[0].parameters["code"]
    assert actions[1].tool_name == "run_tests"
    assert "code" in actions[2].error
    assert "format" in actions[3].error.lower()


def test_code_repair_projection_rejects_prose_wrapped_function_call():
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection

    actions, valids = code_repair_projection(
        [
            """
            I will replace the implementation.
            <function=replace_solution><parameter=code>class Solution:\n    def addOne(self, x: int) -> int:\n        return x + 1</parameter></function>
            """
        ]
    )

    assert valids == [0]
    assert actions[0].valid is False
    assert "exactly one <function=...>...</function> XML tool call" in actions[0].error

def test_code_repair_projection_rejects_concatenated_zero_parameter_tool_calls():
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection

    actions, valids = code_repair_projection(
        ["<function=view_problem></function><function=finish></function>"]
    )

    assert valids == [0]
    assert actions[0].valid is False
    assert "exactly one <function=...>...</function> XML tool call" in actions[0].error



def test_code_repair_projection_rejects_plain_xml_parameter_tags_legacy_behavior_removed():
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection

    actions, valids = code_repair_projection(
        [
            """
            <function=replace_solution>
              <code>class Solution:
    def addOne(self, x: int) -> int:
        return x + 1</code>
            </function>
            """
        ]
    )

    assert valids == [0]
    assert actions[0].valid is False
    assert "missing required parameter" in actions[0].error


def test_code_repair_projection_rejects_markdown_solution_without_xml_legacy_behavior_removed():
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection

    actions, valids = code_repair_projection(
        [
            """
            The repaired solution is:

            ```python
            class Solution:
                def addOne(self, x: int) -> int:
                    return x + 1
            ```
            """
        ]
    )

    assert valids == [0]
    assert actions[0].valid is False
    assert "exactly one <function=...>...</function> XML tool call" in actions[0].error


def test_code_repair_projection_rejects_plain_xml_parameter_tags_without_parameter_wrapper():
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection

    actions, valids = code_repair_projection(
        [
            """
            <function=replace_solution>
              <code>class Solution:
    def addOne(self, x: int) -> int:
        return x + 1</code>
            </function>
            """
        ]
    )

    assert valids == [0]
    assert actions[0].valid is False
    assert "missing required parameter" in actions[0].error


def test_code_repair_projection_rejects_markdown_solution_without_xml_function_call():
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection

    actions, valids = code_repair_projection(
        [
            """
            The repaired solution is:

            ```python
            class Solution:
                def addOne(self, x: int) -> int:
                    return x + 1
            ```
            """
        ]
    )

    assert valids == [0]
    assert actions[0].valid is False
    assert "exactly one <function=...>...</function> XML tool call" in actions[0].error


def test_code_repair_prompts_include_strict_protocol_rules():
    from agent_system.environments.env_package.code_repair.prompts import build_code_repair_initial_prompt
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = create_code_repair_task()
    prompt = build_code_repair_initial_prompt(task, "initial observation")

    assert "CRITICAL FORMAT RULES" in prompt
    assert "Do NOT write Markdown code fences" in prompt
    assert "No text is allowed after </function>" in prompt
    assert "Protocol reminder" in prompt


def test_code_repair_prompt_hard_reject_contract():
    from agent_system.environments.env_package.code_repair.prompts import build_code_repair_initial_prompt

    task = create_code_repair_task()
    prompt = build_code_repair_initial_prompt(task, "initial observation")

    assert_prompt_matches_in_order(
        prompt,
        [
            r"## Repair policy",
            r"Start with\s+view_problem\(section=all\)",
            r"After (?:each|every)\s+replace_solution[\s\S]*?run_tests[\s\S]*?suite=visible",
            r"If visible tests fail[\s\S]*?view_problem\(section=all\)",
            r"If visible tests pass[\s\S]*?(?:call\s+)?finish",
        ],
    )
    assert_text_contains_all(
        prompt,
        [
            "Illegal-order actions are rejected",
            "do not edit code",
            "do not run tests",
            "do not finish the episode",
            "same protocol state remains active",
            "Next required action",
        ],
    )
    assert_prompt_matches_in_order(
        prompt,
        [
            r"Hard rejection examples:",
            r"replace_solution twice in a row",
            r"finish before visible tests pass",
        ],
    )


def test_code_repair_executor_runs_visible_and_full_tests():
    from agent_system.environments.env_package.code_repair.executor import run_code_repair_tests
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = create_code_repair_task()

    starter_result = run_code_repair_tests(task, task.current_starter_code, suite="visible", timeout=5)
    assert starter_result.passed is False
    assert starter_result.total == 2
    assert starter_result.passed_count < starter_result.total
    assert starter_result.error

    full_result = run_code_repair_tests(task, task.raw_record["completion"], suite="full", timeout=5)
    assert full_result.passed is True
    assert full_result.total == 4
    assert full_result.passed_count == 4
    assert full_result.score == 1.0


def test_code_repair_vector_env_supports_multiturn_repair_and_finish():
    from agent_system.environments.env_package.code_repair.envs import CodeRepairVectorEnv
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = create_code_repair_task()
    env = CodeRepairVectorEnv(tasks=[task], env_num=1, group_n=1, seed=0, max_steps=64, history_length=5)

    obs, infos = env.reset()
    assert "Return x + 1" in obs[0]
    assert infos[0]["task_id"] == "dataset:train:0:add-one"

    actions = project_single_action(VIEW_ALL_ACTION)
    next_obs, rewards, dones, infos = env.step(actions)
    assert "Current solution" in next_obs[0]
    assert rewards[0] > 0.0
    assert dones == [False]
    assert infos[0]["code_repair_action"]["tool_name"] == "view_problem"

    actions, _ = code_repair_projection(
        [
            """
            <function=replace_solution>
              <parameter=code>class Solution:
    def addOne(self, x: int) -> int:
        return x + 1</parameter>
            </function>
            """
        ]
    )
    _obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert infos[0]["code_repair_edit_count"] == 1

    actions = project_single_action(RUN_VISIBLE_ACTION)
    _obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert infos[0]["code_repair_visible_score"] == 1.0
    assert rewards[0] > 0.0

    actions = project_single_action(FINISH_READY_ACTION)
    _obs, rewards, dones, infos = env.step(actions)
    assert dones == [True]
    assert infos[0]["won"] is True
    assert infos[0]["code_repair_full_score"] == 1.0
    assert rewards[0] >= 1.0
    env.close()




def test_code_repair_env_rejects_second_replace_without_visible_test_side_effects():
    env = create_code_repair_env()

    try:
        env.reset()
        step_env(env, VIEW_ALL_ACTION)

        _obs, _rewards, dones, infos = step_env(env, WRONG_SOLUTION_ACTION)
        assert dones == [False]
        code_after_first_replace = env.states[0].current_code
        protocol_state_after_first_replace = infos[0]["code_repair_protocol_state_after"]
        code_hash_after_first_replace = infos[0]["code_repair_current_code_hash_after"]

        obs, rewards, dones, infos = step_env(env, CORRECT_SOLUTION_ACTION)

        assert dones == [False]
        assert rewards[0] <= -1.0
        assert_text_contains_all(obs[0], ["run visible tests", "editing again"])
        assert infos[0]["is_action_valid"] is False
        assert infos[0]["code_repair_edit_count"] == 1
        assert infos[0]["code_repair_test_count"] == 0
        assert infos[0]["code_repair_next_required_action"] == "run_tests"
        assert infos[0]["code_repair_protocol_actual_action"] == "replace_solution"
        assert infos[0]["code_repair_protocol_expected_action"] == "run_tests"
        assert infos[0]["code_repair_protocol_allowed_actions"] == ["run_tests"]
        assert infos[0]["code_repair_protocol_accepted"] is False
        assert infos[0]["code_repair_protocol_violation_reason"] == "replace_without_testing"
        assert infos[0]["code_repair_protocol_side_effect_applied"] is False
        assert infos[0]["code_repair_protocol_state_before"] == protocol_state_after_first_replace
        assert infos[0]["code_repair_protocol_state_after"] == protocol_state_after_first_replace
        assert infos[0]["code_repair_current_code_hash_before"] == code_hash_after_first_replace
        assert infos[0]["code_repair_current_code_hash_after"] == code_hash_after_first_replace
        assert env.states[0].current_code == code_after_first_replace
        assert env.states[0].done is False
    finally:
        env.close()


def test_code_repair_env_rejects_finish_before_visible_success_side_effects():
    env = create_code_repair_env()

    try:
        env.reset()
        code_before = env.states[0].current_code
        code_hash_before = hash(code_before)

        obs, rewards, dones, infos = step_env(env, FINISH_READY_ACTION)

        assert dones == [False]
        assert rewards[0] < 0.0
        assert_text_contains_all(obs[0], ["Finish rejected", "Next required action"])
        assert infos[0]["is_action_valid"] is False
        assert infos[0]["code_repair_edit_count"] == 0
        assert infos[0]["code_repair_test_count"] == 0
        assert infos[0]["code_repair_next_required_action"] == "view_problem"
        assert infos[0]["code_repair_protocol_state_before"] == "need_view"
        assert infos[0]["code_repair_protocol_state_after"] == "need_view"
        assert infos[0]["code_repair_protocol_actual_action"] == "finish"
        assert infos[0]["code_repair_protocol_expected_action"] == "view_problem"
        assert infos[0]["code_repair_protocol_allowed_actions"] == ["view_problem"]
        assert infos[0]["code_repair_protocol_accepted"] is False
        assert infos[0]["code_repair_protocol_violation_reason"] == "action_before_initial_view"
        assert infos[0]["code_repair_protocol_side_effect_applied"] is False
        assert infos[0]["code_repair_current_code_hash_before"] == code_hash_before
        assert infos[0]["code_repair_current_code_hash_after"] == code_hash_before
        assert env.states[0].current_code == code_before
        assert env.states[0].done is False
    finally:
        env.close()
def test_code_repair_env_golden_trajectory_and_adversarial_rejection():
    env = create_code_repair_env()

    try:
        env.reset()

        obs, rewards, dones, infos = step_env(env, VIEW_ALL_ACTION)
        assert dones == [False]
        assert rewards[0] > 0.0
        assert infos[0]["code_repair_next_required_action"] == "replace_solution"

        obs, rewards, dones, infos = step_env(env, WRONG_SOLUTION_ACTION)
        assert dones == [False]
        assert rewards[0] > 0.0
        assert infos[0]["code_repair_edit_count"] == 1
        assert infos[0]["code_repair_next_required_action"] == "run_tests"

        obs, rewards, dones, infos = step_env(env, CORRECT_SOLUTION_ACTION)
        assert dones == [False]
        assert rewards[0] <= -1.0
        assert_protocol_rejection_details(
            infos[0],
            expected_action="run_tests",
            actual_action="replace_solution",
            violation_reason="replace_without_testing",
        )

        obs, rewards, dones, infos = step_env(env, RUN_VISIBLE_ACTION)
        assert dones == [False]
        assert infos[0]["code_repair_visible_score"] == 0.0
        assert infos[0]["code_repair_next_required_action"] == "view_problem"

        obs, rewards, dones, infos = step_env(env, VIEW_ALL_ACTION)
        assert dones == [False]
        assert infos[0]["code_repair_next_required_action"] == "replace_solution"

        obs, rewards, dones, infos = step_env(env, CORRECT_SOLUTION_ACTION)
        assert dones == [False]
        assert infos[0]["is_action_valid"] is True
        assert infos[0]["code_repair_edit_count"] == 2
        assert infos[0]["code_repair_next_required_action"] == "run_tests"

        obs, rewards, dones, infos = step_env(env, RUN_VISIBLE_ACTION)
        assert dones == [False]
        assert infos[0]["code_repair_visible_score"] == 1.0
        assert infos[0]["code_repair_next_required_action"] == "finish"

        _obs, rewards, dones, infos = step_env(env, FINISH_READY_ACTION)
        assert dones == [True]
        assert infos[0]["won"] is True
        assert infos[0]["code_repair_full_score"] == 1.0
        assert rewards[0] >= 1.0
    finally:
        env.close()


def test_code_repair_env_enforces_test_after_each_replacement_before_finish():
    from agent_system.environments.env_package.code_repair.envs import CodeRepairVectorEnv

    task = create_code_repair_task()
    env = CodeRepairVectorEnv(tasks=[task], env_num=1, group_n=1, seed=0, max_steps=64, history_length=5)
    env.reset()

    obs, rewards, dones, infos = step_env(env, VIEW_ALL_ACTION)
    assert dones == [False]
    assert_observation_guides_next_action(obs[0], "replace_solution")

    wrong_solution = make_replace_solution_action(WRONG_SOLUTION_CODE)
    correct_solution = make_replace_solution_action(CORRECT_SOLUTION_CODE)

    obs, rewards, dones, infos = step_env(env, WRONG_SOLUTION_ACTION)
    assert dones == [False]
    assert rewards[0] > 0.0
    assert_observation_guides_next_action(obs[0], "run_tests")
    assert infos[0]["code_repair_next_required_action"] == "run_tests"
    assert infos[0]["code_repair_step_policy_violation_count"] == 0

    obs, rewards, dones, infos = step_env(env, CORRECT_SOLUTION_ACTION)
    assert dones == [False]
    assert rewards[0] <= -1.0
    assert_observation_guides_next_action(obs[0], "run_tests")
    assert infos[0]["code_repair_edit_count"] == 1
    assert infos[0]["code_repair_step_policy_violation_count"] == 1
    assert infos[0]["code_repair_policy_violation_count"] == 1
    assert infos[0]["code_repair_order_events"] == ["replace_without_testing"]
    assert_protocol_rejection_details(
        infos[0],
        expected_action="run_tests",
        actual_action="replace_solution",
        violation_reason="replace_without_testing",
    )

    obs, rewards, dones, infos = step_env(env, FINISH_READY_ACTION)
    assert dones == [False]
    assert rewards[0] < 0.0
    assert "Finish rejected" in obs[0]
    assert infos[0]["code_repair_next_required_action"] == "run_tests"
    assert infos[0]["code_repair_order_events"] == ["finish_before_visible_success"]

    obs, rewards, dones, infos = step_env(env, RUN_VISIBLE_ACTION)
    assert dones == [False]
    assert infos[0]["code_repair_visible_score"] == 0.0
    assert infos[0]["code_repair_next_required_action"] == "view_problem"
    assert_observation_guides_next_action(obs[0], "view_problem")

    obs, rewards, dones, infos = step_env(env, VIEW_ALL_ACTION)
    assert dones == [False]
    assert infos[0]["code_repair_next_required_action"] == "replace_solution"
    assert_observation_guides_next_action(obs[0], "replace_solution")

    obs, rewards, dones, infos = step_env(env, CORRECT_SOLUTION_ACTION)
    assert dones == [False]
    assert infos[0]["is_action_valid"] is True
    assert infos[0]["code_repair_edit_count"] == 2
    assert infos[0]["code_repair_next_required_action"] == "run_tests"
    assert_observation_guides_next_action(obs[0], "run_tests")

    obs, rewards, dones, infos = step_env(env, RUN_VISIBLE_ACTION)
    assert dones == [False]
    assert infos[0]["code_repair_visible_score"] == 1.0
    assert infos[0]["code_repair_next_required_action"] == "finish"
    assert_observation_guides_next_action(obs[0], "finish")

    _obs, rewards, dones, infos = step_env(env, FINISH_READY_ACTION)
    assert dones == [True]
    assert infos[0]["won"] is True
    assert rewards[0] >= 1.0
    env.close()


def test_code_repair_env_rejected_finish_on_max_step_does_not_auto_finish_or_run_full_tests():
    env = create_code_repair_env(max_steps=1)

    try:
        env.reset()
        code_before = env.states[0].current_code
        obs, rewards, dones, infos = step_env(env, FINISH_READY_ACTION)

        assert dones == [False]
        assert "Episode finalized by max_steps." not in obs[0]
        assert infos[0]["code_repair_test_count"] == 0
        assert infos[0]["code_repair_full_score"] == 0.0
        assert infos[0]["code_repair_protocol_accepted"] is False
        assert infos[0]["code_repair_protocol_side_effect_applied"] is False
        assert env.states[0].done is False
        assert env.states[0].current_code == code_before
    finally:
        env.close()


def test_code_repair_env_max_step_penalizes_unverified_late_edit_after_visible_pass():
    env = create_code_repair_env(max_steps=4)

    try:
        env.reset()
        step_env(env, VIEW_ALL_ACTION)
        step_env(env, CORRECT_SOLUTION_ACTION)
        step_env(env, RUN_VISIBLE_ACTION)

        assert env.states[0].visible_passed_current_code is True
        assert env.states[0].current_code_tested is True

        late_edit_action = make_replace_solution_action(
            CORRECT_SOLUTION_CODE.replace("return x + 1", "return (x + 1)")
        )
        obs, rewards, dones, infos = step_env(env, late_edit_action)

        assert dones == [True]
        assert "Episode finalized by max_steps." in obs[0]
        assert infos[0]["code_repair_visible_passed_current_code"] is False
        assert infos[0]["code_repair_tested_current_code"] is False
        assert infos[0]["code_repair_step_policy_violation_count"] == 1
        assert "max_steps_without_visible_test_gate" in infos[0]["code_repair_order_events"]
        assert rewards[0] < 2.0
    finally:
        env.close()


def test_code_repair_env_oversized_replace_is_invalid_without_side_effects_or_order_credit():
    env = create_code_repair_env()

    try:
        env.reset()
        step_env(env, VIEW_ALL_ACTION)
        state_before = env.states[0]
        code_before = state_before.current_code
        order_correct_before = state_before.order_correct_count
        violation_streak_before = state_before.order_violation_streak
        shaping_before = state_before.order_shaping_reward_sum

        oversized_code = "class Solution:\\n" + "    # x\\n" * 5000
        oversized_action = make_replace_solution_action(oversized_code)
        obs, rewards, dones, infos = step_env(env, oversized_action)

        assert dones == [False]
        assert rewards[0] == -env.invalid_action_penalty
        assert "Replacement rejected" in obs[0]
        assert infos[0]["is_action_valid"] is False
        assert infos[0]["code_repair_protocol_accepted"] is True
        assert infos[0]["code_repair_protocol_side_effect_applied"] is False
        assert infos[0]["code_repair_edit_count"] == 0
        assert infos[0]["code_repair_order_correct_count"] == order_correct_before
        assert infos[0]["code_repair_order_violation_streak"] == violation_streak_before
        assert infos[0]["code_repair_order_shaping_reward_sum"] == shaping_before
        assert infos[0]["code_repair_protocol_state_before"] == "need_replace"
        assert infos[0]["code_repair_protocol_state_after"] == "need_replace"
        assert env.states[0].current_code == code_before
        assert env.states[0].protocol_snapshot.state.name.lower() == "need_replace"
    finally:
        env.close()


def test_code_repair_env_auto_finish_on_max_steps_clears_next_required_action_guidance():
    from agent_system.environments.env_package.code_repair.envs import CodeRepairVectorEnv
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = create_code_repair_task()
    env = CodeRepairVectorEnv(tasks=[task], env_num=1, group_n=1, seed=0, max_steps=1, history_length=5)

    try:
        env.reset()
        actions = project_single_action(VIEW_ALL_ACTION)
        obs, rewards, dones, infos = env.step(actions)

        assert dones == [True]
        assert rewards[0] < 0.0
        assert "Episode finalized by max_steps." in obs[0]
        assert "Next required action:" not in obs[0]
        assert infos[0]["code_repair_next_required_action"] == "done"
        assert infos[0]["won"] is False
    finally:
        env.close()


def test_code_repair_env_rewards_view_refresh_after_failed_visible_tests():
    from agent_system.environments.env_package.code_repair.envs import CodeRepairVectorEnv
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = create_code_repair_task()
    env = CodeRepairVectorEnv(tasks=[task], env_num=1, group_n=1, seed=0, max_steps=64, history_length=5)
    env.reset()

    wrong_solution = """
    <function=replace_solution>
      <parameter=code>class Solution:
    def addOne(self, x: int) -> int:
        return x</parameter>
    </function>
    """
    correct_solution = """
    <function=replace_solution>
      <parameter=code>class Solution:
    def addOne(self, x: int) -> int:
        return x + 1</parameter>
    </function>
    """

    actions = project_single_action(VIEW_ALL_ACTION)
    env.step(actions)
    actions, _ = code_repair_projection([wrong_solution])
    env.step(actions)

    actions = project_single_action(RUN_VISIBLE_ACTION)
    obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert rewards[0] > 0.0
    assert infos[0]["code_repair_visible_score"] < 1.0
    assert infos[0]["code_repair_next_required_action"] == "view_problem"
    assert_observation_guides_next_action(obs[0], "view_problem")

    actions, _ = code_repair_projection([correct_solution])
    obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert rewards[0] <= -1.0
    assert_observation_guides_next_action(obs[0], "view_problem")
    assert_protocol_rejection_details(
        infos[0],
        expected_action="view_problem",
        actual_action="replace_solution",
        violation_reason="action_before_initial_view",
    )
    assert infos[0]["code_repair_protocol_state_before"] == "need_view"
    assert infos[0]["code_repair_protocol_state_after"] == "need_view"
    assert infos[0]["code_repair_edit_count"] == 1
    assert infos[0]["code_repair_order_events"] == ["replace_before_required_view"]

    env.close()

    env = CodeRepairVectorEnv(tasks=[task], env_num=1, group_n=1, seed=0, max_steps=64, history_length=5)
    env.reset()
    actions = project_single_action(VIEW_ALL_ACTION)
    env.step(actions)
    actions, _ = code_repair_projection([wrong_solution])
    env.step(actions)
    actions = project_single_action(RUN_VISIBLE_ACTION)
    env.step(actions)

    actions = project_single_action(VIEW_ALL_ACTION)
    obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert rewards[0] > 0.0
    assert "Visible test harness" in obs[0]
    assert infos[0]["code_repair_next_required_action"] == "replace_solution"
    assert infos[0]["code_repair_required_action_match"] is True

    actions, _ = code_repair_projection([correct_solution])
    _obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert rewards[0] > 0.0
    assert infos[0]["code_repair_next_required_action"] == "run_tests"
    env.close()


def test_code_repair_followup_prompt_surfaces_next_required_action():
    from agent_system.environments.env_manager import CodeRepairEnvironmentManager
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection

    env = create_code_repair_env()
    cfg = OmegaConf.create({"env": {"history_length": 5}})
    manager = CodeRepairEnvironmentManager(env, code_repair_projection, cfg)

    manager.reset(kwargs=None)
    manager.step([VIEW_ALL_ACTION])
    observations, _rewards, _dones, infos = manager.step([CORRECT_SOLUTION_ACTION])

    assert_observation_guides_next_action(observations["text"][0], "run_tests")
    assert infos[0]["code_repair_next_required_action"] == "run_tests"
    assert infos[0]["code_repair_protocol_actual_action"] == "replace_solution"
    assert infos[0]["code_repair_protocol_expected_action"] == "replace_solution"
    assert infos[0]["code_repair_protocol_state_before"] == "need_replace"
    assert infos[0]["code_repair_protocol_state_after"] == "need_visible_test"
    assert infos[0]["code_repair_required_action_match"] is True
    assert_text_contains_all(
        observations["text"][0],
        [
            "Recent repair history",
            "Step 2 action:",
            "replace_solution(code_chars=",
        ],
    )
    env.close()


def test_rollout_metrics_keep_code_repair_separate_from_r2e_invalid_actions():
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    infos = [
        {
            "is_action_valid": True,
            "code_repair_action": {"tool_name": "replace_solution", "parameters": {}, "valid": True},
            "code_repair_step_policy_violation_count": 1,
            "code_repair_visible_score": 0.5,
            "code_repair_full_score": 0.0,
        }
    ]
    active_masks = np.array([True])

    r2e_metrics = TrajectoryCollector._r2e_step_metrics_from_infos(infos, active_masks, batch_size=1)
    code_repair_metrics = TrajectoryCollector._code_repair_step_metrics_from_infos(
        infos, active_masks, batch_size=1
    )

    assert r2e_metrics["r2e_invalid_action_count"][0] == 0.0
    assert code_repair_metrics["code_repair_tool_replace_solution_count"][0] == 1.0
    assert code_repair_metrics["code_repair_step_policy_violation_count"][0] == 1.0
    assert code_repair_metrics["code_repair_visible_score"][0] == 0.5


def test_code_repair_step_metrics_treat_env_invalid_as_invalid_step_even_when_schema_is_valid():
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    infos = [
        {
            "is_action_valid": False,
            "code_repair_action": {"tool_name": "replace_solution", "parameters": {}, "valid": True},
            "code_repair_protocol_accepted": True,
            "code_repair_protocol_side_effect_applied": False,
            "code_repair_step_policy_violation_count": 0,
            "code_repair_visible_score": 0.0,
            "code_repair_full_score": 0.0,
        }
    ]
    active_masks = np.array([True])

    metrics = TrajectoryCollector._code_repair_step_metrics_from_infos(infos, active_masks, batch_size=1)

    assert metrics["code_repair_invalid_action_step_count"].tolist() == [1.0]
    assert metrics["code_repair_protocol_accept_count"].tolist() == [0.0]
    assert metrics["code_repair_tool_replace_solution_count"].tolist() == [0.0]


def test_code_repair_step_metrics_do_not_overwrite_env_counter_semantics():
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    infos = [
        {
            "is_action_valid": False,
            "code_repair_action": {"tool_name": "replace_solution", "parameters": {}, "valid": False},
            "code_repair_invalid_action_count": 7,
            "code_repair_policy_violation_count": 4,
            "code_repair_step_policy_violation_count": 1,
            "code_repair_visible_score": 0.25,
            "code_repair_full_score": 0.75,
        }
    ]
    active_masks = np.array([True])

    metrics = TrajectoryCollector._code_repair_step_metrics_from_infos(infos, active_masks, batch_size=1)

    assert metrics["code_repair_invalid_action_step_count"].tolist() == [1.0]
    assert metrics["code_repair_step_policy_violation_count"].tolist() == [1.0]
    assert metrics["code_repair_visible_score"].tolist() == [0.25]
    assert metrics["code_repair_full_score"].tolist() == [0.75]
    assert "code_repair_invalid_action_count" not in metrics
    assert "code_repair_policy_violation_count" not in metrics



def test_code_repair_protocol_accepts_and_rejects_metrics_and_payload_block(tmp_path):
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

    infos = [
        {
            "task_id": "task-accept",
            "is_action_valid": True,
            "code_repair_action": {"tool_name": "view_problem", "parameters": {"section": "all"}, "valid": True},
            "code_repair_protocol_state_before": "need_view",
            "code_repair_protocol_allowed_actions": ["view_problem"],
            "code_repair_protocol_expected_action": "view_problem",
            "code_repair_protocol_actual_action": "view_problem",
            "code_repair_protocol_accepted": True,
            "code_repair_protocol_violation_reason": None,
            "code_repair_protocol_side_effect_applied": True,
            "code_repair_protocol_state_after": "need_replace",
            "code_repair_current_code_hash_before": 101,
            "code_repair_current_code_hash_after": 202,
            "code_repair_raw_observation": "accepted raw",
        },
        {
            "task_id": "task-reject",
            "is_action_valid": False,
            "code_repair_action": {"tool_name": "replace_solution", "parameters": {}, "valid": True},
            "code_repair_protocol_state_before": "need_tests",
            "code_repair_protocol_allowed_actions": ["run_tests"],
            "code_repair_protocol_expected_action": "run_tests",
            "code_repair_protocol_actual_action": "replace_solution",
            "code_repair_protocol_accepted": False,
            "code_repair_protocol_violation_reason": "replace_without_testing",
            "code_repair_protocol_side_effect_applied": False,
            "code_repair_protocol_state_after": "need_tests",
            "code_repair_current_code_hash_before": 303,
            "code_repair_current_code_hash_after": 303,
            "code_repair_raw_observation": "rejected raw",
        },
        {
            "task_id": "task-invalid",
            "is_action_valid": False,
            "code_repair_action": {"tool_name": "replace_solution", "parameters": {}, "valid": False},
            "code_repair_raw_observation": "invalid raw",
        },
    ]
    active_masks = np.array([True, True, True])

    metrics = TrajectoryCollector._code_repair_step_metrics_from_infos(infos, active_masks, batch_size=3)

    assert metrics["code_repair_protocol_accept_count"].tolist() == [1.0, 0.0, 0.0]
    assert metrics["code_repair_protocol_reject_count"].tolist() == [0.0, 1.0, 0.0]
    assert metrics["code_repair_protocol_side_effect_applied_count"].tolist() == [1.0, 0.0, 0.0]
    assert metrics["code_repair_invalid_action_step_count"].tolist() == [0.0, 0.0, 1.0]

    class _Logger:
        enabled = True
        def __init__(self):
            self.payloads = []
        def write_step(self, **kwargs):
            self.payloads.append(kwargs["payload"])

    logger = _Logger()
    collector = object.__new__(TrajectoryCollector)
    collector.episode_step_logger = logger

    collector._write_episode_step_logs(
        train_step=9,
        rollout_step=4,
        active_masks=active_masks,
        next_obs={"text": ["accept obs", "reject obs", "invalid obs"], "anchor": [None, None, None]},
        rewards=np.array([0.1, -1.0, -2.0], dtype=np.float32),
        dones=np.array([False, False, False]),
        infos=infos,
        uid_batch=np.array(["u1", "u2", "u3"], dtype=object),
        traj_uid=np.array(["t1", "t2", "t3"], dtype=object),
        text_actions=["a1", "a2", "a3"],
    )

    assert len(logger.payloads) == 3
    assert logger.payloads[0]["env"]["name"] == "code_repair"
    assert logger.payloads[0]["env"]["protocol"] == {
        "state_before": "need_view",
        "allowed_actions": ["view_problem"],
        "expected_action": "view_problem",
        "actual_action": "view_problem",
        "accepted": True,
        "violation_reason": None,
        "side_effect_applied": True,
        "state_after": "need_replace",
        "current_code_hash_before": 101,
        "current_code_hash_after": 202,
    }
    assert logger.payloads[1]["env"]["protocol"] == {
        "state_before": "need_tests",
        "allowed_actions": ["run_tests"],
        "expected_action": "run_tests",
        "actual_action": "replace_solution",
        "accepted": False,
        "violation_reason": "replace_without_testing",
        "side_effect_applied": False,
        "state_after": "need_tests",
        "current_code_hash_before": 303,
        "current_code_hash_after": 303,
    }
    assert logger.payloads[2]["env"]["protocol"] is None


def test_code_repair_manager_limits_followup_history_to_five_steps():
    from agent_system.environments.env_manager import CodeRepairEnvironmentManager
    from agent_system.environments.env_package.code_repair.envs import CodeRepairVectorEnv
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = create_code_repair_task()
    env = CodeRepairVectorEnv(tasks=[task], env_num=1, group_n=1, seed=0, max_steps=64, history_length=5)
    cfg = OmegaConf.create({"env": {"history_length": 5}})
    manager = CodeRepairEnvironmentManager(env, code_repair_projection, cfg)

    manager.reset(kwargs=None)
    action = "<function=view_problem><parameter=section>code</parameter></function>"
    prompt = None
    for _ in range(7):
        observations, _rewards, _dones, _infos = manager.step([action])
        prompt = observations["text"][0]

    assert "Recent repair history" in prompt
    assert "Step 1 action" not in prompt
    assert "Step 2 action" not in prompt
    assert "Step 3 action" in prompt
    assert "Step 7 action" in prompt
    env.close()


def test_make_envs_registers_code_repair(tmp_path):
    from agent_system.environments.env_manager import CodeRepairEnvironmentManager, make_envs

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "test.jsonl"
    train_path.write_text(json.dumps(sample_code_repair_record()) + "\n")
    val_path.write_text(json.dumps(sample_code_repair_record(task_id="val-add-one")) + "\n")

    cfg = OmegaConf.create(
        {
            "data": {"train_batch_size": 1, "val_batch_size": 1},
            "env": {
                "env_name": "code_repair",
                "seed": 0,
                "max_steps": 64,
                "history_length": 5,
                "resources_per_worker": {"num_cpus": 0.1, "num_gpus": 0},
                "rollout": {"n": 2},
                "code_repair": {
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "visible_test_count": 2,
                    "execution_timeout": 5,
                },
            },
        }
    )

    envs, val_envs = make_envs(cfg)
    assert isinstance(envs, CodeRepairEnvironmentManager)
    assert isinstance(val_envs, CodeRepairEnvironmentManager)

    obs, infos = envs.reset(kwargs=None)
    assert len(obs["text"]) == 2
    assert np.array(obs["anchor"]).shape == (2,)
    assert infos[0]["task_id"].endswith("add-one")
    envs.close()
    val_envs.close()


def test_hgpo_make_envs_registers_code_repair_for_training_entrypoint(tmp_path):
    from agent_system.environments.env_manager import CodeRepairEnvironmentManager
    from recipe.hgpo.env_manager import make_envs

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "test.jsonl"
    train_path.write_text(json.dumps(sample_code_repair_record()) + "\n")
    val_path.write_text(json.dumps(sample_code_repair_record(task_id="val-add-one")) + "\n")

    cfg = OmegaConf.create(
        {
            "trainer": {"val_only": False},
            "data": {"train_batch_size": 1, "val_batch_size": 1},
            "env": {
                "env_name": "code_repair",
                "seed": 0,
                "max_steps": 64,
                "history_length": 5,
                "resources_per_worker": {"num_cpus": 0.1, "num_gpus": 0},
                "rollout": {"n": 2},
                "code_repair": {
                    "train_path": str(train_path),
                    "val_path": str(val_path),
                    "visible_test_count": 2,
                    "execution_timeout": 5,
                },
            },
        }
    )

    envs, val_envs = make_envs(cfg)
    assert isinstance(envs, CodeRepairEnvironmentManager)
    assert isinstance(val_envs, CodeRepairEnvironmentManager)
    envs.close()
    val_envs.close()
