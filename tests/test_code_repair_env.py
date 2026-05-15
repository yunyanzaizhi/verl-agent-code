import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


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
            I will replace the implementation.
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


def test_code_repair_projection_recovers_plain_xml_parameter_tags():
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

    assert valids == [1]
    assert actions[0].tool_name == "replace_solution"
    assert actions[0].parameters["code"].startswith("class Solution")
    assert "return x + 1" in actions[0].parameters["code"]


def test_code_repair_projection_recovers_markdown_solution_as_replace_solution():
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

    assert valids == [1]
    assert actions[0].tool_name == "replace_solution"
    assert actions[0].parameters["code"].lstrip().startswith("class Solution")
    assert actions[0].recovered is True
    assert actions[0].recovery_reason == "markdown_solution_to_replace_solution"


def test_code_repair_prompts_include_strict_protocol_rules():
    from agent_system.environments.env_package.code_repair.prompts import build_code_repair_initial_prompt
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = normalize_code_repair_record(sample_code_repair_record(), "dataset", "train", 0, visible_test_count=2)
    prompt = build_code_repair_initial_prompt(task, "initial observation")

    assert "CRITICAL FORMAT RULES" in prompt
    assert "Do NOT write Markdown code fences" in prompt
    assert "No text is allowed after </function>" in prompt
    assert "Protocol reminder" in prompt


def test_code_repair_executor_runs_visible_and_full_tests():
    from agent_system.environments.env_package.code_repair.executor import run_code_repair_tests
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = normalize_code_repair_record(sample_code_repair_record(), "dataset", "train", 0, visible_test_count=2)

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

    task = normalize_code_repair_record(sample_code_repair_record(), "dataset", "train", 0, visible_test_count=2)
    env = CodeRepairVectorEnv(tasks=[task], env_num=1, group_n=1, seed=0, max_steps=64, history_length=5)

    obs, infos = env.reset()
    assert "Return x + 1" in obs[0]
    assert infos[0]["task_id"] == "dataset:train:0:add-one"

    actions, _ = code_repair_projection(["<function=view_problem><parameter=section>all</parameter></function>"])
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

    actions, _ = code_repair_projection(["<function=run_tests><parameter=suite>visible</parameter></function>"])
    _obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert infos[0]["code_repair_visible_score"] == 1.0
    assert rewards[0] > 0.0

    actions, _ = code_repair_projection(["<function=finish><parameter=result>ready</parameter></function>"])
    _obs, rewards, dones, infos = env.step(actions)
    assert dones == [True]
    assert infos[0]["won"] is True
    assert infos[0]["code_repair_full_score"] == 1.0
    assert rewards[0] >= 1.0
    env.close()


def test_code_repair_env_enforces_test_after_each_replacement_before_finish():
    from agent_system.environments.env_package.code_repair.envs import CodeRepairVectorEnv
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = normalize_code_repair_record(sample_code_repair_record(), "dataset", "train", 0, visible_test_count=2)
    env = CodeRepairVectorEnv(tasks=[task], env_num=1, group_n=1, seed=0, max_steps=64, history_length=5)
    env.reset()

    actions, _ = code_repair_projection(["<function=view_problem><parameter=section>all</parameter></function>"])
    obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert "Next required action: replace_solution" in obs[0]

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

    actions, _ = code_repair_projection([wrong_solution])
    obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert rewards[0] > 0.0
    assert "Next required action: run_tests" in obs[0]
    assert infos[0]["code_repair_next_required_action"] == "run_tests"
    assert infos[0]["code_repair_step_policy_violation_count"] == 0

    actions, _ = code_repair_projection([correct_solution])
    obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert rewards[0] <= -1.0
    assert "Please run visible tests before editing again." in obs[0]
    assert infos[0]["is_action_valid"] is True
    assert infos[0]["code_repair_edit_count"] == 2
    assert infos[0]["code_repair_step_policy_violation_count"] == 1
    assert infos[0]["code_repair_policy_violation_count"] == 1
    assert infos[0]["code_repair_order_events"] == ["replace_without_testing"]

    actions, _ = code_repair_projection(["<function=finish><parameter=result>ready</parameter></function>"])
    obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert rewards[0] < 0.0
    assert "Finish rejected" in obs[0]
    assert infos[0]["code_repair_next_required_action"] == "run_tests"
    assert infos[0]["code_repair_order_events"] == ["finish_before_run_tests"]

    actions, _ = code_repair_projection(["<function=run_tests><parameter=suite>visible</parameter></function>"])
    obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert infos[0]["code_repair_visible_score"] == 1.0
    assert infos[0]["code_repair_next_required_action"] == "finish"
    assert "Next required action: finish" in obs[0]

    actions, _ = code_repair_projection(["<function=finish><parameter=result>ready</parameter></function>"])
    _obs, rewards, dones, infos = env.step(actions)
    assert dones == [True]
    assert infos[0]["won"] is True
    assert rewards[0] >= 1.0
    env.close()


def test_code_repair_env_rewards_view_refresh_after_failed_visible_tests():
    from agent_system.environments.env_package.code_repair.envs import CodeRepairVectorEnv
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = normalize_code_repair_record(sample_code_repair_record(), "dataset", "train", 0, visible_test_count=2)
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

    actions, _ = code_repair_projection(["<function=view_problem><parameter=section>all</parameter></function>"])
    env.step(actions)
    actions, _ = code_repair_projection([wrong_solution])
    env.step(actions)

    actions, _ = code_repair_projection(["<function=run_tests><parameter=suite>visible</parameter></function>"])
    obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert rewards[0] > 0.0
    assert infos[0]["code_repair_visible_score"] < 1.0
    assert infos[0]["code_repair_next_required_action"] == "view_problem"
    assert "Next required action: view_problem" in obs[0]

    actions, _ = code_repair_projection([correct_solution])
    obs, rewards, dones, infos = env.step(actions)
    assert dones == [False]
    assert rewards[0] <= -1.0
    assert "Please refresh context with view_problem(section=all)" in obs[0]
    assert infos[0]["is_action_valid"] is True
    assert infos[0]["code_repair_order_events"] == ["replace_before_required_view"]

    env.close()

    env = CodeRepairVectorEnv(tasks=[task], env_num=1, group_n=1, seed=0, max_steps=64, history_length=5)
    env.reset()
    actions, _ = code_repair_projection(["<function=view_problem><parameter=section>all</parameter></function>"])
    env.step(actions)
    actions, _ = code_repair_projection([wrong_solution])
    env.step(actions)
    actions, _ = code_repair_projection(["<function=run_tests><parameter=suite>visible</parameter></function>"])
    env.step(actions)

    actions, _ = code_repair_projection(["<function=view_problem><parameter=section>all</parameter></function>"])
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
    from agent_system.environments.env_package.code_repair.envs import CodeRepairVectorEnv
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = normalize_code_repair_record(sample_code_repair_record(), "dataset", "train", 0, visible_test_count=2)
    env = CodeRepairVectorEnv(tasks=[task], env_num=1, group_n=1, seed=0, max_steps=64, history_length=5)
    cfg = OmegaConf.create({"env": {"history_length": 5}})
    manager = CodeRepairEnvironmentManager(env, code_repair_projection, cfg)

    manager.reset(kwargs=None)
    manager.step(["<function=view_problem><parameter=section>all</parameter></function>"])
    observations, _rewards, _dones, _infos = manager.step(
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

    assert "Next required action: run_tests" in observations["text"][0]
    assert "Do not call replace_solution again until visible tests have run" in observations["text"][0]
    assert "immediately previous action" in observations["text"][0]
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
    assert code_repair_metrics["code_repair_policy_violation_count"][0] == 1.0
    assert code_repair_metrics["code_repair_visible_score"][0] == 0.5


def test_code_repair_manager_limits_followup_history_to_five_steps():
    from agent_system.environments.env_manager import CodeRepairEnvironmentManager
    from agent_system.environments.env_package.code_repair.envs import CodeRepairVectorEnv
    from agent_system.environments.env_package.code_repair.projection import code_repair_projection
    from agent_system.environments.env_package.code_repair.tasks import normalize_code_repair_record

    task = normalize_code_repair_record(sample_code_repair_record(), "dataset", "train", 0, visible_test_count=2)
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
