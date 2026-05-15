import json

import numpy as np


def sample_r2e_record(**overrides):
    record = {
        "repo_name": "demo_repo",
        "docker_image": "example/demo:abc123",
        "commit_hash": "abc123",
        "problem_statement": "[ISSUE]\nFix the parser when input is split.\n[/ISSUE]",
        "expected_output_json": json.dumps({"test_parser": "PASSED"}),
        "relevant_files": ["demo/parser.py"],
        "modified_files": ["demo/parser.py", "tests/test_parser.py"],
    }
    record.update(overrides)
    return record


def test_normalize_r2e_task_record_extracts_issue_and_stable_id():
    from agent_system.environments.env_package.r2e_gym.tasks import normalize_r2e_task_record

    task = normalize_r2e_task_record(sample_r2e_record(), dataset_name="R2E-Gym/R2E-Gym-Subset", split="train", index=7)

    assert task.repo_name == "demo_repo"
    assert task.docker_image == "example/demo:abc123"
    assert task.problem_statement == "Fix the parser when input is split."
    assert task.expected_output_json == json.dumps({"test_parser": "PASSED"})
    assert task.relevant_files == ["demo/parser.py"]
    assert task.task_id == "R2E-Gym/R2E-Gym-Subset:train:7:abc123"
    assert task.raw_record["docker_image"] == "example/demo:abc123"


def test_r2e_gym_projection_parses_valid_xml_tool_call():
    from r2egym.agenthub.action import Action

    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection

    actions, valids = r2e_gym_projection([
        """
        <function=execute_bash>
          <parameter=cmd>python -m pytest -q tests/test_parser.py</parameter>
        </function>
        """
    ])

    assert valids == [1]
    assert isinstance(actions[0], Action)
    assert actions[0].function_name == "execute_bash"
    assert actions[0].parameters == {"cmd": "python -m pytest -q tests/test_parser.py"}


def test_r2e_gym_projection_marks_malformed_output_invalid():
    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection

    actions, valids = r2e_gym_projection(["I will inspect the repository first."])

    assert valids == [0]
    assert actions[0]["function_name"] == ""
    assert "format" in actions[0]["error"].lower()


def test_r2e_gym_projection_rejects_tool_calls_missing_argparse_parameters():
    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection

    actions, valids = r2e_gym_projection(
        [
            "<function=execute_bash><parameter=name>pwd</parameter></function>",
            "<function=execute_bash><parameter>cmd='pwd'</parameter></function>",
            "<function=file_editor><parameter=path>/testbed/a.py</parameter></function>",
            "<function=file_editor><parameter=command>view</parameter><parameter=file_path>/testbed/a.py</parameter></function>",
            "<function=search><parameter=query>needle</parameter></function>",
        ]
    )

    assert valids == [0, 0, 0, 0, 0]
    assert "cmd" in actions[0]["error"]
    assert "cmd" in actions[1]["error"]
    assert "command" in actions[2]["error"]
    assert "path" in actions[3]["error"]
    assert "search_term" in actions[4]["error"]


def test_r2e_gym_projection_uses_first_tool_call_when_multiple_blocks():
    from r2egym.agenthub.action import Action

    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection

    actions, valids = r2e_gym_projection(
        [
            """
            I should inspect first.
            <function=execute_bash>
              <parameter=cmd>pwd</parameter>
            </function>
            <function=finish>
              <parameter=command>submit</parameter>
            </function>
            """
        ]
    )

    assert valids == [1]
    assert isinstance(actions[0], Action)
    assert actions[0].function_name == "execute_bash"
    assert actions[0].parameters == {"cmd": "pwd"}
    assert "only the first tool call was executed" in actions[0].parse_warning
    assert "please output exactly one" in actions[0].parse_warning


def test_r2e_prompt_documents_exact_argparse_aligned_xml_schema():
    from agent_system.environments.env_package.r2e_gym.prompts import R2E_ACTION_RULES, R2E_TOOL_SPEC

    assert "BEGIN FUNCTION #1: file_editor" in R2E_TOOL_SPEC
    assert "Notes for using the str_replace command" in R2E_TOOL_SPEC
    assert "cmd (string, required)" in R2E_TOOL_SPEC
    assert "Can be \"ctrl+c\"" in R2E_TOOL_SPEC
    assert "search_term (string, required)" in R2E_TOOL_SPEC
    assert "Currently allowed value: [submit]" in R2E_TOOL_SPEC

    assert "Each response must include both reasoning (as natural text) and exactly one function call" in R2E_ACTION_RULES
    assert "Your response must be exactly one XML tool call and nothing else" not in R2E_ACTION_RULES
    assert "<parameter=name>value</parameter>" not in R2E_ACTION_RULES


def test_r2e_initial_prompt_uses_r2e_code_repair_workflow():
    from agent_system.environments.env_package.r2e_gym.prompts import build_r2e_initial_prompt
    from agent_system.environments.env_package.r2e_gym.tasks import normalize_r2e_task_record

    task = normalize_r2e_task_record(sample_r2e_record(), "dataset", "train", 0)
    prompt = build_r2e_initial_prompt(task, "Fix the parser when input is split.")

    assert "You are a software engineering agent inside a Docker environment at /testbed." in prompt
    assert "Your task: fix the github issue below by editing source files." in prompt
    assert "<github_issue>" in prompt
    assert "1. EXPLORE" in prompt
    assert "2. REPRODUCE" in prompt
    assert "3. ANALYZE" in prompt
    assert "4. IMPLEMENT" in prompt
    assert "5. VERIFY" in prompt
    assert "6. TEST" in prompt
    assert "7. SUBMIT" in prompt
    assert "Do NOT submit until you have actually used file_editor to edit files and verified the fix." in prompt


def test_r2e_followup_prompt_keeps_original_issue_context_with_truncation():
    from agent_system.environments.env_package.r2e_gym.prompts import build_r2e_followup_prompt
    from agent_system.environments.env_package.r2e_gym.tasks import normalize_r2e_task_record

    long_issue = "Fix the parser sentinel-start " + ("x" * 2100) + " sentinel-tail"
    task = normalize_r2e_task_record(
        sample_r2e_record(problem_statement=f"[ISSUE]\n{long_issue}\n[/ISSUE]"),
        "dataset",
        "train",
        0,
    )

    prompt = build_r2e_followup_prompt(
        task=task,
        current_observation="Tool output after inspecting files",
        history=["Step 1 action:\nexecute_bash(cmd=grep parser)\n\nStep 1 observation:\nmatched parser.py"],
        step_count=1,
    )

    assert "Original issue:" in prompt
    assert "<github_issue>" in prompt
    assert "Fix the parser sentinel-start" in prompt
    assert "sentinel-tail" not in prompt
    assert "[truncated]" in prompt
    assert "Tool output after inspecting files" in prompt
    assert "Step 1 action:" in prompt


def test_r2e_projection_accepts_reasoning_before_xml_tool_call():
    from r2egym.agenthub.action import Action

    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection

    actions, valids = r2e_gym_projection(
        [
            """
            I will first inspect the repository root to understand its structure.
            <function=execute_bash>
              <parameter=cmd>ls -la /testbed</parameter>
            </function>
            """
        ]
    )

    assert valids == [1]
    assert isinstance(actions[0], Action)
    assert actions[0].function_name == "execute_bash"
    assert actions[0].parameters == {"cmd": "ls -la /testbed"}


class FakeRepoEnv:
    created = []

    def __init__(self, args, logger=None, backend="docker", verbose=True, step_timeout=90, reward_timeout=300):
        self.args = args
        self.backend = backend
        self.step_timeout = step_timeout
        self.reward_timeout = reward_timeout
        self.commands_added = []
        self.actions = []
        self.closed = False
        FakeRepoEnv.created.append(self)

    def add_commands(self, cmd_files):
        self.commands_added.extend(str(path) for path in cmd_files)

    def get_task_instruction(self):
        return self.args.ds["problem_statement"].replace("[ISSUE]", "").replace("[/ISSUE]", "").strip()

    def step(self, action, timeout=None):
        self.actions.append(action)
        info = {"total_time": 0.01}
        done = action.function_name == "finish"
        observation = f"ran {action.function_name}"
        return observation, 0.0, done, info

    def compute_reward(self, timeout=None):
        return 1.0

    def close(self):
        self.closed = True


def test_r2e_vector_env_reset_step_finish_and_close_without_docker():
    from agent_system.environments.env_package.r2e_gym.envs import R2EGymVectorEnv
    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection
    from agent_system.environments.env_package.r2e_gym.tasks import normalize_r2e_task_record

    FakeRepoEnv.created = []
    task = normalize_r2e_task_record(sample_r2e_record(), "dataset", "train", 0)
    env = R2EGymVectorEnv(
        tasks=[task],
        env_num=1,
        group_n=1,
        seed=0,
        repo_env_cls=FakeRepoEnv,
        command_files=["tool_a.py", "tool_b.py"],
        backend="docker",
        step_timeout=3,
        reward_timeout=5,
    )

    obs, infos = env.reset()
    assert obs == ["Fix the parser when input is split."]
    assert infos[0]["task_id"] == "dataset:train:0:abc123"
    assert FakeRepoEnv.created[0].commands_added == ["tool_a.py", "tool_b.py"]

    parsed, _ = r2e_gym_projection(["<function=execute_bash><parameter=cmd>pwd</parameter></function>"])
    next_obs, rewards, dones, infos = env.step(parsed)
    assert next_obs == ["ran execute_bash"]
    assert rewards == [0.0]
    assert dones == [False]
    assert infos[0]["is_action_valid"] is True

    parsed, _ = r2e_gym_projection(["<function=finish><parameter=command>submit</parameter></function>"])
    next_obs, rewards, dones, infos = env.step(parsed)
    assert rewards == [1.0]
    assert dones == [True]
    assert infos[0]["won"] is True

    env.close()
    assert FakeRepoEnv.created[0].closed is True


def test_r2e_vector_env_invalid_action_returns_observation_without_runtime_step():
    from agent_system.environments.env_package.r2e_gym.envs import R2EGymVectorEnv
    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection
    from agent_system.environments.env_package.r2e_gym.tasks import normalize_r2e_task_record

    FakeRepoEnv.created = []
    task = normalize_r2e_task_record(sample_r2e_record(), "dataset", "train", 0)
    env = R2EGymVectorEnv(tasks=[task], env_num=1, group_n=1, repo_env_cls=FakeRepoEnv)
    env.reset()

    parsed, _ = r2e_gym_projection(["not xml"])
    next_obs, rewards, dones, infos = env.step(parsed)
    assert "invalid action" in next_obs[0].lower()
    assert rewards == [0.0]
    assert dones == [False]
    assert infos[0]["is_action_valid"] is False
    assert FakeRepoEnv.created[0].actions == []
    env.close()


def test_r2e_vector_env_rewards_str_replace_after_successful_explore():
    from agent_system.environments.env_package.r2e_gym.envs import R2EGymVectorEnv
    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection
    from agent_system.environments.env_package.r2e_gym.tasks import normalize_r2e_task_record

    FakeRepoEnv.created = []
    task = normalize_r2e_task_record(sample_r2e_record(), "dataset", "train", 0)
    env = R2EGymVectorEnv(tasks=[task], env_num=1, group_n=1, repo_env_cls=FakeRepoEnv)
    env.reset()

    parsed, _ = r2e_gym_projection(
        ["<function=file_editor><parameter=command>view</parameter><parameter=path>/testbed/demo/parser.py</parameter></function>"]
    )
    _obs, rewards, dones, infos = env.step(parsed)
    assert rewards == [0.0]
    assert dones == [False]
    assert infos[0]["r2e_shaping_reward"] == 0.0

    parsed, _ = r2e_gym_projection(
        [
            """
            <function=file_editor>
              <parameter=command>str_replace</parameter>
              <parameter=path>/testbed/demo/parser.py</parameter>
              <parameter=old_str>old parser code</parameter>
              <parameter=new_str>new parser code</parameter>
            </function>
            """
        ]
    )
    _obs, rewards, dones, infos = env.step(parsed)

    assert rewards == [0.05]
    assert dones == [False]
    assert infos[0]["r2e_shaping_reward"] == 0.05
    assert infos[0]["r2e_shaping_events"] == ["str_replace_after_successful_explore"]
    env.close()


def test_r2e_vector_env_escalates_repeated_failed_action_penalty():
    from agent_system.environments.env_package.r2e_gym.envs import R2EGymVectorEnv
    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection
    from agent_system.environments.env_package.r2e_gym.tasks import normalize_r2e_task_record

    FakeRepoEnv.created = []
    task = normalize_r2e_task_record(sample_r2e_record(), "dataset", "train", 0)
    env = R2EGymVectorEnv(tasks=[task], env_num=1, group_n=1, repo_env_cls=FakeRepoEnv)
    env.reset()

    parsed, _ = r2e_gym_projection(["not xml"])
    _obs, rewards, dones, infos = env.step(parsed)
    assert rewards == [0.0]
    assert dones == [False]
    assert infos[0]["is_action_valid"] is False
    assert infos[0]["r2e_failure_repeat_count"] == 1
    assert infos[0]["r2e_shaping_reward"] == 0.0

    parsed, _ = r2e_gym_projection(["not xml"])
    _obs, rewards, dones, infos = env.step(parsed)
    assert rewards == [-0.02]
    assert dones == [False]
    assert infos[0]["is_action_valid"] is False
    assert infos[0]["r2e_failure_repeat_count"] == 2
    assert infos[0]["r2e_shaping_events"] == ["repeated_failed_action"]
    assert FakeRepoEnv.created[0].actions == []
    env.close()


def test_r2e_vector_env_penalizes_repeated_view_without_invalidating_action():
    from agent_system.environments.env_package.r2e_gym.envs import R2EGymVectorEnv
    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection
    from agent_system.environments.env_package.r2e_gym.tasks import normalize_r2e_task_record

    FakeRepoEnv.created = []
    task = normalize_r2e_task_record(sample_r2e_record(), "dataset", "train", 0)
    env = R2EGymVectorEnv(tasks=[task], env_num=1, group_n=1, repo_env_cls=FakeRepoEnv)
    env.reset()

    view_call = """
    <function=file_editor>
      <parameter=command>view</parameter>
      <parameter=path>/testbed/demo/parser.py</parameter>
    </function>
    """
    parsed, _ = r2e_gym_projection([view_call])
    _obs, rewards, _dones, infos = env.step(parsed)
    assert rewards == [0.0]
    assert infos[0]["is_action_valid"] is True
    assert infos[0]["r2e_shaping_reward"] == 0.0

    parsed, _ = r2e_gym_projection([view_call])
    _obs, rewards, dones, infos = env.step(parsed)

    assert rewards == [-0.2]
    assert dones == [False]
    assert infos[0]["is_action_valid"] is True
    assert infos[0]["r2e_shaping_events"] == ["repeated_no_progress_view"]
    assert infos[0]["r2e_repeated_view_repeat_count"] == 2
    assert FakeRepoEnv.created[0].actions[0].function_name == "file_editor"
    assert FakeRepoEnv.created[0].actions[1].function_name == "file_editor"
    env.close()


def test_r2e_environment_manager_formats_history_and_success():
    from omegaconf import OmegaConf

    from agent_system.environments.env_manager import R2EGymEnvironmentManager
    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection

    class FakeVectorEnv:
        def reset(self, kwargs=None):
            return ["Initial issue"], [{"task_id": "task-1", "repo_name": "demo_repo", "docker_image": "img"}]

        def step(self, actions):
            return ["Tool output"], [0.0], [False], [{"won": False, "is_action_valid": True, "task_id": "task-1"}]

        def close(self):
            pass

    cfg = OmegaConf.create({"env": {"history_length": 2}})
    manager = R2EGymEnvironmentManager(FakeVectorEnv(), r2e_gym_projection, cfg)
    obs, infos = manager.reset(kwargs=None)
    assert "software engineering agent inside a Docker environment at /testbed" in obs["text"][0]
    assert "Initial issue" in obs["text"][0]
    assert obs["anchor"].tolist() == ["task-1"]

    next_obs, rewards, dones, infos = manager.step(["<function=execute_bash><parameter=cmd>pwd</parameter></function>"])
    assert "Tool output" in next_obs["text"][0]
    assert "execute_bash" in next_obs["text"][0]
    assert infos[0]["raw_model_output"] == "<function=execute_bash><parameter=cmd>pwd</parameter></function>"
    assert infos[0]["r2e_action"] == {"function_name": "execute_bash", "parameters": {"cmd": "pwd"}}
    assert infos[0]["r2e_raw_observation"] == "Tool output"
    assert np.array_equal(rewards, np.array([0.0]))
    assert np.array_equal(dones, np.array([False]))

    success = manager.success_evaluator(
        total_infos=[[{"won": False}, {"won": True}]],
        total_batch_list=[[{"active_masks": True}, {"active_masks": True}]],
    )
    assert np.array_equal(success["success_rate"], np.array([1.0]))


def test_r2e_environment_manager_reports_multiple_tool_call_warning_in_feedback():
    from omegaconf import OmegaConf

    from agent_system.environments.env_manager import R2EGymEnvironmentManager
    from agent_system.environments.env_package.r2e_gym.projection import r2e_gym_projection

    class FakeVectorEnv:
        def reset(self, kwargs=None):
            return ["Initial issue"], [{"task_id": "task-1", "repo_name": "demo_repo", "docker_image": "img"}]

        def step(self, actions):
            assert actions[0].function_name == "execute_bash"
            return ["Tool output"], [0.0], [False], [{"won": False, "is_action_valid": True, "task_id": "task-1"}]

        def close(self):
            pass

    manager = R2EGymEnvironmentManager(FakeVectorEnv(), r2e_gym_projection, OmegaConf.create({"env": {"history_length": 2}}))
    manager.reset(kwargs=None)

    next_obs, _rewards, _dones, infos = manager.step(
        [
            """
            I should inspect first.
            <function=execute_bash><parameter=cmd>pwd</parameter></function>
            <function=finish><parameter=command>submit</parameter></function>
            """
        ]
    )

    assert infos[0]["is_action_valid"] is True
    assert "only the first tool call was executed" in infos[0]["r2e_action"]["parse_warning"]
    assert "only the first tool call was executed" in next_obs["text"][0]
    assert "Tool output" in next_obs["text"][0]
