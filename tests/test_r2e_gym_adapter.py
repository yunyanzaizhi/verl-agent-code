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


def test_r2e_prompt_documents_exact_argparse_aligned_xml_schema():
    from agent_system.environments.env_package.r2e_gym.prompts import R2E_ACTION_RULES, R2E_TOOL_SPEC

    assert "BEGIN FUNCTION #1: file_editor" in R2E_TOOL_SPEC
    assert "Notes for using the str_replace command" in R2E_TOOL_SPEC
    assert "cmd (string, required)" in R2E_TOOL_SPEC
    assert "Can be \"ctrl+c\"" in R2E_TOOL_SPEC
    assert "search_term (string, required)" in R2E_TOOL_SPEC
    assert "Currently allowed value: [submit]" in R2E_TOOL_SPEC

    assert "Each response must include both reasoning (as natural text) and function call" in R2E_ACTION_RULES
    assert "Your response must be exactly one XML tool call and nothing else" not in R2E_ACTION_RULES
    assert "<parameter=name>value</parameter>" not in R2E_ACTION_RULES


def test_r2e_initial_prompt_uses_r2e_code_repair_workflow():
    from agent_system.environments.env_package.r2e_gym.prompts import build_r2e_initial_prompt
    from agent_system.environments.env_package.r2e_gym.tasks import normalize_r2e_task_record

    task = normalize_r2e_task_record(sample_r2e_record(), "dataset", "train", 0)
    prompt = build_r2e_initial_prompt(task, "Fix the parser when input is split.")

    assert "I have uploaded a python code repository in the /testbed directory." in prompt
    assert "<github_issue>" in prompt
    assert "1. First, explore the codebase" in prompt
    assert "2. Assess whether you can reproduce the issue" in prompt
    assert "3. Analyze the root cause" in prompt
    assert "4. Implement your solution" in prompt
    assert "5. Verify your solution" in prompt
    assert "6. Run unit tests" in prompt
    assert "7. Test edge cases" in prompt
    assert "8. Refine if necessary" in prompt
    assert "9. Submit your solution" in prompt
    assert "DO NOT MODIFY any of the existing unit tests" in prompt


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
    assert "repository-level software engineering agent" in obs["text"][0]
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
