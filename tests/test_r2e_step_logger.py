import json

from omegaconf import OmegaConf


def test_episode_step_logger_writes_one_structured_file_per_step(tmp_path):
    from agent_system.multi_turn_rollout.episode_step_logger import EpisodeStepLogger

    logger = EpisodeStepLogger(
        root_dir=tmp_path,
        run_log_name="r2e_gym_lora_smoke_20260510_083811.log",
        enabled=True,
    )

    path = logger.write_step(
        train_step=3,
        episode=7,
        step=2,
        payload={
            "task": {"task_id": "task-7"},
            "model_input": {"raw_observation": "initial obs", "raw_prompt_text": "prompt"},
            "model_output": {"raw_response_text": "<function=finish></function>"},
            "env": {"reward": 1.0, "done": True},
        },
    )

    assert path.name == "r2e_gym_lora_smoke_20260510_083811.log-train_step_000003-episode_000007-step_000002.json"
    data = json.loads(path.read_text())
    assert data["train_step"] == 3
    assert data["episode"] == 7
    assert data["step"] == 2
    assert data["task"]["task_id"] == "task-7"
    assert data["model_input"]["raw_observation"] == "initial obs"
    assert data["model_output"]["raw_response_text"] == "<function=finish></function>"
    assert data["env"]["reward"] == 1.0


def test_episode_step_logger_from_config_can_be_disabled(tmp_path):
    from agent_system.multi_turn_rollout.episode_step_logger import EpisodeStepLogger

    config = OmegaConf.create(
        {
            "env": {
                "r2e_gym": {
                    "step_log_enabled": False,
                    "step_log_dir": str(tmp_path),
                    "run_log_name": "run.log",
                }
            }
        }
    )

    logger = EpisodeStepLogger.from_config(config)

    assert logger.enabled is False
    assert logger.write_step(train_step=1, episode=0, step=1, payload={}) is None
    assert list(tmp_path.iterdir()) == []
