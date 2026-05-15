from omegaconf import OmegaConf
from pathlib import Path


def test_prune_r2e_gym_logs_keeps_latest_five_runs(tmp_path):
    from agent_system.multi_turn_rollout.log_retention import prune_r2e_gym_logs

    log_dir = tmp_path / "logs" / "r2e_gym"
    episode_steps_dir = log_dir / "episode_steps"
    episode_steps_dir.mkdir(parents=True)

    for idx in range(1, 8):
        run_name = f"run_{idx:02d}.log"
        log_path = log_dir / run_name
        pid_path = log_dir / f"run_{idx:02d}.pid"
        step_dir = episode_steps_dir / run_name
        log_path.write_text(f"log {idx}")
        pid_path.write_text(str(1000 + idx))
        step_dir.mkdir()

    (log_dir / "latest.log").write_text("pointer")
    (log_dir / "latest.pid").write_text("pointer")

    keep_names = prune_r2e_gym_logs(log_dir=log_dir, episode_steps_dir=episode_steps_dir, keep=5)

    assert keep_names == [
        "run_07.log",
        "run_06.log",
        "run_05.log",
        "run_04.log",
        "run_03.log",
    ]
    assert sorted(path.name for path in log_dir.glob("run_*.log")) == [
        "run_03.log",
        "run_04.log",
        "run_05.log",
        "run_06.log",
        "run_07.log",
    ]
    assert sorted(path.name for path in log_dir.glob("run_*.pid")) == [
        "run_03.pid",
        "run_04.pid",
        "run_05.pid",
        "run_06.pid",
        "run_07.pid",
    ]
    assert (log_dir / "latest.log").exists()
    assert (log_dir / "latest.pid").exists()
    assert sorted(path.name for path in episode_steps_dir.iterdir()) == [
        "run_03.log",
        "run_04.log",
        "run_05.log",
        "run_06.log",
        "run_07.log",
    ]


def test_episode_step_logger_writes_one_readable_log_per_step(tmp_path):
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
            "model_output": {
                "raw_response_text": "<function=finish></function>",
                "response_ids": [1, 2, 3],
            },
            "actor": {
                "parsed_action": {"function_name": "finish", "parameters": {"command": "submit"}},
                "is_action_valid": True,
            },
            "env": {
                "raw_observation": "submitted",
                "reward": 1.0,
                "done": True,
                "info": {"won": True},
            },
        },
    )

    assert path.relative_to(tmp_path) == Path("train_step_000003") / "episode_000007" / "step_000002.log"
    text = path.read_text()
    assert "R2E EPISODE STEP" in text
    assert "train_step: 3" in text
    assert "task_id: task-7" in text
    assert "MODEL OUTPUT" in text
    assert "<function=finish></function>" in text
    assert "ACTOR" in text
    assert "function_name: finish" in text
    assert "ENVIRONMENT" in text
    assert "raw_observation:" in text
    assert "MODEL INPUT" not in text
    assert "response_ids" not in text


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
