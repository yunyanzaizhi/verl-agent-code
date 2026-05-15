from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


def script_text() -> str:
    return (REPO_ROOT / "examples" / "gigpo_trainer" / "run_code_repair_lora_long_first_step_v100.sh").read_text()


def shell_default(script: str, name: str) -> int:
    pattern = rf"^{re.escape(name)}=\$\{{{re.escape(name)}:-([0-9]+)\}}$"
    match = re.search(pattern, script, flags=re.MULTILINE)
    assert match is not None, f"{name} must have a numeric shell default"
    return int(match.group(1))


def test_code_repair_long_first_step_script_uses_single_v100_lora_defaults():
    script = script_text()

    assert shell_default(script, "N_GPUS") == 1
    assert shell_default(script, "TP_SIZE") == 1
    assert shell_default(script, "TRAIN_BATCH_SIZE") == 1
    assert shell_default(script, "VAL_BATCH_SIZE") == 1
    assert shell_default(script, "GROUP_SIZE") == 2
    assert shell_default(script, "MAX_STEPS") == 64
    assert shell_default(script, "HISTORY_LENGTH") == 5
    assert shell_default(script, "TOTAL_TRAINING_STEPS") == 1
    assert shell_default(script, "TOTAL_EPOCHS") == 1

    assert shell_default(script, "MAX_PROMPT_LENGTH") == 4096
    assert shell_default(script, "MAX_RESPONSE_LENGTH") == 1024
    assert shell_default(script, "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU") == 5120
    assert shell_default(script, "ROLLOUT_MAX_MODEL_LEN") == 5120
    assert shell_default(script, "ROLLOUT_MAX_NUM_BATCHED_TOKENS") == 5120
    assert "ACTOR_USE_DYNAMIC_BSZ=${ACTOR_USE_DYNAMIC_BSZ:-True}" in script
    assert shell_default(script, "ROLLOUT_MAX_NUM_SEQS") == 64
    assert "ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}" in script

    assert "actor_rollout_ref.model.lora_rank=32" in script
    assert "actor_rollout_ref.model.lora_alpha=64" in script
    assert "actor_rollout_ref.model.enable_gradient_checkpointing=True" in script
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=True" in script
    assert "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" in script

    assert "env.env_name=code_repair" in script
    assert "env.max_steps=\"${MAX_STEPS}\"" in script
    assert "env.history_length=\"${HISTORY_LENGTH}\"" in script
    assert "trainer.total_training_steps=\"${TOTAL_TRAINING_STEPS}\"" in script
    assert "++env.code_repair.train_path=\"${CODE_REPAIR_TRAIN_PATH}\"" in script
    assert "++env.code_repair.val_path=\"${CODE_REPAIR_VAL_PATH}\"" in script
    assert "++env.code_repair.visible_test_count=\"${VISIBLE_TEST_COUNT}\"" in script
    assert "++env.code_repair.execution_timeout=\"${EXECUTION_TIMEOUT}\"" in script
    assert "++env.code_repair.allow_full_tests_in_loop=False" in script
    assert "++env.code_repair.auto_finish_on_max_steps=True" in script
    assert "++env.code_repair.step_log_enabled=True" in script
    assert "++env.code_repair.step_log_dir=\"${CODE_REPAIR_STEP_LOG_DIR}\"" in script
    assert "++env.code_repair.run_log_name=\"${CODE_REPAIR_RUN_LOG_NAME}\"" in script
    assert "\n    env.code_repair." not in script


def test_code_repair_long_first_step_script_derives_repo_root_from_script_location():
    script = script_text()

    assert "SCRIPT_DIR=$(cd -- \"$(dirname -- \"${BASH_SOURCE[0]}\")\" && pwd)" in script
    assert "REPO_ROOT=$(cd -- \"${SCRIPT_DIR}/../..\" && pwd)" in script
    assert "cd \"${REPO_ROOT}\"" in script
    assert "cd /home/caiting/verl-agent" not in script
    assert "export PYTHONPATH=\"${REPO_ROOT}:${PYTHONPATH:-}\"" in script


def test_code_repair_long_first_step_script_falls_back_from_pipe_stdout_to_repo_logs():
    script = script_text()

    assert "STDOUT_TARGET=$(readlink -f /proc/$$/fd/1 2>/dev/null || true)" in script
    assert "-f \"${STDOUT_TARGET}\"" in script
    assert "CODE_REPAIR_LOG_DIR=\"${REPO_ROOT}/logs/code_repair\"" in script
    assert "[[ -n \"${STDOUT_TARGET}\" && \"${STDOUT_TARGET}\" != /dev/* ]]" not in script
