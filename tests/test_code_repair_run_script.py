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
    assert 'env.max_steps="${MAX_STEPS}"' in script
    assert 'env.history_length="${HISTORY_LENGTH}"' in script
    assert 'trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"' in script
    assert 'env.code_repair.train_path="${CODE_REPAIR_TRAIN_PATH}"' in script
    assert 'env.code_repair.val_path="${CODE_REPAIR_VAL_PATH}"' in script
