from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


def script_text(name: str) -> str:
    return (REPO_ROOT / "examples" / "gigpo_trainer" / name).read_text()


def shell_default(script: str, name: str) -> int:
    pattern = rf"^{re.escape(name)}=\$\{{{re.escape(name)}:-([0-9]+)\}}$"
    match = re.search(pattern, script, flags=re.MULTILINE)
    assert match is not None, f"{name} must have a numeric shell default"
    return int(match.group(1))


def test_r2e_grpo_v100_defaults_use_v100_safe_response_window():
    script = script_text("run_r2e_gym_grpo_v100.sh")

    assert shell_default(script, "MAX_RESPONSE_LENGTH") == 1536
    assert shell_default(script, "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU") == 8192
    assert shell_default(script, "ROLLOUT_MAX_MODEL_LEN") == 8192
    assert shell_default(script, "ROLLOUT_MAX_NUM_BATCHED_TOKENS") == 8192
    assert "ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}" in script
    assert "actor_rollout_ref.actor.fsdp_config.param_offload=True" in script


def test_r2e_lora_smoke_uses_same_response_window_knobs():
    script = script_text("run_r2e_gym_lora_smoke.sh")

    prompt_length = 8192
    response_length = shell_default(script, "MAX_RESPONSE_LENGTH")
    rollout_model_len = shell_default(script, "ROLLOUT_MAX_MODEL_LEN")
    rollout_batched_tokens = shell_default(script, "ROLLOUT_MAX_NUM_BATCHED_TOKENS")
    actor_token_len = shell_default(script, "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU")

    assert 'data.max_response_length="${MAX_RESPONSE_LENGTH}"' in script
    assert 'actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}"' in script
    assert 'actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"' in script
    assert response_length >= 2048
    assert rollout_model_len >= prompt_length + response_length
    assert rollout_batched_tokens >= prompt_length + response_length
    assert actor_token_len >= prompt_length + response_length
