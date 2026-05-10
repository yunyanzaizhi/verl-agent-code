#!/usr/bin/env bash
set -euo pipefail
set -x

cd /home/caiting/verl-agent

PYTHON_BIN=${PYTHON_BIN:-/home/caiting/verl-agent-exp-copy-from-lab-server-20260505/.venv/bin/python}
R2E_GYM_ROOT=${R2E_GYM_ROOT:-/home/caiting/R2E-Gym}
MODEL_PATH=${MODEL_PATH:-/home/caiting/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-/home/caiting/verl-agent/checkpoints}

ENGINE=${ENGINE:-vllm}
N_GPUS=${N_GPUS:-2}
TP_SIZE=${TP_SIZE:-${N_GPUS}}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-3}
GROUP_SIZE=${GROUP_SIZE:-2}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.25}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}
AMP_DTYPE=${AMP_DTYPE:-float16}

MAX_STEPS=${MAX_STEPS:-30}
HISTORY_LENGTH=${HISTORY_LENGTH:-100}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1536}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-2}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
ROLLOUT_ENABLE_PREFIX_CACHING=${ROLLOUT_ENABLE_PREFIX_CACHING:-False}
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-0.35}
TRAIN_TOP_P=${TRAIN_TOP_P:-0.9}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.2}
VAL_TOP_P=${VAL_TOP_P:-0.9}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
PREPARE_DATA=${PREPARE_DATA:-False}

R2E_TRAIN_DATASET=${R2E_TRAIN_DATASET:-R2E-Gym/R2E-Gym-Subset}
R2E_TRAIN_SPLIT=${R2E_TRAIN_SPLIT:-train}
R2E_VAL_DATASET=${R2E_VAL_DATASET:-R2E-Gym/R2E-Gym-Lite}
R2E_VAL_SPLIT=${R2E_VAL_SPLIT:-dev_10pr_v1}

TRAIN_DATA_FILE=${TRAIN_DATA_FILE:-/home/caiting/data/verl-agent/text/train.parquet}
VAL_DATA_FILE=${VAL_DATA_FILE:-/home/caiting/data/verl-agent/text/test.parquet}

export PYTHONPATH="/home/caiting/verl-agent:${R2E_GYM_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/home/caiting/.cache/huggingface}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

STDOUT_TARGET=$(readlink -f /proc/$$/fd/1 2>/dev/null || true)
if [[ -z "${R2E_RUN_LOG_NAME:-}" ]]; then
    if [[ -n "${STDOUT_TARGET}" && "${STDOUT_TARGET}" != /dev/* ]]; then
        R2E_RUN_LOG_NAME=$(basename "${STDOUT_TARGET}")
    else
        R2E_RUN_LOG_NAME="r2e_gym_grpo_v100_manual_$(date +%Y%m%d_%H%M%S).log"
    fi
fi
if [[ -z "${R2E_STEP_LOG_DIR:-}" ]]; then
    if [[ -n "${STDOUT_TARGET}" && "${STDOUT_TARGET}" != /dev/* ]]; then
        R2E_STEP_LOG_DIR="$(dirname "${STDOUT_TARGET}")/episode_steps/${R2E_RUN_LOG_NAME}"
    else
        R2E_STEP_LOG_DIR="/home/caiting/verl-agent/logs/r2e_gym/episode_steps/${R2E_RUN_LOG_NAME}"
    fi
fi
mkdir -p "${R2E_STEP_LOG_DIR}"
export R2E_RUN_LOG_NAME
export R2E_STEP_LOG_DIR
export R2E_STEP_LOG_ENABLED=${R2E_STEP_LOG_ENABLED:-1}
echo "R2E_STEP_LOG_DIR=${R2E_STEP_LOG_DIR}"

if [[ "${PREPARE_DATA}" == "True" || "${PREPARE_DATA}" == "true" || ! -f "${TRAIN_DATA_FILE}" || ! -f "${VAL_DATA_FILE}" ]]; then
    "${PYTHON_BIN}" -m examples.data_preprocess.prepare \
        --mode 'text' \
        --train_data_size "${TRAIN_BATCH_SIZE}" \
        --val_data_size "${VAL_BATCH_SIZE}"
fi

"$PYTHON_BIN" -m recipe.hgpo.main_hgpo \
    algorithm.adv_estimator='grpo' \
    data.train_files="${TRAIN_DATA_FILE}" \
    data.val_files="${VAL_DATA_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length=8192 \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.model.attn_implementation="${ATTN_IMPLEMENTATION}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.05 \
    actor_rollout_ref.actor.amp_dtype="${AMP_DTYPE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=float16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=fp16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.temperature="${TRAIN_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TRAIN_TOP_P}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    +actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.amp_dtype="${AMP_DTYPE}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_activation_offload=True \
    critic.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    critic.model.attn_implementation="${ATTN_IMPLEMENTATION}" \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.forward_micro_batch_size_per_gpu=1 \
    critic.amp_dtype="${AMP_DTYPE}" \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=1.0 \
    env.env_name=r2e_gym \
    env.seed=0 \
    env.history_length="${HISTORY_LENGTH}" \
    env.max_steps="${MAX_STEPS}" \
    env.rollout.n="${GROUP_SIZE}" \
    env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV_WORKER}" \
    env.r2e_gym.train_dataset="${R2E_TRAIN_DATASET}" \
    env.r2e_gym.train_split="${R2E_TRAIN_SPLIT}" \
    env.r2e_gym.val_dataset="${R2E_VAL_DATASET}" \
    env.r2e_gym.val_split="${R2E_VAL_SPLIT}" \
    env.r2e_gym.backend=docker \
    env.r2e_gym.step_timeout=60 \
    env.r2e_gym.reward_timeout=300 \
    env.r2e_gym.auto_submit_on_max_steps=True \
    env.r2e_gym.step_log_enabled=True \
    env.r2e_gym.step_log_dir="${R2E_STEP_LOG_DIR}" \
    env.r2e_gym.run_log_name="${R2E_RUN_LOG_NAME}" \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name='verl_agent_r2e_gym' \
    trainer.experiment_name='qwen2p5_coder_3b_grpo_v100' \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.default_local_dir="${CHECKPOINTS_DIR}/verl_agent_r2e_gym/qwen2p5_coder_3b_grpo_v100" \
    "$@"
