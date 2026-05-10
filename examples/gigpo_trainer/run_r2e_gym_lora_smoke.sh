#!/usr/bin/env bash
set -euo pipefail
set -x

cd /home/caiting/verl-agent

PYTHON_BIN=${PYTHON_BIN:-/home/caiting/verl-agent-exp-copy-from-lab-server-20260505/.venv/bin/python}
R2E_GYM_ROOT=${R2E_GYM_ROOT:-/home/caiting/R2E-Gym}
MODEL_PATH=${MODEL_PATH:-/home/caiting/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-/home/caiting/verl-agent/checkpoints}

ENGINE=${ENGINE:-vllm}
N_GPUS=${N_GPUS:-1}
TP_SIZE=${TP_SIZE:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
GROUP_SIZE=${GROUP_SIZE:-1}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.5}

R2E_TRAIN_DATASET=${R2E_TRAIN_DATASET:-R2E-Gym/R2E-Gym-Subset}
R2E_TRAIN_SPLIT=${R2E_TRAIN_SPLIT:-train}
R2E_VAL_DATASET=${R2E_VAL_DATASET:-R2E-Gym/R2E-Gym-Lite}
R2E_VAL_SPLIT=${R2E_VAL_SPLIT:-dev_10pr_v1}

TRAIN_DATA_FILE=${TRAIN_DATA_FILE:-/home/caiting/data/verl-agent/text/train.parquet}
VAL_DATA_FILE=${VAL_DATA_FILE:-/home/caiting/data/verl-agent/text/test.parquet}

export PYTHONPATH="/home/caiting/verl-agent:${R2E_GYM_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/home/caiting/.cache/huggingface}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

if [[ ! -f "${TRAIN_DATA_FILE}" || ! -f "${VAL_DATA_FILE}" ]]; then
    "$PYTHON_BIN" -m examples.data_preprocess.prepare \
        --mode 'text' \
        --train_data_size "${TRAIN_BATCH_SIZE}" \
        --val_data_size "${VAL_BATCH_SIZE}"
fi

export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

"$PYTHON_BIN" -m recipe.hgpo.main_hgpo \
    algorithm.adv_estimator='hgpo' \
    algorithm.hgpo.weight_type='length' \
    algorithm.hgpo.mode='mean_norm' \
    algorithm.hgpo.length_weight_alpha=1.0 \
    algorithm.hgpo.base_group=False \
    data.train_files="${TRAIN_DATA_FILE}" \
    data.val_files="${VAL_DATA_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=9216 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_activation_offload=True \
    critic.model.use_remove_padding=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.forward_micro_batch_size_per_gpu=1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    env.env_name=r2e_gym \
    env.seed=0 \
    env.history_length=2 \
    env.max_steps=30 \
    env.rollout.n="${GROUP_SIZE}" \
    env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV_WORKER}" \
    env.r2e_gym.train_dataset="${R2E_TRAIN_DATASET}" \
    env.r2e_gym.train_split="${R2E_TRAIN_SPLIT}" \
    env.r2e_gym.val_dataset="${R2E_VAL_DATASET}" \
    env.r2e_gym.val_split="${R2E_VAL_SPLIT}" \
    env.r2e_gym.backend=docker \
    env.r2e_gym.step_timeout=90 \
    env.r2e_gym.reward_timeout=300 \
    env.r2e_gym.auto_submit_on_max_steps=True \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name='verl_agent_r2e_gym' \
    trainer.experiment_name='qwen2p5_coder_3b_lora_smoke' \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.default_local_dir="${CHECKPOINTS_DIR}/verl_agent_r2e_gym/qwen2p5_coder_3b_lora_smoke" \
    "$@"
