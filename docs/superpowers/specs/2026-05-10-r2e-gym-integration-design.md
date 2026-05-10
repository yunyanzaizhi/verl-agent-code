# R2E-Gym Integration Design

## Goal

Add a clean R2E-Gym code-repair mode to verl-agent so existing multi-turn RL training can run repository-level bug-fixing tasks from R2E-Gym datasets and score them with R2E-Gym Docker test rewards.

## Source Boundary

Only two source trees are valid references for the implementation:

- /home/caiting/verl-agent
- /home/caiting/R2E-Gym

The virtual environment under /home/caiting/verl-agent-exp-copy-from-lab-server-20260505/.venv may be used only as an interpreter and dependency environment. Its source code must not be read, copied, or used as a design reference.

## Architecture

The integration will follow verl-agent's existing environment abstraction. A new self-contained environment package, agent_system/environments/env_package/r2e_gym, will adapt R2E-Gym tasks into the same reset and step interface used by Search, AlfWorld, Sokoban, WebShop, and AppWorld. agent_system/environments/env_manager.py will get a dedicated R2EGymEnvironmentManager that formats R2E prompts, maintains turn history, and reports success metrics.

R2E-Gym itself remains the owner of repository setup, Docker execution, and reward calculation. The adapter will import and call R2E-Gym's RepoEnv, EnvArgs, Action, and command tool files through PYTHONPATH=/home/caiting/R2E-Gym/src. It will not duplicate Docker runtime logic in verl-agent.

## Components

tasks.py will load R2E task records from Hugging Face datasets, using the cached copies already present on lab-server-1. It will normalize each row into a small task dataclass containing repo_name, docker_image, problem_statement, expected_output_json, relevant_files, and the original record.

projection.py will parse model responses in R2E's XML command format: <function=...><parameter=...>...</parameter></function>. It will return R2E Action objects or structured invalid-action records. The parser will be strict enough to mark missing or malformed tool calls invalid, while still allowing the environment to emit a useful observation.

envs.py will host a vectorized R2E environment wrapper. On reset it will choose task records, construct one RepoEnv per active slot, add R2E command files, and return the GitHub issue as the initial observation. On step it will execute one tool call per active environment, return tool output as the next observation, and compute terminal reward only when the model finishes or the rollout reaches the configured max steps.

prompts.py will contain the R2E code-repair instruction template and tool documentation. The prompt will tell the model to operate inside /testbed, use one XML tool call per response, edit non-test source files, validate before final submission when possible, and finish with the R2E finish tool.

env_manager.py will add R2EGymEnvironmentManager and a make_envs branch for env.env_name=r2e_gym. The manager will preserve the same observation contract used by TrajectoryCollector: text, image, and anchor. anchor will be the task issue or stable task id so GiGPO grouping remains deterministic.

## Data Flow

Training data still enters verl-agent through its normal RL dataset loader. The R2E environment receives env_kwargs from the repeated batch when available, but it can also sample directly from configured R2E datasets. For the requested run, the environment will be configured with:

- Train dataset: R2E-Gym/R2E-Gym-Subset, split train
- Validation dataset: R2E-Gym/R2E-Gym-Lite, split dev_10pr_v1

Each rollout step is:

1. TrajectoryCollector asks the policy for a response to the current text observation.
2. r2e_gym_projection parses the response into an R2E action and validity flag.
3. The vectorized R2E environment executes the action through RepoEnv.step.
4. Tool output becomes the next observation and zero step reward is emitted for non-terminal actions.
5. On finish or max steps, the environment calls R2E-Gym's reward calculation and returns terminal reward plus info["won"].

## Error Handling

Malformed model output returns an invalid-action observation and sets info["is_action_valid"] = False. Docker/tool exceptions are converted into observations instead of crashing the rollout when possible. Environment startup errors for a task mark that episode done with reward 0.0 and include the failure reason in info, so a bad Docker image does not kill the whole training process.

The adapter will close every active RepoEnv during reset replacement, explicit close, and destructor cleanup.

## Configuration

The default PPO/HGPO config will gain an env.r2e_gym section with dataset, split, backend, command file, timeout, and reward options. The short-run script will override these values for the requested smoke run.

The script will live at examples/gigpo_trainer/run_r2e_gym_lora_smoke.sh and will use:

- LoRA rank 32
- LoRA alpha 64
- Learning rate 3e-6
- float16 rollout precision
- activation offload enabled
- torch compile disabled
- env.max_steps=30
- one training step

The script will use /home/caiting/verl-agent-exp-copy-from-lab-server-20260505/.venv/bin/python as the Python interpreter and set PYTHONPATH=/home/caiting/verl-agent:/home/caiting/R2E-Gym/src.

## Testing

Unit tests will cover task normalization, XML projection, and a fake R2E vector environment that exercises reset, invalid action, normal action, finish, terminal reward, and close behavior without starting Docker.

Integration verification will use the allowed virtual environment to import verl-agent plus R2E-Gym, load cached dataset samples, and instantiate the adapter with a tiny batch. The final smoke script will be syntax-checked and, if resources allow, launched for the requested one-step run.
