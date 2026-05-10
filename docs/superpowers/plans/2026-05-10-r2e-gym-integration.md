# R2E-Gym Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an R2E-Gym code-repair environment to verl-agent and provide a one-step LoRA smoke training script.

**Architecture:** Create a new `agent_system.environments.env_package.r2e_gym` adapter package that calls R2E-Gym's existing `RepoEnv`, `Action`, and Docker reward runtime through `PYTHONPATH`. Add one `R2EGymEnvironmentManager` and one `make_envs` branch so the existing `TrajectoryCollector` can run R2E repair tasks without trainer changes.

**Tech Stack:** Python 3.10, pytest, Hugging Face datasets, R2E-Gym `RepoEnv`, Docker, verl-agent multi-turn PPO/HGPO environment interface, shell smoke script.

---

## File Structure

- Create `agent_system/environments/env_package/r2e_gym/__init__.py`: public exports for the new package.
- Create `agent_system/environments/env_package/r2e_gym/tasks.py`: dataset loading and task normalization.
- Create `agent_system/environments/env_package/r2e_gym/projection.py`: strict XML action parsing.
- Create `agent_system/environments/env_package/r2e_gym/prompts.py`: prompt and history formatting.
- Create `agent_system/environments/env_package/r2e_gym/envs.py`: vectorized R2E environment wrapper around R2E-Gym `RepoEnv`.
- Modify `agent_system/environments/env_manager.py`: add `R2EGymEnvironmentManager` and `make_envs` branch.
- Modify `verl/trainer/config/ppo_trainer.yaml`: add `env.r2e_gym` defaults.
- Create `examples/gigpo_trainer/run_r2e_gym_lora_smoke.sh`: requested short run.
- Create `tests/test_r2e_gym_adapter.py`: TDD coverage for normalization, projection, fake env stepping, and manager formatting.

## Task 1: Task Normalization And Projection

**Files:**
- Create: `agent_system/environments/env_package/r2e_gym/__init__.py`
- Create: `agent_system/environments/env_package/r2e_gym/tasks.py`
- Create: `agent_system/environments/env_package/r2e_gym/projection.py`
- Test: `tests/test_r2e_gym_adapter.py`

- [ ] **Step 1: Write failing tests for task normalization and projection**

Add tests that import `normalize_r2e_task_record`, `parse_r2e_gym_action`, and `r2e_gym_projection`. Test that a sample R2E row preserves `docker_image`, extracts the issue text between `[ISSUE]` tags, keeps `expected_output_json`, and gives a stable `task_id`. Test that one valid XML call returns an R2E `Action` with `function_name="execute_bash"`, and malformed output returns `valid=False`.

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=/home/caiting/verl-agent:/home/caiting/R2E-Gym/src /home/caiting/verl-agent-exp-copy-from-lab-server-20260505/.venv/bin/python -m pytest tests/test_r2e_gym_adapter.py -q`

Expected: import failure because `agent_system.environments.env_package.r2e_gym` does not exist.

- [ ] **Step 3: Implement normalization and projection**

Implement `R2EGymTask`, `normalize_r2e_task_record`, `load_r2e_tasks_from_config`, `ParsedR2EAction`, `parse_r2e_gym_action`, and `r2e_gym_projection`. Use R2E-Gym's `Action.from_string` for valid XML and return structured invalid dictionaries for bad responses.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same pytest command. Expected: task and projection tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add r2e gym task and action adapter`.

## Task 2: Vectorized R2E Environment

**Files:**
- Create: `agent_system/environments/env_package/r2e_gym/envs.py`
- Create: `agent_system/environments/env_package/r2e_gym/prompts.py`
- Modify: `agent_system/environments/env_package/r2e_gym/__init__.py`
- Test: `tests/test_r2e_gym_adapter.py`

- [ ] **Step 1: Write failing fake-runtime env tests**

Add tests using a fake `RepoEnv` class injected into `R2EGymVectorEnv`. Verify reset returns initial issue observations, step executes valid actions, malformed actions return invalid observations, finish computes terminal reward, and close closes runtimes.

- [ ] **Step 2: Run tests and verify RED**

Run the pytest command. Expected: import failure or missing `R2EGymVectorEnv`.

- [ ] **Step 3: Implement vector environment and prompts**

Implement `R2EGymVectorEnv` with `reset(kwargs=None)`, `step(actions)`, and `close()`. Use one `RepoEnv` per active slot, call `add_commands`, return zero reward for non-terminal actions, call `compute_reward` on finish or forced max-step submit, and expose `won`, `task_id`, `repo_name`, `docker_image`, and `is_action_valid` in info. Add prompt helpers for initial and history turns.

- [ ] **Step 4: Run tests and verify GREEN**

Run the pytest command. Expected: vector env tests pass without starting Docker.

- [ ] **Step 5: Commit**

Commit message: `feat: add r2e gym vector environment`.

## Task 3: Environment Manager And Config Wiring

**Files:**
- Modify: `agent_system/environments/env_manager.py`
- Modify: `verl/trainer/config/ppo_trainer.yaml`
- Test: `tests/test_r2e_gym_adapter.py`

- [ ] **Step 1: Write failing manager tests**

Add tests for `R2EGymEnvironmentManager` using a fake vector env. Verify `reset` returns formatted text, `anchor` is stable, `step` stores history and returns new text, and `success_evaluator` reads the final active `info["won"]`.

- [ ] **Step 2: Run tests and verify RED**

Run the pytest command. Expected: missing `R2EGymEnvironmentManager`.

- [ ] **Step 3: Implement manager and config branch**

Add `R2EGymEnvironmentManager` near other manager classes. Add `elif "r2e_gym" in config.env.env_name.lower()` in `make_envs`, calling `build_r2e_gym_envs` for train and validation. Add `env.r2e_gym` defaults to `ppo_trainer.yaml`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the pytest command. Expected: all adapter tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: wire r2e gym environment into verl agent`.

## Task 4: Smoke Script

**Files:**
- Create: `examples/gigpo_trainer/run_r2e_gym_lora_smoke.sh`

- [ ] **Step 1: Write the script**

Create a bash script that sets `PYTHON_BIN=/home/caiting/verl-agent-exp-copy-from-lab-server-20260505/.venv/bin/python`, exports `PYTHONPATH=/home/caiting/verl-agent:/home/caiting/R2E-Gym/src`, uses cached R2E datasets, and invokes `recipe.hgpo.main_hgpo` with `env.env_name=r2e_gym`, LoRA rank 32, alpha 64, lr 3e-6, fp16 rollout dtype, activation offload true, torch compile false, `env.max_steps=30`, and `trainer.total_training_steps=1`.

- [ ] **Step 2: Syntax-check the script**

Run: `bash -n examples/gigpo_trainer/run_r2e_gym_lora_smoke.sh`

Expected: exit code 0.

- [ ] **Step 3: Commit**

Commit message: `feat: add r2e gym smoke run script`.

## Task 5: Integration Verification

**Files:**
- No new source files unless verification exposes a bug.

- [ ] **Step 1: Import and dataset check**

Run: `PYTHONPATH=/home/caiting/verl-agent:/home/caiting/R2E-Gym/src /home/caiting/verl-agent-exp-copy-from-lab-server-20260505/.venv/bin/python - <<'PY'` with imports for `build_r2e_gym_envs`, `R2EGymEnvironmentManager`, and `load_dataset(..., split=...)` for both requested R2E datasets.

Expected: imports succeed and cached datasets load.

- [ ] **Step 2: Full adapter pytest**

Run: `PYTHONPATH=/home/caiting/verl-agent:/home/caiting/R2E-Gym/src /home/caiting/verl-agent-exp-copy-from-lab-server-20260505/.venv/bin/python -m pytest tests/test_r2e_gym_adapter.py -q`

Expected: all tests pass.

- [ ] **Step 3: Optional one-step launch**

Run the smoke script only if GPU/Ray resources are available and the user has not asked to skip launch. If it fails because of model path or cluster resource availability, report the exact blocker and leave the script ready.

- [ ] **Step 4: Final status**

Report commits, changed files, verification commands, and any runtime blockers.
