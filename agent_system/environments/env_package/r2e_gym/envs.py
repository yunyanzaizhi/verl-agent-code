from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np

from .tasks import R2EGymTask, load_r2e_tasks_from_config, normalize_r2e_task_record


@dataclass(frozen=True)
class R2EEnvArgs:
    ds: Dict[str, Any]


DEFAULT_R2E_COMMAND_FILES = [
    "/home/caiting/R2E-Gym/src/r2egym/agenthub/tools/r2egym/file_editor.py",
    "/home/caiting/R2E-Gym/src/r2egym/agenthub/tools/search.py",
    "/home/caiting/R2E-Gym/src/r2egym/agenthub/tools/r2egym/execute_bash.py",
    "/home/caiting/R2E-Gym/src/r2egym/agenthub/tools/finish.py",
]


class R2EGymVectorEnv(gym.Env):
    def __init__(
        self,
        tasks: Sequence[R2EGymTask],
        env_num: int = 1,
        group_n: int = 1,
        seed: int = 0,
        is_train: bool = True,
        repo_env_cls=None,
        command_files: Optional[Sequence[str]] = None,
        backend: str = "docker",
        step_timeout: int = 90,
        reward_timeout: int = 300,
        max_steps: Optional[int] = None,
        auto_submit_on_max_steps: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("R2EGymVectorEnv requires at least one task.")
        self.tasks = list(tasks)
        self.env_num = int(env_num)
        self.group_n = int(group_n)
        self.batch_size = self.env_num * self.group_n
        self.is_train = is_train
        self.repo_env_cls = repo_env_cls
        self.command_files = [str(path) for path in (command_files or DEFAULT_R2E_COMMAND_FILES)]
        self.backend = backend
        self.step_timeout = int(step_timeout)
        self.reward_timeout = int(reward_timeout)
        self.max_steps = max_steps
        self.auto_submit_on_max_steps = bool(auto_submit_on_max_steps)
        self.verbose = bool(verbose)
        self._rng = np.random.RandomState(seed)
        self._cursor = 0
        self.current_tasks: List[Optional[R2EGymTask]] = []
        self.runtimes: List[Any] = []
        self.dones: List[bool] = []
        self.steps: List[int] = []

    def _repo_env_cls(self):
        if self.repo_env_cls is not None:
            return self.repo_env_cls
        from r2egym.agenthub.environment.env import RepoEnv
        return RepoEnv

    @staticmethod
    def _env_args(task: R2EGymTask):
        return R2EEnvArgs(ds=task.raw_record)

    def _sample_base_tasks(self, count: int) -> List[R2EGymTask]:
        if self.is_train:
            replace = len(self.tasks) < count
            indices = self._rng.choice(len(self.tasks), size=count, replace=replace)
            return [self.tasks[int(i)] for i in indices]
        selected = []
        for _ in range(count):
            selected.append(self.tasks[self._cursor % len(self.tasks)])
            self._cursor += 1
        return selected

    def _tasks_from_kwargs(self, kwargs) -> Optional[List[R2EGymTask]]:
        if not kwargs:
            return None
        if not isinstance(kwargs, list):
            return None
        if not kwargs or not isinstance(kwargs[0], dict) or "docker_image" not in kwargs[0]:
            return None
        tasks = []
        for idx, row in enumerate(kwargs[: self.batch_size]):
            tasks.append(normalize_r2e_task_record(row, "env_kwargs", "runtime", idx))
        return tasks

    def _select_tasks(self, kwargs=None) -> List[R2EGymTask]:
        kwarg_tasks = self._tasks_from_kwargs(kwargs)
        if kwarg_tasks is not None:
            if len(kwarg_tasks) < self.batch_size:
                kwarg_tasks = kwarg_tasks + self._sample_base_tasks(self.batch_size - len(kwarg_tasks))
            return kwarg_tasks[: self.batch_size]
        base_tasks = self._sample_base_tasks(self.env_num)
        repeated: List[R2EGymTask] = []
        for task in base_tasks:
            repeated.extend([task] * self.group_n)
        return repeated[: self.batch_size]

    def _base_info(self, task: Optional[R2EGymTask], is_action_valid: bool = True) -> Dict[str, Any]:
        if task is None:
            return {"task_id": None, "repo_name": None, "docker_image": None, "won": False, "is_action_valid": is_action_valid}
        return {
            "task_id": task.task_id,
            "repo_name": task.repo_name,
            "docker_image": task.docker_image,
            "won": False,
            "is_action_valid": is_action_valid,
        }

    def reset(self, kwargs=None) -> Tuple[List[str], List[Dict[str, Any]]]:
        self.close()
        self.current_tasks = self._select_tasks(kwargs)
        self.runtimes = []
        self.dones = [False] * len(self.current_tasks)
        self.steps = [0] * len(self.current_tasks)
        observations: List[str] = []
        infos: List[Dict[str, Any]] = []
        repo_env_cls = self._repo_env_cls()
        for task in self.current_tasks:
            try:
                runtime = repo_env_cls(
                    self._env_args(task),
                    backend=self.backend,
                    verbose=self.verbose,
                    step_timeout=self.step_timeout,
                    reward_timeout=self.reward_timeout,
                )
                runtime.add_commands(self.command_files)
                observation = runtime.get_task_instruction()
                info = self._base_info(task)
            except Exception as exc:
                runtime = None
                observation = f"R2E-Gym environment setup failed: {exc}"
                info = self._base_info(task, is_action_valid=False)
                info.update({"setup_failed": True, "error": str(exc), "won": False})
            self.runtimes.append(runtime)
            observations.append(str(observation))
            infos.append(info)
        return observations, infos

    def _invalid_step(self, idx: int, action: Dict[str, Any]):
        task = self.current_tasks[idx] if idx < len(self.current_tasks) else None
        error = action.get("error", "Invalid action format.") if isinstance(action, dict) else "Invalid action format."
        info = self._base_info(task, is_action_valid=False)
        info.update({"error": error, "won": False})
        return f"Invalid action: {error}", 0.0, False, info

    def _terminal_reward(self, idx: int, runtime) -> float:
        if runtime is None:
            return 0.0
        try:
            reward = runtime.compute_reward(timeout=self.reward_timeout)
        except TypeError:
            reward = runtime.compute_reward()
        except Exception:
            return 0.0
        if isinstance(reward, tuple):
            reward = reward[0]
        return float(reward)

    def step(self, actions: List[Any]):
        if len(actions) > len(self.current_tasks):
            raise ValueError(f"Got {len(actions)} actions for {len(self.current_tasks)} active R2E environments.")
        observations: List[str] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []
        for idx, action in enumerate(actions):
            task = self.current_tasks[idx]
            runtime = self.runtimes[idx]
            if self.dones[idx]:
                info = self._base_info(task)
                info["won"] = False
                observations.append("Episode already completed.")
                rewards.append(0.0)
                dones.append(True)
                infos.append(info)
                continue
            if isinstance(action, dict):
                observation, reward, done, info = self._invalid_step(idx, action)
            elif runtime is None:
                observation = "R2E-Gym runtime is unavailable for this task."
                reward = 0.0
                done = True
                info = self._base_info(task, is_action_valid=False)
            else:
                try:
                    observation_obj, _step_reward, done, step_info = runtime.step(action, timeout=self.step_timeout)
                except TypeError:
                    observation_obj, _step_reward, done, step_info = runtime.step(action)
                observation = str(observation_obj)
                info = self._base_info(task, is_action_valid=True)
                info.update(step_info or {})
                reward = 0.0
                self.steps[idx] += 1
                forced_submit = (
                    self.max_steps is not None
                    and self.steps[idx] >= int(self.max_steps)
                    and self.auto_submit_on_max_steps
                    and not done
                )
                if done or forced_submit:
                    reward = self._terminal_reward(idx, runtime)
                    done = True
                    info["won"] = reward >= 1.0
                    info["terminal_r2e_reward"] = reward
            self.dones[idx] = bool(done)
            observations.append(observation)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)
        return observations, rewards, dones, infos

    def close(self) -> None:
        runtimes = getattr(self, "runtimes", [])
        for runtime in runtimes:
            if runtime is None:
                continue
            try:
                runtime.close()
            except Exception:
                pass
        self.runtimes = []

    def __del__(self):
        self.close()


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def build_r2e_gym_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config=None,
):
    r2e_config = _cfg_get(env_config, "r2e_gym", env_config)
    tasks = load_r2e_tasks_from_config(r2e_config, is_train=is_train)
    command_files = _cfg_get(r2e_config, "command_files", DEFAULT_R2E_COMMAND_FILES)
    return R2EGymVectorEnv(
        tasks=tasks,
        env_num=env_num,
        group_n=group_n,
        seed=seed,
        is_train=is_train,
        command_files=command_files,
        backend=str(_cfg_get(r2e_config, "backend", "docker")),
        step_timeout=int(_cfg_get(r2e_config, "step_timeout", 90)),
        reward_timeout=int(_cfg_get(r2e_config, "reward_timeout", 300)),
        max_steps=_cfg_get(env_config, "max_steps", None),
        auto_submit_on_max_steps=bool(_cfg_get(r2e_config, "auto_submit_on_max_steps", True)),
        verbose=bool(_cfg_get(r2e_config, "verbose", False)),
    )
