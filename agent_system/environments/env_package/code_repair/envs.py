from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gym
import numpy as np

from .executor import TestRunResult, run_code_repair_tests
from .projection import CodeRepairAction
from .protocol import (
    ProtocolSnapshot,
    ProtocolState,
    allowed_actions,
    apply_action_transition,
    apply_visible_test_result,
    check_protocol_action,
    expected_action,
    initial_protocol_snapshot,
)
from .tasks import CodeRepairTask, load_code_repair_tasks_from_config, normalize_code_repair_record


INVALID_ACTION_PENALTY = 0.05
SYNTAX_ERROR_PENALTY = 0.05
REPEATED_NO_PROGRESS_PENALTY = 0.02
TEST_IMPROVEMENT_REWARD = 0.5
TERMINAL_SUCCESS_REWARD = 1.0
TERMINAL_SCORE_REWARD = 1.0
ORDER_VIOLATION_PENALTY = 1.0
PREMATURE_FINISH_PENALTY = 0.2
UNTESTED_MAX_STEP_PENALTY = 0.5
ORDER_CORRECT_REWARDS = {
    "view_problem": 0.05,
    "replace_solution": 0.10,
    "run_tests": 0.20,
    "view_after_failed_tests": 0.20,
}
ORDER_CORRECT_STREAK_BONUS = 0.02
ORDER_CORRECT_STREAK_BONUS_CAP = 0.10
ORDER_VIOLATION_REPEAT_PENALTY = 0.20
ORDER_VIOLATION_REPEAT_PENALTY_CAP = 1.00


@dataclass
class CodeRepairEpisodeState:
    task: CodeRepairTask
    current_code: str
    done: bool = False
    won: bool = False
    steps: int = 0
    edit_count: int = 0
    test_count: int = 0
    invalid_action_count: int = 0
    best_visible_score: float = 0.0
    best_full_score: float = 0.0
    last_observation: str = ""
    last_code_hash: int = 0
    no_progress_count: int = 0
    current_code_tested: bool = False
    visible_passed_current_code: bool = False
    order_correct_count: int = 0
    order_violation_streak: int = 0
    order_shaping_reward_sum: float = 0.0
    policy_violation_count: int = 0
    protocol_snapshot: ProtocolSnapshot = field(default_factory=initial_protocol_snapshot)
    history: List[Dict[str, str]] = field(default_factory=list)


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    if hasattr(config, "get"):
        try:
            return config.get(name, default)
        except Exception:
            pass
    return getattr(config, name, default)


class CodeRepairVectorEnv(gym.Env):
    def __init__(
        self,
        tasks: Sequence[CodeRepairTask],
        env_num: int = 1,
        group_n: int = 1,
        seed: int = 0,
        is_train: bool = True,
        max_steps: int = 64,
        history_length: int = 5,
        execution_timeout: int = 8,
        max_code_chars: int = 20000,
        allow_full_tests_in_loop: bool = False,
        auto_finish_on_max_steps: bool = True,
        invalid_action_penalty: float = INVALID_ACTION_PENALTY,
        syntax_error_penalty: float = SYNTAX_ERROR_PENALTY,
        repeated_no_progress_penalty: float = REPEATED_NO_PROGRESS_PENALTY,
        test_improvement_reward: float = TEST_IMPROVEMENT_REWARD,
        terminal_success_reward: float = TERMINAL_SUCCESS_REWARD,
        terminal_score_reward: float = TERMINAL_SCORE_REWARD,
        order_violation_penalty: float = ORDER_VIOLATION_PENALTY,
        premature_finish_penalty: float = PREMATURE_FINISH_PENALTY,
        untested_max_step_penalty: float = UNTESTED_MAX_STEP_PENALTY,
    ) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("CodeRepairVectorEnv requires at least one task.")
        self.tasks = list(tasks)
        self.env_num = int(env_num)
        self.group_n = int(group_n)
        self.batch_size = self.env_num * self.group_n
        self.is_train = bool(is_train)
        self.max_steps = int(max_steps)
        self.history_length = int(history_length)
        self.execution_timeout = int(execution_timeout)
        self.max_code_chars = int(max_code_chars)
        self.allow_full_tests_in_loop = bool(allow_full_tests_in_loop)
        self.auto_finish_on_max_steps = bool(auto_finish_on_max_steps)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.syntax_error_penalty = float(syntax_error_penalty)
        self.repeated_no_progress_penalty = float(repeated_no_progress_penalty)
        self.test_improvement_reward = float(test_improvement_reward)
        self.terminal_success_reward = float(terminal_success_reward)
        self.terminal_score_reward = float(terminal_score_reward)
        self.order_violation_penalty = float(order_violation_penalty)
        self.premature_finish_penalty = float(premature_finish_penalty)
        self.untested_max_step_penalty = float(untested_max_step_penalty)
        self._rng = np.random.RandomState(seed)
        self._cursor = 0
        self.current_tasks: List[CodeRepairTask] = []
        self.states: List[CodeRepairEpisodeState] = []

    def _sample_base_tasks(self, count: int) -> List[CodeRepairTask]:
        if self.is_train:
            replace = len(self.tasks) < count
            indices = self._rng.choice(len(self.tasks), size=count, replace=replace)
            return [self.tasks[int(idx)] for idx in indices]
        selected = []
        for _ in range(count):
            selected.append(self.tasks[self._cursor % len(self.tasks)])
            self._cursor += 1
        return selected

    def _tasks_from_kwargs(self, kwargs: Any) -> Optional[List[CodeRepairTask]]:
        if not kwargs or not isinstance(kwargs, list) or not kwargs:
            return None
        if not isinstance(kwargs[0], dict) or "problem_description" not in kwargs[0]:
            return None
        tasks = [
            normalize_code_repair_record(row, dataset_name="env_kwargs", split="runtime", index=idx)
            for idx, row in enumerate(kwargs[: self.batch_size])
        ]
        if len(tasks) < self.batch_size:
            tasks.extend(self._sample_base_tasks(self.batch_size - len(tasks)))
        return tasks[: self.batch_size]

    def _select_tasks(self, kwargs: Any = None) -> List[CodeRepairTask]:
        kwarg_tasks = self._tasks_from_kwargs(kwargs)
        if kwarg_tasks is not None:
            return kwarg_tasks
        base_tasks = self._sample_base_tasks(self.env_num)
        repeated: List[CodeRepairTask] = []
        for task in base_tasks:
            repeated.extend([task] * self.group_n)
        return repeated[: self.batch_size]

    @staticmethod
    def _snapshot_state_name(snapshot: ProtocolSnapshot) -> str:
        return snapshot.state.value

    @staticmethod
    def _next_required_action(state: CodeRepairEpisodeState) -> str:
        if state.done:
            return "done"
        return expected_action(state.protocol_snapshot)

    @staticmethod
    def _allowed_actions(state: CodeRepairEpisodeState) -> List[str]:
        if state.done:
            return []
        return sorted(allowed_actions(state.protocol_snapshot))

    def _base_info(self, state: CodeRepairEpisodeState, is_action_valid: bool = True) -> Dict[str, Any]:
        task = state.task
        current_hash = hash(state.current_code)
        protocol_state = self._snapshot_state_name(state.protocol_snapshot)
        expected = self._next_required_action(state)
        allowed = self._allowed_actions(state)
        return {
            "task_id": task.task_id,
            "dataset_task_id": task.dataset_task_id,
            "question_id": task.question_id,
            "difficulty": task.difficulty,
            "won": bool(state.won),
            "is_action_valid": bool(is_action_valid),
            "code_repair_step": int(state.steps),
            "code_repair_edit_count": int(state.edit_count),
            "code_repair_test_count": int(state.test_count),
            "code_repair_invalid_action_count": int(state.invalid_action_count),
            "code_repair_visible_score": float(state.best_visible_score),
            "code_repair_full_score": float(state.best_full_score),
            "code_repair_next_required_action": expected,
            "code_repair_tested_current_code": bool(state.current_code_tested),
            "code_repair_visible_passed_current_code": bool(state.visible_passed_current_code),
            "code_repair_policy_violation_count": int(state.policy_violation_count),
            "code_repair_step_policy_violation_count": 0,
            "code_repair_order_events": [],
            "code_repair_order_correct_count": int(state.order_correct_count),
            "code_repair_order_violation_streak": int(state.order_violation_streak),
            "code_repair_order_shaping_reward_sum": float(state.order_shaping_reward_sum),
            "code_repair_required_action_match": False,
            "code_repair_protocol_state_before": protocol_state,
            "code_repair_protocol_state_after": protocol_state,
            "code_repair_protocol_expected_action": expected,
            "code_repair_protocol_allowed_actions": allowed,
            "code_repair_protocol_actual_action": None,
            "code_repair_protocol_accepted": None,
            "code_repair_protocol_violation_reason": None,
            "code_repair_protocol_side_effect_applied": False,
            "code_repair_current_code_hash_before": current_hash,
            "code_repair_current_code_hash_after": current_hash,
        }

    def reset(self, kwargs: Any = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        self.current_tasks = self._select_tasks(kwargs)
        self.states = [
            CodeRepairEpisodeState(
                task=task,
                current_code=task.current_starter_code,
                last_code_hash=hash(task.current_starter_code),
            )
            for task in self.current_tasks
        ]
        observations = [self._task_summary(state) for state in self.states]
        infos = [self._base_info(state) for state in self.states]
        return observations, infos

    def _task_summary(self, state: CodeRepairEpisodeState) -> str:
        task = state.task
        return (
            f"Task {task.dataset_task_id}: {task.problem_description}\n\n"
            f"Current solution:\n```python\n{state.current_code.rstrip()}\n```"
        )

    def _visible_tests_text(self, state: CodeRepairEpisodeState) -> str:
        return f"Visible test harness:\n```python\n{state.task.visible_test_code.rstrip()}\n```"

    def _history_text(self, state: CodeRepairEpisodeState) -> str:
        if not state.history:
            return "No repair history yet."
        records = state.history[-self.history_length :]
        lines = []
        start = len(state.history) - len(records) + 1
        for offset, record in enumerate(records):
            lines.append(f"Step {start + offset}: {record['action']}\n{record['observation']}")
        return "\n\n".join(lines)

    def _view(self, state: CodeRepairEpisodeState, section: str) -> str:
        section = section or "all"
        parts = []
        if section in {"problem", "all"}:
            parts.append(f"Problem:\n{state.task.problem_description}")
        if section in {"code", "all"}:
            parts.append(f"Current solution:\n```python\n{state.current_code.rstrip()}\n```")
        if section in {"tests", "all"}:
            parts.append(self._visible_tests_text(state))
        if section in {"history", "all"}:
            parts.append(f"Recent history:\n{self._history_text(state)}")
        return "\n\n".join(parts)

    @staticmethod
    def _serialize_action(action: Any) -> Dict[str, Any]:
        if isinstance(action, CodeRepairAction):
            data = {
                "tool_name": action.tool_name,
                "parameters": dict(action.parameters or {}),
                "valid": bool(action.valid),
            }
            if action.error:
                data["error"] = action.error
            if action.parse_warning:
                data["parse_warning"] = action.parse_warning
            if "code" in data["parameters"]:
                data["parameters"] = dict(data["parameters"])
                data["parameters"]["code"] = f"<{len(action.parameters.get('code', ''))} chars>"
            return data
        return {"tool_name": "", "parameters": {}, "valid": False, "error": str(action)}

    def _invalid_step(self, state: CodeRepairEpisodeState, action: Any) -> Tuple[str, float, bool, Dict[str, Any]]:
        state.invalid_action_count += 1
        error = action.error if isinstance(action, CodeRepairAction) else str(action)
        observation = f"Invalid action: {error}"
        reward = -self.invalid_action_penalty
        info = self._base_info(state, is_action_valid=False)
        info.update({"error": error, "code_repair_action": self._serialize_action(action)})
        return observation, reward, False, info

    def _record_history(self, state: CodeRepairEpisodeState, action: Any, observation: str) -> None:
        if isinstance(action, CodeRepairAction):
            if action.tool_name == "replace_solution":
                action_text = f"replace_solution(code_chars={len(action.parameters.get('code', ''))})"
            elif action.error:
                action_text = action.error
            else:
                action_text = action.tool_name
                if action.parameters:
                    action_text += "(" + ", ".join(f"{key}={value}" for key, value in action.parameters.items()) + ")"
        else:
            action_text = str(action)
        state.history.append({"action": action_text, "observation": observation})

    def _append_next_required_action(self, state: CodeRepairEpisodeState, observation: str) -> str:
        next_required_action = self._next_required_action(state)
        if next_required_action == "done":
            return observation
        guidance = f"Next required action: {next_required_action}."
        if next_required_action == "view_problem":
            guidance += "\nCall view_problem with section=all before editing again."
            guidance += "\nBefore choosing an action, check the immediately previous action in Recent repair history."
        elif next_required_action == "run_tests":
            guidance += "\nDo not call replace_solution again until visible tests have run."
            guidance += "\nBefore choosing an action, check the immediately previous action in Recent repair history."
        elif next_required_action == "finish":
            guidance += "\nVisible tests passed for the current code; finish now instead of editing again."
            guidance += "\nBefore choosing an action, check the immediately previous action in Recent repair history."
        elif next_required_action == "replace_solution":
            guidance += "\nUse replace_solution with a complete class Solution implementation."
            guidance += "\nBefore choosing an action, check the immediately previous action in Recent repair history."
        return f"{observation}\n\n{guidance}"

    def _record_order_violation(self, state: CodeRepairEpisodeState, events: List[str], event: str) -> float:
        state.policy_violation_count += 1
        events.append(event)
        state.order_violation_streak += 1
        state.order_correct_count = 0
        repeat_penalty = min(
            max(0, state.order_violation_streak - 1) * ORDER_VIOLATION_REPEAT_PENALTY,
            ORDER_VIOLATION_REPEAT_PENALTY_CAP,
        )
        penalty = self.order_violation_penalty + repeat_penalty
        state.order_shaping_reward_sum -= penalty
        return -penalty

    def _protocol_violation_event(
        self,
        state: CodeRepairEpisodeState,
        action_name: str,
        violation_reason: Optional[str],
    ) -> str:
        if violation_reason == "action_before_initial_view" and action_name == "replace_solution" and state.test_count > 0:
            return "replace_before_required_view"
        return violation_reason or f"{action_name}_protocol_reject"

    def _accepted_order_reward(
        self,
        state: CodeRepairEpisodeState,
        action_name: str,
        snapshot_before: ProtocolSnapshot,
    ) -> float:
        key = action_name
        if action_name == "view_problem" and snapshot_before.state is ProtocolState.NEED_VIEW and state.test_count > 0:
            key = "view_after_failed_tests"
        reward = ORDER_CORRECT_REWARDS.get(key, 0.0)
        reward += min(state.order_correct_count * ORDER_CORRECT_STREAK_BONUS, ORDER_CORRECT_STREAK_BONUS_CAP)
        state.order_correct_count += 1
        state.order_violation_streak = 0
        state.order_shaping_reward_sum += reward
        return reward

    def _run_tests(self, state: CodeRepairEpisodeState, suite: str) -> Tuple[str, float, TestRunResult]:
        result = run_code_repair_tests(state.task, state.current_code, suite=suite, timeout=self.execution_timeout)
        state.test_count += 1
        score_attr = "best_visible_score" if suite == "visible" else "best_full_score"
        previous = float(getattr(state, score_attr))
        if result.score > previous:
            setattr(state, score_attr, float(result.score))
        improvement = max(0.0, result.score - previous)
        reward = improvement * self.test_improvement_reward
        if not result.passed and result.error:
            reward -= self.syntax_error_penalty if "SyntaxError" in result.error else 0.0
        state.current_code_tested = True
        if suite == "visible":
            state.visible_passed_current_code = bool(result.passed)
            state.protocol_snapshot = apply_visible_test_result(state.protocol_snapshot, passed=bool(result.passed))
        observation = (
            f"{suite.title()} tests: {result.passed_count}/{result.total} passed "
            f"(score={result.score:.3f})."
        )
        if result.error:
            observation += f"\nError:\n{result.error}"
        return observation, reward, result

    def _finish(self, state: CodeRepairEpisodeState, reason: str = "finish") -> Tuple[str, float, TestRunResult]:
        result = run_code_repair_tests(state.task, state.current_code, suite="full", timeout=self.execution_timeout)
        state.test_count += 1
        state.best_full_score = max(state.best_full_score, float(result.score))
        state.won = bool(result.passed)
        state.done = True
        reward = result.score * self.terminal_score_reward
        if result.passed:
            reward += self.terminal_success_reward
        observation = (
            f"Episode finalized by {reason}. Full tests: {result.passed_count}/{result.total} passed "
            f"(score={result.score:.3f})."
        )
        if result.error:
            observation += f"\nError:\n{result.error}"
        return observation, reward, result

    def _maybe_auto_finish(self, state: CodeRepairEpisodeState, reward: float, done: bool, observation: str, info: Dict[str, Any]):
        if done or not self.auto_finish_on_max_steps or state.steps < self.max_steps:
            return observation, reward, done, info
        if not state.visible_passed_current_code:
            state.policy_violation_count += 1
            reward -= self.untested_max_step_penalty
            info["code_repair_step_policy_violation_count"] = int(
                info.get("code_repair_step_policy_violation_count", 0)
            ) + 1
            info.setdefault("code_repair_order_events", []).append("max_steps_without_visible_test_gate")
        final_observation, final_reward, result = self._finish(state, reason="max_steps")
        info.update(
            {
                "won": bool(state.won),
                "code_repair_full_score": float(state.best_full_score),
                "code_repair_final_result": result.__dict__,
            }
        )
        return f"{observation}\n\n{final_observation}", reward + final_reward, True, info

    def _refresh_info(
        self,
        info: Dict[str, Any],
        state: CodeRepairEpisodeState,
        action: Any,
        step_policy_violation_count: int,
        order_events: List[str],
        protocol_state_before: ProtocolSnapshot,
        protocol_state_after: ProtocolSnapshot,
        protocol_actual_action: Optional[str],
        protocol_accepted: Optional[bool],
        protocol_violation_reason: Optional[str],
        protocol_side_effect_applied: bool,
        current_code_hash_before: int,
        current_code_hash_after: int,
    ) -> None:
        info.setdefault("code_repair_action", self._serialize_action(action))
        info["code_repair_step"] = int(state.steps)
        info["code_repair_edit_count"] = int(state.edit_count)
        info["code_repair_test_count"] = int(state.test_count)
        info["code_repair_invalid_action_count"] = int(state.invalid_action_count)
        info["code_repair_visible_score"] = float(state.best_visible_score)
        info["code_repair_full_score"] = float(state.best_full_score)
        info["code_repair_next_required_action"] = self._next_required_action(state)
        info["code_repair_tested_current_code"] = bool(state.current_code_tested)
        info["code_repair_visible_passed_current_code"] = bool(state.visible_passed_current_code)
        info["code_repair_policy_violation_count"] = int(state.policy_violation_count)
        info["code_repair_step_policy_violation_count"] = int(step_policy_violation_count)
        info["code_repair_order_events"] = list(order_events)
        info["code_repair_order_correct_count"] = int(state.order_correct_count)
        info["code_repair_order_violation_streak"] = int(state.order_violation_streak)
        info["code_repair_order_shaping_reward_sum"] = float(state.order_shaping_reward_sum)
        info["code_repair_required_action_match"] = bool(
            protocol_accepted is True and step_policy_violation_count == 0 and not order_events
        )
        info["code_repair_protocol_state_before"] = self._snapshot_state_name(protocol_state_before)
        info["code_repair_protocol_state_after"] = self._snapshot_state_name(protocol_state_after)
        info["code_repair_protocol_expected_action"] = expected_action(protocol_state_before)
        info["code_repair_protocol_allowed_actions"] = sorted(allowed_actions(protocol_state_before))
        info["code_repair_protocol_actual_action"] = protocol_actual_action
        info["code_repair_protocol_accepted"] = protocol_accepted
        info["code_repair_protocol_violation_reason"] = protocol_violation_reason
        info["code_repair_protocol_side_effect_applied"] = bool(protocol_side_effect_applied)
        info["code_repair_current_code_hash_before"] = int(current_code_hash_before)
        info["code_repair_current_code_hash_after"] = int(current_code_hash_after)
        info["won"] = bool(state.won)

    def _protocol_rejection_observation(
        self,
        state: CodeRepairEpisodeState,
        action: CodeRepairAction,
        violation_reason: Optional[str],
    ) -> str:
        if action.tool_name == "finish":
            return "Finish rejected: run visible tests and pass them before finish."
        if action.tool_name == "replace_solution":
            if violation_reason == "replace_without_testing":
                return "Please run visible tests before editing again.\nSolution replacement rejected by protocol."
            if violation_reason == "action_before_initial_view" and state.test_count > 0:
                return "Please refresh context with view_problem(section=all) before editing again.\nSolution replacement rejected by protocol."
            return "Solution replacement rejected by protocol."
        if action.tool_name == "view_problem" and violation_reason == "view_problem_requires_all":
            return "view_problem rejected: section must be all."
        return f"Protocol rejected action: {violation_reason or action.tool_name}."

    def step(self, actions: List[Any]):
        if len(actions) > len(self.states):
            raise ValueError(f"Got {len(actions)} actions for {len(self.states)} active CodeRepair environments.")
        observations: List[str] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []
        for idx, action in enumerate(actions):
            state = self.states[idx]
            if state.done:
                info = self._base_info(state)
                observations.append("Episode already completed.")
                rewards.append(0.0)
                dones.append(True)
                infos.append(info)
                continue

            state.steps += 1
            order_events: List[str] = []
            step_policy_violation_count = 0
            protocol_state_before = state.protocol_snapshot
            protocol_state_after = state.protocol_snapshot
            protocol_actual_action: Optional[str] = None
            protocol_accepted: Optional[bool] = None
            protocol_violation_reason: Optional[str] = None
            protocol_side_effect_applied = False
            current_code_hash_before = hash(state.current_code)

            if not isinstance(action, CodeRepairAction) or not action.valid:
                observation, reward, done, info = self._invalid_step(state, action)
            else:
                protocol_actual_action = action.tool_name
                section = action.parameters.get("section") if action.tool_name == "view_problem" else None
                decision = check_protocol_action(protocol_state_before, action.tool_name, section=section)
                protocol_accepted = bool(decision.accepted)
                protocol_violation_reason = decision.violation_reason

                if not decision.accepted:
                    event = self._protocol_violation_event(state, action.tool_name, decision.violation_reason)
                    reward = self._record_order_violation(state, order_events, event)
                    if action.tool_name == "finish":
                        reward -= self.premature_finish_penalty
                    step_policy_violation_count = 1
                    observation = self._protocol_rejection_observation(state, action, decision.violation_reason)
                    done = False
                    info = self._base_info(state, is_action_valid=False)
                elif action.tool_name == "view_problem":
                    reward = self._accepted_order_reward(state, action.tool_name, protocol_state_before)
                    section = action.parameters.get("section", "all")
                    observation = self._view(state, section)
                    state.protocol_snapshot = apply_action_transition(protocol_state_before, action.tool_name, section=section)
                    protocol_state_after = state.protocol_snapshot
                    protocol_side_effect_applied = True
                    done = False
                    info = self._base_info(state)
                elif action.tool_name == "replace_solution":
                    code = action.parameters.get("code", "")
                    info = self._base_info(state)
                    if len(code) > self.max_code_chars:
                        observation = f"Replacement rejected: code is {len(code)} chars, limit is {self.max_code_chars}."
                        reward = -self.invalid_action_penalty
                        done = False
                        info["is_action_valid"] = False
                        info["error"] = observation
                    else:
                        reward = self._accepted_order_reward(state, action.tool_name, protocol_state_before)
                        state.current_code = code.rstrip() + "\n"
                        state.edit_count += 1
                        state.current_code_tested = False
                        state.visible_passed_current_code = False
                        state.protocol_snapshot = apply_action_transition(protocol_state_before, action.tool_name)
                        code_hash = hash(state.current_code)
                        if code_hash == state.last_code_hash:
                            state.no_progress_count += 1
                            reward -= self.repeated_no_progress_penalty * state.no_progress_count
                            observation = "Solution replacement accepted, but it is identical to the previous code."
                        else:
                            state.no_progress_count = 0
                            observation = "Solution replacement accepted."
                        state.last_code_hash = code_hash
                        protocol_state_after = state.protocol_snapshot
                        protocol_side_effect_applied = True
                        done = False
                elif action.tool_name == "run_tests":
                    reward = self._accepted_order_reward(state, action.tool_name, protocol_state_before)
                    suite = action.parameters.get("suite", "visible")
                    prefix = ""
                    if state.edit_count == 0:
                        prefix += "No replacement has been submitted yet; tests are running against starter code.\n"
                    if suite == "full" and not self.allow_full_tests_in_loop:
                        prefix += "Full tests are reserved for finish; running visible tests instead.\n"
                        suite = "visible"
                    run_observation, run_reward, result = self._run_tests(state, suite)
                    reward += run_reward
                    observation = prefix + run_observation
                    protocol_state_after = state.protocol_snapshot
                    protocol_side_effect_applied = True
                    done = False
                    info = self._base_info(state)
                    info["code_repair_test_result"] = result.__dict__
                elif action.tool_name == "finish":
                    observation, terminal_reward, result = self._finish(state, reason="finish")
                    reward = terminal_reward + self._accepted_order_reward(state, action.tool_name, protocol_state_before)
                    protocol_state_after = state.protocol_snapshot
                    protocol_side_effect_applied = True
                    done = True
                    info = self._base_info(state)
                    info["code_repair_final_result"] = result.__dict__
                else:
                    observation, reward, done, info = self._invalid_step(state, action)

            current_code_hash_after = hash(state.current_code)
            protocol_state_after = state.protocol_snapshot
            state.done = bool(done)
            self._refresh_info(
                info,
                state,
                action,
                step_policy_violation_count,
                order_events,
                protocol_state_before,
                protocol_state_after,
                protocol_actual_action,
                protocol_accepted,
                protocol_violation_reason,
                protocol_side_effect_applied,
                current_code_hash_before,
                current_code_hash_after,
            )
            if info.get("is_action_valid", True):
                observation, reward, done, info = self._maybe_auto_finish(state, float(reward), bool(done), observation, info)
            else:
                reward = float(reward)
                done = bool(done)
            state.done = bool(done)
            self._refresh_info(
                info,
                state,
                action,
                info["code_repair_step_policy_violation_count"],
                info["code_repair_order_events"],
                protocol_state_before,
                state.protocol_snapshot,
                protocol_actual_action,
                protocol_accepted,
                protocol_violation_reason,
                protocol_side_effect_applied,
                current_code_hash_before,
                hash(state.current_code),
            )
            observation = self._append_next_required_action(state, observation)
            state.last_observation = observation
            self._record_history(state, action, observation)
            observations.append(observation)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)
        return observations, rewards, dones, infos

    def close(self) -> None:
        self.states = []


def build_code_repair_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config: Any = None,
):
    code_config = _cfg_get(env_config, "code_repair", env_config)
    tasks = load_code_repair_tasks_from_config(code_config, is_train=is_train)
    return CodeRepairVectorEnv(
        tasks=tasks,
        env_num=env_num,
        group_n=group_n,
        seed=seed,
        is_train=is_train,
        max_steps=int(_cfg_get(env_config, "max_steps", 64)),
        history_length=int(_cfg_get(env_config, "history_length", 5)),
        execution_timeout=int(_cfg_get(code_config, "execution_timeout", 8)),
        max_code_chars=int(_cfg_get(code_config, "max_code_chars", 20000)),
        allow_full_tests_in_loop=bool(_cfg_get(code_config, "allow_full_tests_in_loop", False)),
        auto_finish_on_max_steps=bool(_cfg_get(code_config, "auto_finish_on_max_steps", True)),
        invalid_action_penalty=float(_cfg_get(code_config, "invalid_action_penalty", INVALID_ACTION_PENALTY)),
        syntax_error_penalty=float(_cfg_get(code_config, "syntax_error_penalty", SYNTAX_ERROR_PENALTY)),
        repeated_no_progress_penalty=float(
            _cfg_get(code_config, "repeated_no_progress_penalty", REPEATED_NO_PROGRESS_PENALTY)
        ),
        test_improvement_reward=float(_cfg_get(code_config, "test_improvement_reward", TEST_IMPROVEMENT_REWARD)),
        terminal_success_reward=float(_cfg_get(code_config, "terminal_success_reward", TERMINAL_SUCCESS_REWARD)),
        terminal_score_reward=float(_cfg_get(code_config, "terminal_score_reward", TERMINAL_SCORE_REWARD)),
        order_violation_penalty=float(_cfg_get(code_config, "order_violation_penalty", ORDER_VIOLATION_PENALTY)),
        premature_finish_penalty=float(_cfg_get(code_config, "premature_finish_penalty", PREMATURE_FINISH_PENALTY)),
        untested_max_step_penalty=float(_cfg_get(code_config, "untested_max_step_penalty", UNTESTED_MAX_STEP_PENALTY)),
    )
