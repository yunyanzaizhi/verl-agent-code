from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ProtocolState(str, Enum):
    NEED_VIEW = "need_view"
    NEED_REPLACE = "need_replace"
    NEED_VISIBLE_TEST = "need_visible_test"
    CAN_FINISH = "can_finish"


@dataclass(frozen=True)
class ProtocolSnapshot:
    state: ProtocolState


@dataclass(frozen=True)
class ProtocolDecision:
    accepted: bool
    violation_reason: Optional[str] = None


def initial_protocol_snapshot() -> ProtocolSnapshot:
    return ProtocolSnapshot(state=ProtocolState.NEED_VIEW)


def allowed_actions(snapshot: ProtocolSnapshot) -> set[str]:
    if snapshot.state is ProtocolState.NEED_VIEW:
        return {"view_problem"}
    if snapshot.state is ProtocolState.NEED_REPLACE:
        return {"replace_solution"}
    if snapshot.state is ProtocolState.NEED_VISIBLE_TEST:
        return {"run_tests"}
    if snapshot.state is ProtocolState.CAN_FINISH:
        return {"finish", "replace_solution"}
    raise ValueError(f"Unsupported protocol state: {snapshot.state}")


def expected_action(snapshot: ProtocolSnapshot) -> str:
    if snapshot.state is ProtocolState.NEED_VIEW:
        return "view_problem"
    if snapshot.state is ProtocolState.NEED_REPLACE:
        return "replace_solution"
    if snapshot.state is ProtocolState.NEED_VISIBLE_TEST:
        return "run_tests"
    if snapshot.state is ProtocolState.CAN_FINISH:
        return "finish"
    raise ValueError(f"Unsupported protocol state: {snapshot.state}")


def check_protocol_action(snapshot: ProtocolSnapshot, action: str, *, section: Optional[str] = None) -> ProtocolDecision:
    if snapshot.state is ProtocolState.NEED_VIEW:
        if action != "view_problem":
            return ProtocolDecision(accepted=False, violation_reason="action_before_initial_view")
        if section != "all":
            return ProtocolDecision(accepted=False, violation_reason="view_problem_requires_all")
        return ProtocolDecision(accepted=True)

    if snapshot.state is ProtocolState.NEED_REPLACE:
        if action == "replace_solution":
            return ProtocolDecision(accepted=True)
        if action == "finish":
            return ProtocolDecision(accepted=False, violation_reason="finish_before_visible_success")
        return ProtocolDecision(accepted=False, violation_reason="expected_replace_solution")

    if snapshot.state is ProtocolState.NEED_VISIBLE_TEST:
        if action == "run_tests":
            return ProtocolDecision(accepted=True)
        if action == "replace_solution":
            return ProtocolDecision(accepted=False, violation_reason="replace_without_testing")
        if action == "finish":
            return ProtocolDecision(accepted=False, violation_reason="finish_before_visible_success")
        return ProtocolDecision(accepted=False, violation_reason="expected_visible_test")

    if snapshot.state is ProtocolState.CAN_FINISH:
        if action in {"finish", "replace_solution"}:
            return ProtocolDecision(accepted=True)
        return ProtocolDecision(accepted=False, violation_reason="expected_finish_or_replace")

    raise ValueError(f"Unsupported protocol state: {snapshot.state}")


def apply_action_transition(snapshot: ProtocolSnapshot, action: str, *, section: Optional[str] = None) -> ProtocolSnapshot:
    decision = check_protocol_action(snapshot, action, section=section)
    if not decision.accepted:
        return snapshot

    if snapshot.state is ProtocolState.NEED_VIEW and action == "view_problem":
        return ProtocolSnapshot(state=ProtocolState.NEED_REPLACE)
    if snapshot.state is ProtocolState.NEED_REPLACE and action == "replace_solution":
        return ProtocolSnapshot(state=ProtocolState.NEED_VISIBLE_TEST)
    if snapshot.state is ProtocolState.CAN_FINISH and action == "replace_solution":
        return ProtocolSnapshot(state=ProtocolState.NEED_VISIBLE_TEST)
    return snapshot


def apply_visible_test_result(snapshot: ProtocolSnapshot, *, passed: bool) -> ProtocolSnapshot:
    if snapshot.state is not ProtocolState.NEED_VISIBLE_TEST:
        return snapshot
    if passed:
        return ProtocolSnapshot(state=ProtocolState.CAN_FINISH)
    return ProtocolSnapshot(state=ProtocolState.NEED_VIEW)
