import pytest

from agent_system.environments.env_package.code_repair.protocol import (
    ProtocolState,
    apply_action_transition,
    apply_visible_test_result,
    allowed_actions,
    check_protocol_action,
    expected_action,
    initial_protocol_snapshot,
)


def test_initial_need_view_rejects_replace_solution_with_reason():
    snapshot = initial_protocol_snapshot()

    decision = check_protocol_action(snapshot, "replace_solution")

    assert snapshot.state is ProtocolState.NEED_VIEW
    assert decision.accepted is False
    assert decision.violation_reason == "action_before_initial_view"


def test_view_problem_all_is_accepted_and_transitions_to_need_replace():
    snapshot = initial_protocol_snapshot()

    decision = check_protocol_action(snapshot, "view_problem", section="all")
    next_snapshot = apply_action_transition(snapshot, "view_problem", section="all")

    assert decision.accepted is True
    assert next_snapshot.state is ProtocolState.NEED_REPLACE
    assert allowed_actions(next_snapshot) == {"replace_solution"}
    assert expected_action(next_snapshot) == "replace_solution"


def test_replace_solution_is_accepted_and_transitions_to_need_visible_test():
    snapshot = apply_action_transition(initial_protocol_snapshot(), "view_problem", section="all")

    decision = check_protocol_action(snapshot, "replace_solution")
    next_snapshot = apply_action_transition(snapshot, "replace_solution")

    assert decision.accepted is True
    assert next_snapshot.state is ProtocolState.NEED_VISIBLE_TEST
    assert allowed_actions(next_snapshot) == {"run_tests"}
    assert expected_action(next_snapshot) == "run_tests"


def test_need_visible_test_rejects_replace_solution_with_reason():
    snapshot = apply_action_transition(
        apply_action_transition(initial_protocol_snapshot(), "view_problem", section="all"),
        "replace_solution",
    )

    decision = check_protocol_action(snapshot, "replace_solution")

    assert snapshot.state is ProtocolState.NEED_VISIBLE_TEST
    assert decision.accepted is False
    assert decision.violation_reason == "replace_without_testing"


def test_visible_test_failure_transitions_back_to_need_view():
    snapshot = apply_action_transition(
        apply_action_transition(initial_protocol_snapshot(), "view_problem", section="all"),
        "replace_solution",
    )

    next_snapshot = apply_visible_test_result(snapshot, passed=False)

    assert next_snapshot.state is ProtocolState.NEED_VIEW
    assert allowed_actions(next_snapshot) == {"view_problem"}
    assert expected_action(next_snapshot) == "view_problem"


def test_visible_test_success_transitions_to_can_finish():
    snapshot = apply_action_transition(
        apply_action_transition(initial_protocol_snapshot(), "view_problem", section="all"),
        "replace_solution",
    )

    next_snapshot = apply_visible_test_result(snapshot, passed=True)

    assert next_snapshot.state is ProtocolState.CAN_FINISH
    assert allowed_actions(next_snapshot) == {"finish", "replace_solution"}
    assert expected_action(next_snapshot) == "finish"


def test_finish_before_visible_success_is_rejected():
    snapshot = apply_action_transition(initial_protocol_snapshot(), "view_problem", section="all")

    decision = check_protocol_action(snapshot, "finish")

    assert decision.accepted is False
    assert decision.violation_reason == "finish_before_visible_success"
