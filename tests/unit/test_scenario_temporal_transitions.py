from __future__ import annotations

from collections import defaultdict

import pytest
from pydantic import ValidationError

from synthetic_enterprise.contracts.scenario import (
    ScenarioEntityStateTransition,
    ScenarioEntityStateType,
    ScenarioFamily,
)
from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.scenario_engine import ScenarioEngine


def build_inputs(seed: int = 9090) -> tuple[GeneratorContext, EnterpriseGraph]:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=30, max_employees=30),
        ),
    )
    enterprise = CompanyBuilder().build(context, account_count=12)
    return context, enterprise


def test_temporal_state_transitions_occur_in_chronological_order() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)

    for scenario in scenarios:
        flattened_times = [
            transition.occurred_at
            for transition in scenario.transitions
            for _ in transition.entity_state_transitions
        ]
        assert flattened_times == sorted(flattened_times)


def test_invalid_transitions_are_rejected() -> None:
    with pytest.raises(ValidationError, match="invalid opportunity_stage transition"):
        ScenarioEntityStateTransition(
            transition_type=ScenarioEntityStateType.OPPORTUNITY_STAGE,
            entity_id="opportunity_deadbeef",
            from_state="closed_won",
            to_state="discovery",
        )

    with pytest.raises(ValidationError, match="invalid event_participation transition"):
        ScenarioEntityStateTransition(
            transition_type=ScenarioEntityStateType.EVENT_PARTICIPATION,
            entity_id="event_deadbeef",
            from_state="invited",
            to_state="attended",
            actor_id="contact_deadbeef",
        )

    with pytest.raises(ValidationError, match="invalid follow_up_status transition"):
        ScenarioEntityStateTransition(
            transition_type=ScenarioEntityStateType.FOLLOW_UP_STATUS,
            entity_id="account_deadbeef",
            from_state="completed",
            to_state="pending",
        )


def test_transitions_can_generate_downstream_communication_triggers() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)

    triggering_transitions = [
        (scenario, transition)
        for scenario in scenarios
        for transition in scenario.transitions
        if transition.entity_state_transitions
    ]

    assert triggering_transitions
    for scenario, transition in triggering_transitions:
        assert transition.communication_triggers
        assert all(trigger.reason for trigger in transition.communication_triggers)
        assert {
            trigger.category for trigger in transition.communication_triggers
        }.issubset(set(scenario.expected_labels))


def test_event_attendance_progression_is_explicit() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)
    attendance_scenario = next(
        scenario
        for scenario in scenarios
        if scenario.kind == ScenarioFamily.EVENT_INVITE_TO_ATTENDANCE
    )

    progression = [
        transition.to_state
        for scenario_transition in attendance_scenario.transitions
        for transition in scenario_transition.entity_state_transitions
        if transition.transition_type == ScenarioEntityStateType.EVENT_PARTICIPATION
    ]

    assert progression == ["invited", "accepted", "tentative", "attended"]


def test_account_and_opportunity_state_can_diverge_realistically() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)
    renewal_risk = next(
        scenario
        for scenario in scenarios
        if scenario.kind == ScenarioFamily.RENEWAL_RISK
    )

    latest_state_by_type: dict[ScenarioEntityStateType, str] = {}
    for scenario_transition in renewal_risk.transitions:
        for transition in scenario_transition.entity_state_transitions:
            latest_state_by_type[transition.transition_type] = transition.to_state

    assert latest_state_by_type[ScenarioEntityStateType.ACCOUNT_HEALTH] == "at_risk"
    assert latest_state_by_type[ScenarioEntityStateType.OPPORTUNITY_STAGE] in {
        "evaluation",
        "proposal",
    }


def test_temporal_transition_output_is_deterministic_under_seed() -> None:
    first_context, first_enterprise = build_inputs(seed=9090)
    second_context, second_enterprise = build_inputs(seed=9090)
    third_context, third_enterprise = build_inputs(seed=9091)

    first = ScenarioEngine(context=first_context).build_for_enterprise(first_enterprise)
    second = ScenarioEngine(context=second_context).build_for_enterprise(second_enterprise)
    third = ScenarioEngine(context=third_context).build_for_enterprise(third_enterprise)

    assert first == second
    assert first != third


def test_required_temporal_transition_types_are_present_across_scenarios() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)
    seen_types: defaultdict[ScenarioEntityStateType, int] = defaultdict(int)

    for scenario in scenarios:
        for scenario_transition in scenario.transitions:
            for transition in scenario_transition.entity_state_transitions:
                seen_types[transition.transition_type] += 1

    assert set(seen_types) == {
        ScenarioEntityStateType.OPPORTUNITY_STAGE,
        ScenarioEntityStateType.TICKET_SEVERITY,
        ScenarioEntityStateType.ACCOUNT_HEALTH,
        ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT,
        ScenarioEntityStateType.EVENT_PARTICIPATION,
        ScenarioEntityStateType.OWNERSHIP,
        ScenarioEntityStateType.FOLLOW_UP_STATUS,
    }
