from __future__ import annotations

from collections import Counter

from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.scenario_engine import ScenarioEngine
from synthetic_enterprise.labeling.grounding import LabelGrounding, build_relevance_reason


def build_inputs(seed: int = 6060) -> tuple[GeneratorContext, EnterpriseGraph]:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=30, max_employees=30),
        ),
    )
    enterprise = CompanyBuilder().build(context, account_count=12)
    return context, enterprise


def test_scenario_engine_emits_requested_families() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)

    families = {
        scenario.kind.value if hasattr(scenario.kind, "value") else str(scenario.kind)
        for scenario in scenarios
    }

    assert families == {
        "pre_sales_discovery",
        "procurement_delay",
        "pilot_success",
        "rollout_risk",
        "support_escalation",
        "feature_gap_review",
        "renewal_risk",
        "executive_sponsor_review",
        "event_invite_to_attendance",
        "champion_departure",
        "stakeholder_expansion",
        "contract_redline_delay",
    }


def test_scenarios_have_valid_lifecycle_stages() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)

    assert scenarios
    for scenario in scenarios:
        assert scenario.transitions
        assert scenario.current_stage == scenario.transitions[-1].stage
        assert all(transition.state_changes for transition in scenario.transitions)


def test_scenarios_generate_temporally_ordered_events() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)

    for scenario in scenarios:
        occurred_at = [transition.occurred_at for transition in scenario.transitions]
        assert occurred_at == sorted(occurred_at)


def test_scenario_transitions_are_deterministic_under_seed() -> None:
    first_context, first_enterprise = build_inputs(seed=7070)
    second_context, second_enterprise = build_inputs(seed=7070)
    third_context, third_enterprise = build_inputs(seed=7071)

    first = ScenarioEngine(context=first_context).build_for_enterprise(first_enterprise)
    second = ScenarioEngine(context=second_context).build_for_enterprise(second_enterprise)
    third = ScenarioEngine(context=third_context).build_for_enterprise(third_enterprise)

    assert first == second
    assert first != third


def test_linked_entities_remain_valid() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)
    valid_ids = {
        "account": {account.id for account in enterprise.customer_accounts},
        "employee": {employee.id for employee in enterprise.employees},
        "contact": {contact.id for contact in enterprise.contacts},
        "opportunity": {opportunity.id for opportunity in enterprise.opportunities},
        "event": {event.id for event in enterprise.events},
        "ticket": {ticket.id for ticket in enterprise.ticket_issues},
    }

    for scenario in scenarios:
        assert set(scenario.account_ids).issubset(valid_ids["account"])
        assert set(scenario.employee_ids).issubset(valid_ids["employee"])
        assert set(scenario.contact_ids).issubset(valid_ids["contact"])
        assert set(scenario.opportunity_ids).issubset(valid_ids["opportunity"])
        assert set(scenario.event_ids).issubset(valid_ids["event"])
        assert set(scenario.ticket_ids).issubset(valid_ids["ticket"])

        for transition in scenario.transitions:
            for grounding in transition.category_groundings:
                assert grounding.provenance.object_id in valid_ids[
                    grounding.provenance.object_type.value
                ]


def test_relevant_categories_can_be_grounded_to_scenario_state() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)

    for scenario in scenarios:
        assert scenario.expected_labels
        for category in scenario.expected_labels:
            grounding = scenario.grounding_for(category)
            assert grounding is not None
            LabelGrounding(
                primary_category=category,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=category,
                    is_relevant=True,
                    provenance=grounding.provenance,
                ),
                provenance=grounding.provenance,
            )


def test_scenarios_can_end_stall_or_reopen() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)
    current_stage_counts = Counter(scenario.current_stage.value for scenario in scenarios)

    assert current_stage_counts["completed"] > 0
    assert current_stage_counts["stalled"] > 0
    assert current_stage_counts["reopened"] > 0


def test_multiple_scenarios_can_coexist_without_id_collisions() -> None:
    context, enterprise = build_inputs()
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)

    scenario_ids = [scenario.scenario_id for scenario in scenarios]
    transition_ids = [
        transition.transition_id
        for scenario in scenarios
        for transition in scenario.transitions
    ]

    assert len(scenario_ids) == len(set(scenario_ids))
    assert len(transition_ids) == len(set(transition_ids))
    assert len(set(scenario_ids) & set(transition_ids)) == 0


def test_engine_preserves_company_state_build_compatibility() -> None:
    context, enterprise = build_inputs()
    company = enterprise.companies[0]

    scenarios = ScenarioEngine().build_legacy(
        company_id=company.id,
        company_name=company.name,
        seed=context.seed,
    )

    assert len(scenarios) == 1
    assert scenarios[0].company_id == company.id
