from __future__ import annotations

from synthetic_enterprise.generation.account_profiles import (
    AccountBehaviorProfileType,
    AccountBehaviorResolver,
)
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.cross_system import (
    CrossSystemEventBundle,
    CrossSystemRenderer,
)
from synthetic_enterprise.generation.scenario_engine import ScenarioEngine
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory


def build_profiled_outputs(
    seed: int = 12010,
) -> tuple[
    GeneratorContext,
    CrossSystemRenderer,
    AccountBehaviorResolver,
]:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=12, max_employees=12),
            cross_system_ratio=1.0,
        ),
    )
    enterprise = CompanyBuilder().build(context, account_count=7)
    renderer = CrossSystemRenderer(context=context, enterprise=enterprise)
    resolver = AccountBehaviorResolver(context=context, enterprise=enterprise)
    return context, renderer, resolver


def test_account_profiles_affect_message_volume_and_type() -> None:
    _, renderer, resolver = build_profiled_outputs()
    bundles_by_profile = _bundles_by_profile(renderer=renderer, resolver=resolver)

    champion_bundle = bundles_by_profile[
        AccountBehaviorProfileType.HIGHLY_ENGAGED_CHAMPION_LED
    ]
    dormant_bundle = bundles_by_profile[AccountBehaviorProfileType.LOW_TOUCH_DORMANT]
    unhappy_bundle = bundles_by_profile[
        AccountBehaviorProfileType.SUPPORT_INTENSIVE_UNHAPPY
    ]

    assert len(champion_bundle.all_records) > len(dormant_bundle.all_records)
    assert _category_count(
        unhappy_bundle,
        CommunicationCategory.BLOCKER,
    ) >= 2
    assert _category_count(
        unhappy_bundle,
        CommunicationCategory.ESCALATION,
    ) >= 1


def test_account_profiles_affect_source_mix() -> None:
    _, renderer, resolver = build_profiled_outputs()
    bundles_by_profile = _bundles_by_profile(renderer=renderer, resolver=resolver)

    procurement_bundle = bundles_by_profile[
        AccountBehaviorProfileType.PROCUREMENT_HEAVY_SLOW_MOVER
    ]
    implementation_bundle = bundles_by_profile[
        AccountBehaviorProfileType.IMPLEMENTATION_STRUGGLING
    ]

    assert procurement_bundle.email_records
    assert procurement_bundle.salesforce_records
    assert procurement_bundle.slack_records == []
    assert implementation_bundle.slack_records
    assert implementation_bundle.teams_records


def test_account_profiles_affect_relevant_category_distribution() -> None:
    _, renderer, resolver = build_profiled_outputs()
    bundles_by_profile = _bundles_by_profile(renderer=renderer, resolver=resolver)

    expansion_bundle = bundles_by_profile[
        AccountBehaviorProfileType.EXECUTIVE_SPONSORED_EXPANSION
    ]
    event_active_bundle = bundles_by_profile[
        AccountBehaviorProfileType.EVENT_ACTIVE_LOW_CONVERSION
    ]

    assert _category_count(
        expansion_bundle,
        CommunicationCategory.DECISION_MAKER_SIGNAL,
    ) >= 1
    assert _category_count(
        event_active_bundle,
        CommunicationCategory.EVENT_ATTENDANCE,
    ) > _category_count(
        event_active_bundle,
        CommunicationCategory.BUYING_SIGNAL,
    )


def test_profile_behavior_is_measurable_in_generated_output_and_scheduling() -> None:
    context, renderer, resolver = build_profiled_outputs()
    enterprise = renderer.enterprise
    scenarios = ScenarioEngine(context=context).build_for_enterprise(enterprise)
    scenarios_by_account: dict[str, set[object]] = {}

    for scenario in scenarios:
        for account_id in scenario.account_ids:
            scenarios_by_account.setdefault(account_id, set()).add(scenario.kind)

    expected_families = {
        AccountBehaviorProfileType.HIGHLY_ENGAGED_CHAMPION_LED: "stakeholder_expansion",
        AccountBehaviorProfileType.PROCUREMENT_HEAVY_SLOW_MOVER: "procurement_delay",
        AccountBehaviorProfileType.SUPPORT_INTENSIVE_UNHAPPY: "support_escalation",
        AccountBehaviorProfileType.EXECUTIVE_SPONSORED_EXPANSION: (
            "executive_sponsor_review"
        ),
        AccountBehaviorProfileType.LOW_TOUCH_DORMANT: "renewal_risk",
        AccountBehaviorProfileType.EVENT_ACTIVE_LOW_CONVERSION: (
            "event_invite_to_attendance"
        ),
        AccountBehaviorProfileType.IMPLEMENTATION_STRUGGLING: "rollout_risk",
    }

    for account in enterprise.customer_accounts:
        profile = resolver.profile_for_account(account.id)
        scenario_values = {
            kind.value if hasattr(kind, "value") else str(kind)
            for kind in scenarios_by_account.get(account.id, set())
        }
        assert expected_families[profile.profile_type] in scenario_values


def test_account_profile_behavior_is_deterministic_under_seed() -> None:
    _, first_renderer, first_resolver = build_profiled_outputs(seed=12011)
    _, second_renderer, second_resolver = build_profiled_outputs(seed=12011)
    _, third_renderer, third_resolver = build_profiled_outputs(seed=12012)

    first_profiles = _profile_sequence(first_renderer, first_resolver)
    second_profiles = _profile_sequence(second_renderer, second_resolver)
    third_profiles = _profile_sequence(third_renderer, third_resolver)

    assert first_profiles == second_profiles
    assert first_renderer.render_all_events() == second_renderer.render_all_events()
    assert first_profiles != third_profiles


def _bundles_by_profile(
    *,
    renderer: CrossSystemRenderer,
    resolver: AccountBehaviorResolver,
) -> dict[AccountBehaviorProfileType, CrossSystemEventBundle]:
    bundles = renderer.render_all_events()
    mapping = {}

    for bundle in bundles:
        profile = resolver.profile_for_account(bundle.account_id)
        mapping[profile.profile_type] = bundle

    return mapping


def _profile_sequence(
    renderer: CrossSystemRenderer,
    resolver: AccountBehaviorResolver,
) -> tuple[AccountBehaviorProfileType, ...]:
    bundles = renderer.render_all_events()
    return tuple(
        resolver.profile_for_account(bundle.account_id).profile_type
        for bundle in bundles
    )


def _category_count(
    bundle: CrossSystemEventBundle,
    category: CommunicationCategory,
) -> int:
    records = bundle.all_records
    return sum(
        1
        for record in records
        if getattr(record, "primary_category", None) == category
    )
