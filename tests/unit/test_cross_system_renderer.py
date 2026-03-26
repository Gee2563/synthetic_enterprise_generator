from __future__ import annotations

from datetime import timedelta

from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.cross_system import CrossSystemRenderer
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory


def build_renderer(cross_system_ratio: float = 1.0) -> CrossSystemRenderer:
    context = GeneratorContext(
        seed=10010,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            cross_system_ratio=cross_system_ratio,
        ),
    )
    enterprise = CompanyBuilder().build(context)
    return CrossSystemRenderer(context=context, enterprise=enterprise)


def test_linked_cross_system_records_share_stable_entity_references() -> None:
    renderer = build_renderer(cross_system_ratio=1.0)
    event = renderer.enterprise.events[0]
    bundle = renderer.render_event_bundle(event.id)

    assert bundle.event_id == event.id
    assert bundle.account_id == event.account_id
    assert bundle.email_records
    assert bundle.slack_records
    assert bundle.teams_records
    assert bundle.salesforce_records

    for record in bundle.all_records:
        account_id = _linked_account_id(record)
        event_id = _linked_event_id(record)
        if account_id is not None:
            assert account_id == bundle.account_id
        if event_id is not None:
            assert event_id == bundle.event_id


def test_timestamps_across_systems_are_plausible() -> None:
    renderer = build_renderer(cross_system_ratio=1.0)
    event = renderer.enterprise.events[0]
    bundle = renderer.render_event_bundle(event.id)
    timestamps = [record.timestamp for record in bundle.all_records]

    assert timestamps
    assert min(timestamps) >= event.starts_at - timedelta(days=3)
    assert max(timestamps) <= event.ends_at + timedelta(days=3)


def test_message_wording_differs_by_channel_while_facts_remain_consistent() -> None:
    renderer = build_renderer(cross_system_ratio=1.0)
    event = renderer.enterprise.events[0]
    bundle = renderer.render_event_bundle(event.id)
    email = next(
        record
        for record in bundle.email_records
        if record.primary_category != CommunicationCategory.EVENT_ATTENDANCE
    )
    slack = next(
        record
        for record in bundle.slack_records
        if record.is_relevant
    )
    teams = next(
        record
        for record in bundle.teams_records
        if record.is_relevant
    )

    assert email.body != slack.body
    assert slack.body != teams.body
    assert email.body != teams.body
    assert email.account_id == slack.linked_account_id == teams.linked_account_id
    assert email.event_id == slack.linked_event_id == teams.linked_event_id


def test_missing_data_scenarios_are_supported() -> None:
    renderer = build_renderer(cross_system_ratio=1.0)
    event = renderer.enterprise.events[0]
    account_id = event.account_id
    assert account_id is not None

    enterprise = renderer.enterprise.model_copy(
        update={
            "opportunities": [
                opportunity
                for opportunity in renderer.enterprise.opportunities
                if opportunity.account_id != account_id
            ],
            "ticket_issues": [
                ticket
                for ticket in renderer.enterprise.ticket_issues
                if ticket.account_id != account_id
            ],
        }
    )
    renderer_without_follow_up_state = CrossSystemRenderer(
        context=renderer.context,
        enterprise=enterprise,
    )
    bundle = renderer_without_follow_up_state.render_event_bundle(event.id)

    assert bundle.email_records
    assert bundle.teams_records
    assert bundle.salesforce_records
    assert all(_linked_opportunity_id(record) is None for record in bundle.all_records)
    assert all(_linked_ticket_id(record) is None for record in bundle.all_records)


def test_configurable_percentage_of_events_appear_in_multiple_systems() -> None:
    no_coverage = build_renderer(cross_system_ratio=0.0).render_all_events()
    partial_coverage = build_renderer(cross_system_ratio=0.5).render_all_events()
    full_renderer = build_renderer(cross_system_ratio=1.0)
    full_coverage = full_renderer.render_all_events()

    assert no_coverage == []
    assert len(partial_coverage) == 1
    assert len(full_coverage) == len(full_renderer.enterprise.events)
    assert all(bundle.rendered_system_count >= 3 for bundle in full_coverage)


def _linked_account_id(record: object) -> str | None:
    for field_name in ("account_id", "linked_account_id"):
        value = getattr(record, field_name, None)
        if value is not None:
            return value
    return None


def _linked_event_id(record: object) -> str | None:
    for field_name in ("event_id", "linked_event_id"):
        value = getattr(record, field_name, None)
        if value is not None:
            return value
    return None


def _linked_opportunity_id(record: object) -> str | None:
    for field_name in ("opportunity_id", "linked_opportunity_id"):
        value = getattr(record, field_name, None)
        if value is not None:
            return value
    return None


def _linked_ticket_id(record: object) -> str | None:
    for field_name in ("ticket_id", "linked_ticket_id", "case_id"):
        value = getattr(record, field_name, None)
        if value is not None:
            return value
    return None
