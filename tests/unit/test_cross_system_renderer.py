from __future__ import annotations

from datetime import timedelta

from synthetic_enterprise.contracts.records.salesforce import SalesforceObjectType
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


def test_linked_events_can_appear_at_different_times_across_systems() -> None:
    renderer = build_renderer(cross_system_ratio=1.0)
    event = renderer.enterprise.events[0]
    bundle = renderer.render_event_bundle(event.id)

    slack_blocker = next(
        record
        for record in bundle.slack_records
        if record.primary_category == CommunicationCategory.BLOCKER
    )
    email_attendance = next(
        record
        for record in bundle.email_records
        if record.primary_category == CommunicationCategory.EVENT_ATTENDANCE
    )
    campaign_member = next(
        record
        for record in bundle.salesforce_records
        if record.object_type == "CampaignMember"
        and record.structured_fields.get("member_status") == "Attended"
    )
    salesforce_note = next(
        record
        for record in bundle.salesforce_records
        if record.object_type == "Note"
    )
    teams_attendance = next(
        record
        for record in bundle.teams_records
        if record.primary_category == CommunicationCategory.EVENT_ATTENDANCE
    )

    assert slack_blocker.timestamp < email_attendance.timestamp
    assert email_attendance.timestamp < campaign_member.timestamp
    assert teams_attendance.timestamp < salesforce_note.timestamp


def test_facts_remain_logically_consistent_despite_lag() -> None:
    renderer = build_renderer(cross_system_ratio=1.0)
    event = renderer.enterprise.events[0]
    bundle = renderer.render_event_bundle(event.id)

    email_attendance = next(
        record
        for record in bundle.email_records
        if record.primary_category == CommunicationCategory.EVENT_ATTENDANCE
    )
    slack_attendance = next(
        record
        for record in bundle.slack_records
        if record.primary_category == CommunicationCategory.EVENT_ATTENDANCE
    )
    teams_attendance = next(
        record
        for record in bundle.teams_records
        if record.primary_category == CommunicationCategory.EVENT_ATTENDANCE
    )
    salesforce_statuses = {
        record.structured_fields.get("member_status")
        for record in bundle.salesforce_records
        if record.object_type == SalesforceObjectType.CAMPAIGN_MEMBER
    }

    assert email_attendance.event_id == slack_attendance.linked_event_id
    assert email_attendance.event_id == teams_attendance.linked_event_id
    assert {"Attended", "No Show"} <= salesforce_statuses
    assert (
        "joined" in email_attendance.body.lower()
        or "attended" in email_attendance.body.lower()
    )
    assert (
        "never made it" in slack_attendance.body.lower()
        or "missed" in teams_attendance.body.lower()
    )


def test_some_events_intentionally_appear_in_only_a_subset_of_systems() -> None:
    context = GeneratorContext(
        seed=10011,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            cross_system_ratio=1.0,
        ),
    )
    enterprise = CompanyBuilder().build(context, account_count=4)
    bundles = CrossSystemRenderer(context=context, enterprise=enterprise).render_all_events()

    assert bundles
    assert any(bundle.rendered_system_count == 3 for bundle in bundles)
    assert any(bundle.rendered_system_count == 4 for bundle in bundles)


def test_cross_system_linkage_still_works_with_partial_visibility() -> None:
    context = GeneratorContext(
        seed=10011,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            cross_system_ratio=1.0,
        ),
    )
    enterprise = CompanyBuilder().build(context, account_count=4)
    bundles = CrossSystemRenderer(context=context, enterprise=enterprise).render_all_events()

    for bundle in bundles:
        assert bundle.rendered_system_count >= 3
        for record in bundle.all_records:
            account_id = _linked_account_id(record)
            event_id = _linked_event_id(record)
            if account_id is not None:
                assert account_id == bundle.account_id
            if event_id is not None:
                assert event_id == bundle.event_id


def test_cross_system_rendering_remains_deterministic_under_lag() -> None:
    context_a = GeneratorContext(
        seed=10012,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            cross_system_ratio=1.0,
        ),
    )
    enterprise_a = CompanyBuilder().build(context_a, account_count=4)
    first = CrossSystemRenderer(context=context_a, enterprise=enterprise_a).render_all_events()

    context_b = GeneratorContext(
        seed=10012,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            cross_system_ratio=1.0,
        ),
    )
    enterprise_b = CompanyBuilder().build(context_b, account_count=4)
    second = CrossSystemRenderer(context=context_b, enterprise=enterprise_b).render_all_events()

    context_c = GeneratorContext(
        seed=10013,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            cross_system_ratio=1.0,
        ),
    )
    enterprise_c = CompanyBuilder().build(context_c, account_count=4)
    third = CrossSystemRenderer(context=context_c, enterprise=enterprise_c).render_all_events()

    assert first == second
    assert first != third


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
