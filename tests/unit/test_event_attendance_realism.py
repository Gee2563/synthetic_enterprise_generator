from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from synthetic_enterprise.contracts.records.email import EmailRecord
from synthetic_enterprise.contracts.records.salesforce import (
    SalesforceObjectType,
    SalesforceRecord,
)
from synthetic_enterprise.contracts.records.slack import SlackRecord
from synthetic_enterprise.contracts.records.teams import TeamsRecord
from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.generation.attendance import EventAttendanceBuilder
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory
from synthetic_enterprise.sources.email.renderer import EmailRenderer
from synthetic_enterprise.sources.salesforce.renderer import SalesforceRenderer
from synthetic_enterprise.sources.slack.renderer import SlackRenderer
from synthetic_enterprise.sources.teams.renderer import TeamsRenderer


@dataclass(frozen=True, slots=True)
class AttendanceOutputs:
    email_rows: Sequence[EmailRecord]
    slack_rows: Sequence[SlackRecord]
    teams_rows: Sequence[TeamsRecord]
    salesforce_rows: Sequence[SalesforceRecord]


PostEventAttendanceRow: TypeAlias = EmailRecord | TeamsRecord


def test_event_attendance_is_backed_by_valid_event_lifecycle_state() -> None:
    context, enterprise = build_inputs()
    event = enterprise.events[0]
    lifecycle = EventAttendanceBuilder(context=context, enterprise=enterprise).build(
        event_id=event.id
    )

    assert lifecycle.invite_sent_at < lifecycle.rsvp_requested_at
    assert lifecycle.rsvp_requested_at < lifecycle.reminder_sent_at
    assert lifecycle.reminder_sent_at < lifecycle.internal_coordination_at
    assert lifecycle.internal_coordination_at < lifecycle.event_started_at
    assert lifecycle.event_started_at < lifecycle.post_event_follow_up_at
    assert lifecycle.attended_contact_ids
    assert lifecycle.no_show_contact_ids

    for participant in lifecycle.participants:
        assert participant.invited_at <= participant.reminder_sent_at
        if participant.rsvp_status == "declined":
            assert participant.final_status == "declined"
        else:
            assert participant.final_status in {"attended", "no_show"}


def test_attendance_may_appear_explicitly_or_implicitly_depending_on_source() -> None:
    outputs = build_outputs()

    email_rows = [
        row
        for row in outputs.email_rows
        if row.primary_category == CommunicationCategory.EVENT_ATTENDANCE
    ]
    slack_rows = [
        row
        for row in outputs.slack_rows
        if row.primary_category == CommunicationCategory.EVENT_ATTENDANCE
    ]
    teams_rows = [
        row
        for row in outputs.teams_rows
        if row.primary_category == CommunicationCategory.EVENT_ATTENDANCE
    ]
    salesforce_rows = [
        row
        for row in outputs.salesforce_rows
        if row.primary_category == CommunicationCategory.EVENT_ATTENDANCE
    ]

    assert any(
        any(
            marker in f"{row.subject} {row.body}".lower()
            for marker in ("joined", "missed", "thanks for joining")
        )
        for row in email_rows
    )
    assert any(
        any(
            marker in row.body.lower()
            for marker in ("on the bridge", "never made it", "already in")
        )
        for row in slack_rows
    )
    assert any("Recap:" in row.body and "\n-" in row.body for row in teams_rows)
    assert any(
        row.object_type in {
            SalesforceObjectType.EVENT,
            SalesforceObjectType.CAMPAIGN_MEMBER,
        }
        for row in salesforce_rows
    )


def test_no_show_and_attended_are_distinguishable() -> None:
    outputs = build_outputs()

    salesforce_member_statuses = {
        row.structured_fields.get("member_status")
        for row in outputs.salesforce_rows
        if row.object_type == SalesforceObjectType.CAMPAIGN_MEMBER
    }
    combined_text = " ".join(
        f"{getattr(row, 'subject', '')} {getattr(row, 'body', getattr(row, 'text_body', ''))}"
        for row in [
            *outputs.email_rows,
            *outputs.slack_rows,
            *outputs.teams_rows,
            *outputs.salesforce_rows,
        ]
        if getattr(row, "primary_category", None) == CommunicationCategory.EVENT_ATTENDANCE
    ).lower()

    assert {"Attended", "No Show"} <= salesforce_member_statuses
    assert "joined" in combined_text or "attended" in combined_text
    assert "missed" in combined_text or "no show" in combined_text


def test_post_event_follow_up_can_reference_attendance_plausibly() -> None:
    context, enterprise = build_inputs()
    event = enterprise.events[0]
    outputs = build_outputs(context=context, enterprise=enterprise)

    post_event_rows: list[PostEventAttendanceRow] = [
        row
        for row in outputs.email_rows
        if row.primary_category == CommunicationCategory.EVENT_ATTENDANCE
        and row.timestamp > event.ends_at
    ]
    post_event_rows.extend(
        row
        for row in outputs.teams_rows
        if row.primary_category == CommunicationCategory.EVENT_ATTENDANCE
        and row.timestamp > event.ends_at
    )

    assert post_event_rows
    assert any(
        any(
            marker in getattr(row, "body", "").lower()
            for marker in ("joined", "missed", "follow up", "recap")
        )
        for row in post_event_rows
    )


def test_event_attendance_can_appear_across_email_teams_slack_and_crm() -> None:
    outputs = build_outputs()

    assert any(
        row.primary_category == CommunicationCategory.EVENT_ATTENDANCE
        for row in outputs.email_rows
    )
    assert any(
        row.primary_category == CommunicationCategory.EVENT_ATTENDANCE
        for row in outputs.slack_rows
    )
    assert any(
        row.primary_category == CommunicationCategory.EVENT_ATTENDANCE
        for row in outputs.teams_rows
    )
    assert any(
        row.primary_category == CommunicationCategory.EVENT_ATTENDANCE
        for row in outputs.salesforce_rows
    )


def test_label_grounding_remains_correct() -> None:
    outputs = build_outputs()

    for email_row in outputs.email_rows:
        if email_row.primary_category != CommunicationCategory.EVENT_ATTENDANCE:
            continue
        assert email_row.provenance is not None
        assert email_row.provenance.object_id == email_row.event_id

    for slack_row in outputs.slack_rows:
        if slack_row.primary_category != CommunicationCategory.EVENT_ATTENDANCE:
            continue
        assert slack_row.provenance is not None
        assert slack_row.provenance.object_id == slack_row.linked_event_id

    for teams_row in outputs.teams_rows:
        if teams_row.primary_category != CommunicationCategory.EVENT_ATTENDANCE:
            continue
        assert teams_row.provenance is not None
        assert teams_row.provenance.object_id == teams_row.linked_event_id

    for crm_row in outputs.salesforce_rows:
        if crm_row.primary_category != CommunicationCategory.EVENT_ATTENDANCE:
            continue
        assert crm_row.provenance is not None
        assert crm_row.provenance.object_id in {
            crm_row.event_id,
            crm_row.campaign_id,
            crm_row.record_id,
        }


def build_inputs() -> tuple[GeneratorContext, EnterpriseGraph]:
    context = GeneratorContext(
        seed=9494,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_ratio=0.8,
            hard_negative_ratio=0.25,
        ),
    )
    enterprise = CompanyBuilder().build(context)
    return context, enterprise


def build_outputs(
    *,
    context: GeneratorContext | None = None,
    enterprise: EnterpriseGraph | None = None,
) -> AttendanceOutputs:
    if context is None or enterprise is None:
        context, enterprise = build_inputs()
    event = enterprise.events[0]

    return AttendanceOutputs(
        email_rows=EmailRenderer(context=context, enterprise=enterprise).generate_messages(
            event_id=event.id
        ),
        slack_rows=SlackRenderer(context=context, enterprise=enterprise).generate_messages(),
        teams_rows=TeamsRenderer(context=context, enterprise=enterprise).generate_messages(),
        salesforce_rows=SalesforceRenderer(
            context=context,
            enterprise=enterprise,
        ).generate_records(),
    )
