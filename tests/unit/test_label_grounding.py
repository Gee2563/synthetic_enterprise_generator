from __future__ import annotations

from typing import Protocol

import pytest
from pydantic import ValidationError

from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.grounding import (
    LabelGrounding,
    LabelProvenance,
    ProvenanceObjectType,
    ProvenanceStrength,
    build_relevance_reason,
)
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory
from synthetic_enterprise.sources.email.renderer import EmailRenderer
from synthetic_enterprise.sources.salesforce.renderer import SalesforceRenderer
from synthetic_enterprise.sources.slack.renderer import SlackRenderer
from synthetic_enterprise.sources.teams.renderer import TeamsRenderer


class HasGrounding(Protocol):
    is_relevant: bool
    primary_category: CommunicationCategory
    relevance_reason: str
    provenance: LabelProvenance | None


def test_relevant_rows_always_have_a_valid_provenance_object() -> None:
    rows = _generated_rows()
    relevant_rows = [row for row in rows if row.is_relevant]

    assert relevant_rows

    for row in relevant_rows:
        assert row.provenance is not None
        assert row.provenance.strength == ProvenanceStrength.STRONG
        assert row.provenance.object_id in _row_context_ids(row)


def test_provenance_explains_why_the_label_was_assigned() -> None:
    row = next(
        current
        for current in _generated_rows()
        if current.is_relevant
        and current.primary_category == CommunicationCategory.BLOCKER
    )

    assert row.provenance is not None
    assert row.provenance.explanation
    assert row.relevance_reason == build_relevance_reason(
        primary_category=row.primary_category,
        is_relevant=row.is_relevant,
        provenance=row.provenance,
    )
    assert row.provenance.object_id in row.relevance_reason


def test_noise_rows_may_have_null_provenance_or_weak_provenance() -> None:
    rows = _generated_rows()
    noise_rows = [row for row in rows if not row.is_relevant]

    assert noise_rows
    assert all(
        row.provenance is None or row.provenance.strength == ProvenanceStrength.WEAK
        for row in noise_rows
    )


def test_contradictory_labels_are_rejected() -> None:
    provenance = LabelProvenance(
        object_type=ProvenanceObjectType.TICKET,
        object_id="ticket_issue_001",
        explanation="Open support issue is still active.",
    )

    with pytest.raises(ValidationError):
        LabelGrounding(
            primary_category=CommunicationCategory.BUYING_SIGNAL,
            is_relevant=True,
            provenance=provenance,
            relevance_reason="Commercial motion is grounded in ticket ticket_issue_001.",
        )


def test_label_reason_pairs_are_consistent() -> None:
    provenance = LabelProvenance(
        object_type=ProvenanceObjectType.EVENT,
        object_id="event_001",
        explanation="Attendance is captured in the meeting record.",
    )
    expected_reason = build_relevance_reason(
        primary_category=CommunicationCategory.EVENT_ATTENDANCE,
        is_relevant=True,
        provenance=provenance,
    )

    grounding = LabelGrounding(
        primary_category=CommunicationCategory.EVENT_ATTENDANCE,
        is_relevant=True,
        provenance=provenance,
        relevance_reason=expected_reason,
    )

    assert grounding.relevance_reason == expected_reason

    with pytest.raises(ValidationError):
        LabelGrounding(
            primary_category=CommunicationCategory.EVENT_ATTENDANCE,
            is_relevant=True,
            provenance=provenance,
            relevance_reason="This reason does not match the grounded label.",
        )


def _generated_rows() -> list[HasGrounding]:
    context = GeneratorContext(
        seed=7070,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_ratio=0.8,
        ),
    )
    enterprise = CompanyBuilder().build(context)
    email_renderer = EmailRenderer(context=context, enterprise=enterprise)
    slack_renderer = SlackRenderer(context=context, enterprise=enterprise)
    teams_renderer = TeamsRenderer(context=context, enterprise=enterprise)
    salesforce_renderer = SalesforceRenderer(context=context, enterprise=enterprise)
    email_event = enterprise.events[0]
    email_row = email_renderer.render_from_event(
        event_id=email_event.id,
        primary_category=CommunicationCategory.FOLLOW_UP,
        message_index_in_thread=0,
    )

    return [
        email_row,
        *slack_renderer.generate_messages(),
        *teams_renderer.generate_messages(),
        *salesforce_renderer.generate_records(),
    ]


def _row_context_ids(row: HasGrounding) -> set[str]:
    candidate_fields = (
        "record_id",
        "account_id",
        "contact_id",
        "lead_id",
        "opportunity_id",
        "event_id",
        "case_id",
        "campaign_id",
        "ticket_id",
        "linked_account_id",
        "linked_opportunity_id",
        "linked_event_id",
        "linked_ticket_id",
        "parent_record_id",
    )
    context_ids: set[str] = set()

    for field_name in candidate_fields:
        field_value = getattr(row, field_name, None)
        if field_value is not None:
            context_ids.add(field_value)

    return context_ids
