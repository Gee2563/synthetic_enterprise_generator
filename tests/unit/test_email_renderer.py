from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from synthetic_enterprise.contracts.records.email import EmailRecord
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory
from synthetic_enterprise.sources.email.renderer import EmailRenderer


def build_renderer() -> tuple[GeneratorContext, EmailRenderer]:
    context = GeneratorContext(
        seed=2026,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
        ),
    )
    graph = CompanyBuilder().build(context)
    renderer = EmailRenderer(context=context, enterprise=graph)
    return context, renderer


def test_email_record_has_required_fields() -> None:
    required_fields = {
        "email_id",
        "thread_id",
        "message_index_in_thread",
        "timestamp",
        "sender_employee_id",
        "sender_contact_id",
        "to",
        "cc",
        "bcc",
        "subject",
        "body",
        "attachments",
        "account_id",
        "opportunity_id",
        "event_id",
        "ticket_id",
        "primary_category",
        "is_relevant",
        "relevance_reason",
        "source_system",
    }

    assert required_fields.issubset(EmailRecord.model_fields)


def test_subject_and_body_are_non_empty() -> None:
    with pytest.raises(ValidationError):
        EmailRecord(
            email_id="email_001",
            thread_id="thread_001",
            message_index_in_thread=0,
            timestamp=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
            sender_employee_id="employee_001",
            sender_contact_id=None,
            to=["contact_001"],
            cc=[],
            bcc=[],
            subject="",
            body="   ",
            attachments=[],
            account_id="account_001",
            opportunity_id=None,
            event_id="event_001",
            ticket_id=None,
            primary_category=CommunicationCategory.SCHEDULING_ONLY,
            is_relevant=False,
            relevance_reason="Calendar logistics only.",
            source_system="email",
        )


def test_participant_ids_refer_to_known_entities() -> None:
    _, renderer = build_renderer()
    event = renderer.enterprise.events[0]
    email = renderer.render_from_event(
        event_id=event.id,
        primary_category=CommunicationCategory.FOLLOW_UP,
        message_index_in_thread=0,
    )
    valid_participants = {employee.id for employee in renderer.enterprise.employees}
    valid_participants.update(contact.id for contact in renderer.enterprise.contacts)
    participants = {
        participant_id
        for participant_id in [
            email.sender_employee_id,
            email.sender_contact_id,
            *email.to,
            *email.cc,
            *email.bcc,
        ]
        if participant_id is not None
    }

    assert participants
    assert participants.issubset(valid_participants)


def test_thread_ordering_works() -> None:
    _, renderer = build_renderer()
    event = renderer.enterprise.events[0]

    scheduling_email = renderer.render_from_event(
        event_id=event.id,
        primary_category=CommunicationCategory.SCHEDULING_ONLY,
        message_index_in_thread=0,
    )
    follow_up_email = renderer.render_from_event(
        event_id=event.id,
        primary_category=CommunicationCategory.FOLLOW_UP,
        message_index_in_thread=1,
    )

    assert scheduling_email.thread_id == follow_up_email.thread_id
    assert scheduling_email.message_index_in_thread < follow_up_email.message_index_in_thread
    assert scheduling_email.timestamp < follow_up_email.timestamp


def test_relevant_messages_contain_traceable_business_context() -> None:
    _, renderer = build_renderer()
    event = renderer.enterprise.events[0]
    account = next(
        account
        for account in renderer.enterprise.customer_accounts
        if account.id == event.account_id
    )
    email = renderer.render_from_event(
        event_id=event.id,
        primary_category=CommunicationCategory.FOLLOW_UP,
        message_index_in_thread=1,
    )

    assert email.is_relevant is True
    assert email.primary_category == CommunicationCategory.FOLLOW_UP
    assert email.account_id == account.id
    assert email.event_id == event.id
    assert email.opportunity_id is not None
    assert email.ticket_id is not None
    assert account.name in f"{email.subject} {email.body}"
    assert event.title in email.body
    assert event.id in email.relevance_reason


def test_noise_emails_do_not_accidentally_leak_strong_relevance_signals() -> None:
    _, renderer = build_renderer()
    event = renderer.enterprise.events[0]
    email = renderer.render_from_event(
        event_id=event.id,
        primary_category=CommunicationCategory.SCHEDULING_ONLY,
        message_index_in_thread=0,
    )

    combined_text = f"{email.subject} {email.body} {email.relevance_reason}".lower()
    blocked_terms = [
        "pain",
        "complaint",
        "feature request",
        "buying",
        "churn",
        "follow up",
        "blocker",
        "escalation",
        "decision maker",
    ]

    assert email.is_relevant is False
    assert email.primary_category == CommunicationCategory.SCHEDULING_ONLY
    assert email.opportunity_id is None
    assert email.ticket_id is None
    for blocked_term in blocked_terms:
        assert blocked_term not in combined_text


def test_email_records_are_serializable_to_dataframe_rows() -> None:
    _, renderer = build_renderer()
    event = renderer.enterprise.events[0]
    email = renderer.render_from_event(
        event_id=event.id,
        primary_category=CommunicationCategory.FOLLOW_UP,
        message_index_in_thread=0,
    )

    frame = pd.DataFrame([email.to_dataframe_row()])

    assert frame.loc[0, "email_id"] == email.email_id
    assert frame.loc[0, "source_system"] == "email"
    assert frame.loc[0, "primary_category"] == email.primary_category.value
