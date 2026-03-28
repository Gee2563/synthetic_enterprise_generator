from __future__ import annotations

from synthetic_enterprise.contracts.records.email import EmailRecord
from synthetic_enterprise.evaluation.quality_report import DatasetQualityEvaluator
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory
from synthetic_enterprise.sources.email.renderer import EmailRenderer
from synthetic_enterprise.sources.email.templates import (
    EmailAudience,
    EmailDetailLevel,
    EmailSeniority,
    EmailStyleEngine,
    EmailStyleRequest,
    EmailSubjectMode,
)


def build_request(**overrides: object) -> EmailStyleRequest:
    request = EmailStyleRequest(
        thread_id="thread_compose",
        message_index_in_thread=1,
        base_subject="Next steps after account review",
        sender_name="Avery Stone",
        sender_role="Customer Success Manager",
        recipient_name="Jordan",
        subject_mode=EmailSubjectMode.REPLY,
        audience=EmailAudience.EXTERNAL_CUSTOMER,
        seniority=EmailSeniority.IC,
        detail_level=EmailDetailLevel.DETAILED,
        include_signature=False,
        context_lines=(
            "We need owners for the migration plan.",
            "The customer asked for a timing update by Friday.",
        ),
        prior_thread_summary="We already aligned on the review scope in the earlier thread.",
        objection_line="Procurement is still pushing on timing.",
        action_ask="Please reply with owners by Friday.",
        disclaimer_line="This note is for planning only.",
    )
    return request.model_copy(update=overrides)


def test_same_intent_can_render_into_multiple_distinct_surface_forms() -> None:
    request = build_request()
    rendered = [
        EmailStyleEngine(context=GeneratorContext(seed=seed)).render(request)
        for seed in range(300, 306)
    ]

    assert len({item.body for item in rendered}) >= 4
    ordering_flags = [
        item.body.lower().index("review scope") < item.body.lower().index("procurement")
        for item in rendered
    ]
    assert len(set(ordering_flags)) > 1
    for item in rendered:
        body = item.body.lower()
        assert "review scope" in body
        assert "procurement" in body
        assert "owners by friday" in body
        assert "planning only" in body


def test_outputs_remain_semantically_consistent_with_provenance() -> None:
    context = GeneratorContext(
        seed=4041,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
        ),
    )
    enterprise = CompanyBuilder().build(context)
    renderer = EmailRenderer(context=context, enterprise=enterprise)
    event = enterprise.events[0]
    account = next(
        current_account
        for current_account in enterprise.customer_accounts
        if current_account.id == event.account_id
    )

    email = renderer.render_from_event(
        event_id=event.id,
        primary_category=CommunicationCategory.FOLLOW_UP,
        message_index_in_thread=1,
    )

    assert email.is_relevant is True
    assert email.provenance is not None
    assert email.provenance.object_id == email.event_id
    assert account.name in f"{email.subject} {email.body}"
    assert event.title in f"{email.subject} {email.body}"
    assert "owner" in email.body.lower() or "next step" in email.body.lower()


def test_duplicate_rate_decreases_on_controlled_benchmark() -> None:
    request = build_request()
    baseline_rows = [
        _as_email_row(
            seed=seed,
            subject_body=_legacy_monolithic_render(seed=seed, request=request),
        )
        for seed in range(500, 516)
    ]
    compositional_rows = [
        _as_email_row(
            seed=seed,
            subject_body=(
                EmailStyleEngine(context=GeneratorContext(seed=seed)).render(request).subject,
                EmailStyleEngine(context=GeneratorContext(seed=seed)).render(request).body,
            ),
        )
        for seed in range(500, 516)
    ]

    evaluator = DatasetQualityEvaluator()
    baseline_report = evaluator.evaluate(
        {"email": baseline_rows, "slack": [], "teams": [], "salesforce": []}
    )
    compositional_report = evaluator.evaluate(
        {"email": compositional_rows, "slack": [], "teams": [], "salesforce": []}
    )

    assert compositional_report.duplicate_rate < baseline_report.duplicate_rate
    assert compositional_report.lexical_diversity > baseline_report.lexical_diversity


def test_generated_messages_still_remain_valid_for_email_source() -> None:
    context = GeneratorContext(
        seed=4042,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
        ),
    )
    enterprise = CompanyBuilder().build(context)
    renderer = EmailRenderer(context=context, enterprise=enterprise)
    event = enterprise.events[0]

    messages = renderer.generate_messages(event_id=event.id)

    assert messages
    assert all(EmailRecord.model_validate(message.to_dict()) == message for message in messages)


def test_deterministic_seeded_generation_still_holds() -> None:
    request = build_request(sender_role="Implementation Manager")
    first = EmailStyleEngine(context=GeneratorContext(seed=601)).render(request)
    second = EmailStyleEngine(context=GeneratorContext(seed=601)).render(request)
    third = EmailStyleEngine(context=GeneratorContext(seed=602)).render(request)

    assert first == second
    assert first != third


def _legacy_monolithic_render(
    *,
    seed: int,
    request: EmailStyleRequest,
) -> tuple[str, str]:
    greeting = f"Hi {request.recipient_name},"
    opener = (
        "Thanks again for the time today."
        if seed % 2 == 0
        else "Thanks again for the discussion today."
    )
    body = "\n".join(
        [
            greeting,
            "",
            opener,
            "",
            f"First, {request.prior_thread_summary}",
            f"Also, {request.context_lines[0]}",
            f"Finally, {request.action_ask}",
            request.disclaimer_line or "",
        ]
    ).strip()
    subject = f"Re: {request.base_subject}"
    return subject, body


def _as_email_row(
    *,
    seed: int,
    subject_body: tuple[str, str],
) -> dict[str, object]:
    subject, body = subject_body
    return {
        "source_system": "email",
        "email_id": f"email_benchmark_{seed}",
        "thread_id": f"thread_benchmark_{seed // 4}",
        "message_index_in_thread": seed % 4,
        "sender_employee_id": "employee_benchmark",
        "sender_contact_id": None,
        "to": ["contact_benchmark"],
        "cc": [],
        "bcc": [],
        "subject": subject,
        "body": body,
        "attachments": [],
        "account_id": None,
        "opportunity_id": None,
        "event_id": None,
        "ticket_id": None,
        "primary_category": "status_updates",
        "is_relevant": False,
        "relevance_reason": "benchmark row",
        "provenance": None,
    }
