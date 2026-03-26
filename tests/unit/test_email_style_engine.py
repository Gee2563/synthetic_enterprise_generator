from __future__ import annotations

import re

from synthetic_enterprise.generation.context import GeneratorContext
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
        thread_id="thread_001",
        message_index_in_thread=0,
        base_subject="Platform review next steps",
        sender_name="Avery Stone",
        sender_role="Customer Success Manager",
        recipient_name="Jordan",
        subject_mode=EmailSubjectMode.NEW,
        audience=EmailAudience.INTERNAL,
        seniority=EmailSeniority.IC,
        detail_level=EmailDetailLevel.DETAILED,
        include_signature=True,
        cc=("employee_002",),
        bcc=("employee_003",),
        context_lines=(
            "We need owners for the migration plan.",
            "The customer asked for a timing update by Friday.",
        ),
    )
    return request.model_copy(update=overrides)


def test_generated_subjects_follow_allowed_patterns() -> None:
    engine = EmailStyleEngine(context=GeneratorContext(seed=7))

    new_email = engine.render(build_request(subject_mode=EmailSubjectMode.NEW))
    reply_email = engine.render(
        build_request(subject_mode=EmailSubjectMode.REPLY, message_index_in_thread=1)
    )
    forward_email = engine.render(
        build_request(subject_mode=EmailSubjectMode.FORWARD, message_index_in_thread=2)
    )

    assert re.match(r"^(Re: |FW: )?.+", new_email.subject)
    assert re.match(r"^(Re: |FW: )?.+", reply_email.subject)
    assert re.match(r"^(Re: |FW: )?.+", forward_email.subject)
    assert not new_email.subject.startswith(("Re: ", "FW: "))
    assert reply_email.subject.startswith("Re: ")
    assert forward_email.subject.startswith("FW: ")


def test_signatures_can_be_toggled_on_off() -> None:
    engine = EmailStyleEngine(context=GeneratorContext(seed=8))

    with_signature = engine.render(build_request(include_signature=True))
    without_signature = engine.render(build_request(include_signature=False))

    assert with_signature.signature_block is not None
    assert "Avery Stone" in with_signature.body
    assert without_signature.signature_block is None
    assert "Avery Stone" not in without_signature.body


def test_tone_varies_by_persona_and_role() -> None:
    executive_engine = EmailStyleEngine(context=GeneratorContext(seed=9))
    internal_exec = executive_engine.render(
        build_request(
            sender_role="Chief Executive Officer",
            seniority=EmailSeniority.EXECUTIVE,
            audience=EmailAudience.INTERNAL,
            detail_level=EmailDetailLevel.SHORT,
        )
    )
    internal_ic = executive_engine.render(
        build_request(
            sender_role="Support Engineer",
            seniority=EmailSeniority.IC,
            audience=EmailAudience.INTERNAL,
            detail_level=EmailDetailLevel.DETAILED,
        )
    )
    external_customer = executive_engine.render(
        build_request(
            audience=EmailAudience.EXTERNAL_CUSTOMER,
            recipient_name="Jordan",
            detail_level=EmailDetailLevel.DETAILED,
        )
    )

    assert internal_exec.body != internal_ic.body
    assert internal_ic.body != external_customer.body
    assert "At a high level" in internal_exec.body
    assert "details below" in internal_ic.body
    assert "Thanks again" in external_customer.body


def test_reply_chains_preserve_thread_metadata() -> None:
    engine = EmailStyleEngine(context=GeneratorContext(seed=10))

    first = engine.render(build_request(message_index_in_thread=0))
    reply = engine.render(
        build_request(
            message_index_in_thread=1,
            subject_mode=EmailSubjectMode.REPLY,
        )
    )

    assert first.thread_id == reply.thread_id == "thread_001"
    assert first.message_index_in_thread == 0
    assert reply.message_index_in_thread == 1
    assert reply.subject.startswith("Re: ")


def test_body_text_is_not_identical_across_repeated_generations_with_different_seeds() -> None:
    first = EmailStyleEngine(context=GeneratorContext(seed=11)).render(build_request())
    second = EmailStyleEngine(context=GeneratorContext(seed=12)).render(build_request())

    assert first.body != second.body


def test_same_seed_yields_deterministic_output() -> None:
    request = build_request(
        subject_mode=EmailSubjectMode.REPLY,
        message_index_in_thread=2,
        audience=EmailAudience.EXTERNAL_CUSTOMER,
    )
    first = EmailStyleEngine(context=GeneratorContext(seed=13)).render(request)
    second = EmailStyleEngine(context=GeneratorContext(seed=13)).render(request)

    assert first == second


def test_cc_bcc_and_detail_level_are_supported() -> None:
    engine = EmailStyleEngine(context=GeneratorContext(seed=14))
    short_email = engine.render(build_request(detail_level=EmailDetailLevel.SHORT))
    detailed_email = engine.render(build_request(detail_level=EmailDetailLevel.DETAILED))

    assert short_email.cc == ("employee_002",)
    assert short_email.bcc == ("employee_003",)
    assert len(short_email.body) < len(detailed_email.body)
