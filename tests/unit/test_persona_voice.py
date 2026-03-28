from __future__ import annotations

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
        thread_id="thread_persona",
        message_index_in_thread=0,
        base_subject="Next steps after account review",
        sender_name="Avery Stone",
        sender_role="Customer Success Manager",
        recipient_name="Jordan",
        subject_mode=EmailSubjectMode.NEW,
        audience=EmailAudience.INTERNAL,
        seniority=EmailSeniority.IC,
        detail_level=EmailDetailLevel.DETAILED,
        include_signature=False,
        context_lines=(
            "We need owners for the migration plan.",
            "The customer asked for a timing update by Friday.",
        ),
    )
    return request.model_copy(update=overrides)


def test_role_persona_affects_output_style() -> None:
    engine = EmailStyleEngine(context=GeneratorContext(seed=51))

    ae = engine.render(build_request(sender_role="Account Executive"))
    product_manager = engine.render(build_request(sender_role="Product Manager"))
    implementation_manager = engine.render(build_request(sender_role="Implementation Manager"))

    assert "Commercial:" in ae.body
    assert "Roadmap:" in product_manager.body
    assert "Workstream:" in implementation_manager.body


def test_internal_vs_external_communication_changes_tone() -> None:
    engine = EmailStyleEngine(context=GeneratorContext(seed=52))

    internal_message = engine.render(
        build_request(
            sender_role="Customer Success Manager",
            audience=EmailAudience.INTERNAL,
        )
    )
    external_message = engine.render(
        build_request(
            sender_role="Customer Success Manager",
            audience=EmailAudience.EXTERNAL_CUSTOMER,
        )
    )

    assert "Hi team," in internal_message.body or "Hi all," in internal_message.body
    assert "Hi Jordan," in external_message.body or "Hello Jordan," in external_message.body
    assert "team" not in external_message.body.lower().splitlines()[0]


def test_executive_emails_differ_from_ic_messages() -> None:
    engine = EmailStyleEngine(context=GeneratorContext(seed=53))

    executive = engine.render(
        build_request(
            sender_role="VP Customer Success",
            seniority=EmailSeniority.EXECUTIVE,
        )
    )
    ic = engine.render(
        build_request(
            sender_role="Support Engineer",
            seniority=EmailSeniority.IC,
        )
    )

    assert "Summary:" in executive.body
    assert "At a high level" in executive.body
    assert "Log:" in ic.body
    assert "At a high level" not in ic.body


def test_sales_personas_differ_from_support_personas() -> None:
    engine = EmailStyleEngine(context=GeneratorContext(seed=54))

    sales = engine.render(
        build_request(
            sender_role="Account Executive",
            audience=EmailAudience.EXTERNAL_CUSTOMER,
        )
    )
    support = engine.render(
        build_request(
            sender_role="Support Engineer",
            audience=EmailAudience.EXTERNAL_CUSTOMER,
        )
    )

    assert "commercial" in sales.body.lower()
    assert "frustrating" in support.body.lower()
    assert "log" in support.body.lower() or "fix" in support.body.lower()


def test_persona_variation_does_not_break_determinism() -> None:
    request = build_request(sender_role="Product Manager", audience=EmailAudience.INTERNAL)
    first = EmailStyleEngine(context=GeneratorContext(seed=55)).render(request)
    second = EmailStyleEngine(context=GeneratorContext(seed=55)).render(request)
    third = EmailStyleEngine(context=GeneratorContext(seed=56)).render(request)

    assert first == second
    assert first != third


def test_persona_styles_do_not_collapse_into_repeated_phrases() -> None:
    engine = EmailStyleEngine(context=GeneratorContext(seed=57))
    roles = (
        "Account Executive",
        "Customer Success Manager",
        "Support Engineer",
        "Product Manager",
        "Implementation Manager",
        "Operations Lead",
    )

    bodies = [
        engine.render(build_request(sender_role=role, include_signature=False)).body
        for role in roles
    ]
    first_detail_lines = {_first_detail_line(body) for body in bodies}

    assert len(set(bodies)) == len(roles)
    assert len(first_detail_lines) >= 5


def _first_detail_line(body: str) -> str:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return lines[2]
