from __future__ import annotations

from synthetic_enterprise.evaluation.quality_report import DatasetQualityEvaluator
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory
from synthetic_enterprise.sources.email.paraphrases import (
    EmailParaphraseBlock,
    SafeEmailParaphraser,
)
from synthetic_enterprise.sources.email.renderer import EmailRenderer


def test_paraphrase_variants_preserve_key_facts() -> None:
    paraphraser = SafeEmailParaphraser(context=GeneratorContext(seed=701))
    block = EmailParaphraseBlock(
        key="follow_up_fact",
        templates=(
            "Following {event_title} with {account_name}, please confirm owners by {deadline}.",
            "After {event_title} with {account_name}, please confirm owners by {deadline}.",
            "For {account_name}, please confirm owners by {deadline} after {event_title}.",
        ),
        facts={
            "event_title": "Quarterly Review",
            "account_name": "Northwind Health",
            "deadline": "Friday 14:00 UTC",
        },
    )

    variants = paraphraser.render_all(block)

    assert len(variants) == 3
    for variant in variants:
        assert "Quarterly Review" in variant
        assert "Northwind Health" in variant
        assert "Friday 14:00 UTC" in variant


def test_entity_mentions_are_not_lost_or_corrupted() -> None:
    paraphraser = SafeEmailParaphraser(context=GeneratorContext(seed=702))
    block = EmailParaphraseBlock(
        key="entity_preservation",
        templates=(
            "{contact_name} asked about {product_name}.",
            "Question from {contact_name} on {product_name}.",
        ),
        facts={
            "contact_name": "Jordan Lee",
            "product_name": "Signal Cloud workspace lane (Atlas)",
        },
    )

    rendered = paraphraser.render(block)

    assert "Jordan Lee" in rendered
    assert "Signal Cloud workspace lane (Atlas)" in rendered


def test_dates_and_times_remain_stable() -> None:
    paraphraser = SafeEmailParaphraser(context=GeneratorContext(seed=703))
    block = EmailParaphraseBlock(
        key="date_stability",
        templates=(
            "The review stays on {event_date}.",
            "Please keep the review on {event_date}.",
        ),
        facts={"event_date": "Tuesday 09:30 UTC"},
    )

    variants = paraphraser.render_all(block)

    assert all("Tuesday 09:30 UTC" in variant for variant in variants)


def test_paraphrasing_increases_unique_text_rate() -> None:
    context = GeneratorContext(seed=704)
    paraphraser = SafeEmailParaphraser(context=context)
    blocks = [
        EmailParaphraseBlock(
            key=f"benchmark_{index}",
            templates=(
                "Following {event_title}, please confirm owners by {deadline}.",
                "After {event_title}, please confirm owners by {deadline}.",
                "Please confirm owners by {deadline} after {event_title}.",
            ),
            facts={
                "event_title": "Quarterly Review",
                "deadline": "Friday 14:00 UTC",
            },
        )
        for index in range(12)
    ]

    baseline_rows = [
        _benchmark_row(
            index=index,
            body=(
                "Following Quarterly Review, "
                "please confirm owners by Friday 14:00 UTC."
            ),
        )
        for index in range(12)
    ]
    paraphrased_rows = [
        _benchmark_row(index=index, body=paraphraser.render(block))
        for index, block in enumerate(blocks)
    ]

    evaluator = DatasetQualityEvaluator()
    baseline_report = evaluator.evaluate(
        {"email": baseline_rows, "slack": [], "teams": [], "salesforce": []}
    )
    paraphrased_report = evaluator.evaluate(
        {"email": paraphrased_rows, "slack": [], "teams": [], "salesforce": []}
    )

    assert paraphrased_report.duplicate_rate < baseline_report.duplicate_rate
    assert paraphrased_report.lexical_diversity > baseline_report.lexical_diversity


def test_paraphrasing_does_not_alter_labels() -> None:
    context = GeneratorContext(
        seed=705,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
        ),
    )
    enterprise = CompanyBuilder().build(context)
    renderer = EmailRenderer(context=context, enterprise=enterprise)
    event = enterprise.events[0]

    relevant = renderer.render_from_event(
        event_id=event.id,
        primary_category=CommunicationCategory.FOLLOW_UP,
        message_index_in_thread=1,
    )
    noise = renderer.render_from_event(
        event_id=event.id,
        primary_category=CommunicationCategory.SCHEDULING_ONLY,
        message_index_in_thread=0,
    )

    assert relevant.primary_category == CommunicationCategory.FOLLOW_UP
    assert relevant.is_relevant is True
    assert noise.primary_category == CommunicationCategory.SCHEDULING_ONLY
    assert noise.is_relevant is False


def test_deterministic_generation_is_preserved() -> None:
    block = EmailParaphraseBlock(
        key="determinism",
        templates=(
            "Please confirm owners by {deadline}.",
            "Please send owners by {deadline}.",
            "Please share owners by {deadline}.",
        ),
        facts={"deadline": "Friday 14:00 UTC"},
    )

    first = SafeEmailParaphraser(context=GeneratorContext(seed=706)).render(block)
    second = SafeEmailParaphraser(context=GeneratorContext(seed=706)).render(block)
    third = SafeEmailParaphraser(context=GeneratorContext(seed=707)).render(block)

    assert first == second
    assert first != third


def _benchmark_row(*, index: int, body: str) -> dict[str, object]:
    return {
        "source_system": "email",
        "email_id": f"email_paraphrase_{index}",
        "thread_id": f"thread_paraphrase_{index}",
        "message_index_in_thread": 0,
        "sender_employee_id": "employee_benchmark",
        "sender_contact_id": None,
        "to": ["contact_benchmark"],
        "cc": [],
        "bcc": [],
        "subject": "Benchmark subject",
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
