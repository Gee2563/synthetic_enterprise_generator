from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, cast

from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.hard_negatives import (
    HardNegativeEngine,
    HardNegativeFamily,
)
from synthetic_enterprise.labeling.grounding import LabelProvenance, ProvenanceStrength
from synthetic_enterprise.sources.email.renderer import EmailRenderer
from synthetic_enterprise.sources.salesforce.renderer import SalesforceRenderer
from synthetic_enterprise.sources.slack.renderer import SlackRenderer
from synthetic_enterprise.sources.teams.renderer import TeamsRenderer


class HardNegativeRow(Protocol):
    is_relevant: bool
    provenance: LabelProvenance | None
    relevance_reason: str


def test_hard_negatives_contain_lexical_overlap_with_positives() -> None:
    rows_by_source = build_rows_by_source(hard_negative_ratio=1.0)

    for source_name, rows in rows_by_source.items():
        positives = [_text(row) for row in rows if row.is_relevant]
        hard_negatives = hard_negative_rows(rows)

        assert positives, source_name
        assert hard_negatives, source_name
        assert any(
            max(_token_overlap(_text(hard_negative), positive) for positive in positives) >= 2
            for hard_negative in hard_negatives
        ), source_name


def test_hard_negatives_remain_correctly_labeled_non_relevant() -> None:
    rows_by_source = build_rows_by_source(hard_negative_ratio=1.0)

    for source_name, rows in rows_by_source.items():
        hard_negatives = hard_negative_rows(rows)

        assert hard_negatives, source_name
        assert all(not row.is_relevant for row in hard_negatives), source_name


def test_hard_negative_proportion_is_configurable() -> None:
    low_ratio = build_rows_by_source(hard_negative_ratio=0.0)
    mid_ratio = build_rows_by_source(hard_negative_ratio=0.5)
    high_ratio = build_rows_by_source(hard_negative_ratio=1.0)

    for source_name in low_ratio:
        low_count = len(hard_negative_rows(low_ratio[source_name]))
        mid_count = len(hard_negative_rows(mid_ratio[source_name]))
        high_count = len(hard_negative_rows(high_ratio[source_name]))

        assert low_count == 0, source_name
        assert mid_count <= high_count, source_name
        assert high_count > 0, source_name


def test_hard_negative_distribution_is_configurable() -> None:
    context = GeneratorContext(
        seed=9191,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            hard_negative_family_weights={
                HardNegativeFamily.FOLLOW_UP_NO_ACTION.value: 4.0,
                HardNegativeFamily.BUDGET_NON_BUYING.value: 2.0,
                HardNegativeFamily.EVENT_NO_ATTENDANCE.value: 0.0,
            },
        ),
    )

    families = HardNegativeEngine(context).plan(count=12)
    counts = Counter(families)

    assert set(families) == {
        HardNegativeFamily.FOLLOW_UP_NO_ACTION,
        HardNegativeFamily.BUDGET_NON_BUYING,
    }
    assert counts[HardNegativeFamily.FOLLOW_UP_NO_ACTION] > counts[
        HardNegativeFamily.BUDGET_NON_BUYING
    ] > 0


def test_generator_does_not_collapse_into_repetitive_templates() -> None:
    rows_by_source = build_rows_by_source(hard_negative_ratio=1.0)

    for source_name, rows in rows_by_source.items():
        normalized_texts = {
            _normalize_text(_text(row))
            for row in hard_negative_rows(rows)
        }

        assert len(normalized_texts) >= 2, source_name


def test_rationale_explains_why_they_are_not_relevant() -> None:
    rows_by_source = build_rows_by_source(hard_negative_ratio=1.0)

    for source_name, rows in rows_by_source.items():
        hard_negatives = hard_negative_rows(rows)

        assert hard_negatives, source_name
        assert all(
            any(
                marker in row.relevance_reason.lower()
                for marker in (
                    "without",
                    "rather than",
                    "not evidence",
                    "no attendee",
                    "no required action",
                    "visibility only",
                    "not an engaged",
                    "not a live blocker",
                )
            )
            for row in hard_negatives
        ), source_name


def test_hard_negatives_appear_in_all_four_source_systems() -> None:
    rows_by_source = build_rows_by_source(hard_negative_ratio=1.0)

    assert set(rows_by_source) == {"email", "slack", "teams", "salesforce"}
    assert all(hard_negative_rows(rows) for rows in rows_by_source.values())


def test_benchmark_shows_increased_classification_difficulty() -> None:
    rows_by_source = build_rows_by_source(hard_negative_ratio=1.0)

    for source_name, rows in rows_by_source.items():
        positives = [_text(row) for row in rows if row.is_relevant]
        hard_negatives = hard_negative_rows(rows)
        generic_noise = generic_noise_rows(rows)

        assert positives, source_name
        assert hard_negatives, source_name
        assert generic_noise, source_name
        assert _average_max_overlap(hard_negatives, positives) > _average_max_overlap(
            generic_noise,
            positives,
        ), source_name


def build_rows_by_source(
    hard_negative_ratio: float,
    *,
    hard_negative_family_weights: dict[str, float] | None = None,
) -> dict[str, Sequence[HardNegativeRow]]:
    context = GeneratorContext(
        seed=9090,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_ratio=0.8,
            hard_negative_ratio=hard_negative_ratio,
            hard_negative_family_weights=hard_negative_family_weights,
        ),
    )
    enterprise = CompanyBuilder().build(context)
    event = enterprise.events[0]

    return {
        "email": cast(
            Sequence[HardNegativeRow],
            EmailRenderer(context=context, enterprise=enterprise).generate_messages(
                event_id=event.id
            ),
        ),
        "slack": cast(
            Sequence[HardNegativeRow],
            SlackRenderer(context=context, enterprise=enterprise).generate_messages(),
        ),
        "teams": cast(
            Sequence[HardNegativeRow],
            TeamsRenderer(context=context, enterprise=enterprise).generate_messages(),
        ),
        "salesforce": cast(
            Sequence[HardNegativeRow],
            SalesforceRenderer(
                context=context,
                enterprise=enterprise,
            ).generate_records(),
        ),
    }


def hard_negative_rows(rows: Sequence[HardNegativeRow]) -> list[HardNegativeRow]:
    return [
        row
        for row in rows
        if not row.is_relevant
        and row.provenance is not None
        and row.provenance.strength == ProvenanceStrength.WEAK
    ]


def generic_noise_rows(rows: Sequence[HardNegativeRow]) -> list[HardNegativeRow]:
    return [
        row
        for row in rows
        if not row.is_relevant and row.provenance is None
    ]


def _text(row: HardNegativeRow) -> str:
    parts = [
        getattr(row, "subject", None),
        getattr(row, "body", None),
        getattr(row, "text_body", None),
    ]
    return " ".join(part for part in parts if part is not None)


def _token_overlap(left: str, right: str) -> int:
    return len(_meaningful_tokens(left) & _meaningful_tokens(right))


def _meaningful_tokens(value: str) -> set[str]:
    stopwords = {
        "about",
        "after",
        "before",
        "there",
        "their",
        "still",
        "please",
        "only",
        "notes",
        "update",
        "customer",
        "account",
    }
    return {
        token
        for token in re.findall(r"[a-z]{4,}", value.lower())
        if token not in stopwords
    }


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _average_max_overlap(
    candidates: Sequence[HardNegativeRow],
    positives: Sequence[str],
) -> float:
    if not candidates or not positives:
        return 0.0

    overlaps = [
        max(_token_overlap(_text(candidate), positive) for positive in positives)
        for candidate in candidates
    ]
    return sum(overlaps) / len(overlaps)
