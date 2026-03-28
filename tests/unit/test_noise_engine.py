from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.noise_engine import NoiseEngine, NoiseFamily
from synthetic_enterprise.labeling.grounding import LabelProvenance, ProvenanceStrength
from synthetic_enterprise.sources.email.renderer import EmailRenderer
from synthetic_enterprise.sources.salesforce.renderer import SalesforceRenderer
from synthetic_enterprise.sources.slack.renderer import SlackRenderer
from synthetic_enterprise.sources.teams.renderer import TeamsRenderer


class NoiseRow(Protocol):
    is_relevant: bool
    provenance: LabelProvenance | None
    primary_category: object


def test_noise_family_distribution_is_configurable() -> None:
    context = GeneratorContext(
        seed=8181,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_family_weights={
                NoiseFamily.SOCIAL_BANTER.value: 5.0,
                NoiseFamily.ACCESS_REQUEST.value: 2.0,
                NoiseFamily.DOCUMENT_REVIEW_PING.value: 0.0,
            },
        ),
    )

    families = NoiseEngine(context).plan(source_system="slack", count=12)
    counts = Counter(families)

    assert set(families) == {
        NoiseFamily.SOCIAL_BANTER,
        NoiseFamily.ACCESS_REQUEST,
    }
    assert counts[NoiseFamily.SOCIAL_BANTER] > counts[NoiseFamily.ACCESS_REQUEST] > 0


def test_outputs_from_different_families_are_distinguishable() -> None:
    rows_by_source = build_rows_by_source(noise_ratio=1.0, hard_negative_ratio=0.0)

    for source_name, rows in rows_by_source.items():
        generic_noise = generic_noise_rows(rows)
        normalized_texts = {_normalize_text(_text(row)) for row in generic_noise}
        categories = {str(row.primary_category) for row in generic_noise}

        assert len(generic_noise) >= 4, source_name
        assert len(normalized_texts) >= 4, source_name
        assert len(categories) >= 3, source_name


def test_overall_relevance_rate_remains_in_target_range() -> None:
    rows_by_source = build_rows_by_source(
        noise_ratio=0.85,
        hard_negative_ratio=0.25,
    )
    all_rows = [row for rows in rows_by_source.values() for row in rows]
    relevance_rate = sum(row.is_relevant for row in all_rows) / len(all_rows)

    assert 0.05 <= relevance_rate <= 0.35


def test_noise_messages_do_not_accidentally_create_false_provenance() -> None:
    rows_by_source = build_rows_by_source(noise_ratio=1.0, hard_negative_ratio=0.0)

    for source_name, rows in rows_by_source.items():
        generic_noise = generic_noise_rows(rows)

        assert generic_noise, source_name
        assert all(row.provenance is None for row in generic_noise), source_name


def test_lexical_overlap_with_relevant_examples_exists_for_some_hard_negatives() -> None:
    rows_by_source = build_rows_by_source(noise_ratio=0.85, hard_negative_ratio=1.0)

    for source_name, rows in rows_by_source.items():
        positives = [_text(row) for row in rows if row.is_relevant]
        hard_negatives = hard_negative_rows(rows)

        assert positives, source_name
        assert hard_negatives, source_name
        assert any(
            max(_token_overlap(_text(candidate), positive) for positive in positives) >= 2
            for candidate in hard_negatives
        ), source_name


def test_noise_generation_remains_deterministic() -> None:
    first = build_rows_by_source(noise_ratio=1.0, hard_negative_ratio=0.0, seed=9898)
    second = build_rows_by_source(noise_ratio=1.0, hard_negative_ratio=0.0, seed=9898)

    assert row_fingerprints(first) == row_fingerprints(second)


def build_rows_by_source(
    *,
    noise_ratio: float,
    hard_negative_ratio: float,
    seed: int = 9191,
    noise_family_weights: Mapping[str, float] | None = None,
) -> dict[str, Sequence[NoiseRow]]:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_ratio=noise_ratio,
            hard_negative_ratio=hard_negative_ratio,
            noise_family_weights=noise_family_weights,
        ),
    )
    enterprise = CompanyBuilder().build(context)
    event = enterprise.events[0]

    return {
        "email": cast(
            Sequence[NoiseRow],
            EmailRenderer(context=context, enterprise=enterprise).generate_messages(
                event_id=event.id
            ),
        ),
        "slack": cast(
            Sequence[NoiseRow],
            SlackRenderer(context=context, enterprise=enterprise).generate_messages(),
        ),
        "teams": cast(
            Sequence[NoiseRow],
            TeamsRenderer(context=context, enterprise=enterprise).generate_messages(),
        ),
        "salesforce": cast(
            Sequence[NoiseRow],
            SalesforceRenderer(
                context=context,
                enterprise=enterprise,
            ).generate_records(),
        ),
    }


def generic_noise_rows(rows: Sequence[NoiseRow]) -> list[NoiseRow]:
    return [
        row
        for row in rows
        if not row.is_relevant and row.provenance is None
    ]


def hard_negative_rows(rows: Sequence[NoiseRow]) -> list[NoiseRow]:
    return [
        row
        for row in rows
        if not row.is_relevant
        and row.provenance is not None
        and row.provenance.strength == ProvenanceStrength.WEAK
    ]


def row_fingerprints(rows_by_source: Mapping[str, Sequence[NoiseRow]]) -> dict[str, list[str]]:
    fingerprints: dict[str, list[str]] = {}
    for source_name, rows in rows_by_source.items():
        fingerprints[source_name] = [
            "|".join(
                [
                    getattr(row, "subject", None) or "",
                    getattr(row, "body", None) or "",
                    getattr(row, "text_body", None) or "",
                    str(getattr(row, "primary_category")),
                    str(getattr(row, "provenance", None)),
                ]
            )
            for row in rows
            if not row.is_relevant
        ]
    return fingerprints


def _text(row: NoiseRow) -> str:
    return " ".join(
        part
        for part in (
            getattr(row, "subject", None),
            getattr(row, "body", None),
            getattr(row, "text_body", None),
        )
        if part is not None
    )


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
