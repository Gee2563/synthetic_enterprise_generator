from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.grounding import LabelProvenance, ProvenanceStrength
from synthetic_enterprise.sources.email.renderer import EmailRenderer
from synthetic_enterprise.sources.salesforce.renderer import SalesforceRenderer
from synthetic_enterprise.sources.slack.renderer import SlackRenderer
from synthetic_enterprise.sources.teams.renderer import TeamsRenderer


class HardNegativeRow(Protocol):
    is_relevant: bool
    provenance: LabelProvenance | None


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


def test_generator_does_not_collapse_into_repetitive_templates() -> None:
    rows_by_source = build_rows_by_source(hard_negative_ratio=1.0)

    for source_name, rows in rows_by_source.items():
        normalized_texts = {
            _normalize_text(_text(row))
            for row in hard_negative_rows(rows)
        }

        assert len(normalized_texts) >= 2, source_name


def build_rows_by_source(hard_negative_ratio: float) -> dict[str, Sequence[HardNegativeRow]]:
    context = GeneratorContext(
        seed=9090,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_ratio=0.8,
            hard_negative_ratio=hard_negative_ratio,
        ),
    )
    enterprise = CompanyBuilder().build(context)
    event = enterprise.events[0]

    return {
        "email": EmailRenderer(context=context, enterprise=enterprise).generate_messages(
            event_id=event.id
        ),
        "slack": SlackRenderer(context=context, enterprise=enterprise).generate_messages(),
        "teams": TeamsRenderer(context=context, enterprise=enterprise).generate_messages(),
        "salesforce": SalesforceRenderer(
            context=context,
            enterprise=enterprise,
        ).generate_records(),
    }


def hard_negative_rows(rows: Sequence[HardNegativeRow]) -> list[HardNegativeRow]:
    return [
        row
        for row in rows
        if not row.is_relevant
        and row.provenance is not None
        and row.provenance.strength == ProvenanceStrength.WEAK
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
