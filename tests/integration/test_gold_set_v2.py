from __future__ import annotations

from collections import Counter
from pathlib import Path

from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.evaluation.gold_set import GoldSetBuilder, GoldSetV2Config
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import (
    DatasetTargets,
    GenerationPipeline,
    Phase2BenchmarkTargets,
)
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory

REQUIRED_V2_CATEGORIES = (
    CommunicationCategory.BUYING_SIGNAL,
    CommunicationCategory.BLOCKER,
    CommunicationCategory.EVENT_ATTENDANCE,
    CommunicationCategory.DECISION_MAKER_SIGNAL,
)


def build_phase2_benchmark(
    tmp_path: Path,
) -> tuple[dict[str, list[dict[str, object]]], EnterpriseGraph]:
    pipeline = GenerationPipeline(
        context=GeneratorContext(
            seed=20260420,
            config=GeneratorConfig(
                company_size=CompanySizeConfig(min_employees=18, max_employees=18),
                noise_ratio=0.82,
                hard_negative_ratio=0.25,
                cross_system_ratio=0.5,
                messiness_rate=0.0,
            ),
        )
    )
    benchmark = pipeline.generate_phase2_benchmark(
        targets=Phase2BenchmarkTargets(
            company_count=3,
            per_company_targets=DatasetTargets(
                account_count=12,
                email_count=18,
                slack_count=18,
                teams_count=18,
                salesforce_count=18,
                chunk_size=9,
            ),
        ),
        destination_root=tmp_path / "phase2-benchmark",
    )
    return benchmark.rows_by_source, benchmark.enterprise


def test_gold_set_v2_includes_required_categories_and_hard_negatives(
    tmp_path: Path,
) -> None:
    rows_by_source, enterprise = build_phase2_benchmark(tmp_path)
    gold_set = GoldSetBuilder(enterprise=enterprise).create_v2(
        rows_by_source=rows_by_source,
        config=GoldSetV2Config(
            relevant_categories=REQUIRED_V2_CATEGORIES,
            samples_per_relevant_category=1,
            hard_negative_count=4,
            min_cross_system_examples=3,
            min_sources=4,
            min_companies=3,
            min_messy_examples=2,
            min_event_attendance_edge_cases=1,
        ),
    )

    relevant_counts = Counter(
        example.primary_category
        for example in gold_set.examples
        if example.is_relevant
    )

    assert set(relevant_counts) == set(REQUIRED_V2_CATEGORIES)
    assert all(relevant_counts[category] == 1 for category in REQUIRED_V2_CATEGORIES)
    assert sum(example.is_hard_negative for example in gold_set.examples) >= 4


def test_gold_set_v2_rationales_are_present_and_coherent(tmp_path: Path) -> None:
    rows_by_source, enterprise = build_phase2_benchmark(tmp_path)
    gold_set = GoldSetBuilder(enterprise=enterprise).create_v2(
        rows_by_source=rows_by_source,
        config=GoldSetV2Config(
            relevant_categories=REQUIRED_V2_CATEGORIES,
            samples_per_relevant_category=1,
            hard_negative_count=4,
            min_cross_system_examples=3,
            min_sources=4,
            min_companies=3,
            min_messy_examples=2,
            min_event_attendance_edge_cases=1,
        ),
    )

    assert all(example.rationale.strip() for example in gold_set.examples)
    assert all("because" in example.rationale.lower() for example in gold_set.examples)
    assert all(
        example.rationale.startswith("Relevant because")
        for example in gold_set.examples
        if example.is_relevant
    )
    assert all(
        example.rationale.startswith("Hard negative because")
        for example in gold_set.examples
        if example.is_hard_negative
    )


def test_gold_set_v2_enforces_source_and_company_diversity_and_no_broken_links(
    tmp_path: Path,
) -> None:
    rows_by_source, enterprise = build_phase2_benchmark(tmp_path)
    gold_set = GoldSetBuilder(enterprise=enterprise).create_v2(
        rows_by_source=rows_by_source,
        config=GoldSetV2Config(
            relevant_categories=REQUIRED_V2_CATEGORIES,
            samples_per_relevant_category=1,
            hard_negative_count=4,
            min_cross_system_examples=3,
            min_sources=4,
            min_companies=3,
            min_messy_examples=2,
            min_event_attendance_edge_cases=1,
        ),
    )

    assert {example.source_system for example in gold_set.examples} == {
        "email",
        "slack",
        "teams",
        "salesforce",
    }
    assert len({str(example.row_data["company_id"]) for example in gold_set.examples}) == 3
    assert sum(example.cross_system_group_id is not None for example in gold_set.examples) >= 3


def test_gold_set_v2_captures_messy_and_event_attendance_edge_cases(
    tmp_path: Path,
) -> None:
    rows_by_source, enterprise = build_phase2_benchmark(tmp_path)
    gold_set = GoldSetBuilder(enterprise=enterprise).create_v2(
        rows_by_source=rows_by_source,
        config=GoldSetV2Config(
            relevant_categories=REQUIRED_V2_CATEGORIES,
            samples_per_relevant_category=1,
            hard_negative_count=4,
            min_cross_system_examples=3,
            min_sources=4,
            min_companies=3,
            min_messy_examples=2,
            min_event_attendance_edge_cases=1,
        ),
    )

    messy_examples = [
        example
        for example in gold_set.examples
        if example.source_system == "salesforce"
        and isinstance(example.row_data.get("structured_fields"), dict)
        and any(
            key in example.row_data["structured_fields"]
            for key in (
                "missing_fields",
                "late_entry_days",
                "contradicts_record_id",
                "stage_sync_delay_days",
                "stale_owner_snapshot_days",
                "attendance_reconciliation_pending",
                "follow_up_capture_state",
            )
        )
    ]
    attendance_edge_cases = [
        example
        for example in gold_set.examples
        if example.primary_category == CommunicationCategory.EVENT_ATTENDANCE
        and any(
            token in _gold_text(example).lower()
            for token in ("missed", "no show", "tentative", "joined")
        )
    ]

    assert len(messy_examples) >= 2
    assert attendance_edge_cases


def _gold_text(example: object) -> str:
    row_data = getattr(example, "row_data")
    parts = [
        row_data.get("subject"),
        row_data.get("body"),
        row_data.get("text_body"),
        row_data.get("relevance_reason"),
    ]
    return " ".join(str(part) for part in parts if isinstance(part, str))
