from __future__ import annotations

from pathlib import Path

import pandas as pd

from synthetic_enterprise.evaluation.gold_set import GoldSetBuilder, GoldSetConfig
from synthetic_enterprise.evaluation.quality_report import DatasetQualityEvaluator
from synthetic_enterprise.evaluation.regression import (
    PHASE1_REGRESSION_TARGETS,
    build_regression_context,
    build_regression_fixture,
    build_regression_snapshot,
    collect_reference_errors,
)
from synthetic_enterprise.evaluation.training_formats import TrainingFormatExporter
from synthetic_enterprise.generation.pipeline import GenerationPipeline
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory

GOLD_RELEVANT_CATEGORIES = (
    CommunicationCategory.BUYING_SIGNAL,
    CommunicationCategory.EVENT_ATTENDANCE,
    CommunicationCategory.BLOCKER,
)


def test_small_regression_fixture_snapshots_are_deterministic(tmp_path: Path) -> None:
    fixture_a = build_regression_fixture(
        destination_root=tmp_path / "run-a",
        seed=20260430,
    )
    fixture_b = build_regression_fixture(
        destination_root=tmp_path / "run-b",
        seed=20260430,
    )
    fixture_c = build_regression_fixture(
        destination_root=tmp_path / "run-c",
        seed=20260431,
    )

    snapshot_a = build_regression_snapshot(fixture_a.rows_by_source)
    snapshot_b = build_regression_snapshot(fixture_b.rows_by_source)
    snapshot_c = build_regression_snapshot(fixture_c.rows_by_source)

    assert snapshot_a.to_dict() == snapshot_b.to_dict()
    assert snapshot_a.to_dict() != snapshot_c.to_dict()
    assert snapshot_a.row_counts == {
        "email": PHASE1_REGRESSION_TARGETS.email_count,
        "slack": PHASE1_REGRESSION_TARGETS.slack_count,
        "teams": PHASE1_REGRESSION_TARGETS.teams_count,
        "salesforce": PHASE1_REGRESSION_TARGETS.salesforce_count,
    }
    assert snapshot_a.total_rows == sum(snapshot_a.row_counts.values())


def test_relevant_rows_remain_grounded_and_entity_references_are_valid(
    tmp_path: Path,
) -> None:
    fixture = build_regression_fixture(
        destination_root=tmp_path / "fixture",
        seed=20260430,
    )
    all_rows = [
        row
        for rows in fixture.rows_by_source.values()
        for row in rows
    ]
    relevant_rows = [row for row in all_rows if row["is_relevant"] is True]

    assert relevant_rows
    assert all(row["provenance"] is not None for row in relevant_rows)
    assert collect_reference_errors(
        enterprise=fixture.dataset.enterprise,
        rows_by_source=fixture.rows_by_source,
    ) == []


def test_regression_exports_remain_readable_and_quality_report_builds(
    tmp_path: Path,
) -> None:
    fixture = build_regression_fixture(
        destination_root=tmp_path / "fixture",
        seed=20260430,
    )

    for source_name, expected_count in build_regression_snapshot(
        fixture.rows_by_source
    ).row_counts.items():
        parquet_root = tmp_path / "fixture" / f"source={source_name}" / "parquet"
        frame = pd.concat(
            [
                pd.read_parquet(parquet_file)
                for parquet_file in sorted(parquet_root.glob("*.parquet"))
            ],
            ignore_index=True,
        )

        assert len(frame) == expected_count
        assert frame["source_system"].eq(source_name).all()

    report = DatasetQualityEvaluator(
        enterprise=fixture.dataset.enterprise,
        context=fixture.context,
    ).evaluate(fixture.rows_by_source)
    snapshot = build_regression_snapshot(fixture.rows_by_source)

    assert report.total_rows == sum(snapshot.row_counts.values())
    assert 0.0 <= report.relevance_rate <= 1.0
    assert report.source_row_counts["email"] == PHASE1_REGRESSION_TARGETS.email_count


def test_chunked_generation_regression_counts_and_schemas_hold(
    tmp_path: Path,
) -> None:
    context = build_regression_context(seed=20260430)
    pipeline = GenerationPipeline(context=context)

    in_memory = pipeline.generate_dataset(
        targets=PHASE1_REGRESSION_TARGETS,
        destination_root=tmp_path / "in-memory",
    )
    streamed = pipeline.generate_dataset_streaming(
        targets=PHASE1_REGRESSION_TARGETS,
        destination_root=tmp_path / "streamed",
        generation_chunk_size=PHASE1_REGRESSION_TARGETS.chunk_size or 20,
    )

    for source_name in ("email", "slack", "teams", "salesforce"):
        expected_count = PHASE1_REGRESSION_TARGETS.rows_for_source(
            source_name
        )
        assert in_memory.manifests[source_name].row_count == expected_count
        assert streamed.manifests[source_name].row_count == expected_count
        assert (
            in_memory.manifests[source_name].columns
            == streamed.manifests[source_name].columns
        )


def test_training_format_exports_remain_valid_on_regression_fixture(
    tmp_path: Path,
) -> None:
    fixture = build_regression_fixture(
        destination_root=tmp_path / "fixture",
        seed=20260430,
    )
    gold_set = GoldSetBuilder(enterprise=fixture.dataset.enterprise).create(
        rows_by_source=fixture.rows_by_source,
        config=GoldSetConfig(
            relevant_categories=GOLD_RELEVANT_CATEGORIES,
            samples_per_relevant_category=1,
            hard_negative_count=2,
            min_cross_system_examples=1,
        ),
    )
    exporter = TrainingFormatExporter(enterprise=fixture.dataset.enterprise)

    row_dataset = exporter.build_row_classification(rows_by_source=fixture.rows_by_source)
    message_dataset = exporter.build_message_classification(rows_by_source=fixture.rows_by_source)
    rationale_dataset = exporter.build_rationale_extraction(gold_set=gold_set)
    grouped_dataset = exporter.build_grouped_thread_classification(
        rows_by_source=fixture.rows_by_source
    )
    linking_dataset = exporter.build_cross_document_linking(
        rows_by_source=fixture.rows_by_source
    )

    assert row_dataset.samples
    assert message_dataset.samples
    assert rationale_dataset.samples
    assert grouped_dataset.samples
    assert linking_dataset.samples
