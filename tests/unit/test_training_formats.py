from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.evaluation.gold_set import GoldSet, GoldSetBuilder, GoldSetConfig
from synthetic_enterprise.evaluation.training_formats import (
    TrainingFormatExporter,
)
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import DatasetTargets, GenerationPipeline
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory

GOLD_RELEVANT_CATEGORIES = (
    CommunicationCategory.BUYING_SIGNAL,
    CommunicationCategory.EVENT_ATTENDANCE,
    CommunicationCategory.BLOCKER,
)


def build_training_inputs(
    tmp_path: Path,
) -> tuple[dict[str, list[dict[str, object]]], EnterpriseGraph, GoldSet]:
    context = GeneratorContext(
        seed=20260402,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=30, max_employees=30),
            noise_ratio=0.85,
            hard_negative_ratio=0.25,
            cross_system_ratio=0.5,
        ),
    )
    dataset = GenerationPipeline(context=context).generate_dataset(
        targets=DatasetTargets(
            account_count=20,
            email_count=120,
            slack_count=120,
            teams_count=120,
            salesforce_count=120,
            chunk_size=40,
        ),
        destination_root=tmp_path / "training-source",
    )
    rows_by_source = {
        "email": [record.to_dict() for record in dataset.email_records],
        "slack": [record.to_dict() for record in dataset.slack_records],
        "teams": [record.to_dict() for record in dataset.teams_records],
        "salesforce": [record.to_dict() for record in dataset.salesforce_records],
    }
    gold_set = GoldSetBuilder(enterprise=dataset.enterprise).create(
        rows_by_source=rows_by_source,
        config=GoldSetConfig(
            relevant_categories=GOLD_RELEVANT_CATEGORIES,
            samples_per_relevant_category=1,
            hard_negative_count=2,
            min_cross_system_examples=1,
        ),
    )
    return rows_by_source, dataset.enterprise, gold_set


def test_training_format_output_schemas_are_valid(tmp_path: Path) -> None:
    rows_by_source, enterprise, gold_set = build_training_inputs(tmp_path)
    exporter = TrainingFormatExporter(enterprise=enterprise)

    row_dataset = exporter.build_row_classification(rows_by_source=rows_by_source)
    message_dataset = exporter.build_message_classification(rows_by_source=rows_by_source)
    rationale_dataset = exporter.build_rationale_extraction(gold_set=gold_set)
    grouped_dataset = exporter.build_grouped_thread_classification(rows_by_source=rows_by_source)
    linking_dataset = exporter.build_cross_document_linking(rows_by_source=rows_by_source)

    assert row_dataset.samples
    assert message_dataset.samples
    assert rationale_dataset.samples
    assert grouped_dataset.samples
    assert linking_dataset.samples
    assert "samples" in row_dataset.to_dict()
    assert "samples" in message_dataset.to_dict()
    assert "samples" in rationale_dataset.to_dict()
    assert "samples" in grouped_dataset.to_dict()
    assert "samples" in linking_dataset.to_dict()


def test_labels_map_correctly_from_provenance(tmp_path: Path) -> None:
    rows_by_source, enterprise, _ = build_training_inputs(tmp_path)
    exporter = TrainingFormatExporter(enterprise=enterprise)
    row_dataset = exporter.build_row_classification(
        rows_by_source=rows_by_source,
        category_filter={CommunicationCategory.BLOCKER},
    )

    sample = next(sample for sample in row_dataset.samples if sample.is_relevant)
    source_row = next(
        row
        for rows in rows_by_source.values()
        for row in rows
        if exporter.source_row_id(row) == sample.source_row_id
    )
    provenance = cast(dict[str, Any], source_row["provenance"])

    assert sample.label == sample.primary_category.value
    assert provenance is not None
    assert sample.target_object_type == provenance["object_type"]
    assert sample.target_object_id == provenance["object_id"]


def test_grouped_thread_builder_preserves_ordering(tmp_path: Path) -> None:
    rows_by_source, enterprise, _ = build_training_inputs(tmp_path)
    exporter = TrainingFormatExporter(enterprise=enterprise)
    grouped_dataset = exporter.build_grouped_thread_classification(
        rows_by_source=rows_by_source,
        source_filter={"email"},
    )

    sample = next(sample for sample in grouped_dataset.samples if len(sample.messages) > 1)
    ordered_positions = [message.message_order for message in sample.messages]
    source_rows = sorted(
        (
            row
            for row in rows_by_source["email"]
            if row["thread_id"] == sample.thread_id
        ),
        key=lambda row: int(cast(int | str, row["message_index_in_thread"])),
    )

    assert ordered_positions == sorted(ordered_positions)
    assert [message.source_row_id for message in sample.messages] == [
        row["email_id"] for row in source_rows
    ]


def test_training_samples_can_be_filtered_by_source_and_category(tmp_path: Path) -> None:
    rows_by_source, enterprise, _ = build_training_inputs(tmp_path)
    exporter = TrainingFormatExporter(enterprise=enterprise)
    dataset = exporter.build_message_classification(
        rows_by_source=rows_by_source,
        source_filter={"slack"},
        category_filter={CommunicationCategory.BLOCKER},
    )

    assert dataset.samples
    assert {sample.source_system for sample in dataset.samples} == {"slack"}
    assert {sample.primary_category for sample in dataset.samples} == {
        CommunicationCategory.BLOCKER
    }


def test_rationales_are_present_when_required(tmp_path: Path) -> None:
    _, enterprise, gold_set = build_training_inputs(tmp_path)
    exporter = TrainingFormatExporter(enterprise=enterprise)
    dataset = exporter.build_rationale_extraction(gold_set=gold_set, require_rationale=True)

    assert dataset.samples
    assert all(sample.rationale.strip() for sample in dataset.samples)

    broken_examples = [
        gold_set.examples[0].model_copy(update={"rationale": ""}),
        *gold_set.examples[1:],
    ]
    broken_gold_set = GoldSet(config=gold_set.config, examples=broken_examples)

    with pytest.raises(ValueError, match="rationale"):
        exporter.build_rationale_extraction(
            gold_set=broken_gold_set,
            require_rationale=True,
        )
