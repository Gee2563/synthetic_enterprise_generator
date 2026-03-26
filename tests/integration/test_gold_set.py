from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from synthetic_enterprise.evaluation.gold_set import GoldSetBuilder, GoldSetConfig
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import DatasetTargets, GenerationPipeline
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory

REQUIRED_RELEVANT_CATEGORIES = (
    CommunicationCategory.BUYING_SIGNAL,
    CommunicationCategory.EVENT_ATTENDANCE,
    CommunicationCategory.FOLLOW_UP,
    CommunicationCategory.BLOCKER,
)


def build_dataset(tmp_path: Path) -> tuple[dict[str, list[dict[str, object]]], object]:
    context = GeneratorContext(
        seed=20260401,
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
        destination_root=tmp_path / "source-dataset",
    )
    rows_by_source = {
        "email": [record.to_dict() for record in dataset.email_records],
        "slack": [record.to_dict() for record in dataset.slack_records],
        "teams": [record.to_dict() for record in dataset.teams_records],
        "salesforce": [record.to_dict() for record in dataset.salesforce_records],
    }
    return rows_by_source, dataset.enterprise


def test_gold_set_contains_required_categories_and_stratified_sampling(
    tmp_path: Path,
) -> None:
    rows_by_source, enterprise = build_dataset(tmp_path)
    gold_set = GoldSetBuilder(enterprise=enterprise).create(
        rows_by_source=rows_by_source,
        config=GoldSetConfig(
            relevant_categories=REQUIRED_RELEVANT_CATEGORIES,
            samples_per_relevant_category=2,
            hard_negative_count=4,
            min_cross_system_examples=2,
        ),
    )

    relevant_counts = Counter(
        example.primary_category
        for example in gold_set.examples
        if example.is_relevant
    )

    assert len(gold_set.examples) == 12
    assert set(relevant_counts) == set(REQUIRED_RELEVANT_CATEGORIES)
    for category in REQUIRED_RELEVANT_CATEGORIES:
        assert relevant_counts[category] == 2
    assert sum(example.is_hard_negative for example in gold_set.examples) == 4


def test_gold_set_has_human_rationales_and_cross_system_examples(tmp_path: Path) -> None:
    rows_by_source, enterprise = build_dataset(tmp_path)
    gold_set = GoldSetBuilder(enterprise=enterprise).create(
        rows_by_source=rows_by_source,
        config=GoldSetConfig(
            relevant_categories=REQUIRED_RELEVANT_CATEGORIES,
            samples_per_relevant_category=1,
            hard_negative_count=2,
            min_cross_system_examples=2,
        ),
    )

    assert all(example.rationale.strip() for example in gold_set.examples if example.is_relevant)
    assert sum(example.cross_system_group_id is not None for example in gold_set.examples) >= 2


def test_gold_set_rejects_broken_references(tmp_path: Path) -> None:
    rows_by_source, enterprise = build_dataset(tmp_path)
    broken_row = next(
        row
        for row in rows_by_source["email"]
        if row["primary_category"] == CommunicationCategory.FOLLOW_UP.value
        and row["is_relevant"] is True
    ).copy()
    broken_row["account_id"] = "account_missing"

    with pytest.raises(ValueError, match="broken reference"):
        GoldSetBuilder(enterprise=enterprise).create(
            rows_by_source={"email": [broken_row]},
            config=GoldSetConfig(
                relevant_categories=(CommunicationCategory.FOLLOW_UP,),
                samples_per_relevant_category=1,
                hard_negative_count=0,
                min_cross_system_examples=0,
            ),
        )


def test_gold_set_exports_parquet_and_jsonl(tmp_path: Path) -> None:
    rows_by_source, enterprise = build_dataset(tmp_path)
    builder = GoldSetBuilder(enterprise=enterprise)
    gold_set = builder.create(
        rows_by_source=rows_by_source,
        config=GoldSetConfig(
            relevant_categories=REQUIRED_RELEVANT_CATEGORIES,
            samples_per_relevant_category=1,
            hard_negative_count=2,
            min_cross_system_examples=1,
        ),
    )

    export_paths = builder.export(gold_set=gold_set, destination_root=tmp_path / "gold")

    parquet_frame = pd.read_parquet(export_paths["parquet"])
    jsonl_lines = (tmp_path / "gold" / "gold_set.jsonl").read_text(encoding="utf-8").splitlines()

    assert export_paths["parquet"].exists()
    assert export_paths["jsonl"].exists()
    assert len(parquet_frame) == len(gold_set.examples)
    assert len(jsonl_lines) == len(gold_set.examples)
