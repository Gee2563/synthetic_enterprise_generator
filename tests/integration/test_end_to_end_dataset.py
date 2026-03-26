from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import pandas as pd

from synthetic_enterprise.contracts.records.email import EmailRecord
from synthetic_enterprise.contracts.records.salesforce import SalesforceRecord
from synthetic_enterprise.contracts.records.slack import SlackRecord
from synthetic_enterprise.contracts.records.teams import TeamsRecord
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import DatasetTargets, GenerationPipeline

DatasetRow: TypeAlias = EmailRecord | SlackRecord | TeamsRecord | SalesforceRecord


def _parquet_row_count(parquet_root: Path) -> int:
    return sum(
        len(pd.read_parquet(parquet_file))
        for parquet_file in sorted(parquet_root.glob("*.parquet"))
    )


def test_generation_pipeline_builds_small_end_to_end_dataset(tmp_path: Path) -> None:
    config = GeneratorConfig(
        company_size=CompanySizeConfig(min_employees=30, max_employees=30),
        noise_ratio=0.85,
        hard_negative_ratio=0.25,
        cross_system_ratio=0.5,
    )
    context = GeneratorContext(seed=20260326, config=config)
    pipeline = GenerationPipeline(context=context)

    dataset = pipeline.generate_dataset(
        targets=DatasetTargets(
            account_count=20,
            email_count=200,
            slack_count=200,
            teams_count=200,
            salesforce_count=200,
            chunk_size=50,
        ),
        destination_root=tmp_path,
        write_csv=True,
    )

    assert len(dataset.enterprise.companies) == 1
    assert len(dataset.enterprise.employees) == 30
    assert len(dataset.enterprise.customer_accounts) == 20

    assert len(dataset.email_records) == 200
    assert len(dataset.slack_records) == 200
    assert len(dataset.teams_records) == 200
    assert len(dataset.salesforce_records) == 200

    assert set(dataset.manifests) == {"email", "slack", "teams", "salesforce"}
    assert {record.source_system for record in dataset.email_records} == {"email"}
    assert {record.source_system for record in dataset.slack_records} == {"slack"}
    assert {record.source_system for record in dataset.teams_records} == {"teams"}
    assert {record.source_system for record in dataset.salesforce_records} == {"salesforce"}

    email_account_by_event = {
        record.event_id: record.account_id
        for record in dataset.email_records
        if record.event_id is not None and record.account_id is not None
    }
    slack_account_by_event = {
        record.linked_event_id: record.linked_account_id
        for record in dataset.slack_records
        if record.linked_event_id is not None and record.linked_account_id is not None
    }
    teams_account_by_event = {
        record.linked_event_id: record.linked_account_id
        for record in dataset.teams_records
        if record.linked_event_id is not None and record.linked_account_id is not None
    }
    salesforce_account_by_event = {
        record.event_id: record.account_id
        for record in dataset.salesforce_records
        if record.event_id is not None and record.account_id is not None
    }

    shared_event_ids = (
        set(email_account_by_event)
        & set(slack_account_by_event)
        & set(teams_account_by_event)
        & set(salesforce_account_by_event)
    )
    assert shared_event_ids

    shared_event_id = next(iter(shared_event_ids))
    assert {
        email_account_by_event[shared_event_id],
        slack_account_by_event[shared_event_id],
        teams_account_by_event[shared_event_id],
        salesforce_account_by_event[shared_event_id],
    } == {email_account_by_event[shared_event_id]}

    all_rows: list[DatasetRow] = [
        *dataset.email_records,
        *dataset.slack_records,
        *dataset.teams_records,
        *dataset.salesforce_records,
    ]
    relevant_count = sum(record.is_relevant for record in all_rows)
    relevance_ratio = relevant_count / len(all_rows)

    assert 0.10 <= relevance_ratio <= 0.20
    assert relevant_count < len(all_rows) - relevant_count
    assert all(record.provenance is not None for record in all_rows if record.is_relevant)

    expected_counts = {
        "email": 200,
        "slack": 200,
        "teams": 200,
        "salesforce": 200,
    }
    for source_name, expected_count in expected_counts.items():
        manifest = dataset.manifests[source_name]
        parquet_root = tmp_path / f"source={source_name}" / "parquet"
        csv_root = tmp_path / f"source={source_name}" / "csv"
        manifest_path = tmp_path / f"source={source_name}" / "manifest.json"

        assert manifest.row_count == expected_count
        assert manifest.chunk_count == 4
        assert manifest_path.exists()
        assert parquet_root.exists()
        assert csv_root.exists()
        assert _parquet_row_count(parquet_root) == expected_count
