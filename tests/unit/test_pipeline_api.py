from __future__ import annotations

import pytest

from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import (
    DatasetSourceName,
    DatasetTargets,
    GenerationPipeline,
)


def build_pipeline() -> GenerationPipeline:
    context = GeneratorContext(
        seed=20260329,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=30, max_employees=30),
        ),
    )
    return GenerationPipeline(context=context)


@pytest.mark.parametrize(
    ("source_name", "expected_count"),
    [
        ("email", 12),
        ("slack", 13),
        ("teams", 14),
        ("salesforce", 15),
    ],
)
def test_public_source_build_api_returns_sorted_source_specific_rows(
    source_name: DatasetSourceName,
    expected_count: int,
) -> None:
    pipeline = build_pipeline()
    targets = DatasetTargets(
        account_count=4,
        email_count=12,
        slack_count=13,
        teams_count=14,
        salesforce_count=15,
    )
    enterprise = pipeline.build_enterprise(account_count=targets.account_count)
    bundles = pipeline.select_cross_system_bundles(enterprise=enterprise, targets=targets)

    records = pipeline.build_source_records(
        source_name=source_name,
        enterprise=enterprise,
        targets=targets,
        bundles=bundles,
    )

    assert len(records) == expected_count
    assert {record.source_system for record in records} == {source_name}

    sort_fingerprint = [
        (record.timestamp, _record_identity(record))
        for record in records
    ]
    assert sort_fingerprint == sorted(sort_fingerprint)


def _record_identity(record: object) -> str:
    for field_name in (
        "email_id",
        "slack_message_id",
        "teams_message_id",
        "salesforce_record_id",
    ):
        value = getattr(record, field_name, None)
        if isinstance(value, str):
            return value
    raise AssertionError("record is missing a supported identity field")
