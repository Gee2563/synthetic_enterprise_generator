from __future__ import annotations

import pytest

from synthetic_enterprise.contracts.records.salesforce import (
    SalesforceObjectType,
    SalesforceRecord,
)
from synthetic_enterprise.evaluation.quality_report import DatasetQualityEvaluator
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.sources.salesforce.renderer import SalesforceRenderer

MESSY_STRUCTURED_KEYS = {
    "missing_fields",
    "late_entry_days",
    "contradicts_record_id",
    "stage_sync_delay_days",
    "stale_owner_snapshot_days",
    "attendance_reconciliation_pending",
    "follow_up_capture_state",
}


def test_configured_messiness_rate_is_respected() -> None:
    clean_records = build_records(messiness_rate=0.0)
    messy_records = build_records(messiness_rate=0.35)

    clean_rate = _messy_rate(clean_records)
    messy_rate = _messy_rate(messy_records)

    assert clean_rate == pytest.approx(0.0)
    assert messy_rate >= 0.2
    assert messy_rate <= 0.45


def test_data_remains_schema_valid_where_intended() -> None:
    records = build_records(messiness_rate=0.4)

    assert records
    assert all(record.source_system == "salesforce" for record in records)
    assert any(
        record.object_type == SalesforceObjectType.CONTACT
        and "missing_fields" in record.structured_fields
        for record in records
    )
    assert all(record.record_id for record in records)


def test_some_inconsistencies_are_allowed_but_bounded() -> None:
    records = build_records(messiness_rate=0.4)
    contradictory_notes = [
        record
        for record in records
        if record.object_type == SalesforceObjectType.NOTE
        and "contradicts_record_id" in record.structured_fields
    ]
    reopened_cases = [
        record
        for record in records
        if record.object_type == SalesforceObjectType.CASE and record.status == "Reopened"
    ]
    outdated_opportunities = [
        record
        for record in records
        if record.object_type == SalesforceObjectType.OPPORTUNITY
        and "stage_sync_delay_days" in record.structured_fields
    ]

    assert contradictory_notes
    assert len(contradictory_notes) <= 2
    assert reopened_cases
    assert outdated_opportunities
    assert _messy_count(records) < len(records) / 2


def test_relevant_provenance_can_still_be_traced_where_required() -> None:
    records = build_records(messiness_rate=0.45)
    relevant_records = [record for record in records if record.is_relevant]

    assert relevant_records
    assert all(record.provenance is not None for record in relevant_records)


def test_quality_reports_can_detect_presence_of_messy_data() -> None:
    clean_records = build_records(messiness_rate=0.0)
    messy_records = build_records(messiness_rate=0.4)
    evaluator = DatasetQualityEvaluator()

    clean_report = evaluator.evaluate(
        {"salesforce": [record.to_dict() for record in clean_records]}
    )
    messy_report = evaluator.evaluate(
        {"salesforce": [record.to_dict() for record in messy_records]}
    )

    assert clean_report.messy_data_rate == pytest.approx(0.0)
    assert messy_report.messy_data_rate > 0.0
    assert messy_report.messy_data_rate > clean_report.messy_data_rate


def test_deterministic_generation_holds() -> None:
    first = build_records(seed=9393, messiness_rate=0.4)
    second = build_records(seed=9393, messiness_rate=0.4)
    third = build_records(seed=9394, messiness_rate=0.4)

    assert first == second
    assert first != third


def build_records(
    *,
    seed: int = 9292,
    messiness_rate: float = 0.0,
) -> list[SalesforceRecord]:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_ratio=0.8,
            hard_negative_ratio=0.25,
            messiness_rate=messiness_rate,
        ),
    )
    enterprise = CompanyBuilder().build(context)
    return SalesforceRenderer(context=context, enterprise=enterprise).generate_records()


def _messy_rate(records: list[SalesforceRecord]) -> float:
    return _messy_count(records) / len(records)


def _messy_count(records: list[SalesforceRecord]) -> int:
    return sum(_is_messy(record) for record in records)


def _is_messy(record: SalesforceRecord) -> bool:
    if record.status == "Reopened":
        return True
    return any(key in record.structured_fields for key in MESSY_STRUCTURED_KEYS)
