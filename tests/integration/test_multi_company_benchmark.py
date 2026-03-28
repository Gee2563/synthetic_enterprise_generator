from __future__ import annotations

from pathlib import Path

import pandas as pd

from synthetic_enterprise.evaluation.quality_report import DatasetQualityEvaluator
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.language_profiles import resolve_company_language_profile
from synthetic_enterprise.generation.pipeline import (
    DatasetTargets,
    GenerationPipeline,
    MultiCompanyBenchmarkTargets,
)


def build_pipeline(seed: int = 20260401) -> GenerationPipeline:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=12, max_employees=12),
            noise_ratio=0.8,
            hard_negative_ratio=0.25,
            cross_system_ratio=1.0,
        ),
    )
    return GenerationPipeline(context=context)


def test_multi_company_generation_produces_distinct_companies_and_safe_ids(
    tmp_path: Path,
) -> None:
    dataset = build_pipeline().generate_multi_company_benchmark(
        targets=MultiCompanyBenchmarkTargets(
            company_count=3,
            per_company_targets=DatasetTargets(
                account_count=4,
                email_count=12,
                slack_count=12,
                teams_count=12,
                salesforce_count=12,
                chunk_size=6,
            ),
        ),
        destination_root=tmp_path,
    )

    company_ids = [company.id for company in dataset.enterprise.companies]
    assert len(company_ids) == 3
    assert len(set(company_ids)) == 3
    assert len(set(dataset.company_profiles.values())) == 3

    all_entity_ids = [
        entity.id
        for entity_group in (
            dataset.enterprise.companies,
            dataset.enterprise.departments,
            dataset.enterprise.employees,
            dataset.enterprise.customer_accounts,
            dataset.enterprise.contacts,
            dataset.enterprise.opportunities,
            dataset.enterprise.events,
            dataset.enterprise.products,
            dataset.enterprise.ticket_issues,
            dataset.enterprise.campaigns,
            dataset.enterprise.message_envelopes,
            dataset.enterprise.crm_activities,
        )
        for entity in entity_group
    ]
    assert len(all_entity_ids) == len(set(all_entity_ids))

    for source_name, field_name in (
        ("email", "email_id"),
        ("slack", "slack_message_id"),
        ("teams", "teams_message_id"),
        ("salesforce", "salesforce_record_id"),
    ):
        ids = [
            str(row[field_name])
            for row in dataset.rows_by_source[source_name]
        ]
        assert len(ids) == len(set(ids))


def test_company_lexical_patterns_and_source_behavior_differ_per_company(
    tmp_path: Path,
) -> None:
    dataset = build_pipeline().generate_multi_company_benchmark(
        targets=MultiCompanyBenchmarkTargets(
            company_count=3,
            per_company_targets=DatasetTargets(
                account_count=4,
                email_count=10,
                slack_count=10,
                teams_count=10,
                salesforce_count=10,
            ),
        ),
        destination_root=tmp_path,
    )

    combined_texts: dict[str, str] = {}
    for company_id, profile_type in dataset.company_profiles.items():
        profile = resolve_company_language_profile(seed=0, explicit_profile=profile_type)
        company_rows = [
            row
            for rows in dataset.rows_by_source.values()
            for row in rows
            if row["company_id"] == company_id
        ]
        combined_text = " ".join(_row_text(row) for row in company_rows).lower()
        combined_texts[company_id] = combined_text

        markers = (
            profile.email_marker.lower(),
            profile.slack_marker.lower(),
            profile.teams_marker.lower(),
            profile.crm_marker.lower(),
        )
        assert sum(marker in combined_text for marker in markers) >= 2

    assert len(set(combined_texts.values())) == len(combined_texts)

    for company_id in dataset.company_profiles:
        email_rows = _rows_for_company(dataset.rows_by_source["email"], company_id)
        slack_rows = _rows_for_company(dataset.rows_by_source["slack"], company_id)
        teams_rows = _rows_for_company(dataset.rows_by_source["teams"], company_id)
        salesforce_rows = _rows_for_company(dataset.rows_by_source["salesforce"], company_id)

        assert email_rows and all("thread_id" in row for row in email_rows)
        assert slack_rows and all("channel_name" in row for row in slack_rows)
        assert teams_rows and all("chat_or_channel" in row for row in teams_rows)
        assert salesforce_rows and all("object_type" in row for row in salesforce_rows)


def test_multi_company_exports_preserve_company_id_and_quality_reports_can_compare_companies(
    tmp_path: Path,
) -> None:
    targets = MultiCompanyBenchmarkTargets(
        company_count=3,
        per_company_targets=DatasetTargets(
            account_count=4,
            email_count=8,
            slack_count=8,
            teams_count=8,
            salesforce_count=8,
            chunk_size=4,
        ),
    )
    dataset = build_pipeline().generate_multi_company_benchmark(
        targets=targets,
        destination_root=tmp_path,
        write_csv=True,
    )

    expected_company_ids = set(dataset.company_profiles)
    for source_name in ("email", "slack", "teams", "salesforce"):
        parquet_files = sorted((tmp_path / f"source={source_name}" / "parquet").glob("*.parquet"))
        assert parquet_files
        frame = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
        assert "company_id" in frame.columns
        assert set(frame["company_id"]) == expected_company_ids

    reports = DatasetQualityEvaluator(enterprise=dataset.enterprise).evaluate_by_company(
        dataset.rows_by_source
    )

    assert set(reports) == expected_company_ids
    for company_id, report in reports.items():
        assert report.total_rows == 32
        assert report.source_row_counts == {
            "email": 8,
            "slack": 8,
            "teams": 8,
            "salesforce": 8,
        }
        assert company_id in expected_company_ids

    email_lengths = {
        report.average_message_length_by_source["email"]
        for report in reports.values()
    }
    assert len(email_lengths) > 1


def _row_text(row: dict[str, object]) -> str:
    parts = [row.get("subject"), row.get("body"), row.get("text_body")]
    return " ".join(str(part) for part in parts if isinstance(part, str))


def _rows_for_company(
    rows: list[dict[str, object]],
    company_id: str,
) -> list[dict[str, object]]:
    return [row for row in rows if row["company_id"] == company_id]
