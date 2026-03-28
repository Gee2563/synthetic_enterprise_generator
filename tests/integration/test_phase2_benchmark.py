from __future__ import annotations

from pathlib import Path

import pandas as pd

from synthetic_enterprise.contracts.scenario import ScenarioFamily
from synthetic_enterprise.evaluation.quality_report import DatasetQualityEvaluator
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import (
    DatasetTargets,
    GenerationPipeline,
    Phase2BenchmarkTargets,
)


def build_pipeline(seed: int = 20260415) -> GenerationPipeline:
    return GenerationPipeline(
        context=GeneratorContext(
            seed=seed,
            config=GeneratorConfig(
                company_size=CompanySizeConfig(min_employees=18, max_employees=18),
                noise_ratio=0.82,
                hard_negative_ratio=0.25,
                cross_system_ratio=0.5,
                messiness_rate=0.0,
            ),
        )
    )


def test_phase2_benchmark_contains_required_scenario_families_and_profiles(
    tmp_path: Path,
) -> None:
    benchmark = build_pipeline().generate_phase2_benchmark(
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
        destination_root=tmp_path,
    )

    assert set(benchmark.benchmark_manifest.scenario_families) == {
        family.value
        for family in ScenarioFamily
    }
    assert len(benchmark.company_profiles) == 3
    assert len(set(benchmark.company_profiles.values())) == 3
    assert len(benchmark.benchmark_manifest.account_behavior_profiles) >= 3
    assert benchmark.benchmark_manifest.hard_negative_row_count > 0
    assert benchmark.benchmark_manifest.messy_row_count > 0
    assert benchmark.benchmark_manifest.event_attendance_row_count > 0


def test_phase2_benchmark_represents_all_companies_and_sources(tmp_path: Path) -> None:
    benchmark = build_pipeline().generate_phase2_benchmark(
        targets=Phase2BenchmarkTargets(
            company_count=3,
            per_company_targets=DatasetTargets(
                account_count=12,
                email_count=12,
                slack_count=12,
                teams_count=12,
                salesforce_count=12,
                chunk_size=6,
            ),
        ),
        destination_root=tmp_path,
    )

    company_ids = set(benchmark.company_profiles)
    assert company_ids == {company.id for company in benchmark.enterprise.companies}
    assert set(benchmark.rows_by_source) == {"email", "slack", "teams", "salesforce"}

    for source_name, rows in benchmark.rows_by_source.items():
        assert rows
        assert {str(row["company_id"]) for row in rows} == company_ids


def test_phase2_benchmark_improves_category_coverage_and_duplicate_rate(
    tmp_path: Path,
) -> None:
    pipeline = build_pipeline()
    baseline = pipeline.generate_dataset(
        targets=DatasetTargets(
            account_count=4,
            email_count=54,
            slack_count=54,
            teams_count=54,
            salesforce_count=54,
            chunk_size=18,
        ),
        destination_root=tmp_path / "baseline",
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
        destination_root=tmp_path / "benchmark",
    )

    baseline_rows = {
        "email": [record.to_dict() for record in baseline.email_records],
        "slack": [record.to_dict() for record in baseline.slack_records],
        "teams": [record.to_dict() for record in baseline.teams_records],
        "salesforce": [record.to_dict() for record in baseline.salesforce_records],
    }
    baseline_report = DatasetQualityEvaluator(
        enterprise=baseline.enterprise,
        context=pipeline.context,
    ).evaluate(baseline_rows)
    benchmark_report = DatasetQualityEvaluator(
        enterprise=benchmark.enterprise,
    ).evaluate(benchmark.rows_by_source)

    baseline_category_count = sum(
        rate > 0.0
        for rate in baseline_report.category_distribution.values()
    )
    benchmark_category_count = sum(
        rate > 0.0
        for rate in benchmark_report.category_distribution.values()
    )

    assert benchmark_category_count > baseline_category_count
    assert benchmark_report.duplicate_rate < baseline_report.duplicate_rate


def test_phase2_benchmark_exports_successfully_and_writes_manifest(
    tmp_path: Path,
) -> None:
    benchmark = build_pipeline().generate_phase2_benchmark(
        targets=Phase2BenchmarkTargets(
            company_count=3,
            per_company_targets=DatasetTargets(
                account_count=12,
                email_count=12,
                slack_count=12,
                teams_count=12,
                salesforce_count=12,
                chunk_size=6,
            ),
        ),
        destination_root=tmp_path,
        write_csv=True,
    )

    assert benchmark.benchmark_manifest_path.exists()
    for source_name, manifest in benchmark.manifests.items():
        parquet_files = sorted((tmp_path / f"source={source_name}" / "parquet").glob("*.parquet"))
        assert parquet_files
        frame = pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)
        assert len(frame) == manifest.row_count
        assert frame["source_system"].eq(source_name).all()
