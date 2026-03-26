from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import DatasetTargets, GenerationPipeline

ID_COLUMN_BY_SOURCE = {
    "email": "email_id",
    "slack": "slack_message_id",
    "teams": "teams_message_id",
    "salesforce": "salesforce_record_id",
}


def build_pipeline(seed: int = 20260327) -> GenerationPipeline:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=30, max_employees=30),
            noise_ratio=0.85,
            hard_negative_ratio=0.25,
            cross_system_ratio=0.5,
        ),
    )
    return GenerationPipeline(context=context)


def _read_source_frame(root: Path, source_name: str) -> pd.DataFrame:
    parquet_root = root / f"source={source_name}" / "parquet"
    frames = [
        pd.read_parquet(parquet_file)
        for parquet_file in sorted(parquet_root.glob("*.parquet"))
    ]
    return pd.concat(frames, ignore_index=True)


def test_streaming_generation_avoids_duplicate_ids_and_aggregates_counts(
    tmp_path: Path,
) -> None:
    pipeline = build_pipeline()
    targets = DatasetTargets(
        account_count=20,
        email_count=420,
        slack_count=420,
        teams_count=420,
        salesforce_count=420,
        chunk_size=100,
    )

    streamed = pipeline.generate_dataset_streaming(
        targets=targets,
        destination_root=tmp_path,
        generation_chunk_size=100,
    )

    for source_name, expected_count in {
        "email": 420,
        "slack": 420,
        "teams": 420,
        "salesforce": 420,
    }.items():
        frame = _read_source_frame(tmp_path, source_name)
        id_column = ID_COLUMN_BY_SOURCE[source_name]

        assert len(frame) == expected_count
        assert len(frame[id_column]) == frame[id_column].nunique()
        assert streamed.manifests[source_name].row_count == expected_count
        assert streamed.manifests[source_name].chunk_count == 5


def test_chunked_and_non_chunked_outputs_are_schema_compatible(tmp_path: Path) -> None:
    targets = DatasetTargets(
        account_count=20,
        email_count=200,
        slack_count=200,
        teams_count=200,
        salesforce_count=200,
        chunk_size=50,
    )
    pipeline = build_pipeline()

    in_memory = pipeline.generate_dataset(
        targets=targets,
        destination_root=tmp_path / "in-memory",
    )
    streamed = pipeline.generate_dataset_streaming(
        targets=targets,
        destination_root=tmp_path / "streamed",
        generation_chunk_size=50,
    )

    for source_name in ("email", "slack", "teams", "salesforce"):
        assert in_memory.manifests[source_name].columns == streamed.manifests[source_name].columns


def test_streaming_generation_supports_medium_benchmark_dataset(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = build_pipeline(seed=20260329)
    targets = DatasetTargets(
        account_count=20,
        email_count=1200,
        slack_count=1200,
        teams_count=1200,
        salesforce_count=1200,
        chunk_size=250,
    )
    caplog.set_level(logging.INFO)

    streamed = pipeline.generate_dataset_streaming(
        targets=targets,
        destination_root=tmp_path,
        generation_chunk_size=250,
    )

    assert streamed.enterprise is not None
    assert streamed.manifests["email"].row_count == 1200
    assert streamed.manifests["slack"].row_count == 1200
    assert streamed.manifests["teams"].row_count == 1200
    assert streamed.manifests["salesforce"].row_count == 1200
    assert "Completed source email" in caplog.text
    assert "Completed source salesforce" in caplog.text
