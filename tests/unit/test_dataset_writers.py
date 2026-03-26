from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.sources.email.renderer import EmailRenderer
from synthetic_enterprise.sources.salesforce.renderer import SalesforceRenderer
from synthetic_enterprise.sources.slack.renderer import SlackRenderer
from synthetic_enterprise.storage.parquet_writer import (
    rows_to_dataframe,
    write_source_dataset,
)


def build_context() -> GeneratorContext:
    return GeneratorContext(
        seed=1111,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
        ),
    )


def test_exported_parquet_can_be_read_back(tmp_path: Path) -> None:
    context = build_context()
    enterprise = CompanyBuilder().build(context)
    event = enterprise.events[0]
    rows = EmailRenderer(context=context, enterprise=enterprise).generate_messages(
        event_id=event.id
    )

    manifest = write_source_dataset(
        rows=rows,
        destination_root=tmp_path,
        source_name="email",
        seed=context.seed,
        config=context.config,
        schema_version="2026.1",
    )

    parquet_path = tmp_path / "source=email" / "parquet" / "part-00000.parquet"
    table = pq.read_table(parquet_path)
    frame = table.to_pandas()

    assert len(frame) == manifest.row_count == len(rows)
    assert table.schema.metadata is not None
    assert table.schema.metadata[b"schema_version"] == b"2026.1"


def test_row_counts_match_expectations(tmp_path: Path) -> None:
    context = build_context()
    enterprise = CompanyBuilder().build(context)
    rows = SlackRenderer(context=context, enterprise=enterprise).generate_messages()

    manifest = write_source_dataset(
        rows=rows,
        destination_root=tmp_path,
        source_name="slack",
        seed=context.seed,
        config=context.config,
    )

    parquet_files = sorted((tmp_path / "source=slack" / "parquet").glob("*.parquet"))
    total_rows = sum(len(pq.read_table(parquet_file).to_pandas()) for parquet_file in parquet_files)

    assert manifest.row_count == len(rows)
    assert total_rows == len(rows)


def test_schema_columns_are_stable() -> None:
    context = build_context()
    enterprise = CompanyBuilder().build(context)
    event = enterprise.events[0]
    rows = EmailRenderer(context=context, enterprise=enterprise).generate_messages(
        event_id=event.id
    )

    frame = rows_to_dataframe(rows)

    assert list(frame.columns) == sorted(rows[0].to_dataframe_row().keys())


def test_large_writes_can_be_chunked(tmp_path: Path) -> None:
    context = build_context()
    enterprise = CompanyBuilder().build(context)
    rows = SalesforceRenderer(context=context, enterprise=enterprise).generate_records()[:5]

    manifest = write_source_dataset(
        rows=rows,
        destination_root=tmp_path,
        source_name="salesforce",
        seed=context.seed,
        config=context.config,
        chunk_size=2,
        write_csv=True,
    )

    parquet_files = sorted((tmp_path / "source=salesforce" / "parquet").glob("*.parquet"))
    csv_files = sorted((tmp_path / "source=salesforce" / "csv").glob("*.csv"))

    assert manifest.chunk_count == 3
    assert len(parquet_files) == 3
    assert len(csv_files) == 3


def test_manifest_is_created(tmp_path: Path) -> None:
    context = build_context()
    enterprise = CompanyBuilder().build(context)
    rows = SlackRenderer(context=context, enterprise=enterprise).generate_messages()

    manifest = write_source_dataset(
        rows=rows,
        destination_root=tmp_path,
        source_name="slack",
        seed=context.seed,
        config=context.config,
        schema_version="2026.2",
    )

    manifest_path = tmp_path / "source=slack" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.exists()
    assert payload["source_name"] == "slack"
    assert payload["seed"] == context.seed
    assert payload["schema_version"] == "2026.2"
    assert manifest.row_count == payload["row_count"]


def test_source_system_specific_exports_work_independently(tmp_path: Path) -> None:
    context = build_context()
    enterprise = CompanyBuilder().build(context)
    event = enterprise.events[0]
    email_rows = EmailRenderer(context=context, enterprise=enterprise).generate_messages(
        event_id=event.id
    )
    slack_rows = SlackRenderer(context=context, enterprise=enterprise).generate_messages()

    email_manifest = write_source_dataset(
        rows=email_rows,
        destination_root=tmp_path,
        source_name="email",
        seed=context.seed,
        config=context.config,
    )
    slack_manifest = write_source_dataset(
        rows=slack_rows,
        destination_root=tmp_path,
        source_name="slack",
        seed=context.seed,
        config=context.config,
    )

    email_frame = pd.read_parquet(tmp_path / "source=email" / "parquet" / "part-00000.parquet")
    slack_frame = pd.read_parquet(tmp_path / "source=slack" / "parquet" / "part-00000.parquet")

    assert email_manifest.source_name == "email"
    assert slack_manifest.source_name == "slack"
    assert email_frame["source_system"].eq("email").all()
    assert slack_frame["source_system"].eq("slack").all()
