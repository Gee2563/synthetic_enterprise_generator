from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from synthetic_enterprise.contracts.manifest import DatasetManifest
from synthetic_enterprise.storage.partitions import source_partition


class SupportsDataFrameRow(Protocol):
    def to_dataframe_row(self) -> dict[str, object]:
        ...


def rows_to_dataframe(
    rows: Sequence[SupportsDataFrameRow | dict[str, object]],
) -> pd.DataFrame:
    """Convert exported rows into a stable pandas DataFrame."""

    row_dicts = [_row_to_dict(row) for row in rows]
    if not row_dicts:
        return pd.DataFrame()

    columns = sorted({column for row in row_dicts for column in row})
    normalized_rows = [
        {column: row.get(column) for column in columns}
        for row in row_dicts
    ]
    return pd.DataFrame(normalized_rows, columns=columns)


def dataframe_to_arrow_table(
    frame: pd.DataFrame,
    *,
    schema_metadata: dict[str, str] | None = None,
) -> pa.Table:
    """Convert a pandas DataFrame into an Arrow table with optional metadata."""

    storage_frame = _normalize_frame_for_storage(frame)
    table = pa.Table.from_pandas(storage_frame, preserve_index=False)
    if schema_metadata is None:
        return table

    encoded_metadata = {
        key.encode("utf-8"): value.encode("utf-8")
        for key, value in schema_metadata.items()
    }
    return table.replace_schema_metadata(encoded_metadata)


def write_parquet(
    frame: pd.DataFrame,
    destination: Path,
    *,
    schema_metadata: dict[str, str] | None = None,
) -> None:
    """Write a DataFrame to parquet via pyarrow."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    table = dataframe_to_arrow_table(frame, schema_metadata=schema_metadata)
    pq.write_table(table, destination)


def write_csv(frame: pd.DataFrame, destination: Path) -> None:
    """Write a DataFrame to CSV with normalized nested values."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    _normalize_frame_for_storage(frame).to_csv(destination, index=False)


def write_source_dataset(
    *,
    rows: Sequence[SupportsDataFrameRow | dict[str, object]],
    destination_root: Path,
    source_name: str,
    seed: int,
    config: object,
    schema_version: str = "1.0",
    chunk_size: int | None = None,
    write_csv: bool = False,
) -> DatasetManifest:
    """Write one source dataset with manifest and optional chunking."""

    frame = rows_to_dataframe(rows)
    partition_root = source_partition(destination_root, source_name)
    partition_root.mkdir(parents=True, exist_ok=True)

    effective_chunk_size = chunk_size or max(len(frame), 1)
    chunks = _dataframe_chunks(frame, chunk_size=effective_chunk_size)
    parquet_root = partition_root / "parquet"
    csv_root = partition_root / "csv"
    output_files: list[str] = []

    for index, chunk in enumerate(chunks):
        parquet_path = parquet_root / f"part-{index:05d}.parquet"
        write_parquet(
            chunk,
            parquet_path,
            schema_metadata={
                "schema_version": schema_version,
                "source_name": source_name,
            },
        )
        output_files.append(str(parquet_path.relative_to(destination_root)))

        if write_csv:
            csv_path = csv_root / f"part-{index:05d}.csv"
            write_csv_file(chunk, csv_path)
            output_files.append(str(csv_path.relative_to(destination_root)))

    manifest = DatasetManifest(
        source_name=source_name,
        schema_version=schema_version,
        seed=seed,
        config=_normalize_config(config),
        row_count=len(frame),
        chunk_count=len(chunks),
        columns=tuple(frame.columns),
        export_formats=_export_formats(write_csv=write_csv),
        output_files=tuple(output_files),
        partitions=(partition_root.name,),
    )
    manifest.write_json(partition_root / "manifest.json")
    return manifest


def write_source_dataset_streaming(
    *,
    row_chunks: Iterable[Sequence[SupportsDataFrameRow | dict[str, object]]],
    destination_root: Path,
    source_name: str,
    seed: int,
    config: object,
    schema_version: str = "1.0",
    write_csv: bool = False,
) -> DatasetManifest:
    """Write one source dataset from an iterable of row chunks."""

    partition_root = source_partition(destination_root, source_name)
    partition_root.mkdir(parents=True, exist_ok=True)
    parquet_root = partition_root / "parquet"
    csv_root = partition_root / "csv"
    output_files: list[str] = []
    row_count = 0
    chunk_count = 0
    columns: tuple[str, ...] = ()

    for index, rows in enumerate(row_chunks):
        frame = rows_to_dataframe(rows)
        if chunk_count == 0:
            columns = tuple(frame.columns)
        else:
            frame = _align_frame_columns(frame, columns)

        parquet_path = parquet_root / f"part-{index:05d}.parquet"
        write_parquet(
            frame,
            parquet_path,
            schema_metadata={
                "schema_version": schema_version,
                "source_name": source_name,
            },
        )
        output_files.append(str(parquet_path.relative_to(destination_root)))

        if write_csv:
            csv_path = csv_root / f"part-{index:05d}.csv"
            write_csv_file(frame, csv_path)
            output_files.append(str(csv_path.relative_to(destination_root)))

        row_count += len(frame)
        chunk_count += 1

    if chunk_count == 0:
        empty_frame = pd.DataFrame()
        parquet_path = parquet_root / "part-00000.parquet"
        write_parquet(
            empty_frame,
            parquet_path,
            schema_metadata={
                "schema_version": schema_version,
                "source_name": source_name,
            },
        )
        output_files.append(str(parquet_path.relative_to(destination_root)))
        chunk_count = 1

    manifest = DatasetManifest(
        source_name=source_name,
        schema_version=schema_version,
        seed=seed,
        config=_normalize_config(config),
        row_count=row_count,
        chunk_count=chunk_count,
        columns=columns,
        export_formats=_export_formats(write_csv=write_csv),
        output_files=tuple(output_files),
        partitions=(partition_root.name,),
    )
    manifest.write_json(partition_root / "manifest.json")
    return manifest


def write_csv_file(frame: pd.DataFrame, destination: Path) -> None:
    """Compatibility wrapper for CSV export."""

    write_csv(frame, destination)


def _row_to_dict(row: SupportsDataFrameRow | dict[str, object]) -> dict[str, object]:
    if isinstance(row, dict):
        return dict(row)
    return row.to_dataframe_row()


def _normalize_frame_for_storage(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    return frame.map(_normalize_scalar)


def _align_frame_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if not columns:
        return frame

    current_columns = tuple(frame.columns)
    if current_columns == columns:
        return frame

    extra_columns = sorted(set(current_columns) - set(columns))
    missing_columns = sorted(set(columns) - set(current_columns))
    if extra_columns:
        raise ValueError(
            f"streamed chunk columns must match the initial schema, found extras: {extra_columns}"
        )

    aligned = frame.copy()
    for column in missing_columns:
        aligned[column] = None

    return aligned.loc[:, list(columns)]


def _normalize_scalar(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, ZoneInfo):
        return str(value)
    if isinstance(value, list | dict | tuple):
        return json.dumps(_normalize_config(value), sort_keys=True)
    return value


def _normalize_config(config: object) -> Any:
    if is_dataclass(config) and not isinstance(config, type):
        return _normalize_config(asdict(config))
    if isinstance(config, datetime):
        return config.isoformat()
    if isinstance(config, ZoneInfo):
        return str(config)
    if isinstance(config, dict):
        return {
            key: _normalize_config(value)
            for key, value in sorted(config.items())
        }
    if isinstance(config, list):
        return [_normalize_config(value) for value in config]
    if isinstance(config, tuple):
        return [_normalize_config(value) for value in config]
    return config


def _dataframe_chunks(frame: pd.DataFrame, *, chunk_size: int) -> list[pd.DataFrame]:
    if len(frame) == 0:
        return [frame.copy()]

    return [
        frame.iloc[start:start + chunk_size].reset_index(drop=True)
        for start in range(0, len(frame), chunk_size)
    ]


def _export_formats(*, write_csv: bool) -> tuple[str, ...]:
    if write_csv:
        return ("parquet", "csv")
    return ("parquet",)
