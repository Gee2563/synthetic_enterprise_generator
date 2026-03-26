from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Describes an exported dataset batch."""

    source_name: str
    schema_version: str
    seed: int
    config: dict[str, Any] = field(default_factory=dict)
    row_count: int = 0
    chunk_count: int = 0
    columns: tuple[str, ...] = field(default_factory=tuple)
    export_formats: tuple[str, ...] = field(default_factory=tuple)
    output_files: tuple[str, ...] = field(default_factory=tuple)
    partitions: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, destination: Path) -> None:
        destination.write_text(
            json.dumps(_normalize_value(self.to_dict()), sort_keys=True, indent=2),
            encoding="utf-8",
        )


def _normalize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            key: _normalize_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value
