from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write_jsonl(rows: Iterable[Mapping[str, Any]], destination: Path) -> None:
    """Write rows as newline-delimited JSON."""

    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")
