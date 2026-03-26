from __future__ import annotations

from pathlib import Path


def source_partition(root: Path, source_name: str) -> Path:
    """Return a partition path for a source."""

    return root / f"source={source_name}"
