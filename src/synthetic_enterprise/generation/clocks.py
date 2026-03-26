from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class FixedClock:
    """Deterministic clock placeholder."""

    now_value: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.now_value
