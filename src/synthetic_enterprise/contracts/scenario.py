from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Scenario:
    """A minimal scenario contract for future vertical slices."""

    scenario_id: str
    kind: str
    company_id: str
    expected_labels: tuple[str, ...] = field(default_factory=tuple)
