from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Lineage:
    """Tracks how a record maps back to its simulated origin."""

    company_id: str
    scenario_ids: tuple[str, ...] = field(default_factory=tuple)
    seed_path: tuple[int, ...] = field(default_factory=tuple)
