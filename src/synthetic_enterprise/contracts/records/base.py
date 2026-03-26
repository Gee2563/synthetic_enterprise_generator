from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BaseRecord:
    """Shared fields across exported records."""

    record_id: str
    source: str
    company_id: str
    content_text: str
    relevance_labels: tuple[str, ...] = field(default_factory=tuple)
