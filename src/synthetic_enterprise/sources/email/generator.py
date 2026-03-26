from __future__ import annotations

from dataclasses import dataclass, field

from synthetic_enterprise.contracts.records.email import EmailRecord
from synthetic_enterprise.generation.context import GeneratorContext


@dataclass(slots=True)
class EmailGenerator:
    """Placeholder email generator."""

    context: GeneratorContext
    records: list[EmailRecord] = field(default_factory=list)

    def generate(self) -> list[EmailRecord]:
        return list(self.records)
