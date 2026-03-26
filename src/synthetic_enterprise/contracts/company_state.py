from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompanyState:
    """Minimal company-level state backing synthetic record generation."""

    company_id: str
    company_name: str
    seed: int
