from __future__ import annotations

from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, StringConstraints, field_validator

from synthetic_enterprise.domain.base import EnterpriseEntity

RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
DomainName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=3,
        pattern=r"^[a-z0-9.-]+\.[a-z]{2,}$",
    ),
]


class Company(EnterpriseEntity):
    """Represents the synthetic vendor organization being simulated."""

    name: RequiredText
    domain: DomainName
    timezone: RequiredText

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {value}") from exc
        return value


class Department(EnterpriseEntity):
    """Represents a department inside a company."""

    company_id: str = Field(pattern=r"^company_[a-z0-9]+$")
    name: RequiredText
    parent_department_id: str | None = Field(default=None, pattern=r"^department_[a-z0-9]+$")
    leader_employee_id: str | None = Field(default=None, pattern=r"^employee_[a-z0-9]+$")
