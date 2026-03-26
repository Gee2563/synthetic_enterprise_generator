from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from synthetic_enterprise.domain.base import EnterpriseEntity

RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
IssueStatus = Literal["open", "in_progress", "resolved", "closed"]
IssueSeverity = Literal["low", "medium", "high", "critical"]


class TicketIssue(EnterpriseEntity):
    """Represents a support issue or escalation tied to a customer account."""

    account_id: str = Field(pattern=r"^account_[a-z0-9]+$")
    opened_by_contact_id: str = Field(pattern=r"^contact_[a-z0-9]+$")
    owner_employee_id: str | None = Field(default=None, pattern=r"^employee_[a-z0-9]+$")
    related_product_id: str | None = Field(default=None, pattern=r"^product_[a-z0-9]+$")
    status: IssueStatus
    severity: IssueSeverity
    summary: RequiredText
