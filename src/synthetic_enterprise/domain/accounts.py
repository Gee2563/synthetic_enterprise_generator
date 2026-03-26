from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from synthetic_enterprise.domain.base import EnterpriseEntity

RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CustomerAccount(EnterpriseEntity):
    """Represents a customer account owned by the vendor organization."""

    entity_name = "account"
    company_id: str = Field(pattern=r"^company_[a-z0-9]+$")
    name: RequiredText
    industry: RequiredText
    owner_employee_id: str = Field(pattern=r"^employee_[a-z0-9]+$")
