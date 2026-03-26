from __future__ import annotations

from typing import Annotated

from pydantic import EmailStr, Field, StringConstraints

from synthetic_enterprise.domain.base import EnterpriseEntity

RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Employee(EnterpriseEntity):
    """Represents an internal employee of the simulated company."""

    company_id: str = Field(pattern=r"^company_[a-z0-9]+$")
    department_id: str = Field(pattern=r"^department_[a-z0-9]+$")
    email: EmailStr
    first_name: RequiredText
    last_name: RequiredText
    title: RequiredText
    manager_employee_id: str | None = Field(default=None, pattern=r"^employee_[a-z0-9]+$")


class Contact(EnterpriseEntity):
    """Represents an external stakeholder tied to a customer account."""

    account_id: str = Field(pattern=r"^account_[a-z0-9]+$")
    first_name: RequiredText
    last_name: RequiredText
    email: EmailStr
    title: RequiredText
