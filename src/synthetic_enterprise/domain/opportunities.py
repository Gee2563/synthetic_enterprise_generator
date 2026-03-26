from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from synthetic_enterprise.domain.base import EnterpriseEntity

RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
OpportunityStage = Literal[
    "qualification",
    "discovery",
    "evaluation",
    "proposal",
    "closed_won",
    "closed_lost",
]


class Opportunity(EnterpriseEntity):
    """Represents a sales opportunity tied to an account."""

    account_id: str = Field(pattern=r"^account_[a-z0-9]+$")
    owner_employee_id: str = Field(pattern=r"^employee_[a-z0-9]+$")
    primary_contact_id: str = Field(pattern=r"^contact_[a-z0-9]+$")
    product_ids: list[str] = Field(min_length=1)
    stage: OpportunityStage
    amount: float = Field(ge=0.0)

    @property
    def has_products(self) -> bool:
        return bool(self.product_ids)
