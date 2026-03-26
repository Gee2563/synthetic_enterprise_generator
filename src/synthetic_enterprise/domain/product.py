from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from synthetic_enterprise.domain.base import EnterpriseEntity

RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Product(EnterpriseEntity):
    """Represents a product sold by the simulated company."""

    company_id: str = Field(pattern=r"^company_[a-z0-9]+$")
    name: RequiredText
    sku: RequiredText
    family: RequiredText
