from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from synthetic_enterprise.domain.base import EnterpriseEntity

RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EventType = Literal["customer_meeting", "webinar", "conference", "internal_meeting"]
CampaignChannel = Literal["email", "webinar", "paid_social", "event"]


class Event(EnterpriseEntity):
    """Represents a planned or completed event in the simulated timeline."""

    company_id: str = Field(pattern=r"^company_[a-z0-9]+$")
    account_id: str | None = Field(default=None, pattern=r"^account_[a-z0-9]+$")
    organizer_employee_id: str = Field(pattern=r"^employee_[a-z0-9]+$")
    attendee_contact_ids: list[str] = Field(default_factory=list)
    title: RequiredText
    event_type: EventType
    starts_at: AwareDatetime
    ends_at: AwareDatetime

    @model_validator(mode="after")
    def validate_window(self) -> Event:
        if self.ends_at <= self.starts_at:
            raise ValueError("ends_at must be greater than starts_at")
        return self


class Campaign(EnterpriseEntity):
    """Represents a marketing or customer outreach campaign."""

    company_id: str = Field(pattern=r"^company_[a-z0-9]+$")
    owner_employee_id: str = Field(pattern=r"^employee_[a-z0-9]+$")
    name: RequiredText
    channel: CampaignChannel
    target_account_ids: list[str] = Field(default_factory=list)
    product_ids: list[str] = Field(default_factory=list)
