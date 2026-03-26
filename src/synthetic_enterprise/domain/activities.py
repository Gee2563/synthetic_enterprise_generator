from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from synthetic_enterprise.domain.base import EnterpriseEntity

RequiredText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
MessageSource = Literal["email", "slack", "teams", "salesforce"]


class MessageEnvelope(EnterpriseEntity):
    """Represents shared metadata around a message rendered from any source."""

    entity_name = "message"
    company_id: str = Field(pattern=r"^company_[a-z0-9]+$")
    source: MessageSource
    subject: RequiredText
    body_text: RequiredText
    sender_employee_id: str | None = Field(default=None, pattern=r"^employee_[a-z0-9]+$")
    sender_contact_id: str | None = Field(default=None, pattern=r"^contact_[a-z0-9]+$")
    recipient_employee_ids: list[str] = Field(default_factory=list)
    recipient_contact_ids: list[str] = Field(default_factory=list)
    related_account_id: str | None = Field(default=None, pattern=r"^account_[a-z0-9]+$")
    related_opportunity_id: str | None = Field(default=None, pattern=r"^opportunity_[a-z0-9]+$")
    sent_at: AwareDatetime

    @model_validator(mode="after")
    def validate_participants(self) -> MessageEnvelope:
        has_sender = self.sender_employee_id is not None or self.sender_contact_id is not None
        has_recipient = bool(self.recipient_employee_ids or self.recipient_contact_ids)

        if not has_sender:
            raise ValueError("message envelopes require at least one sender")
        if not has_recipient:
            raise ValueError("message envelopes require at least one recipient")
        return self


class CRMActivity(EnterpriseEntity):
    """Represents a structured CRM activity tied to other enterprise entities."""

    company_id: str = Field(pattern=r"^company_[a-z0-9]+$")
    account_id: str = Field(pattern=r"^account_[a-z0-9]+$")
    employee_id: str = Field(pattern=r"^employee_[a-z0-9]+$")
    contact_id: str | None = Field(default=None, pattern=r"^contact_[a-z0-9]+$")
    opportunity_id: str | None = Field(default=None, pattern=r"^opportunity_[a-z0-9]+$")
    campaign_id: str | None = Field(default=None, pattern=r"^campaign_[a-z0-9]+$")
    message_id: str | None = Field(default=None, pattern=r"^message_[a-z0-9]+$")
    activity_type: RequiredText
    occurred_at: AwareDatetime
