from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.labeling.grounding import LabelGrounding, LabelProvenance
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory, CommunicationTaxonomy

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EmployeeId = Annotated[str, StringConstraints(pattern=r"^employee_[a-z0-9]+$")]
ContactId = Annotated[str, StringConstraints(pattern=r"^contact_[a-z0-9]+$")]
ParticipantId = Annotated[str, StringConstraints(pattern=r"^(employee|contact)_[a-z0-9]+$")]
EmailId = Annotated[str, StringConstraints(pattern=r"^email_[a-z0-9]+$")]
ThreadId = Annotated[str, StringConstraints(pattern=r"^thread_[a-z0-9]+$")]
AccountId = Annotated[str, StringConstraints(pattern=r"^account_[a-z0-9]+$")]
OpportunityId = Annotated[str, StringConstraints(pattern=r"^opportunity_[a-z0-9]+$")]
EventId = Annotated[str, StringConstraints(pattern=r"^event_[a-z0-9]+$")]
TicketId = Annotated[str, StringConstraints(pattern=r"^ticket_issue_[a-z0-9]+$")]


class EmailRecord(EnterpriseModel):
    """One serialized email dataset row."""

    email_id: EmailId
    thread_id: ThreadId
    message_index_in_thread: int = Field(ge=0)
    timestamp: AwareDatetime
    sender_employee_id: EmployeeId | None = None
    sender_contact_id: ContactId | None = None
    to: list[ParticipantId] = Field(min_length=1)
    cc: list[ParticipantId] = Field(default_factory=list)
    bcc: list[ParticipantId] = Field(default_factory=list)
    subject: NonEmptyText
    body: NonEmptyText
    attachments: list[str] = Field(default_factory=list)
    account_id: AccountId | None = None
    opportunity_id: OpportunityId | None = None
    event_id: EventId | None = None
    ticket_id: TicketId | None = None
    primary_category: CommunicationCategory
    is_relevant: bool
    relevance_reason: NonEmptyText
    provenance: LabelProvenance | None = None
    source_system: Literal["email"] = "email"

    @model_validator(mode="after")
    def validate_email_record(self) -> EmailRecord:
        sender_count = int(self.sender_employee_id is not None) + int(
            self.sender_contact_id is not None
        )
        if sender_count != 1:
            raise ValueError("exactly one sender id must be provided")

        taxonomy = CommunicationTaxonomy(
            primary_category=self.primary_category,
            is_relevant=self.is_relevant,
        )
        if taxonomy.is_relevant and not any(
            value is not None
            for value in (
                self.account_id,
                self.opportunity_id,
                self.event_id,
                self.ticket_id,
            )
        ):
            raise ValueError("relevant email records must include traceable business context ids")

        if self.provenance is not None:
            valid_provenance_ids = {
                value
                for value in (
                    self.account_id,
                    self.opportunity_id,
                    self.event_id,
                    self.ticket_id,
                )
                if value is not None
            }
            if self.provenance.object_id not in valid_provenance_ids:
                raise ValueError("email provenance must reference one of the linked business ids")

        LabelGrounding(
            primary_category=self.primary_category,
            is_relevant=self.is_relevant,
            relevance_reason=self.relevance_reason,
            provenance=self.provenance,
        )

        return self

    def to_dataframe_row(self) -> dict[str, object]:
        return self.model_dump(mode="json")
