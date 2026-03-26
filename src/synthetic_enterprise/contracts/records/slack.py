from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.labeling.grounding import LabelGrounding, LabelProvenance
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory, CommunicationTaxonomy

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SlackMessageId = Annotated[str, StringConstraints(pattern=r"^slack_message_[a-z0-9]+$")]
ChannelId = Annotated[str, StringConstraints(pattern=r"^channel_[a-z0-9]+$")]
ThreadId = Annotated[str, StringConstraints(pattern=r"^slack_thread_[a-z0-9]+$")]
EmployeeId = Annotated[str, StringConstraints(pattern=r"^employee_[a-z0-9]+$")]
AccountId = Annotated[str, StringConstraints(pattern=r"^account_[a-z0-9]+$")]
OpportunityId = Annotated[str, StringConstraints(pattern=r"^opportunity_[a-z0-9]+$")]
EventId = Annotated[str, StringConstraints(pattern=r"^event_[a-z0-9]+$")]
TicketId = Annotated[str, StringConstraints(pattern=r"^ticket_issue_[a-z0-9]+$")]


class SlackRecord(EnterpriseModel):
    """One serialized Slack message dataset row."""

    slack_message_id: SlackMessageId
    channel_id: ChannelId
    channel_name: NonEmptyText
    thread_id: ThreadId | None = None
    parent_message_id: SlackMessageId | None = None
    timestamp: AwareDatetime
    sender_employee_id: EmployeeId
    body: NonEmptyText
    mentions: list[EmployeeId] = Field(default_factory=list)
    reactions: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    linked_account_id: AccountId | None = None
    linked_opportunity_id: OpportunityId | None = None
    linked_event_id: EventId | None = None
    linked_ticket_id: TicketId | None = None
    primary_category: CommunicationCategory
    is_relevant: bool
    relevance_reason: NonEmptyText
    provenance: LabelProvenance | None = None
    source_system: Literal["slack"] = "slack"

    @model_validator(mode="after")
    def validate_slack_record(self) -> SlackRecord:
        taxonomy = CommunicationTaxonomy(
            primary_category=self.primary_category,
            is_relevant=self.is_relevant,
        )

        if self.parent_message_id is not None and self.thread_id is None:
            raise ValueError("thread replies must include a thread_id")

        if self.thread_id is None and self.parent_message_id is not None:
            raise ValueError("parent_message_id requires a thread_id")

        if taxonomy.is_relevant and not any(
            value is not None
            for value in (
                self.linked_account_id,
                self.linked_opportunity_id,
                self.linked_event_id,
                self.linked_ticket_id,
            )
        ):
            raise ValueError("relevant slack records must include traceable business context ids")

        if self.provenance is not None:
            valid_provenance_ids = {
                value
                for value in (
                    self.linked_account_id,
                    self.linked_opportunity_id,
                    self.linked_event_id,
                    self.linked_ticket_id,
                )
                if value is not None
            }
            if self.provenance.object_id not in valid_provenance_ids:
                raise ValueError("slack provenance must reference one of the linked business ids")

        LabelGrounding(
            primary_category=self.primary_category,
            is_relevant=self.is_relevant,
            relevance_reason=self.relevance_reason,
            provenance=self.provenance,
        )

        return self

    def to_dataframe_row(self) -> dict[str, object]:
        return self.model_dump(mode="json")
