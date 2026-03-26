from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import AwareDatetime, Field, StringConstraints, model_validator

from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.labeling.grounding import LabelGrounding, LabelProvenance
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory, CommunicationTaxonomy

ScalarFieldValue: TypeAlias = bool | float | int | str | None
StructuredFieldValue: TypeAlias = ScalarFieldValue | list[str]

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SalesforceRecordId = Annotated[str, StringConstraints(pattern=r"^sf_record_[a-z0-9]+$")]
GenericRecordId = Annotated[str, StringConstraints(pattern=r"^[a-z_]+_[a-z0-9]+$")]
EmployeeId = Annotated[str, StringConstraints(pattern=r"^employee_[a-z0-9]+$")]
AccountId = Annotated[str, StringConstraints(pattern=r"^account_[a-z0-9]+$")]
ContactId = Annotated[str, StringConstraints(pattern=r"^contact_[a-z0-9]+$")]
LeadId = Annotated[str, StringConstraints(pattern=r"^lead_[a-z0-9]+$")]
OpportunityId = Annotated[str, StringConstraints(pattern=r"^opportunity_[a-z0-9]+$")]
EventId = Annotated[str, StringConstraints(pattern=r"^event_[a-z0-9]+$")]
CaseId = Annotated[str, StringConstraints(pattern=r"^ticket_issue_[a-z0-9]+$")]
CampaignId = Annotated[str, StringConstraints(pattern=r"^campaign_[a-z0-9]+$")]


class SalesforceObjectType(str, Enum):
    ACCOUNT = "Account"
    CONTACT = "Contact"
    LEAD = "Lead"
    OPPORTUNITY = "Opportunity"
    TASK = "Task"
    EVENT = "Event"
    CASE = "Case"
    CAMPAIGN = "Campaign"
    CAMPAIGN_MEMBER = "CampaignMember"
    NOTE = "Note"


class SalesforceOpportunityStage(str, Enum):
    QUALIFICATION = "qualification"
    DISCOVERY = "discovery"
    EVALUATION = "evaluation"
    PROPOSAL = "proposal"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"


class SalesforceRecord(EnterpriseModel):
    """One serialized Salesforce CRM row."""

    salesforce_record_id: SalesforceRecordId
    object_type: SalesforceObjectType
    record_id: GenericRecordId
    timestamp: AwareDatetime
    owner_employee_id: EmployeeId | None = None
    account_id: AccountId | None = None
    contact_id: ContactId | None = None
    lead_id: LeadId | None = None
    opportunity_id: OpportunityId | None = None
    event_id: EventId | None = None
    case_id: CaseId | None = None
    campaign_id: CampaignId | None = None
    parent_record_id: GenericRecordId | None = None
    attendee_contact_ids: list[ContactId] = Field(default_factory=list)
    stage: SalesforceOpportunityStage | None = None
    status: str | None = None
    subject: str | None = None
    text_body: str | None = None
    structured_fields: dict[str, StructuredFieldValue] = Field(default_factory=dict)
    primary_category: CommunicationCategory
    is_relevant: bool
    relevance_reason: NonEmptyText
    provenance: LabelProvenance | None = None
    source_system: Literal["salesforce"] = "salesforce"

    @model_validator(mode="after")
    def validate_salesforce_record(self) -> SalesforceRecord:
        taxonomy = CommunicationTaxonomy(
            primary_category=self.primary_category,
            is_relevant=self.is_relevant,
        )

        if (
            self.object_type == SalesforceObjectType.ACCOUNT
            and self.account_id != self.record_id
        ):
            raise ValueError("Account records must store their account_id in record_id")
        if self.object_type == SalesforceObjectType.CONTACT:
            if self.contact_id != self.record_id or self.account_id is None:
                raise ValueError(
                    "Contact records must include matching contact_id and account_id"
                )
        if (
            self.object_type == SalesforceObjectType.LEAD
            and self.lead_id != self.record_id
        ):
            raise ValueError("Lead records must store their lead_id in record_id")
        if self.object_type == SalesforceObjectType.OPPORTUNITY:
            if (
                self.opportunity_id != self.record_id
                or self.account_id is None
                or self.stage is None
            ):
                raise ValueError(
                    "Opportunity records must include matching opportunity_id, "
                    "account_id, and stage"
                )
        if self.object_type == SalesforceObjectType.EVENT:
            if self.event_id != self.record_id:
                raise ValueError("Event records must store their event_id in record_id")
        if self.object_type == SalesforceObjectType.CASE:
            if self.case_id != self.record_id or self.account_id is None:
                raise ValueError("Case records must include matching case_id and account_id")
        if (
            self.object_type == SalesforceObjectType.CAMPAIGN
            and self.campaign_id != self.record_id
        ):
            raise ValueError("Campaign records must store their campaign_id in record_id")
        if self.object_type == SalesforceObjectType.CAMPAIGN_MEMBER:
            if self.campaign_id is None or (
                (self.contact_id is None) == (self.lead_id is None)
            ):
                raise ValueError(
                    "CampaignMember records must include campaign_id and exactly "
                    "one of contact_id/lead_id"
                )
        if self.object_type == SalesforceObjectType.TASK and self.subject is None:
            raise ValueError("Task records require a subject")
        if self.object_type == SalesforceObjectType.NOTE and self.text_body is None:
            raise ValueError("Note records require text_body")

        if taxonomy.is_relevant and not any(
            value is not None
            for value in (
                self.account_id,
                self.opportunity_id,
                self.event_id,
                self.case_id,
                self.campaign_id,
            )
        ):
            raise ValueError(
                "Relevant Salesforce records must include traceable business context ids"
            )

        if self.provenance is not None:
            valid_provenance_ids = {
                value
                for value in (
                    self.record_id,
                    self.account_id,
                    self.contact_id,
                    self.lead_id,
                    self.opportunity_id,
                    self.event_id,
                    self.case_id,
                    self.campaign_id,
                    self.parent_record_id,
                )
                if value is not None
            }
            if self.provenance.object_id not in valid_provenance_ids:
                raise ValueError(
                    "Salesforce provenance must reference one of the row's linked ids"
                )

        LabelGrounding(
            primary_category=self.primary_category,
            is_relevant=self.is_relevant,
            relevance_reason=self.relevance_reason,
            provenance=self.provenance,
        )

        return self

    def to_dataframe_row(self) -> dict[str, object]:
        return self.model_dump(mode="json")
