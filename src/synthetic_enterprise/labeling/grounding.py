from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import Field, StringConstraints, model_validator

from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory, CommunicationTaxonomy

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ProvenanceId = Annotated[str, StringConstraints(pattern=r"^[a-z_]+_[a-z0-9]+$")]


class ProvenanceObjectType(str, Enum):
    ACCOUNT = "account"
    CONTACT = "contact"
    LEAD = "lead"
    OPPORTUNITY = "opportunity"
    EVENT = "event"
    TICKET = "ticket"
    CAMPAIGN = "campaign"
    TASK = "task"
    NOTE = "note"


class ProvenanceStrength(str, Enum):
    STRONG = "strong"
    WEAK = "weak"


PROVENANCE_PREFIXES = {
    ProvenanceObjectType.ACCOUNT: "account_",
    ProvenanceObjectType.CONTACT: "contact_",
    ProvenanceObjectType.LEAD: "lead_",
    ProvenanceObjectType.OPPORTUNITY: "opportunity_",
    ProvenanceObjectType.EVENT: "event_",
    ProvenanceObjectType.TICKET: "ticket_issue_",
    ProvenanceObjectType.CAMPAIGN: "campaign_",
    ProvenanceObjectType.TASK: "task_",
    ProvenanceObjectType.NOTE: "note_",
}

ALLOWED_PROVENANCE_BY_CATEGORY = {
    CommunicationCategory.PAIN_POINT: {
        ProvenanceObjectType.TICKET,
        ProvenanceObjectType.ACCOUNT,
        ProvenanceObjectType.CONTACT,
    },
    CommunicationCategory.COMPLAINT: {
        ProvenanceObjectType.TICKET,
        ProvenanceObjectType.ACCOUNT,
        ProvenanceObjectType.CONTACT,
    },
    CommunicationCategory.FEATURE_REQUEST: {
        ProvenanceObjectType.TICKET,
        ProvenanceObjectType.OPPORTUNITY,
        ProvenanceObjectType.ACCOUNT,
        ProvenanceObjectType.CONTACT,
    },
    CommunicationCategory.BUYING_SIGNAL: {
        ProvenanceObjectType.OPPORTUNITY,
        ProvenanceObjectType.ACCOUNT,
        ProvenanceObjectType.CONTACT,
        ProvenanceObjectType.LEAD,
    },
    CommunicationCategory.CHURN_RISK: {
        ProvenanceObjectType.ACCOUNT,
        ProvenanceObjectType.OPPORTUNITY,
        ProvenanceObjectType.TICKET,
        ProvenanceObjectType.CONTACT,
    },
    CommunicationCategory.EVENT_ATTENDANCE: {
        ProvenanceObjectType.EVENT,
        ProvenanceObjectType.CAMPAIGN,
        ProvenanceObjectType.CONTACT,
        ProvenanceObjectType.LEAD,
    },
    CommunicationCategory.FOLLOW_UP: {
        ProvenanceObjectType.EVENT,
        ProvenanceObjectType.OPPORTUNITY,
        ProvenanceObjectType.TASK,
        ProvenanceObjectType.ACCOUNT,
    },
    CommunicationCategory.BLOCKER: {
        ProvenanceObjectType.TICKET,
        ProvenanceObjectType.OPPORTUNITY,
        ProvenanceObjectType.TASK,
        ProvenanceObjectType.EVENT,
    },
    CommunicationCategory.ESCALATION: {
        ProvenanceObjectType.TICKET,
        ProvenanceObjectType.ACCOUNT,
        ProvenanceObjectType.CONTACT,
    },
    CommunicationCategory.DECISION_MAKER_SIGNAL: {
        ProvenanceObjectType.CONTACT,
        ProvenanceObjectType.ACCOUNT,
        ProvenanceObjectType.OPPORTUNITY,
        ProvenanceObjectType.EVENT,
    },
}

RELEVANT_REASON_PREFIXES = {
    CommunicationCategory.PAIN_POINT: "Customer pain is grounded in",
    CommunicationCategory.COMPLAINT: "Customer complaint is grounded in",
    CommunicationCategory.FEATURE_REQUEST: "Requested capability is grounded in",
    CommunicationCategory.BUYING_SIGNAL: "Commercial motion is grounded in",
    CommunicationCategory.CHURN_RISK: "Renewal risk is grounded in",
    CommunicationCategory.EVENT_ATTENDANCE: "Attendance is grounded in",
    CommunicationCategory.FOLLOW_UP: "Next action is grounded in",
    CommunicationCategory.BLOCKER: "Constraint is grounded in",
    CommunicationCategory.ESCALATION: "Escalation is grounded in",
    CommunicationCategory.DECISION_MAKER_SIGNAL: "Stakeholder authority is grounded in",
}

NOISE_REASON_PREFIXES = {
    CommunicationCategory.GREETINGS: "Greeting only",
    CommunicationCategory.STATUS_UPDATES: "Routine status is weakly tied to",
    CommunicationCategory.SOCIAL_CHATTER: "Social chatter only",
    CommunicationCategory.SCHEDULING_ONLY: "Calendar logistics are weakly tied to",
    CommunicationCategory.FYI_FORWARD: "Low-context forward is weakly tied to",
    CommunicationCategory.AUTOMATED_NOTIFICATION: "Automated notice is weakly tied to",
    CommunicationCategory.DUPLICATE_SUMMARY: "Duplicate summary is weakly tied to",
    CommunicationCategory.LOW_SIGNAL_CHECKIN: "Light check-in is weakly tied to",
    CommunicationCategory.IRRELEVANT_MARKETING: "Low-value marketing context is weakly tied to",
    CommunicationCategory.ADMIN_OPS: "Administrative coordination is weakly tied to",
}

NOISE_REASON_WITHOUT_PROVENANCE = {
    CommunicationCategory.GREETINGS: "Greeting only.",
    CommunicationCategory.STATUS_UPDATES: "Routine status only.",
    CommunicationCategory.SOCIAL_CHATTER: "Social chatter only.",
    CommunicationCategory.SCHEDULING_ONLY: "Calendar logistics only.",
    CommunicationCategory.FYI_FORWARD: "Low-context forward only.",
    CommunicationCategory.AUTOMATED_NOTIFICATION: "Automated notice only.",
    CommunicationCategory.DUPLICATE_SUMMARY: "Duplicate summary only.",
    CommunicationCategory.LOW_SIGNAL_CHECKIN: "Light check-in only.",
    CommunicationCategory.IRRELEVANT_MARKETING: "Low-value marketing only.",
    CommunicationCategory.ADMIN_OPS: "Administrative coordination only.",
}


class LabelProvenance(EnterpriseModel):
    """Primary grounding object used to justify a generated label."""

    object_type: ProvenanceObjectType
    object_id: ProvenanceId
    strength: ProvenanceStrength = ProvenanceStrength.STRONG
    explanation: NonEmptyText

    @model_validator(mode="after")
    def validate_provenance(self) -> LabelProvenance:
        prefix = PROVENANCE_PREFIXES[self.object_type]
        if not self.object_id.startswith(prefix):
            raise ValueError("provenance object_id must match the object_type prefix")
        return self


class LabelGrounding(EnterpriseModel):
    """Validated pairing of a label, its reason, and supporting provenance."""

    primary_category: CommunicationCategory
    is_relevant: bool
    relevance_reason: NonEmptyText
    provenance: LabelProvenance | None = Field(default=None)

    @model_validator(mode="after")
    def validate_grounding(self) -> LabelGrounding:
        taxonomy = CommunicationTaxonomy(
            primary_category=self.primary_category,
            is_relevant=self.is_relevant,
        )

        if taxonomy.is_relevant:
            if self.provenance is None:
                raise ValueError("relevant labels require a provenance object")
            if self.provenance.strength != ProvenanceStrength.STRONG:
                raise ValueError("relevant labels require strong provenance")
            allowed_types = ALLOWED_PROVENANCE_BY_CATEGORY[self.primary_category]
            if self.provenance.object_type not in allowed_types:
                raise ValueError("provenance object_type contradicts the assigned label")
        elif self.provenance is not None and self.provenance.strength == ProvenanceStrength.STRONG:
            raise ValueError("noise labels cannot use strong provenance")

        expected_reason = build_relevance_reason(
            primary_category=self.primary_category,
            is_relevant=self.is_relevant,
            provenance=self.provenance,
        )
        if self.relevance_reason != expected_reason:
            raise ValueError("relevance_reason must match the grounded label explanation")

        return self


def build_relevance_reason(
    *,
    primary_category: CommunicationCategory,
    is_relevant: bool,
    provenance: LabelProvenance | None,
) -> str:
    if is_relevant:
        if provenance is None:
            raise ValueError("relevant labels require provenance to build a reason")
        prefix = RELEVANT_REASON_PREFIXES[primary_category]
        return (
            f"{prefix} {provenance.object_type.value} {provenance.object_id}: "
            f"{provenance.explanation}"
        )

    if provenance is None:
        return NOISE_REASON_WITHOUT_PROVENANCE[primary_category]

    prefix = NOISE_REASON_PREFIXES[primary_category]
    return (
        f"{prefix} {provenance.object_type.value} {provenance.object_id}: "
        f"{provenance.explanation}"
    )
