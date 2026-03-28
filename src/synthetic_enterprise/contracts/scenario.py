from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.labeling.grounding import LabelProvenance, ProvenanceStrength
from synthetic_enterprise.labeling.taxonomy import RELEVANT_CATEGORIES, CommunicationCategory

ScenarioSystem = Literal["email", "slack", "teams", "salesforce"]


class ScenarioFamily(str, Enum):
    PRE_SALES_DISCOVERY = "pre_sales_discovery"
    PROCUREMENT_DELAY = "procurement_delay"
    PILOT_SUCCESS = "pilot_success"
    ROLLOUT_RISK = "rollout_risk"
    SUPPORT_ESCALATION = "support_escalation"
    FEATURE_GAP_REVIEW = "feature_gap_review"
    RENEWAL_RISK = "renewal_risk"
    EXECUTIVE_SPONSOR_REVIEW = "executive_sponsor_review"
    EVENT_INVITE_TO_ATTENDANCE = "event_invite_to_attendance"
    CHAMPION_DEPARTURE = "champion_departure"
    STAKEHOLDER_EXPANSION = "stakeholder_expansion"
    CONTRACT_REDLINE_DELAY = "contract_redline_delay"


class ScenarioLifecycleStage(str, Enum):
    IDENTIFIED = "identified"
    ACTIVE = "active"
    STALLED = "stalled"
    REOPENED = "reopened"
    COMPLETED = "completed"
    CLOSED = "closed"


class ScenarioEntityStateType(str, Enum):
    OPPORTUNITY_STAGE = "opportunity_stage"
    TICKET_SEVERITY = "ticket_severity"
    ACCOUNT_HEALTH = "account_health"
    STAKEHOLDER_ENGAGEMENT = "stakeholder_engagement"
    EVENT_PARTICIPATION = "event_participation"
    OWNERSHIP = "ownership"
    FOLLOW_UP_STATUS = "follow_up_status"


class ScenarioCommunicationTrigger(EnterpriseModel):
    """Downstream communication cue caused by a state transition."""

    category: CommunicationCategory
    reason: str = Field(min_length=1)


OPPORTUNITY_STAGE_TRANSITIONS = {
    None: {"qualification", "discovery"},
    "qualification": {"discovery", "closed_lost"},
    "discovery": {"evaluation", "closed_lost"},
    "evaluation": {"proposal", "closed_lost"},
    "proposal": {"closed_won", "closed_lost", "evaluation"},
    "closed_won": set(),
    "closed_lost": set(),
}

TICKET_SEVERITY_TRANSITIONS = {
    None: {"low", "medium", "high"},
    "low": {"medium"},
    "medium": {"high", "low"},
    "high": {"critical", "medium"},
    "critical": {"high"},
}

ACCOUNT_HEALTH_TRANSITIONS = {
    None: {"healthy", "watchlist"},
    "healthy": {"watchlist", "at_risk"},
    "watchlist": {"healthy", "at_risk"},
    "at_risk": {"healthy", "watchlist"},
}

STAKEHOLDER_ENGAGEMENT_TRANSITIONS = {
    None: {"engaged", "neutral"},
    "neutral": {"engaged", "disengaged"},
    "engaged": {"expanding", "disengaged"},
    "expanding": {"engaged", "disengaged"},
    "disengaged": {"engaged"},
}

EVENT_PARTICIPATION_TRANSITIONS = {
    None: {"invited"},
    "invited": {"accepted", "tentative"},
    "accepted": {"tentative", "attended", "no_show"},
    "tentative": {"accepted", "attended", "no_show"},
    "attended": set(),
    "no_show": set(),
}

FOLLOW_UP_STATUS_TRANSITIONS = {
    None: {"pending"},
    "pending": {"completed", "slipped"},
    "slipped": {"completed"},
    "completed": set(),
}

ENTITY_ID_PREFIXES = {
    ScenarioEntityStateType.OPPORTUNITY_STAGE: ("opportunity_",),
    ScenarioEntityStateType.TICKET_SEVERITY: ("ticket_issue_",),
    ScenarioEntityStateType.ACCOUNT_HEALTH: ("account_",),
    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT: ("contact_",),
    ScenarioEntityStateType.EVENT_PARTICIPATION: ("event_",),
    ScenarioEntityStateType.OWNERSHIP: ("account_", "opportunity_", "ticket_issue_"),
    ScenarioEntityStateType.FOLLOW_UP_STATUS: (
        "account_",
        "opportunity_",
        "event_",
        "ticket_issue_",
    ),
}

STATE_TRANSITION_RULES = {
    ScenarioEntityStateType.OPPORTUNITY_STAGE: OPPORTUNITY_STAGE_TRANSITIONS,
    ScenarioEntityStateType.TICKET_SEVERITY: TICKET_SEVERITY_TRANSITIONS,
    ScenarioEntityStateType.ACCOUNT_HEALTH: ACCOUNT_HEALTH_TRANSITIONS,
    ScenarioEntityStateType.STAKEHOLDER_ENGAGEMENT: STAKEHOLDER_ENGAGEMENT_TRANSITIONS,
    ScenarioEntityStateType.EVENT_PARTICIPATION: EVENT_PARTICIPATION_TRANSITIONS,
    ScenarioEntityStateType.FOLLOW_UP_STATUS: FOLLOW_UP_STATUS_TRANSITIONS,
}


class ScenarioEntityStateTransition(EnterpriseModel):
    """One explicit business-state transition inside a scenario lifecycle step."""

    transition_type: ScenarioEntityStateType
    entity_id: str = Field(min_length=1)
    from_state: str | None = None
    to_state: str = Field(min_length=1)
    actor_id: str | None = None

    @model_validator(mode="after")
    def validate_transition(self) -> ScenarioEntityStateTransition:
        valid_prefixes = ENTITY_ID_PREFIXES[self.transition_type]
        if not self.entity_id.startswith(valid_prefixes):
            raise ValueError("entity_id does not match the transition type")

        rules = STATE_TRANSITION_RULES.get(self.transition_type)
        if rules is not None:
            allowed_to_states = rules.get(self.from_state)
            if allowed_to_states is None or self.to_state not in allowed_to_states:
                raise ValueError(f"invalid {self.transition_type.value} transition")
        elif self.transition_type == ScenarioEntityStateType.OWNERSHIP:
            if self.from_state is not None and self.from_state == self.to_state:
                raise ValueError("invalid ownership transition")
            if self.from_state is not None and not self.from_state.startswith("employee_"):
                raise ValueError("invalid ownership transition")
            if not self.to_state.startswith("employee_"):
                raise ValueError("invalid ownership transition")

        if self.transition_type == ScenarioEntityStateType.EVENT_PARTICIPATION:
            if self.actor_id is None or not self.actor_id.startswith("contact_"):
                raise ValueError("event_participation transitions require a contact actor_id")
        elif self.actor_id is not None and not self.actor_id.startswith(("contact_", "employee_")):
            raise ValueError("actor_id must reference a contact or employee")

        return self


ALLOWED_LIFECYCLE_STAGE_TRANSITIONS = {
    ScenarioLifecycleStage.IDENTIFIED: {ScenarioLifecycleStage.ACTIVE},
    ScenarioLifecycleStage.ACTIVE: {
        ScenarioLifecycleStage.STALLED,
        ScenarioLifecycleStage.COMPLETED,
        ScenarioLifecycleStage.CLOSED,
    },
    ScenarioLifecycleStage.STALLED: {
        ScenarioLifecycleStage.REOPENED,
        ScenarioLifecycleStage.CLOSED,
    },
    ScenarioLifecycleStage.REOPENED: {
        ScenarioLifecycleStage.ACTIVE,
        ScenarioLifecycleStage.COMPLETED,
        ScenarioLifecycleStage.CLOSED,
    },
    ScenarioLifecycleStage.COMPLETED: set(),
    ScenarioLifecycleStage.CLOSED: set(),
}


class ScenarioCategoryGrounding(EnterpriseModel):
    """Scenario-state grounding for one relevant category."""

    category: CommunicationCategory
    provenance: LabelProvenance

    @model_validator(mode="after")
    def validate_grounding(self) -> ScenarioCategoryGrounding:
        if self.category not in RELEVANT_CATEGORIES:
            raise ValueError("scenario category groundings must use relevant labels")
        if self.provenance.strength != ProvenanceStrength.STRONG:
            raise ValueError("scenario category groundings require strong provenance")
        return self


class ScenarioStateTransition(EnterpriseModel):
    """One timestamped scenario lifecycle transition."""

    transition_id: str = Field(pattern=r"^scenario_transition_[a-z0-9]+$")
    stage: ScenarioLifecycleStage
    occurred_at: AwareDatetime
    summary: str = Field(min_length=1)
    state_changes: dict[str, str] = Field(default_factory=dict)
    entity_state_transitions: tuple[ScenarioEntityStateTransition, ...] = Field(
        default_factory=tuple
    )
    communication_triggers: tuple[ScenarioCommunicationTrigger, ...] = Field(
        default_factory=tuple
    )
    category_groundings: tuple[ScenarioCategoryGrounding, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_state_changes(self) -> ScenarioStateTransition:
        if not self.state_changes:
            raise ValueError("scenario transitions must emit at least one state change")
        if self.entity_state_transitions and not self.communication_triggers:
            raise ValueError(
                "scenario transitions with entity state changes require communication triggers"
            )
        return self


class Scenario(EnterpriseModel):
    """Lifecycle-based business scenario that can drive multi-system rendering."""

    scenario_id: str = Field(pattern=r"^scenario_[a-z0-9]+$")
    kind: ScenarioFamily | str
    company_id: str = Field(pattern=r"^company_[a-z0-9]+$")
    account_ids: tuple[str, ...] = Field(default_factory=tuple)
    employee_ids: tuple[str, ...] = Field(default_factory=tuple)
    contact_ids: tuple[str, ...] = Field(default_factory=tuple)
    opportunity_ids: tuple[str, ...] = Field(default_factory=tuple)
    event_ids: tuple[str, ...] = Field(default_factory=tuple)
    ticket_ids: tuple[str, ...] = Field(default_factory=tuple)
    surface_systems: tuple[ScenarioSystem, ...] = (
        "email",
        "slack",
        "teams",
        "salesforce",
    )
    transitions: tuple[ScenarioStateTransition, ...] = Field(default_factory=tuple)
    current_stage: ScenarioLifecycleStage = ScenarioLifecycleStage.IDENTIFIED
    expected_labels: tuple[CommunicationCategory, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Scenario:
        if not self.transitions:
            raise ValueError("scenarios must contain at least one transition")
        if self.current_stage != self.transitions[-1].stage:
            raise ValueError("current_stage must match the last transition stage")

        previous_transition: ScenarioStateTransition | None = None
        for transition in self.transitions:
            if previous_transition is not None:
                if transition.occurred_at <= previous_transition.occurred_at:
                    raise ValueError("scenario transitions must be strictly time ordered")
                allowed_next_stages = ALLOWED_LIFECYCLE_STAGE_TRANSITIONS[
                    previous_transition.stage
                ]
                if transition.stage not in allowed_next_stages:
                    raise ValueError("scenario contains an invalid lifecycle stage transition")
            previous_transition = transition

        for category in self.expected_labels:
            if category not in RELEVANT_CATEGORIES:
                raise ValueError("scenario expected_labels must use relevant labels")
            if self.grounding_for(category) is None:
                raise ValueError("scenario expected_labels must be grounded in scenario state")

        return self

    def grounding_for(
        self,
        category: CommunicationCategory,
    ) -> ScenarioCategoryGrounding | None:
        for transition in self.transitions:
            for grounding in transition.category_groundings:
                if grounding.category == category:
                    return grounding
        return None
