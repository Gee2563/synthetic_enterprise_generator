from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RelevantCategory(str, Enum):
    PAIN_POINT = "pain_point"
    COMPLAINT = "complaint"
    FEATURE_REQUEST = "feature_request"
    BUYING_SIGNAL = "buying_signal"
    CHURN_RISK = "churn_risk"
    EVENT_ATTENDANCE = "event_attendance"
    FOLLOW_UP = "follow_up"
    BLOCKER = "blocker"
    ESCALATION = "escalation"
    DECISION_MAKER_SIGNAL = "decision_maker_signal"


class NoiseCategory(str, Enum):
    GREETINGS = "greetings"
    STATUS_UPDATES = "status_updates"
    SOCIAL_CHATTER = "social_chatter"
    SCHEDULING_ONLY = "scheduling_only"
    FYI_FORWARD = "fyi_forward"
    AUTOMATED_NOTIFICATION = "automated_notification"
    DUPLICATE_SUMMARY = "duplicate_summary"
    LOW_SIGNAL_CHECKIN = "low_signal_checkin"
    IRRELEVANT_MARKETING = "irrelevant_marketing"
    ADMIN_OPS = "admin_ops"


class CommunicationCategory(str, Enum):
    PAIN_POINT = RelevantCategory.PAIN_POINT.value
    COMPLAINT = RelevantCategory.COMPLAINT.value
    FEATURE_REQUEST = RelevantCategory.FEATURE_REQUEST.value
    BUYING_SIGNAL = RelevantCategory.BUYING_SIGNAL.value
    CHURN_RISK = RelevantCategory.CHURN_RISK.value
    EVENT_ATTENDANCE = RelevantCategory.EVENT_ATTENDANCE.value
    FOLLOW_UP = RelevantCategory.FOLLOW_UP.value
    BLOCKER = RelevantCategory.BLOCKER.value
    ESCALATION = RelevantCategory.ESCALATION.value
    DECISION_MAKER_SIGNAL = RelevantCategory.DECISION_MAKER_SIGNAL.value
    GREETINGS = NoiseCategory.GREETINGS.value
    STATUS_UPDATES = NoiseCategory.STATUS_UPDATES.value
    SOCIAL_CHATTER = NoiseCategory.SOCIAL_CHATTER.value
    SCHEDULING_ONLY = NoiseCategory.SCHEDULING_ONLY.value
    FYI_FORWARD = NoiseCategory.FYI_FORWARD.value
    AUTOMATED_NOTIFICATION = NoiseCategory.AUTOMATED_NOTIFICATION.value
    DUPLICATE_SUMMARY = NoiseCategory.DUPLICATE_SUMMARY.value
    LOW_SIGNAL_CHECKIN = NoiseCategory.LOW_SIGNAL_CHECKIN.value
    IRRELEVANT_MARKETING = NoiseCategory.IRRELEVANT_MARKETING.value
    ADMIN_OPS = NoiseCategory.ADMIN_OPS.value


RELEVANT_CATEGORIES = frozenset(
    CommunicationCategory(category.value) for category in RelevantCategory
)
NOISE_CATEGORIES = frozenset(CommunicationCategory(category.value) for category in NoiseCategory)


def is_relevant_category(category: CommunicationCategory) -> bool:
    return category in RELEVANT_CATEGORIES


class CommunicationTaxonomy(BaseModel):
    """Primary taxonomy assignment for a generated communication record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    primary_category: CommunicationCategory
    is_relevant: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def derive_and_validate_relevance(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_category = data.get("primary_category")
        if raw_category is None:
            return data

        category = CommunicationCategory(raw_category)
        expected_flag = is_relevant_category(category)
        provided_flag = data.get("is_relevant")

        if provided_flag is None:
            return {**data, "is_relevant": expected_flag}

        if provided_flag != expected_flag:
            raise ValueError("is_relevant must align with the primary_category mapping")

        return data


class TaxonomyConfig(BaseModel):
    """Configuration for relevance/noise mix in generated datasets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    noise_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
