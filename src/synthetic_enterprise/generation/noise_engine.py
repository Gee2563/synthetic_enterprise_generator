from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory

NoiseSourceSystem = Literal["email", "slack", "teams", "salesforce"]


class NoiseFamily(str, Enum):
    VAGUE_FOLLOW_UP = "vague_follow_up"
    DOCUMENT_REVIEW_PING = "document_review_ping"
    AGENDA_COORDINATION = "agenda_coordination"
    INTERNAL_FYI_SUMMARY = "internal_fyi_summary"
    DUPLICATE_REMINDER = "duplicate_reminder"
    OWNERSHIP_AMBIGUITY = "ownership_ambiguity"
    SOCIAL_BANTER = "social_banter"
    COMPLIANCE_ADMIN_REMINDER = "compliance_admin_reminder"
    TRAVEL_LOGISTICS = "travel_logistics"
    ACCESS_REQUEST = "access_request"
    EMPTY_CHECK_IN = "empty_check_in"
    STATUS_NUDGE = "status_nudge"
    PARTIAL_HANDOFF = "partial_handoff"
    IRRELEVANT_FORWARDED_CHAIN = "irrelevant_forwarded_chain"
    TOOL_OUTAGE_CHATTER = "tool_outage_chatter"
    RECRUITING_HR_CHATTER = "recruiting_hr_chatter"
    PROCUREMENT_ADMIN = "procurement_admin"
    STALE_CRM_UPDATE = "stale_crm_update"


DEFAULT_NOISE_FAMILY_WEIGHTS: Mapping[NoiseFamily, float] = {
    NoiseFamily.VAGUE_FOLLOW_UP: 1.3,
    NoiseFamily.DOCUMENT_REVIEW_PING: 1.0,
    NoiseFamily.AGENDA_COORDINATION: 1.1,
    NoiseFamily.INTERNAL_FYI_SUMMARY: 0.9,
    NoiseFamily.DUPLICATE_REMINDER: 0.8,
    NoiseFamily.OWNERSHIP_AMBIGUITY: 1.0,
    NoiseFamily.SOCIAL_BANTER: 1.0,
    NoiseFamily.COMPLIANCE_ADMIN_REMINDER: 0.9,
    NoiseFamily.TRAVEL_LOGISTICS: 0.8,
    NoiseFamily.ACCESS_REQUEST: 0.9,
    NoiseFamily.EMPTY_CHECK_IN: 1.0,
    NoiseFamily.STATUS_NUDGE: 1.1,
    NoiseFamily.PARTIAL_HANDOFF: 0.8,
    NoiseFamily.IRRELEVANT_FORWARDED_CHAIN: 0.8,
    NoiseFamily.TOOL_OUTAGE_CHATTER: 0.7,
    NoiseFamily.RECRUITING_HR_CHATTER: 0.7,
    NoiseFamily.PROCUREMENT_ADMIN: 0.8,
    NoiseFamily.STALE_CRM_UPDATE: 0.8,
}

NOISE_FAMILY_CATEGORIES: Mapping[NoiseFamily, CommunicationCategory] = {
    NoiseFamily.VAGUE_FOLLOW_UP: CommunicationCategory.LOW_SIGNAL_CHECKIN,
    NoiseFamily.DOCUMENT_REVIEW_PING: CommunicationCategory.STATUS_UPDATES,
    NoiseFamily.AGENDA_COORDINATION: CommunicationCategory.SCHEDULING_ONLY,
    NoiseFamily.INTERNAL_FYI_SUMMARY: CommunicationCategory.FYI_FORWARD,
    NoiseFamily.DUPLICATE_REMINDER: CommunicationCategory.DUPLICATE_SUMMARY,
    NoiseFamily.OWNERSHIP_AMBIGUITY: CommunicationCategory.LOW_SIGNAL_CHECKIN,
    NoiseFamily.SOCIAL_BANTER: CommunicationCategory.SOCIAL_CHATTER,
    NoiseFamily.COMPLIANCE_ADMIN_REMINDER: CommunicationCategory.ADMIN_OPS,
    NoiseFamily.TRAVEL_LOGISTICS: CommunicationCategory.SCHEDULING_ONLY,
    NoiseFamily.ACCESS_REQUEST: CommunicationCategory.ADMIN_OPS,
    NoiseFamily.EMPTY_CHECK_IN: CommunicationCategory.GREETINGS,
    NoiseFamily.STATUS_NUDGE: CommunicationCategory.STATUS_UPDATES,
    NoiseFamily.PARTIAL_HANDOFF: CommunicationCategory.STATUS_UPDATES,
    NoiseFamily.IRRELEVANT_FORWARDED_CHAIN: CommunicationCategory.FYI_FORWARD,
    NoiseFamily.TOOL_OUTAGE_CHATTER: CommunicationCategory.AUTOMATED_NOTIFICATION,
    NoiseFamily.RECRUITING_HR_CHATTER: CommunicationCategory.SOCIAL_CHATTER,
    NoiseFamily.PROCUREMENT_ADMIN: CommunicationCategory.ADMIN_OPS,
    NoiseFamily.STALE_CRM_UPDATE: CommunicationCategory.STATUS_UPDATES,
}

SOURCE_NOISE_FAMILIES: Mapping[NoiseSourceSystem, tuple[NoiseFamily, ...]] = {
    "email": (
        NoiseFamily.VAGUE_FOLLOW_UP,
        NoiseFamily.DOCUMENT_REVIEW_PING,
        NoiseFamily.AGENDA_COORDINATION,
        NoiseFamily.INTERNAL_FYI_SUMMARY,
        NoiseFamily.DUPLICATE_REMINDER,
        NoiseFamily.OWNERSHIP_AMBIGUITY,
        NoiseFamily.COMPLIANCE_ADMIN_REMINDER,
        NoiseFamily.TRAVEL_LOGISTICS,
        NoiseFamily.EMPTY_CHECK_IN,
        NoiseFamily.IRRELEVANT_FORWARDED_CHAIN,
        NoiseFamily.PROCUREMENT_ADMIN,
    ),
    "slack": (
        NoiseFamily.VAGUE_FOLLOW_UP,
        NoiseFamily.DOCUMENT_REVIEW_PING,
        NoiseFamily.OWNERSHIP_AMBIGUITY,
        NoiseFamily.SOCIAL_BANTER,
        NoiseFamily.COMPLIANCE_ADMIN_REMINDER,
        NoiseFamily.ACCESS_REQUEST,
        NoiseFamily.EMPTY_CHECK_IN,
        NoiseFamily.STATUS_NUDGE,
        NoiseFamily.PARTIAL_HANDOFF,
        NoiseFamily.TOOL_OUTAGE_CHATTER,
        NoiseFamily.RECRUITING_HR_CHATTER,
        NoiseFamily.PROCUREMENT_ADMIN,
    ),
    "teams": (
        NoiseFamily.DOCUMENT_REVIEW_PING,
        NoiseFamily.AGENDA_COORDINATION,
        NoiseFamily.INTERNAL_FYI_SUMMARY,
        NoiseFamily.DUPLICATE_REMINDER,
        NoiseFamily.OWNERSHIP_AMBIGUITY,
        NoiseFamily.COMPLIANCE_ADMIN_REMINDER,
        NoiseFamily.ACCESS_REQUEST,
        NoiseFamily.STATUS_NUDGE,
        NoiseFamily.PARTIAL_HANDOFF,
        NoiseFamily.IRRELEVANT_FORWARDED_CHAIN,
        NoiseFamily.TOOL_OUTAGE_CHATTER,
    ),
    "salesforce": (
        NoiseFamily.VAGUE_FOLLOW_UP,
        NoiseFamily.DOCUMENT_REVIEW_PING,
        NoiseFamily.DUPLICATE_REMINDER,
        NoiseFamily.OWNERSHIP_AMBIGUITY,
        NoiseFamily.COMPLIANCE_ADMIN_REMINDER,
        NoiseFamily.ACCESS_REQUEST,
        NoiseFamily.PARTIAL_HANDOFF,
        NoiseFamily.IRRELEVANT_FORWARDED_CHAIN,
        NoiseFamily.PROCUREMENT_ADMIN,
        NoiseFamily.STALE_CRM_UPDATE,
    ),
}


def noise_category_for_family(family: NoiseFamily) -> CommunicationCategory:
    return NOISE_FAMILY_CATEGORIES[family]


@dataclass(slots=True)
class NoiseEngine:
    """Deterministically plan broad noise families for each source system."""

    context: GeneratorContext

    def plan(
        self,
        *,
        source_system: NoiseSourceSystem,
        count: int,
    ) -> tuple[NoiseFamily, ...]:
        if count <= 0:
            return ()

        supported_families = SOURCE_NOISE_FAMILIES[source_system]
        weights = self._supported_weights(supported_families)
        if not weights:
            fallback_family = supported_families[0]
            return tuple(fallback_family for _ in range(count))

        allocations = self._allocated_counts(
            source_system=source_system,
            weights=weights,
            count=count,
        )
        planned: list[tuple[int, NoiseFamily]] = []

        for family, family_count in allocations.items():
            for occurrence in range(family_count):
                planned.append(
                    (
                        self.context.derive_seed(
                            f"noise-order:{source_system}:{family.value}:{occurrence}"
                        ),
                        family,
                    )
                )

        return tuple(
            family
            for _, family in sorted(
                planned,
                key=lambda item: (item[0], item[1].value),
            )
        )

    def _supported_weights(
        self,
        supported_families: Sequence[NoiseFamily],
    ) -> dict[NoiseFamily, float]:
        resolved: dict[NoiseFamily, float] = {}

        for family in supported_families:
            weight = self._weight_for_family(family)
            if weight > 0.0:
                resolved[family] = weight

        return resolved

    def _weight_for_family(self, family: NoiseFamily) -> float:
        configured = self.context.config.noise_family_weights
        if configured is None:
            return DEFAULT_NOISE_FAMILY_WEIGHTS[family]

        return max(0.0, _lookup_configured_weight(configured, family))

    def _allocated_counts(
        self,
        *,
        source_system: NoiseSourceSystem,
        weights: Mapping[NoiseFamily, float],
        count: int,
    ) -> dict[NoiseFamily, int]:
        total_weight = sum(weights.values())
        allocations = {family: 0 for family in weights}
        remainders: list[tuple[float, int, NoiseFamily]] = []

        for family, weight in weights.items():
            exact = count * (weight / total_weight)
            whole = int(exact)
            allocations[family] = whole
            remainders.append(
                (
                    exact - whole,
                    self.context.derive_seed(
                        f"noise-remainder:{source_system}:{family.value}"
                    ),
                    family,
                )
            )

        remaining = count - sum(allocations.values())
        for _, _, family in sorted(
            remainders,
            key=lambda item: (-item[0], item[1], item[2].value),
        )[:remaining]:
            allocations[family] += 1

        return allocations


def _lookup_configured_weight(
    configured: Mapping[str, float],
    family: NoiseFamily,
) -> float:
    if family.value in configured:
        return configured[family.value]
    return configured.get(str(family), 0.0)
