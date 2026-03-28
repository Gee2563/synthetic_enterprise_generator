from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from synthetic_enterprise.generation.context import GeneratorContext


class HardNegativeFamily(str, Enum):
    EVENT_NO_ATTENDANCE = "event_no_attendance"
    BUDGET_NON_BUYING = "budget_non_buying"
    COMPLAINT_NON_PRODUCT = "complaint_non_product"
    ESCALATION_NO_BLOCKER = "escalation_no_blocker"
    EXECUTIVE_VISIBILITY_ONLY = "executive_visibility_only"
    FOLLOW_UP_NO_ACTION = "follow_up_no_action"


DEFAULT_HARD_NEGATIVE_FAMILY_WEIGHTS: Mapping[HardNegativeFamily, float] = {
    HardNegativeFamily.EVENT_NO_ATTENDANCE: 1.0,
    HardNegativeFamily.BUDGET_NON_BUYING: 1.0,
    HardNegativeFamily.COMPLAINT_NON_PRODUCT: 0.9,
    HardNegativeFamily.ESCALATION_NO_BLOCKER: 0.9,
    HardNegativeFamily.EXECUTIVE_VISIBILITY_ONLY: 0.8,
    HardNegativeFamily.FOLLOW_UP_NO_ACTION: 1.1,
}


def hard_negative_count(*, template_count: int, ratio: float) -> int:
    """Return the deterministic number of hard-negative rows to emit."""

    if template_count <= 0 or ratio <= 0.0:
        return 0

    return min(template_count, max(0, int(round(template_count * ratio))))


def hard_negative_explanation(family: HardNegativeFamily) -> str:
    if family == HardNegativeFamily.EVENT_NO_ATTENDANCE:
        return "Event is mentioned without attendance or RSVP change."
    if family == HardNegativeFamily.BUDGET_NON_BUYING:
        return "Budget wording is administrative rather than evidence of buying intent."
    if family == HardNegativeFamily.COMPLAINT_NON_PRODUCT:
        return "Complaint concerns travel or logistics rather than product pain."
    if family == HardNegativeFamily.ESCALATION_NO_BLOCKER:
        return "Escalation wording reflects schedule pressure and not a live blocker."
    if family == HardNegativeFamily.EXECUTIVE_VISIBILITY_ONLY:
        return (
            "Executive mention is for visibility only and not an engaged "
            "decision-maker signal."
        )
    return "Follow up is noted with no required action or owner."


@dataclass(slots=True)
class HardNegativeEngine:
    """Deterministically allocate hard-negative families for near-miss rows."""

    context: GeneratorContext

    def plan(self, *, count: int) -> tuple[HardNegativeFamily, ...]:
        if count <= 0:
            return ()

        weights = self._family_weights()
        if not weights:
            return ()

        allocations = self._allocated_counts(weights=weights, count=count)
        planned: list[tuple[int, HardNegativeFamily]] = []

        for family, family_count in allocations.items():
            for occurrence in range(family_count):
                planned.append(
                    (
                        self.context.derive_seed(
                            f"hard-negative-order:{family.value}:{occurrence}"
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

    def _family_weights(self) -> dict[HardNegativeFamily, float]:
        weights: dict[HardNegativeFamily, float] = {}
        configured = self.context.config.hard_negative_family_weights

        for family in HardNegativeFamily:
            if configured is None:
                weight = DEFAULT_HARD_NEGATIVE_FAMILY_WEIGHTS[family]
            else:
                weight = max(0.0, _configured_weight(configured, family))
            if weight > 0.0:
                weights[family] = weight

        return weights

    def _allocated_counts(
        self,
        *,
        weights: Mapping[HardNegativeFamily, float],
        count: int,
    ) -> dict[HardNegativeFamily, int]:
        total_weight = sum(weights.values())
        allocations = {family: 0 for family in weights}
        remainders: list[tuple[float, int, HardNegativeFamily]] = []

        for family, weight in weights.items():
            exact = count * (weight / total_weight)
            whole = int(exact)
            allocations[family] = whole
            remainders.append(
                (
                    exact - whole,
                    self.context.derive_seed(f"hard-negative-remainder:{family.value}"),
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


def _configured_weight(
    configured: Mapping[str, float],
    family: HardNegativeFamily,
) -> float:
    if family.value in configured:
        return configured[family.value]
    return configured.get(str(family), 0.0)
