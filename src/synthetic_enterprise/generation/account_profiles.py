from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from synthetic_enterprise.contracts.scenario import ScenarioFamily
from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory


class AccountBehaviorProfileType(str, Enum):
    HIGHLY_ENGAGED_CHAMPION_LED = "highly_engaged_champion_led"
    PROCUREMENT_HEAVY_SLOW_MOVER = "procurement_heavy_slow_mover"
    SUPPORT_INTENSIVE_UNHAPPY = "support_intensive_unhappy"
    EXECUTIVE_SPONSORED_EXPANSION = "executive_sponsored_expansion"
    LOW_TOUCH_DORMANT = "low_touch_dormant"
    EVENT_ACTIVE_LOW_CONVERSION = "event_active_low_conversion"
    IMPLEMENTATION_STRUGGLING = "implementation_struggling"


@dataclass(frozen=True, slots=True)
class AccountBehaviorProfile:
    profile_type: AccountBehaviorProfileType
    preferred_systems: tuple[str, ...]
    scheduled_family: ScenarioFamily
    emphasized_categories: tuple[CommunicationCategory, ...]


PROFILE_CATALOG: dict[AccountBehaviorProfileType, AccountBehaviorProfile] = {
    AccountBehaviorProfileType.HIGHLY_ENGAGED_CHAMPION_LED: AccountBehaviorProfile(
        profile_type=AccountBehaviorProfileType.HIGHLY_ENGAGED_CHAMPION_LED,
        preferred_systems=("email", "slack", "teams", "salesforce"),
        scheduled_family=ScenarioFamily.STAKEHOLDER_EXPANSION,
        emphasized_categories=(
            CommunicationCategory.BUYING_SIGNAL,
            CommunicationCategory.FOLLOW_UP,
        ),
    ),
    AccountBehaviorProfileType.PROCUREMENT_HEAVY_SLOW_MOVER: AccountBehaviorProfile(
        profile_type=AccountBehaviorProfileType.PROCUREMENT_HEAVY_SLOW_MOVER,
        preferred_systems=("email", "teams", "salesforce"),
        scheduled_family=ScenarioFamily.PROCUREMENT_DELAY,
        emphasized_categories=(
            CommunicationCategory.BLOCKER,
            CommunicationCategory.FOLLOW_UP,
        ),
    ),
    AccountBehaviorProfileType.SUPPORT_INTENSIVE_UNHAPPY: AccountBehaviorProfile(
        profile_type=AccountBehaviorProfileType.SUPPORT_INTENSIVE_UNHAPPY,
        preferred_systems=("email", "slack", "teams", "salesforce"),
        scheduled_family=ScenarioFamily.SUPPORT_ESCALATION,
        emphasized_categories=(
            CommunicationCategory.BLOCKER,
            CommunicationCategory.ESCALATION,
        ),
    ),
    AccountBehaviorProfileType.EXECUTIVE_SPONSORED_EXPANSION: AccountBehaviorProfile(
        profile_type=AccountBehaviorProfileType.EXECUTIVE_SPONSORED_EXPANSION,
        preferred_systems=("email", "teams", "salesforce"),
        scheduled_family=ScenarioFamily.EXECUTIVE_SPONSOR_REVIEW,
        emphasized_categories=(
            CommunicationCategory.DECISION_MAKER_SIGNAL,
            CommunicationCategory.BUYING_SIGNAL,
        ),
    ),
    AccountBehaviorProfileType.LOW_TOUCH_DORMANT: AccountBehaviorProfile(
        profile_type=AccountBehaviorProfileType.LOW_TOUCH_DORMANT,
        preferred_systems=("email", "teams", "salesforce"),
        scheduled_family=ScenarioFamily.RENEWAL_RISK,
        emphasized_categories=(CommunicationCategory.FOLLOW_UP,),
    ),
    AccountBehaviorProfileType.EVENT_ACTIVE_LOW_CONVERSION: AccountBehaviorProfile(
        profile_type=AccountBehaviorProfileType.EVENT_ACTIVE_LOW_CONVERSION,
        preferred_systems=("email", "slack", "teams", "salesforce"),
        scheduled_family=ScenarioFamily.EVENT_INVITE_TO_ATTENDANCE,
        emphasized_categories=(CommunicationCategory.EVENT_ATTENDANCE,),
    ),
    AccountBehaviorProfileType.IMPLEMENTATION_STRUGGLING: AccountBehaviorProfile(
        profile_type=AccountBehaviorProfileType.IMPLEMENTATION_STRUGGLING,
        preferred_systems=("email", "slack", "teams"),
        scheduled_family=ScenarioFamily.ROLLOUT_RISK,
        emphasized_categories=(
            CommunicationCategory.BLOCKER,
            CommunicationCategory.FOLLOW_UP,
        ),
    ),
}

PROFILE_ORDER: tuple[AccountBehaviorProfileType, ...] = tuple(PROFILE_CATALOG)


@dataclass(slots=True)
class AccountBehaviorResolver:
    context: GeneratorContext
    enterprise: EnterpriseGraph
    _profiles_by_account_id: dict[str, AccountBehaviorProfile] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        ordered_accounts = sorted(
            self.enterprise.customer_accounts,
            key=lambda account: account.id,
        )
        if not ordered_accounts:
            self._profiles_by_account_id = {}
            return

        offset = self.context.derive_seed("account-behavior-profile-offset") % len(
            PROFILE_ORDER
        )
        self._profiles_by_account_id = {
            account.id: PROFILE_CATALOG[
                PROFILE_ORDER[(index + offset) % len(PROFILE_ORDER)]
            ]
            for index, account in enumerate(ordered_accounts)
        }

    def profile_for_account(self, account_id: str) -> AccountBehaviorProfile:
        try:
            return self._profiles_by_account_id[account_id]
        except KeyError as exc:
            raise ValueError(f"unknown account id {account_id!r}") from exc
