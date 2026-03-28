from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CompanyLanguageProfileType(str, Enum):
    ENTERPRISE_B2B_FORMAL = "enterprise_b2b_formal"
    FAST_MOVING_STARTUP = "fast_moving_startup"
    FIELD_OPS_HEAVY = "field_ops_heavy"
    SECURITY_SENSITIVE_ENTERPRISE = "security_sensitive_enterprise"
    PRODUCT_LED_SAAS = "product_led_saas"
    CONSULTING_LED_ORG = "consulting_led_org"


@dataclass(frozen=True, slots=True)
class CompanyLanguageProfile:
    profile_type: CompanyLanguageProfileType
    lexical_markers: tuple[str, ...]
    email_marker: str
    slack_marker: str
    teams_marker: str
    crm_marker: str
    meeting_language: str
    team_alias: str
    code_name: str
    product_alias_suffix: str
    abbreviation: str

    def product_reference(self, product_name: str) -> str:
        return f"{product_name} {self.product_alias_suffix} ({self.code_name})"


_PROFILE_LIBRARY = {
    CompanyLanguageProfileType.ENTERPRISE_B2B_FORMAL: CompanyLanguageProfile(
        profile_type=CompanyLanguageProfileType.ENTERPRISE_B2B_FORMAL,
        lexical_markers=("governance", "programme", "accordingly", "steering", "portfolio"),
        email_marker="governance review",
        slack_marker="programme lane",
        teams_marker="steering committee",
        crm_marker="executive summary",
        meeting_language="portfolio review",
        team_alias="programme office",
        code_name="Northbridge",
        product_alias_suffix="programme track",
        abbreviation="EBR",
    ),
    CompanyLanguageProfileType.FAST_MOVING_STARTUP: CompanyLanguageProfile(
        profile_type=CompanyLanguageProfileType.FAST_MOVING_STARTUP,
        lexical_markers=("ship", "unblock", "quick", "squad", "loop"),
        email_marker="quick unblock",
        slack_marker="ship lane",
        teams_marker="squad check",
        crm_marker="founder note",
        meeting_language="squad sync",
        team_alias="ship squad",
        code_name="Comet",
        product_alias_suffix="launch lane",
        abbreviation="ETA",
    ),
    CompanyLanguageProfileType.FIELD_OPS_HEAVY: CompanyLanguageProfile(
        profile_type=CompanyLanguageProfileType.FIELD_OPS_HEAVY,
        lexical_markers=("route", "dispatch", "onsite", "crew", "handoff"),
        email_marker="onsite handoff",
        slack_marker="dispatch board",
        teams_marker="crew standup",
        crm_marker="field note",
        meeting_language="onsite review",
        team_alias="route crew",
        code_name="Waypoint",
        product_alias_suffix="field pack",
        abbreviation="SLA",
    ),
    CompanyLanguageProfileType.SECURITY_SENSITIVE_ENTERPRISE: CompanyLanguageProfile(
        profile_type=CompanyLanguageProfileType.SECURITY_SENSITIVE_ENTERPRISE,
        lexical_markers=("control", "risk", "access", "review", "policy"),
        email_marker="control review",
        slack_marker="risk lane",
        teams_marker="policy review",
        crm_marker="control note",
        meeting_language="access review",
        team_alias="control desk",
        code_name="Cipher",
        product_alias_suffix="trust layer",
        abbreviation="CAB",
    ),
    CompanyLanguageProfileType.PRODUCT_LED_SAAS: CompanyLanguageProfile(
        profile_type=CompanyLanguageProfileType.PRODUCT_LED_SAAS,
        lexical_markers=("adoption", "usage", "self-serve", "workspace", "activation"),
        email_marker="adoption path",
        slack_marker="usage pulse",
        teams_marker="activation review",
        crm_marker="product note",
        meeting_language="workspace review",
        team_alias="growth pod",
        code_name="Atlas",
        product_alias_suffix="workspace lane",
        abbreviation="PLG",
    ),
    CompanyLanguageProfileType.CONSULTING_LED_ORG: CompanyLanguageProfile(
        profile_type=CompanyLanguageProfileType.CONSULTING_LED_ORG,
        lexical_markers=("workstream", "milestone", "readout", "client", "steerco"),
        email_marker="workstream readout",
        slack_marker="milestone lane",
        teams_marker="client readout",
        crm_marker="engagement note",
        meeting_language="steerco",
        team_alias="delivery pod",
        code_name="Summit",
        product_alias_suffix="engagement track",
        abbreviation="RAID",
    ),
}


def resolve_company_language_profile(
    *,
    seed: int,
    explicit_profile: CompanyLanguageProfileType | None = None,
) -> CompanyLanguageProfile:
    if explicit_profile is not None:
        return _PROFILE_LIBRARY[explicit_profile]

    profile_types = tuple(CompanyLanguageProfileType)
    profile_type = profile_types[seed % len(profile_types)]
    return _PROFILE_LIBRARY[profile_type]
