from __future__ import annotations

from dataclasses import dataclass

from synthetic_enterprise.contracts.records.email import EmailRecord
from synthetic_enterprise.contracts.records.salesforce import SalesforceObjectType, SalesforceRecord
from synthetic_enterprise.contracts.records.slack import SlackRecord
from synthetic_enterprise.contracts.records.teams import TeamsRecord
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.language_profiles import (
    CompanyLanguageProfileType,
    resolve_company_language_profile,
)
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory
from synthetic_enterprise.sources.email.renderer import EmailRenderer
from synthetic_enterprise.sources.salesforce.renderer import SalesforceRenderer
from synthetic_enterprise.sources.slack.renderer import SlackRenderer
from synthetic_enterprise.sources.teams.renderer import TeamsRenderer


@dataclass(frozen=True, slots=True)
class ProfileRenderSamples:
    email: EmailRecord
    slack: SlackRecord
    teams: TeamsRecord
    salesforce: SalesforceRecord


def build_samples(
    profile_type: CompanyLanguageProfileType,
    *,
    seed: int = 8080,
) -> ProfileRenderSamples:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=8, max_employees=10),
            company_language_profile=profile_type,
        ),
    )
    enterprise = CompanyBuilder().build(context)
    event = enterprise.events[0]

    email = EmailRenderer(context=context, enterprise=enterprise).render_from_event(
        event_id=event.id,
        primary_category=CommunicationCategory.FOLLOW_UP,
        message_index_in_thread=1,
    )
    slack = next(
        message
        for message in SlackRenderer(context=context, enterprise=enterprise).generate_messages()
        if message.is_relevant and message.primary_category == CommunicationCategory.FOLLOW_UP
    )
    teams = next(
        message
        for message in TeamsRenderer(context=context, enterprise=enterprise).generate_messages()
        if message.is_relevant and message.primary_category == CommunicationCategory.FOLLOW_UP
    )
    salesforce = next(
        record
        for record in SalesforceRenderer(context=context, enterprise=enterprise).generate_records()
        if record.object_type == SalesforceObjectType.NOTE and record.is_relevant
    )
    return ProfileRenderSamples(
        email=email,
        slack=slack,
        teams=teams,
        salesforce=salesforce,
    )


def test_different_company_profiles_yield_measurably_different_lexical_patterns() -> None:
    formal = build_samples(CompanyLanguageProfileType.ENTERPRISE_B2B_FORMAL)
    startup = build_samples(CompanyLanguageProfileType.FAST_MOVING_STARTUP)

    formal_profile = resolve_company_language_profile(
        seed=8080,
        explicit_profile=CompanyLanguageProfileType.ENTERPRISE_B2B_FORMAL,
    )
    startup_profile = resolve_company_language_profile(
        seed=8080,
        explicit_profile=CompanyLanguageProfileType.FAST_MOVING_STARTUP,
    )
    formal_text = _combined_text(formal)
    startup_text = _combined_text(startup)

    assert _marker_hits(formal_text, formal_profile.lexical_markers) >= 4
    assert _marker_hits(startup_text, startup_profile.lexical_markers) >= 4
    assert _marker_hits(formal_text, startup_profile.lexical_markers) < _marker_hits(
        formal_text,
        formal_profile.lexical_markers,
    )
    assert _marker_hits(startup_text, formal_profile.lexical_markers) < _marker_hits(
        startup_text,
        startup_profile.lexical_markers,
    )


def test_same_underlying_scenario_renders_differently_under_different_profiles() -> None:
    formal = build_samples(CompanyLanguageProfileType.SECURITY_SENSITIVE_ENTERPRISE)
    product_led = build_samples(CompanyLanguageProfileType.PRODUCT_LED_SAAS)

    assert formal.email.account_id == product_led.email.account_id
    assert formal.email.event_id == product_led.email.event_id
    assert formal.email.body != product_led.email.body
    assert formal.slack.body != product_led.slack.body
    assert formal.teams.body != product_led.teams.body
    assert formal.salesforce.text_body != product_led.salesforce.text_body


def test_deterministic_generation_still_holds_under_seed() -> None:
    first = build_samples(CompanyLanguageProfileType.CONSULTING_LED_ORG, seed=8181)
    second = build_samples(CompanyLanguageProfileType.CONSULTING_LED_ORG, seed=8181)
    third = build_samples(CompanyLanguageProfileType.CONSULTING_LED_ORG, seed=8182)

    assert first == second
    assert first != third


def test_profile_settings_affect_all_four_source_systems() -> None:
    samples = build_samples(CompanyLanguageProfileType.SECURITY_SENSITIVE_ENTERPRISE)
    profile = resolve_company_language_profile(
        seed=8080,
        explicit_profile=CompanyLanguageProfileType.SECURITY_SENSITIVE_ENTERPRISE,
    )

    assert profile.email_marker.lower() in samples.email.body.lower()
    assert profile.slack_marker.lower() in samples.slack.body.lower()
    assert profile.teams_marker.lower() in samples.teams.body.lower()
    assert samples.salesforce.text_body is not None
    assert profile.crm_marker.lower() in samples.salesforce.text_body.lower()


def test_profile_changes_do_not_break_schema_validity() -> None:
    samples = build_samples(CompanyLanguageProfileType.FIELD_OPS_HEAVY, seed=8282)

    assert EmailRecord.model_validate(samples.email.to_dict()) == samples.email
    assert SlackRecord.model_validate(samples.slack.to_dict()) == samples.slack
    assert TeamsRecord.model_validate(samples.teams.to_dict()) == samples.teams
    assert SalesforceRecord.model_validate(samples.salesforce.to_dict()) == samples.salesforce


def _combined_text(samples: ProfileRenderSamples) -> str:
    return " ".join(
        [
            samples.email.subject,
            samples.email.body,
            samples.slack.body,
            samples.teams.body,
            samples.salesforce.subject or "",
            samples.salesforce.text_body or "",
        ]
    ).lower()


def _marker_hits(text: str, markers: tuple[str, ...]) -> int:
    return sum(1 for marker in markers if marker.lower() in text)
