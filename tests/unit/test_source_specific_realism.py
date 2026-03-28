from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

import pandas as pd

from synthetic_enterprise.contracts.records.email import EmailRecord
from synthetic_enterprise.contracts.records.salesforce import (
    SalesforceObjectType,
    SalesforceRecord,
)
from synthetic_enterprise.contracts.records.slack import SlackRecord
from synthetic_enterprise.contracts.records.teams import TeamsRecord
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.sources.email.renderer import EmailRenderer
from synthetic_enterprise.sources.salesforce.renderer import SalesforceRenderer
from synthetic_enterprise.sources.slack.renderer import SlackRenderer
from synthetic_enterprise.sources.teams.renderer import TeamsRenderer


@dataclass(frozen=True, slots=True)
class SourceOutputs:
    email: list[EmailRecord]
    slack: list[SlackRecord]
    teams: list[TeamsRecord]
    salesforce: list[SalesforceRecord]


def test_source_specific_artifacts_appear_in_expected_channels() -> None:
    outputs = build_outputs()

    email_thread = _largest_email_thread(outputs.email)
    assert any(message.subject.startswith("FW:") for message in email_thread)
    assert any(message.subject.startswith("Re:") for message in email_thread)
    assert len({message.subject for message in email_thread}) >= 3
    assert any(len(message.cc) >= 2 for message in email_thread)
    assert any("Regards," in message.body or "Best," in message.body for message in email_thread)
    assert any("Disclaimer:" in message.body for message in email_thread)
    assert any(_all_prefixed(message.to, "employee_") for message in email_thread)
    assert any(_any_prefixed(message.to, "contact_") for message in email_thread)

    account_slack = [
        message for message in outputs.slack if message.channel_name.startswith("#acct-")
    ]
    assert any(message.reactions for message in account_slack)
    assert any(message.body.strip().endswith("?") for message in account_slack)
    assert any(
        any(token in message.body.lower() for token in ("sgtm", "thx", "eta", "yep"))
        for message in account_slack
    )
    assert any(len(message.body.split()) <= 4 for message in account_slack)

    assert any(message.meeting_id is not None for message in outputs.teams)
    assert any(message.file_refs for message in outputs.teams)
    assert any(message.body.startswith("Recap:") for message in outputs.teams)
    assert any("Task list:" in message.body for message in outputs.teams)
    assert any("\n-" in message.body for message in outputs.teams)

    assert any(
        record.object_type == SalesforceObjectType.NOTE
        and record.text_body is not None
        and len(record.text_body.split()) <= 4
        for record in outputs.salesforce
    )
    assert any(
        "stale_days" in record.structured_fields
        for record in outputs.salesforce
        if record.object_type == SalesforceObjectType.OPPORTUNITY
    )
    assert any("duplicate_of" in record.structured_fields for record in outputs.salesforce)
    assert any(
        {"previous_owner_employee_id", "pending_owner_employee_id"}
        <= record.structured_fields.keys()
        for record in outputs.salesforce
    )
    assert any(
        record.object_type
        in {
            SalesforceObjectType.EVENT,
            SalesforceObjectType.CAMPAIGN,
            SalesforceObjectType.CAMPAIGN_MEMBER,
        }
        for record in outputs.salesforce
    )


def test_style_separation_between_channels_increases() -> None:
    outputs = build_outputs()

    email_avg_words = mean(
        len(f"{message.subject} {message.body}".split()) for message in outputs.email
    )
    slack_avg_words = mean(len(message.body.split()) for message in outputs.slack)
    teams_structured_messages = sum("\n-" in message.body for message in outputs.teams)
    salesforce_note_avg_words = mean(
        len((record.text_body or "").split())
        for record in outputs.salesforce
        if record.object_type == SalesforceObjectType.NOTE
    )

    assert slack_avg_words < email_avg_words
    assert teams_structured_messages >= 3
    assert salesforce_note_avg_words < email_avg_words
    assert any(message.subject.startswith(("Re:", "FW:")) for message in outputs.email)
    assert any(
        any(token in message.body.lower() for token in ("sgtm", "thx", "eta"))
        for message in outputs.slack
    )
    assert any(message.body.startswith("Recap:") for message in outputs.teams)


def test_facts_remain_consistent_across_systems() -> None:
    outputs = build_outputs()

    email_row = next(message for message in outputs.email if message.is_relevant)
    slack_row = next(message for message in outputs.slack if message.is_relevant)
    teams_row = next(message for message in outputs.teams if message.is_relevant)
    event_row = next(
        record
        for record in outputs.salesforce
        if record.object_type == SalesforceObjectType.EVENT
    )

    assert email_row.account_id == slack_row.linked_account_id == teams_row.linked_account_id
    assert email_row.event_id == slack_row.linked_event_id == teams_row.linked_event_id
    assert email_row.account_id == event_row.account_id
    assert email_row.event_id == event_row.event_id


def test_source_specific_artifacts_do_not_break_export_schemas() -> None:
    outputs = build_outputs()

    email_frame = pd.DataFrame([row.to_dataframe_row() for row in outputs.email])
    slack_frame = pd.DataFrame([row.to_dataframe_row() for row in outputs.slack])
    teams_frame = pd.DataFrame([row.to_dataframe_row() for row in outputs.teams])
    salesforce_frame = pd.DataFrame([row.to_dataframe_row() for row in outputs.salesforce])

    assert email_frame["source_system"].eq("email").all()
    assert slack_frame["source_system"].eq("slack").all()
    assert teams_frame["source_system"].eq("teams").all()
    assert salesforce_frame["source_system"].eq("salesforce").all()


def test_deterministic_generation_still_holds() -> None:
    first = build_outputs(seed=9191)
    second = build_outputs(seed=9191)
    third = build_outputs(seed=9192)

    assert first == second
    assert first != third


def build_outputs(seed: int = 8088) -> SourceOutputs:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_ratio=0.8,
            hard_negative_ratio=0.25,
        ),
    )
    enterprise = CompanyBuilder().build(context)
    event = enterprise.events[0]

    return SourceOutputs(
        email=EmailRenderer(context=context, enterprise=enterprise).generate_messages(
            event_id=event.id
        ),
        slack=SlackRenderer(context=context, enterprise=enterprise).generate_messages(),
        teams=TeamsRenderer(context=context, enterprise=enterprise).generate_messages(),
        salesforce=SalesforceRenderer(
            context=context,
            enterprise=enterprise,
        ).generate_records(),
    )


def _largest_email_thread(rows: list[EmailRecord]) -> list[EmailRecord]:
    grouped: dict[str, list[EmailRecord]] = defaultdict(list)
    for row in rows:
        grouped[row.thread_id].append(row)
    return max(grouped.values(), key=len)


def _all_prefixed(values: list[str], prefix: str) -> bool:
    return bool(values) and all(value.startswith(prefix) for value in values)


def _any_prefixed(values: list[str], prefix: str) -> bool:
    return any(value.startswith(prefix) for value in values)
