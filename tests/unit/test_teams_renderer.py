from __future__ import annotations

import pandas as pd

from synthetic_enterprise.contracts.records.teams import TeamsRecord
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory
from synthetic_enterprise.sources.slack.renderer import SlackRenderer
from synthetic_enterprise.sources.teams.renderer import TeamsRenderer


def build_renderer(noise_ratio: float = 0.8, seed: int = 4040) -> TeamsRenderer:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_ratio=noise_ratio,
        ),
    )
    graph = CompanyBuilder().build(context)
    return TeamsRenderer(context=context, enterprise=graph)


def test_valid_teams_row_schema() -> None:
    expected_fields = {
        "teams_message_id",
        "team_id",
        "channel_id",
        "chat_or_channel",
        "thread_id",
        "timestamp",
        "sender_employee_id",
        "body",
        "mentions",
        "meeting_id",
        "file_refs",
        "linked_account_id",
        "linked_opportunity_id",
        "linked_event_id",
        "linked_ticket_id",
        "primary_category",
        "is_relevant",
        "relevance_reason",
        "source_system",
    }

    assert expected_fields.issubset(TeamsRecord.model_fields)


def test_meeting_linked_messages_attach_to_valid_events_when_present() -> None:
    renderer = build_renderer()
    messages = renderer.generate_messages()
    event_ids = {event.id for event in renderer.enterprise.events}

    meeting_messages = [message for message in messages if message.meeting_id is not None]

    assert meeting_messages
    assert all(message.linked_event_id in event_ids for message in meeting_messages)


def test_teams_tone_differs_from_slack() -> None:
    teams_renderer = build_renderer()
    slack_renderer = SlackRenderer(
        context=teams_renderer.context,
        enterprise=teams_renderer.enterprise,
    )

    teams_message = next(
        message for message in teams_renderer.generate_messages() if message.is_relevant
    )
    slack_message = next(
        message for message in slack_renderer.generate_messages() if message.is_relevant
    )

    assert teams_message.body != slack_message.body
    assert "Agenda:" in teams_message.body or "Document:" in teams_message.body
    assert "Agenda:" not in slack_message.body


def test_linked_entities_exist() -> None:
    renderer = build_renderer()
    messages = renderer.generate_messages()
    account_ids = {account.id for account in renderer.enterprise.customer_accounts}
    opportunity_ids = {opportunity.id for opportunity in renderer.enterprise.opportunities}
    event_ids = {event.id for event in renderer.enterprise.events}
    ticket_ids = {ticket.id for ticket in renderer.enterprise.ticket_issues}

    for message in messages:
        if message.linked_account_id is not None:
            assert message.linked_account_id in account_ids
        if message.linked_opportunity_id is not None:
            assert message.linked_opportunity_id in opportunity_ids
        if message.linked_event_id is not None:
            assert message.linked_event_id in event_ids
        if message.linked_ticket_id is not None:
            assert message.linked_ticket_id in ticket_ids


def test_irrelevant_coordination_messages_are_common() -> None:
    renderer = build_renderer(noise_ratio=0.85)
    messages = renderer.generate_messages()
    irrelevant_messages = [message for message in messages if not message.is_relevant]
    relevant_messages = [message for message in messages if message.is_relevant]

    assert irrelevant_messages
    assert relevant_messages
    assert len(irrelevant_messages) > len(relevant_messages)


def test_deterministic_generation_works_under_seed() -> None:
    first = build_renderer(seed=5050).generate_messages()
    second = build_renderer(seed=5050).generate_messages()
    third = build_renderer(seed=5051).generate_messages()

    assert first == second
    assert first != third


def test_teams_rows_export_cleanly() -> None:
    renderer = build_renderer()
    rows = [message.to_dataframe_row() for message in renderer.generate_messages()]
    frame = pd.DataFrame(rows)

    assert frame["source_system"].eq("teams").all()
    assert frame["primary_category"].isin(
        {category.value for category in CommunicationCategory}
    ).all()
