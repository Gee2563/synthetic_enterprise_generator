from __future__ import annotations

import pandas as pd

from synthetic_enterprise.contracts.records.slack import SlackRecord
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory
from synthetic_enterprise.sources.slack.renderer import SlackRenderer


def build_renderer(noise_ratio: float = 0.8) -> SlackRenderer:
    context = GeneratorContext(
        seed=3030,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_ratio=noise_ratio,
        ),
    )
    graph = CompanyBuilder().build(context)
    return SlackRenderer(context=context, enterprise=graph)


def test_channel_thread_relationships_are_valid() -> None:
    renderer = build_renderer()
    messages = renderer.generate_messages()

    assert messages

    for message in messages:
        if message.thread_id is None:
            assert message.parent_message_id is None
        else:
            if message.parent_message_id is None:
                continue
            parent = next(
                candidate
                for candidate in messages
                if candidate.slack_message_id == message.parent_message_id
            )
            assert parent.channel_id == message.channel_id
            assert parent.thread_id == message.thread_id


def test_thread_replies_point_to_valid_parent_messages() -> None:
    renderer = build_renderer()
    messages = renderer.generate_messages()
    message_ids = {message.slack_message_id for message in messages}
    replies = [message for message in messages if message.parent_message_id is not None]

    assert replies
    assert all(reply.parent_message_id in message_ids for reply in replies)


def test_messages_differ_by_channel_type() -> None:
    renderer = build_renderer()
    messages = renderer.generate_messages()

    chatter_message = next(
        message for message in messages if message.channel_name == "#watercooler"
    )
    account_message = next(
        message for message in messages if message.channel_name.startswith("#acct-")
    )

    assert chatter_message.body != account_message.body
    assert chatter_message.linked_account_id is None
    assert account_message.linked_account_id is not None


def test_relevant_messages_can_still_be_subtle() -> None:
    renderer = build_renderer()
    messages = renderer.generate_messages()
    subtle_message = next(
        message
        for message in messages
        if message.is_relevant and message.primary_category == CommunicationCategory.FOLLOW_UP
    )

    combined_text = f"{subtle_message.body} {subtle_message.relevance_reason}".lower()

    assert subtle_message.linked_account_id is not None
    assert subtle_message.linked_event_id is not None
    assert "follow up" not in combined_text
    assert "complaint" not in combined_text
    assert "buying signal" not in combined_text


def test_noise_messages_dominate_when_configured() -> None:
    renderer = build_renderer(noise_ratio=0.9)
    messages = renderer.generate_messages()
    noise_messages = [message for message in messages if not message.is_relevant]
    relevant_messages = [message for message in messages if message.is_relevant]

    assert noise_messages
    assert relevant_messages
    assert len(noise_messages) > len(relevant_messages)


def test_slack_rows_export_cleanly() -> None:
    renderer = build_renderer()
    messages = renderer.generate_messages()
    frame = pd.DataFrame([message.to_dataframe_row() for message in messages])

    assert "slack_message_id" in frame.columns
    assert frame["source_system"].eq("slack").all()
    assert frame["primary_category"].isin(
        {category.value for category in CommunicationCategory}
    ).all()


def test_slack_record_has_expected_fields() -> None:
    expected_fields = {
        "slack_message_id",
        "channel_id",
        "channel_name",
        "thread_id",
        "parent_message_id",
        "timestamp",
        "sender_employee_id",
        "body",
        "mentions",
        "reactions",
        "attachments",
        "linked_account_id",
        "linked_opportunity_id",
        "linked_event_id",
        "linked_ticket_id",
        "primary_category",
        "is_relevant",
        "relevance_reason",
        "source_system",
    }

    assert expected_fields.issubset(SlackRecord.model_fields)
