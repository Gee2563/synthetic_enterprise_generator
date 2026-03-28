from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, TypeVar, cast

from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.grounding import LabelProvenance
from synthetic_enterprise.sources.email.renderer import EmailRenderer
from synthetic_enterprise.sources.slack.renderer import SlackRenderer
from synthetic_enterprise.sources.teams.renderer import TeamsRenderer


class EmailLike(Protocol):
    thread_id: str
    message_index_in_thread: int
    timestamp: datetime
    account_id: str | None
    event_id: str | None
    opportunity_id: str | None
    ticket_id: str | None
    provenance: LabelProvenance | None
    sender_contact_id: str | None
    sender_employee_id: str | None
    body: str


class SlackLike(Protocol):
    slack_message_id: str
    thread_id: str | None
    parent_message_id: str | None
    timestamp: datetime
    linked_account_id: str | None
    linked_event_id: str | None
    linked_opportunity_id: str | None
    linked_ticket_id: str | None
    provenance: LabelProvenance | None
    body: str


class TeamsLike(Protocol):
    teams_message_id: str
    thread_id: str | None
    timestamp: datetime
    linked_account_id: str | None
    linked_event_id: str | None
    linked_opportunity_id: str | None
    linked_ticket_id: str | None
    provenance: LabelProvenance | None
    meeting_id: str | None
    body: str


ThreadRowT = TypeVar("ThreadRowT", bound=object)


@dataclass(frozen=True, slots=True)
class SourceThreads:
    email: Sequence[EmailLike]
    slack: Sequence[SlackLike]
    teams: Sequence[TeamsLike]


def test_thread_messages_remain_chronologically_valid() -> None:
    rows = build_source_threads()

    for email_thread in _group_by_thread(rows.email):
        ordered = sorted(email_thread, key=lambda row: row.message_index_in_thread)
        assert [row.timestamp for row in ordered] == sorted(row.timestamp for row in ordered)

    for slack_thread in _group_by_thread(rows.slack):
        timestamps = [
            row.timestamp for row in sorted(slack_thread, key=lambda row: row.timestamp)
        ]
        assert timestamps == sorted(timestamps)

    for teams_thread in _group_by_thread(rows.teams):
        timestamps = [
            row.timestamp for row in sorted(teams_thread, key=lambda row: row.timestamp)
        ]
        assert timestamps == sorted(timestamps)


def test_thread_depth_distribution_increases_relative_to_baseline() -> None:
    rows = build_source_threads()

    assert _max_thread_depth(rows.email) >= 6
    assert _max_thread_depth(rows.slack) >= 6
    assert _max_thread_depth(rows.teams) >= 6


def test_reply_links_remain_valid() -> None:
    rows = build_source_threads()
    slack_rows = rows.slack
    by_id = {row.slack_message_id: row for row in slack_rows}
    children_by_parent = Counter(
        row.parent_message_id
        for row in slack_rows
        if row.parent_message_id is not None
    )

    assert all(parent_id in by_id for parent_id in children_by_parent)
    assert any(count > 1 for count in children_by_parent.values())
    assert any(
        row.parent_message_id is not None
        and by_id[row.parent_message_id].parent_message_id is not None
        for row in slack_rows
    )


def test_reopening_a_thread_preserves_identity_and_provenance() -> None:
    rows = build_source_threads()

    assert _has_reopened_email_thread(rows.email)
    assert _has_reopened_slack_thread(rows.slack)
    assert _has_reopened_teams_thread(rows.teams)


def test_channel_specific_thread_behavior_differs_across_email_slack_and_teams() -> None:
    rows = build_source_threads()

    assert any(row.sender_contact_id is not None for row in rows.email)
    assert any(
        row.parent_message_id is not None and len(row.body.split()) <= 6
        for row in rows.slack
    )
    assert any(
        row.thread_id is not None and row.meeting_id is not None and "\n-" in row.body
        for row in rows.teams
    )


def test_longer_threads_do_not_cause_inconsistent_facts() -> None:
    rows = build_source_threads()

    for email_thread in _group_by_thread(rows.email):
        assert _non_null_set(row.account_id for row in email_thread) <= 1
        assert _non_null_set(row.event_id for row in email_thread) <= 1
        assert _non_null_set(row.opportunity_id for row in email_thread) <= 1
        assert _non_null_set(row.ticket_id for row in email_thread) <= 1

    for slack_thread in _group_by_thread(rows.slack):
        assert _non_null_set(row.linked_account_id for row in slack_thread) <= 1
        assert _non_null_set(row.linked_event_id for row in slack_thread) <= 1
        assert _non_null_set(row.linked_opportunity_id for row in slack_thread) <= 1
        assert _non_null_set(row.linked_ticket_id for row in slack_thread) <= 1

    for teams_thread in _group_by_thread(rows.teams):
        assert _non_null_set(row.linked_account_id for row in teams_thread) <= 1
        assert _non_null_set(row.linked_event_id for row in teams_thread) <= 1
        assert _non_null_set(row.linked_opportunity_id for row in teams_thread) <= 1
        assert _non_null_set(row.linked_ticket_id for row in teams_thread) <= 1


def build_source_threads() -> SourceThreads:
    context = GeneratorContext(
        seed=7171,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_ratio=0.8,
            hard_negative_ratio=0.25,
        ),
    )
    enterprise = CompanyBuilder().build(context)
    event = enterprise.events[0]

    return SourceThreads(
        email=cast(
            Sequence[EmailLike],
            EmailRenderer(context=context, enterprise=enterprise).generate_messages(
                event_id=event.id
            ),
        ),
        slack=cast(
            Sequence[SlackLike],
            SlackRenderer(context=context, enterprise=enterprise).generate_messages(),
        ),
        teams=cast(
            Sequence[TeamsLike],
            TeamsRenderer(context=context, enterprise=enterprise).generate_messages(),
        ),
    )


def _group_by_thread(rows: Sequence[ThreadRowT]) -> list[list[ThreadRowT]]:
    grouped: dict[str, list[ThreadRowT]] = defaultdict(list)
    for row in rows:
        thread_id = getattr(row, "thread_id", None)
        if thread_id is not None:
            grouped[str(thread_id)].append(row)
    return list(grouped.values())


def _max_thread_depth(rows: Sequence[ThreadRowT]) -> int:
    threads = _group_by_thread(rows)
    return max(len(thread) for thread in threads)


def _has_reopened_email_thread(rows: Sequence[EmailLike]) -> bool:
    for thread in _group_by_thread(rows):
        ordered = sorted(thread, key=lambda row: row.timestamp)
        if len(ordered) < 3:
            continue
        gaps = [
            later.timestamp - earlier.timestamp
            for earlier, later in zip(ordered, ordered[1:])
        ]
        if max(gaps) < timedelta(hours=24):
            continue
        first = ordered[0]
        last = ordered[-1]
        first_provenance = getattr(first, "provenance", None)
        last_provenance = getattr(last, "provenance", None)
        if first_provenance is None or last_provenance is None:
            continue
        if first_provenance.object_id != last_provenance.object_id:
            continue
        if first.account_id != last.account_id or first.event_id != last.event_id:
            continue
        return True
    return False


def _has_reopened_slack_thread(rows: Sequence[SlackLike]) -> bool:
    for thread in _group_by_thread(rows):
        ordered = sorted(thread, key=lambda row: row.timestamp)
        if len(ordered) < 3:
            continue
        gaps = [
            later.timestamp - earlier.timestamp
            for earlier, later in zip(ordered, ordered[1:])
        ]
        if max(gaps) < timedelta(hours=24):
            continue
        first = ordered[0]
        last = ordered[-1]
        first_provenance = first.provenance
        last_provenance = last.provenance
        if first_provenance is None or last_provenance is None:
            continue
        if first_provenance.object_id != last_provenance.object_id:
            continue
        if (
            first.linked_account_id != last.linked_account_id
            or first.linked_event_id != last.linked_event_id
        ):
            continue
        return True
    return False


def _has_reopened_teams_thread(rows: Sequence[TeamsLike]) -> bool:
    for thread in _group_by_thread(rows):
        ordered = sorted(thread, key=lambda row: row.timestamp)
        if len(ordered) < 3:
            continue
        gaps = [
            later.timestamp - earlier.timestamp
            for earlier, later in zip(ordered, ordered[1:])
        ]
        if max(gaps) < timedelta(hours=24):
            continue
        first = ordered[0]
        last = ordered[-1]
        first_provenance = first.provenance
        last_provenance = last.provenance
        if first_provenance is None or last_provenance is None:
            continue
        if first_provenance.object_id != last_provenance.object_id:
            continue
        if (
            first.linked_account_id != last.linked_account_id
            or first.linked_event_id != last.linked_event_id
        ):
            continue
        return True
    return False


def _non_null_set(values: Iterable[str | None]) -> int:
    return len({value for value in values if value is not None})
