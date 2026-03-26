from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, TypedDict

from synthetic_enterprise.contracts.records.teams import TeamsRecord
from synthetic_enterprise.domain import (
    CustomerAccount,
    Employee,
    EnterpriseGraph,
    Event,
    EventScenarioResolver,
)
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.hard_negatives import hard_negative_count
from synthetic_enterprise.labeling.grounding import (
    LabelProvenance,
    ProvenanceObjectType,
    ProvenanceStrength,
    build_relevance_reason,
)
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory


class TeamsNoiseSpec(TypedDict):
    suffix: str
    team_name: str
    channel_name: str
    chat_or_channel: Literal["chat", "channel"]
    sender_employee_id: str
    body: str
    mentions: list[str]
    meeting_id: str | None
    file_refs: list[str]
    category: CommunicationCategory
    timestamp: datetime


class TeamsHardNegativeSpec(TypedDict):
    suffix: str
    body: str
    file_refs: list[str]
    category: CommunicationCategory
    timestamp: datetime
    linked_opportunity_id: str | None
    linked_ticket_id: str | None
    provenance: LabelProvenance


@dataclass(slots=True)
class TeamsRenderer:
    """Render a small deterministic Teams dataset from simulated enterprise state."""

    context: GeneratorContext
    enterprise: EnterpriseGraph

    def generate_messages(self) -> list[TeamsRecord]:
        scenario = EventScenarioResolver(self.enterprise).primary_account_event()
        account = scenario.account
        event = scenario.event
        opportunity = scenario.opportunity
        ticket = scenario.ticket
        organizer = scenario.organizer
        account_owner = scenario.account_owner
        support_owner = scenario.support_owner
        if opportunity is None:
            raise ValueError("teams rendering requires an account opportunity")
        if ticket is None:
            raise ValueError("teams rendering requires an account-linked ticket")

        relevant_messages = self._build_relevant_messages(
            account=account,
            event=event,
            organizer=organizer,
            account_owner=account_owner,
            support_owner=support_owner,
            opportunity_id=opportunity.id,
            ticket_id=ticket.id,
        )
        noise_messages = self._build_noise_messages(
            account=account,
            event=event,
            organizer=organizer,
            account_owner=account_owner,
        )
        hard_negative_messages = self._build_hard_negative_messages(
            account=account,
            event=event,
            organizer=organizer,
            opportunity_id=opportunity.id,
            ticket_id=ticket.id,
        )

        return [*relevant_messages, *noise_messages, *hard_negative_messages]

    def _build_relevant_messages(
        self,
        *,
        account: CustomerAccount,
        event: Event,
        organizer: Employee,
        account_owner: Employee,
        support_owner: Employee,
        opportunity_id: str,
        ticket_id: str,
    ) -> list[TeamsRecord]:
        team_name = "Customer Delivery"
        channel_name = f"acct-{_slugify(account.name)}"
        team_id = self._team_id(team_name)
        channel_id = self._channel_id(team_name, channel_name)
        thread_id = self._thread_id(event.id)
        meeting_id = self._meeting_id(event.id)
        root_provenance = LabelProvenance(
            object_type=ProvenanceObjectType.EVENT,
            object_id=event.id,
            explanation=(
                f"Customer review for account {account.id} still needs owners and "
                "documented next steps."
            ),
        )
        reply_provenance = LabelProvenance(
            object_type=ProvenanceObjectType.TICKET,
            object_id=ticket_id,
            explanation=(
                f"Open validation work is constraining the customer review tied to "
                f"event {event.id}."
            ),
        )

        root = TeamsRecord(
            teams_message_id=self._message_id("relevant-root"),
            team_id=team_id,
            channel_id=channel_id,
            chat_or_channel="channel",
            thread_id=thread_id,
            timestamp=event.starts_at - timedelta(hours=4),
            sender_employee_id=organizer.id,
            body=(
                "Agenda:\n"
                f"- confirm owners for {account.name}\n"
                "- document the revised date hold\n"
                "- capture the export test result in the meeting notes\n"
                "Document: rollout-plan.docx"
            ),
            mentions=[account_owner.id, support_owner.id],
            meeting_id=meeting_id,
            file_refs=["rollout-plan.docx", "meeting-notes.docx"],
            linked_account_id=account.id,
            linked_opportunity_id=opportunity_id,
            linked_event_id=event.id,
            linked_ticket_id=None,
            primary_category=CommunicationCategory.FOLLOW_UP,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.FOLLOW_UP,
                is_relevant=True,
                provenance=root_provenance,
            ),
            provenance=root_provenance,
            source_system="teams",
        )
        reply = TeamsRecord(
            teams_message_id=self._message_id("relevant-reply"),
            team_id=team_id,
            channel_id=channel_id,
            chat_or_channel="channel",
            thread_id=thread_id,
            timestamp=root.timestamp + timedelta(minutes=18),
            sender_employee_id=support_owner.id,
            body=(
                "Status update:\n"
                "- export validation still in progress\n"
                "- if the test clears by noon, I will post the file revision here"
            ),
            mentions=[organizer.id],
            meeting_id=meeting_id,
            file_refs=["validation-checklist.xlsx"],
            linked_account_id=account.id,
            linked_opportunity_id=None,
            linked_event_id=event.id,
            linked_ticket_id=ticket_id,
            primary_category=CommunicationCategory.BLOCKER,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.BLOCKER,
                is_relevant=True,
                provenance=reply_provenance,
            ),
            provenance=reply_provenance,
            source_system="teams",
        )
        return [root, reply]

    def _build_noise_messages(
        self,
        *,
        account: CustomerAccount,
        event: Event,
        organizer: Employee,
        account_owner: Employee,
    ) -> list[TeamsRecord]:
        noise_count = max(4, int(round(8 * self.context.noise_ratio)))
        specs = self._noise_templates(
            account=account,
            event=event,
            organizer=organizer,
            account_owner=account_owner,
        )
        messages: list[TeamsRecord] = []

        for index in range(noise_count):
            spec = specs[index % len(specs)]
            messages.append(
                TeamsRecord(
                    teams_message_id=self._message_id(f"{spec['suffix']}-{index}"),
                    team_id=self._team_id(spec["team_name"]),
                    channel_id=self._channel_id(spec["team_name"], spec["channel_name"]),
                    chat_or_channel=spec["chat_or_channel"],
                    thread_id=None,
                    timestamp=spec["timestamp"] + timedelta(minutes=index),
                    sender_employee_id=spec["sender_employee_id"],
                    body=spec["body"],
                    mentions=spec["mentions"],
                    meeting_id=spec["meeting_id"],
                    file_refs=spec["file_refs"],
                    linked_account_id=None,
                    linked_opportunity_id=None,
                    linked_event_id=None,
                    linked_ticket_id=None,
                    primary_category=spec["category"],
                    is_relevant=False,
                    relevance_reason=build_relevance_reason(
                        primary_category=spec["category"],
                        is_relevant=False,
                        provenance=None,
                    ),
                    provenance=None,
                    source_system="teams",
                )
            )

        return messages

    def _build_hard_negative_messages(
        self,
        *,
        account: CustomerAccount,
        event: Event,
        organizer: Employee,
        opportunity_id: str,
        ticket_id: str,
    ) -> list[TeamsRecord]:
        total = hard_negative_count(
            template_count=4,
            ratio=self.context.hard_negative_ratio,
        )
        specs = self._hard_negative_templates(
            account=account,
            event=event,
            opportunity_id=opportunity_id,
            ticket_id=ticket_id,
        )
        team_name = "Customer Delivery"
        channel_name = f"acct-{_slugify(account.name)}"
        team_id = self._team_id(team_name)
        channel_id = self._channel_id(team_name, channel_name)
        thread_id = self._thread_id(event.id)
        meeting_id = self._meeting_id(event.id)
        sender_employee_id = organizer.id
        messages: list[TeamsRecord] = []

        for index in range(total):
            spec = specs[index]
            messages.append(
                TeamsRecord(
                    teams_message_id=self._message_id(f"hard-negative-{spec['suffix']}"),
                    team_id=team_id,
                    channel_id=channel_id,
                    chat_or_channel="channel",
                    thread_id=thread_id,
                    timestamp=spec["timestamp"],
                    sender_employee_id=sender_employee_id,
                    body=spec["body"],
                    mentions=[],
                    meeting_id=meeting_id,
                    file_refs=spec["file_refs"],
                    linked_account_id=account.id,
                    linked_opportunity_id=spec["linked_opportunity_id"],
                    linked_event_id=event.id,
                    linked_ticket_id=spec["linked_ticket_id"],
                    primary_category=spec["category"],
                    is_relevant=False,
                    relevance_reason=build_relevance_reason(
                        primary_category=spec["category"],
                        is_relevant=False,
                        provenance=spec["provenance"],
                    ),
                    provenance=spec["provenance"],
                    source_system="teams",
                )
            )

        return messages

    def _noise_templates(
        self,
        *,
        account: CustomerAccount,
        event: Event,
        organizer: Employee,
        account_owner: Employee,
    ) -> list[TeamsNoiseSpec]:
        employees = self.enterprise.employees
        return [
            {
                "suffix": "coord-0",
                "team_name": "Operations",
                "channel_name": "project-updates",
                "chat_or_channel": "channel",
                "sender_employee_id": employees[0].id,
                "body": (
                    "Status update:\n"
                    "- slides moved to the shared folder\n"
                    "- no review needed today"
                ),
                "mentions": [],
                "meeting_id": None,
                "file_refs": ["deck-v3.pptx"],
                "category": CommunicationCategory.STATUS_UPDATES,
                "timestamp": event.starts_at - timedelta(days=2, hours=3),
            },
            {
                "suffix": "coord-1",
                "team_name": "Operations",
                "channel_name": "meeting-prep",
                "chat_or_channel": "channel",
                "sender_employee_id": organizer.id,
                "body": "Please keep the notes doc open during the session.",
                "mentions": [account_owner.id],
                "meeting_id": None,
                "file_refs": ["meeting-notes.docx"],
                "category": CommunicationCategory.SCHEDULING_ONLY,
                "timestamp": event.starts_at - timedelta(days=1, hours=5),
            },
            {
                "suffix": "coord-2",
                "team_name": "Operations",
                "channel_name": "pm-chat",
                "chat_or_channel": "chat",
                "sender_employee_id": employees[2].id,
                "body": "Can someone rename the workbook tab before I upload it?",
                "mentions": [],
                "meeting_id": None,
                "file_refs": ["status-tracker.xlsx"],
                "category": CommunicationCategory.ADMIN_OPS,
                "timestamp": event.starts_at - timedelta(days=1, hours=2),
            },
            {
                "suffix": "coord-3",
                "team_name": "Operations",
                "channel_name": "pm-chat",
                "chat_or_channel": "chat",
                "sender_employee_id": employees[3].id,
                "body": "same summary as above, copied here for visibility",
                "mentions": [organizer.id],
                "meeting_id": None,
                "file_refs": [],
                "category": CommunicationCategory.DUPLICATE_SUMMARY,
                "timestamp": event.starts_at - timedelta(days=1, hours=1),
            },
            {
                "suffix": "coord-4",
                "team_name": "Operations",
                "channel_name": "meeting-prep",
                "chat_or_channel": "channel",
                "sender_employee_id": account_owner.id,
                "body": (
                    "Document: prep-notes.docx\n"
                    f"Please add comments for {account.name} by end of day."
                ),
                "mentions": [organizer.id],
                "meeting_id": None,
                "file_refs": ["prep-notes.docx"],
                "category": CommunicationCategory.LOW_SIGNAL_CHECKIN,
                "timestamp": event.starts_at - timedelta(hours=20),
            },
            {
                "suffix": "coord-5",
                "team_name": "Operations",
                "channel_name": "ops-bot",
                "chat_or_channel": "channel",
                "sender_employee_id": employees[4].id,
                "body": "Automated notice: recording transcript uploaded.",
                "mentions": [],
                "meeting_id": None,
                "file_refs": ["meeting-transcript.docx"],
                "category": CommunicationCategory.AUTOMATED_NOTIFICATION,
                "timestamp": event.starts_at - timedelta(hours=10),
            },
        ]

    def _hard_negative_templates(
        self,
        *,
        account: CustomerAccount,
        event: Event,
        opportunity_id: str,
        ticket_id: str,
    ) -> list[TeamsHardNegativeSpec]:
        return [
            {
                "suffix": "timing",
                "body": (
                    "Agenda addendum:\n"
                    "- move review logistics to next week after travel rebooking\n"
                    "- no attendee confirmation is needed yet"
                ),
                "file_refs": ["travel-logistics.docx"],
                "category": CommunicationCategory.SCHEDULING_ONLY,
                "timestamp": event.starts_at - timedelta(hours=6),
                "linked_opportunity_id": None,
                "linked_ticket_id": None,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "Meeting logistics mention the review without attendance or "
                        "follow-up action."
                    ),
                ),
            },
            {
                "suffix": "budget",
                "body": (
                    "Document: event-budget.xlsx\n"
                    f"The budget tab for {account.name} covers catering and room blocks, "
                    "not customer spend."
                ),
                "file_refs": ["event-budget.xlsx"],
                "category": CommunicationCategory.STATUS_UPDATES,
                "timestamp": event.starts_at - timedelta(hours=5, minutes=30),
                "linked_opportunity_id": opportunity_id,
                "linked_ticket_id": None,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.OPPORTUNITY,
                    object_id=opportunity_id,
                    strength=ProvenanceStrength.WEAK,
                    explanation="Budget wording is logistical and not evidence of buying intent.",
                ),
            },
            {
                "suffix": "travel",
                "body": (
                    "Document: weather-plan.docx\n"
                    "Travel complaint may move the onsite review check-in by 30 minutes."
                ),
                "file_refs": ["weather-plan.docx"],
                "category": CommunicationCategory.FYI_FORWARD,
                "timestamp": event.starts_at - timedelta(hours=5),
                "linked_opportunity_id": None,
                "linked_ticket_id": None,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "The complaint concerns travel coordination rather than "
                        "customer pain."
                    ),
                ),
            },
            {
                "suffix": "export-ref",
                "body": (
                    "Document: export-faq.docx\n"
                    "Forwarding the prior export FAQ for internal prep only."
                ),
                "file_refs": ["export-faq.docx"],
                "category": CommunicationCategory.DUPLICATE_SUMMARY,
                "timestamp": event.starts_at - timedelta(hours=4, minutes=30),
                "linked_opportunity_id": None,
                "linked_ticket_id": ticket_id,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.TICKET,
                    object_id=ticket_id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "Export terminology appears in reference material without "
                        "a current blocker."
                    ),
                ),
            },
        ]

    def _message_id(self, namespace: str) -> str:
        derived = self.context.derive_seed(f"teams-message:{namespace}")
        return f"teams_message_{derived:016x}"

    def _team_id(self, team_name: str) -> str:
        derived = self.context.derive_seed(f"teams-team:{team_name}")
        return f"team_{derived:016x}"

    def _channel_id(self, team_name: str, channel_name: str) -> str:
        derived = self.context.derive_seed(f"teams-channel:{team_name}:{channel_name}")
        return f"channel_{derived:016x}"

    def _thread_id(self, event_id: str) -> str:
        derived = self.context.derive_seed(f"teams-thread:{event_id}")
        return f"teams_thread_{derived:016x}"

    def _meeting_id(self, event_id: str) -> str:
        derived = self.context.derive_seed(f"teams-meeting:{event_id}")
        return f"meeting_{derived:016x}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "account"
