from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypedDict

from synthetic_enterprise.contracts.records.slack import SlackRecord
from synthetic_enterprise.domain import (
    CustomerAccount,
    Employee,
    EnterpriseGraph,
    Event,
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


class NoiseMessageSpec(TypedDict):
    suffix: str
    channel_name: str
    sender_employee_id: str
    body: str
    mentions: list[str]
    reactions: list[str]
    attachments: list[str]
    category: CommunicationCategory
    timestamp: datetime


class HardNegativeMessageSpec(TypedDict):
    suffix: str
    body: str
    category: CommunicationCategory
    timestamp: datetime
    linked_opportunity_id: str | None
    linked_ticket_id: str | None
    provenance: LabelProvenance


@dataclass(slots=True)
class SlackRenderer:
    """Render a small deterministic Slack dataset from simulated enterprise state."""

    context: GeneratorContext
    enterprise: EnterpriseGraph

    def generate_messages(self) -> list[SlackRecord]:
        account = self.enterprise.customer_accounts[0]
        event = next(
            current
            for current in self.enterprise.events
            if current.account_id == account.id
        )
        opportunity = next(
            current
            for current in self.enterprise.opportunities
            if current.account_id == account.id
        )
        ticket = next(
            current
            for current in self.enterprise.ticket_issues
            if current.account_id == account.id
        )
        organizer = self._employee_by_id[event.organizer_employee_id]
        account_owner = self._employee_by_id[account.owner_employee_id]
        support_owner = organizer
        if ticket.owner_employee_id is not None:
            support_owner = self._employee_by_id[ticket.owner_employee_id]
        primary_contact = next(
            contact for contact in self.enterprise.contacts if contact.account_id == account.id
        )

        thread_id = self._thread_id(event.id)
        account_channel_name = f"#acct-{_slugify(account.name)}"
        account_channel_id = self._channel_id(account_channel_name)

        relevant_root = SlackRecord(
            slack_message_id=self._message_id("relevant-root"),
            channel_id=account_channel_id,
            channel_name=account_channel_name,
            thread_id=thread_id,
            parent_message_id=None,
            timestamp=event.starts_at - timedelta(hours=6),
            sender_employee_id=account_owner.id,
            body=(
                f"{primary_contact.first_name} still needs timing before Friday "
                "or this slips a week."
            ),
            mentions=[organizer.id],
            reactions=[],
            attachments=[],
            linked_account_id=account.id,
            linked_opportunity_id=opportunity.id,
            linked_event_id=event.id,
            linked_ticket_id=None,
            primary_category=CommunicationCategory.FOLLOW_UP,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.FOLLOW_UP,
                is_relevant=True,
                provenance=LabelProvenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    explanation=(
                        f"Customer timing is still open on account {account.id} "
                        "before the scheduled review."
                    ),
                ),
            ),
            provenance=LabelProvenance(
                object_type=ProvenanceObjectType.EVENT,
                object_id=event.id,
                explanation=(
                    f"Customer timing is still open on account {account.id} "
                    "before the scheduled review."
                ),
            ),
            source_system="slack",
        )

        relevant_reply = SlackRecord(
            slack_message_id=self._message_id("relevant-reply"),
            channel_id=account_channel_id,
            channel_name=account_channel_name,
            thread_id=thread_id,
            parent_message_id=relevant_root.slack_message_id,
            timestamp=relevant_root.timestamp + timedelta(minutes=22),
            sender_employee_id=support_owner.id,
            body="Export fix is still in test. I can send the date hold right after it clears.",
            mentions=[account_owner.id],
            reactions=[":eyes:"],
            attachments=[],
            linked_account_id=account.id,
            linked_opportunity_id=None,
            linked_event_id=event.id,
            linked_ticket_id=ticket.id,
            primary_category=CommunicationCategory.BLOCKER,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.BLOCKER,
                is_relevant=True,
                provenance=LabelProvenance(
                    object_type=ProvenanceObjectType.TICKET,
                    object_id=ticket.id,
                    explanation=(
                        f"Open export fix is constraining the customer review tied to "
                        f"event {event.id}."
                    ),
                ),
            ),
            provenance=LabelProvenance(
                object_type=ProvenanceObjectType.TICKET,
                object_id=ticket.id,
                explanation=(
                    f"Open export fix is constraining the customer review tied to "
                    f"event {event.id}."
                ),
            ),
            source_system="slack",
        )

        noise_messages = self._build_noise_messages(
            account=account,
            organizer=organizer,
            account_owner=account_owner,
            event=event,
        )
        hard_negative_messages = self._build_hard_negative_messages(
            account=account,
            event=event,
            opportunity_id=opportunity.id,
            ticket_id=ticket.id,
            thread_id=thread_id,
            parent_message_id=relevant_root.slack_message_id,
            sender_employee_id=organizer.id,
        )

        return [relevant_root, relevant_reply, *noise_messages, *hard_negative_messages]

    @property
    def _employee_by_id(self) -> dict[str, Employee]:
        return {employee.id: employee for employee in self.enterprise.employees}

    def _build_noise_messages(
        self,
        *,
        account: CustomerAccount,
        organizer: Employee,
        account_owner: Employee,
        event: Event,
    ) -> list[SlackRecord]:
        noise_count = max(3, int(round(8 * self.context.noise_ratio)))
        specs = self._noise_templates(
            account=account,
            organizer=organizer,
            account_owner=account_owner,
            event=event,
        )

        noise_messages: list[SlackRecord] = []
        for index in range(noise_count):
            spec = specs[index % len(specs)]
            noise_messages.append(
                SlackRecord(
                    slack_message_id=self._message_id(spec["suffix"]),
                    channel_id=self._channel_id(spec["channel_name"]),
                    channel_name=spec["channel_name"],
                    thread_id=None,
                    parent_message_id=None,
                    timestamp=spec["timestamp"] + timedelta(minutes=index),
                    sender_employee_id=spec["sender_employee_id"],
                    body=spec["body"],
                    mentions=spec["mentions"],
                    reactions=spec["reactions"],
                    attachments=spec["attachments"],
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
                    source_system="slack",
                )
            )

        return noise_messages

    def _build_hard_negative_messages(
        self,
        *,
        account: CustomerAccount,
        event: Event,
        opportunity_id: str,
        ticket_id: str,
        thread_id: str,
        parent_message_id: str,
        sender_employee_id: str,
    ) -> list[SlackRecord]:
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
        channel_name = f"#acct-{_slugify(account.name)}"
        channel_id = self._channel_id(channel_name)
        messages: list[SlackRecord] = []

        for index in range(total):
            spec = specs[index]
            messages.append(
                SlackRecord(
                    slack_message_id=self._message_id(f"hard-negative-{spec['suffix']}"),
                    channel_id=channel_id,
                    channel_name=channel_name,
                    thread_id=thread_id,
                    parent_message_id=parent_message_id,
                    timestamp=spec["timestamp"],
                    sender_employee_id=sender_employee_id,
                    body=spec["body"],
                    mentions=[],
                    reactions=[],
                    attachments=[],
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
                    source_system="slack",
                )
            )

        return messages

    def _noise_templates(
        self,
        *,
        account: CustomerAccount,
        organizer: Employee,
        account_owner: Employee,
        event: Event,
    ) -> list[NoiseMessageSpec]:
        employees = self.enterprise.employees
        return [
            {
                "suffix": "noise-0",
                "channel_name": "#watercooler",
                "sender_employee_id": employees[0].id,
                "body": "morning all",
                "mentions": [],
                "reactions": [":wave:"],
                "attachments": [],
                "category": CommunicationCategory.GREETINGS,
                "timestamp": event.starts_at - timedelta(days=3, hours=1),
            },
            {
                "suffix": "noise-1",
                "channel_name": "#watercooler",
                "sender_employee_id": employees[1].id,
                "body": "anyone doing coffee after standup?",
                "mentions": [],
                "reactions": [":coffee:"],
                "attachments": [],
                "category": CommunicationCategory.SOCIAL_CHATTER,
                "timestamp": event.starts_at - timedelta(days=3, minutes=15),
            },
            {
                "suffix": "noise-2",
                "channel_name": "#deal-desk",
                "sender_employee_id": account_owner.id,
                "body": f"deck is in the folder for {account.name}",
                "mentions": [organizer.id],
                "reactions": [],
                "attachments": ["deck-link.url"],
                "category": CommunicationCategory.STATUS_UPDATES,
                "timestamp": event.starts_at - timedelta(days=2, hours=2),
            },
            {
                "suffix": "noise-3",
                "channel_name": "#deal-desk",
                "sender_employee_id": organizer.id,
                "body": "ping me when you have five",
                "mentions": [account_owner.id],
                "reactions": [],
                "attachments": [],
                "category": CommunicationCategory.LOW_SIGNAL_CHECKIN,
                "timestamp": event.starts_at - timedelta(days=2, hours=1),
            },
            {
                "suffix": "noise-4",
                "channel_name": "#ops-bot",
                "sender_employee_id": employees[2].id,
                "body": "build green. no action needed.",
                "mentions": [],
                "reactions": [":white_check_mark:"],
                "attachments": ["build-log.txt"],
                "category": CommunicationCategory.AUTOMATED_NOTIFICATION,
                "timestamp": event.starts_at - timedelta(days=1, hours=5),
            },
            {
                "suffix": "noise-5",
                "channel_name": "#sales-admin",
                "sender_employee_id": employees[3].id,
                "body": "pto calendar updated for next week",
                "mentions": [],
                "reactions": [],
                "attachments": [],
                "category": CommunicationCategory.ADMIN_OPS,
                "timestamp": event.starts_at - timedelta(days=1, hours=3),
            },
            {
                "suffix": "noise-6",
                "channel_name": "#deal-desk",
                "sender_employee_id": organizer.id,
                "body": "same note as above. no change.",
                "mentions": [account_owner.id],
                "reactions": [],
                "attachments": [],
                "category": CommunicationCategory.DUPLICATE_SUMMARY,
                "timestamp": event.starts_at - timedelta(days=1, hours=2),
            },
            {
                "suffix": "noise-7",
                "channel_name": "#ops-bot",
                "sender_employee_id": employees[4].id,
                "body": "calendar invite sent",
                "mentions": [],
                "reactions": [],
                "attachments": ["invite.ics"],
                "category": CommunicationCategory.SCHEDULING_ONLY,
                "timestamp": event.starts_at - timedelta(days=1, hours=1),
            },
        ]

    def _hard_negative_templates(
        self,
        *,
        account: CustomerAccount,
        event: Event,
        opportunity_id: str,
        ticket_id: str,
    ) -> list[HardNegativeMessageSpec]:
        return [
            {
                "suffix": "timing",
                "body": (
                    "let's discuss the review timing next week after the customer trip is "
                    "rebooked"
                ),
                "category": CommunicationCategory.SCHEDULING_ONLY,
                "timestamp": event.starts_at - timedelta(hours=8),
                "linked_opportunity_id": None,
                "linked_ticket_id": None,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "Review timing is mentioned without attendance or action "
                        "ownership."
                    ),
                ),
            },
            {
                "suffix": "budget",
                "body": (
                    f"budget line for {account.name} is hotel and booth shipping for the "
                    "review, not customer spend"
                ),
                "category": CommunicationCategory.STATUS_UPDATES,
                "timestamp": event.starts_at - timedelta(hours=7, minutes=30),
                "linked_opportunity_id": opportunity_id,
                "linked_ticket_id": None,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.OPPORTUNITY,
                    object_id=opportunity_id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "Budget wording refers to internal event logistics rather "
                        "than buying intent."
                    ),
                ),
            },
            {
                "suffix": "travel",
                "body": (
                    "weather complaint from the travel desk may shift the onsite review by a "
                    "day, nothing product-related"
                ),
                "category": CommunicationCategory.FYI_FORWARD,
                "timestamp": event.starts_at - timedelta(hours=7),
                "linked_opportunity_id": None,
                "linked_ticket_id": None,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    strength=ProvenanceStrength.WEAK,
                    explanation="Complaint is about travel logistics rather than customer pain.",
                ),
            },
            {
                "suffix": "export-ref",
                "body": (
                    "sharing the export checklist from last quarter for reference only, no "
                    "customer action today"
                ),
                "category": CommunicationCategory.DUPLICATE_SUMMARY,
                "timestamp": event.starts_at - timedelta(hours=6, minutes=30),
                "linked_opportunity_id": None,
                "linked_ticket_id": ticket_id,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.TICKET,
                    object_id=ticket_id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "Export keyword appears in archived reference material "
                        "without a live blocker."
                    ),
                ),
            },
        ]

    def _message_id(self, namespace: str) -> str:
        derived = self.context.derive_seed(f"slack-message:{namespace}")
        return f"slack_message_{derived:016x}"

    def _thread_id(self, event_id: str) -> str:
        derived = self.context.derive_seed(f"slack-thread:{event_id}")
        return f"slack_thread_{derived:016x}"

    def _channel_id(self, channel_name: str) -> str:
        derived = self.context.derive_seed(f"slack-channel:{channel_name}")
        return f"channel_{derived:016x}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "account"
