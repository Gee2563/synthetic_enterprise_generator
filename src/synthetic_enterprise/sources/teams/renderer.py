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
from synthetic_enterprise.generation.attendance import EventAttendanceBuilder
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.hard_negatives import (
    HardNegativeEngine,
    HardNegativeFamily,
    hard_negative_count,
    hard_negative_explanation,
)
from synthetic_enterprise.generation.noise_engine import (
    NoiseEngine,
    NoiseFamily,
    noise_category_for_family,
)
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
        attendance_messages = self._build_attendance_messages(
            account=account,
            event=event,
            organizer=organizer,
        )

        return [
            *relevant_messages,
            *attendance_messages,
            *noise_messages,
            *hard_negative_messages,
        ]

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
        profile = self.context.language_profile
        team_name = "Customer Delivery"
        channel_name = f"acct-{_slugify(account.name)}"
        team_id = self._team_id(team_name)
        channel_id = self._channel_id(team_name, channel_name)
        thread_id = self._thread_id(event.id)
        meeting_id = self._meeting_id(event.id)
        product_reference = profile.product_reference(self.enterprise.products[0].name)
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
        status_timestamp = event.starts_at - timedelta(hours=3, minutes=24)
        delayed_timestamp = event.starts_at - timedelta(hours=1)
        reopened_timestamp = delayed_timestamp + timedelta(days=2, hours=2)

        root = TeamsRecord(
            teams_message_id=self._message_id("relevant-root"),
            team_id=team_id,
            channel_id=channel_id,
            chat_or_channel="channel",
            thread_id=thread_id,
            timestamp=event.starts_at - timedelta(hours=4),
            sender_employee_id=organizer.id,
            body=(
                f"Agenda: {profile.teams_marker}\n"
                f"- confirm owners for {account.name}\n"
                f"- document the revised date hold for the {profile.meeting_language}\n"
                f"- capture the export test result for {product_reference}\n"
                f"Document: rollout-plan.docx ({profile.team_alias})"
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
                f"Status update: {profile.teams_marker}\n"
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
        partial_reply = TeamsRecord(
            teams_message_id=self._message_id("relevant-partial-reply"),
            team_id=team_id,
            channel_id=channel_id,
            chat_or_channel="channel",
            thread_id=thread_id,
            timestamp=status_timestamp,
            sender_employee_id=account_owner.id,
            body=(
                "Recap:\n"
                "- keeping the same customer review slot\n"
                "- waiting on one owner name before the resend"
            ),
            mentions=[organizer.id],
            meeting_id=meeting_id,
            file_refs=["owner-log.docx"],
            linked_account_id=account.id,
            linked_opportunity_id=opportunity_id,
            linked_event_id=event.id,
            linked_ticket_id=None,
            primary_category=CommunicationCategory.STATUS_UPDATES,
            is_relevant=False,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.STATUS_UPDATES,
                is_relevant=False,
                provenance=None,
            ),
            provenance=None,
            source_system="teams",
        )
        delayed_follow_up = TeamsRecord(
            teams_message_id=self._message_id("relevant-delayed-follow-up"),
            team_id=team_id,
            channel_id=channel_id,
            chat_or_channel="channel",
            thread_id=thread_id,
            timestamp=delayed_timestamp,
            sender_employee_id=organizer.id,
            body=(
                "Task list:\n"
                f"- same {profile.meeting_language}\n"
                "- resend the notes once the owner line is confirmed\n"
                "- capture the final customer acknowledgment in this thread"
            ),
            mentions=[account_owner.id],
            meeting_id=meeting_id,
            file_refs=["meeting-notes.docx"],
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
        reopened_thread = TeamsRecord(
            teams_message_id=self._message_id("relevant-reopened-thread"),
            team_id=team_id,
            channel_id=channel_id,
            chat_or_channel="channel",
            thread_id=thread_id,
            timestamp=reopened_timestamp,
            sender_employee_id=account_owner.id,
            body=(
                "Status update:\n"
                "- reopening this thread after the weekend gap\n"
                "- same account review, same attendees, still need the final owner"
            ),
            mentions=[organizer.id, support_owner.id],
            meeting_id=meeting_id,
            file_refs=["reopen-log.docx"],
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
        resolution_ack = TeamsRecord(
            teams_message_id=self._message_id("relevant-resolution-ack"),
            team_id=team_id,
            channel_id=channel_id,
            chat_or_channel="channel",
            thread_id=thread_id,
            timestamp=reopened_thread.timestamp + timedelta(minutes=21),
            sender_employee_id=support_owner.id,
            body=(
                "Resolution note:\n"
                "- validation passed\n"
                "- posting the customer-ready revision and closing the loop"
            ),
            mentions=[organizer.id],
            meeting_id=meeting_id,
            file_refs=["validation-checklist.xlsx", "rollout-plan.docx"],
            linked_account_id=account.id,
            linked_opportunity_id=opportunity_id,
            linked_event_id=event.id,
            linked_ticket_id=ticket_id,
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
        return [
            root,
            reply,
            partial_reply,
            delayed_follow_up,
            reopened_thread,
            resolution_ack,
        ]

    def _build_attendance_messages(
        self,
        *,
        account: CustomerAccount,
        event: Event,
        organizer: Employee,
    ) -> list[TeamsRecord]:
        lifecycle = EventAttendanceBuilder(
            context=self.context,
            enterprise=self.enterprise,
        ).build(event_id=event.id)
        contacts_by_id = {
            contact.id: contact for contact in self.enterprise.contacts
        }
        attended_contact = contacts_by_id[lifecycle.attended_contact_ids[0]]
        no_show_contact = contacts_by_id[lifecycle.no_show_contact_ids[0]]
        team_name = "Customer Delivery"
        channel_name = f"acct-{_slugify(account.name)}"
        event_provenance = LabelProvenance(
            object_type=ProvenanceObjectType.EVENT,
            object_id=event.id,
            explanation=(
                "Attendance is grounded in the RSVP trail and meeting recap."
            ),
        )

        return [
            TeamsRecord(
                teams_message_id=self._message_id("attendance-recap"),
                team_id=self._team_id(team_name),
                channel_id=self._channel_id(team_name, channel_name),
                chat_or_channel="channel",
                thread_id=self._thread_id(f"{event.id}:attendance"),
                timestamp=lifecycle.post_event_follow_up_at,
                sender_employee_id=organizer.id,
                body=(
                    "Recap:\n"
                    f"- {attended_contact.first_name} joined the live review\n"
                    f"- {no_show_contact.first_name} missed the live session\n"
                    "- follow up with the recording and attendance recap"
                ),
                mentions=[],
                meeting_id=self._meeting_id(event.id),
                file_refs=["attendance-recap.docx", "meeting-notes.docx"],
                linked_account_id=account.id,
                linked_opportunity_id=None,
                linked_event_id=event.id,
                linked_ticket_id=None,
                primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                    is_relevant=True,
                    provenance=event_provenance,
                ),
                provenance=event_provenance,
                source_system="teams",
            )
        ]

    def _build_noise_messages(
        self,
        *,
        account: CustomerAccount,
        event: Event,
        organizer: Employee,
        account_owner: Employee,
    ) -> list[TeamsRecord]:
        noise_count = max(4, int(round(8 * self.context.noise_ratio)))
        families = NoiseEngine(self.context).plan(source_system="teams", count=noise_count)
        messages: list[TeamsRecord] = []

        for index, family in enumerate(families):
            spec = self._noise_spec_for_family(
                family=family,
                index=index,
                account=account,
                event=event,
                organizer=organizer,
                account_owner=account_owner,
            )
            messages.append(
                TeamsRecord(
                    teams_message_id=self._message_id(f"{spec['suffix']}-{index}"),
                    team_id=self._team_id(spec["team_name"]),
                    channel_id=self._channel_id(spec["team_name"], spec["channel_name"]),
                    chat_or_channel=spec["chat_or_channel"],
                    thread_id=None,
                    timestamp=spec["timestamp"],
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
            template_count=len(HardNegativeFamily),
            ratio=self.context.hard_negative_ratio,
        )
        families = HardNegativeEngine(self.context).plan(count=total)
        team_name = "Customer Delivery"
        channel_name = f"acct-{_slugify(account.name)}"
        team_id = self._team_id(team_name)
        channel_id = self._channel_id(team_name, channel_name)
        meeting_id = self._meeting_id(event.id)
        sender_employee_id = organizer.id
        messages: list[TeamsRecord] = []

        for index, family in enumerate(families):
            spec = self._hard_negative_spec_for_family(
                family=family,
                account=account,
                event=event,
                opportunity_id=opportunity_id,
                ticket_id=ticket_id,
            )
            messages.append(
                TeamsRecord(
                    teams_message_id=self._message_id(
                        f"hard-negative-{spec['suffix']}-{index}"
                    ),
                    team_id=team_id,
                    channel_id=channel_id,
                    chat_or_channel="channel",
                    thread_id=self._thread_id(
                        f"{event.id}:hard-negative:{family.value}:{index}"
                    ),
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

    def _noise_spec_for_family(
        self,
        *,
        family: NoiseFamily,
        index: int,
        account: CustomerAccount,
        event: Event,
        organizer: Employee,
        account_owner: Employee,
    ) -> TeamsNoiseSpec:
        employees = self.enterprise.employees
        fallback_sender_id = employees[index % len(employees)].id
        timestamp = event.starts_at - timedelta(days=2) + timedelta(minutes=23 * index)

        if family == NoiseFamily.DOCUMENT_REVIEW_PING:
            return {
                "suffix": family.value,
                "team_name": "Operations",
                "channel_name": "project-updates",
                "chat_or_channel": "channel",
                "sender_employee_id": account_owner.id,
                "body": (
                    "Document: prep-notes.docx\n"
                    f"Please skim the comments for {account.name} before archive."
                ),
                "mentions": [organizer.id],
                "meeting_id": None,
                "file_refs": ["prep-notes.docx"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.AGENDA_COORDINATION:
            return {
                "suffix": family.value,
                "team_name": "Operations",
                "channel_name": "meeting-prep",
                "chat_or_channel": "channel",
                "sender_employee_id": organizer.id,
                "body": "Agenda note:\n- keep the notes doc open\n- no customer update is needed",
                "mentions": [account_owner.id],
                "meeting_id": None,
                "file_refs": ["meeting-notes.docx"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.INTERNAL_FYI_SUMMARY:
            return {
                "suffix": family.value,
                "team_name": "Operations",
                "channel_name": "pm-chat",
                "chat_or_channel": "chat",
                "sender_employee_id": fallback_sender_id,
                "body": (
                    "FYI summary:\n"
                    "- folder renamed\n"
                    "- workbook archived\n"
                    "- no action requested"
                ),
                "mentions": [],
                "meeting_id": None,
                "file_refs": ["archive-log.docx"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.DUPLICATE_REMINDER:
            return {
                "suffix": family.value,
                "team_name": "Operations",
                "channel_name": "pm-chat",
                "chat_or_channel": "chat",
                "sender_employee_id": fallback_sender_id,
                "body": "same reminder as above copied here for the thread history",
                "mentions": [organizer.id],
                "meeting_id": None,
                "file_refs": [],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.OWNERSHIP_AMBIGUITY:
            return {
                "suffix": family.value,
                "team_name": "Operations",
                "channel_name": "project-updates",
                "chat_or_channel": "channel",
                "sender_employee_id": organizer.id,
                "body": "Owner still unclear on the archive pass. tagging nobody until it lands.",
                "mentions": [],
                "meeting_id": None,
                "file_refs": [],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.COMPLIANCE_ADMIN_REMINDER:
            return {
                "suffix": family.value,
                "team_name": "Operations",
                "channel_name": "ops-admin",
                "chat_or_channel": "channel",
                "sender_employee_id": fallback_sender_id,
                "body": (
                    "Compliance reminder:\n"
                    "- finish annual training\n"
                    "- upload the completion PDF"
                ),
                "mentions": [],
                "meeting_id": None,
                "file_refs": ["training-guide.pdf"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.ACCESS_REQUEST:
            return {
                "suffix": family.value,
                "team_name": "Operations",
                "channel_name": "it-help",
                "chat_or_channel": "chat",
                "sender_employee_id": fallback_sender_id,
                "body": (
                    "Access request:\n"
                    "- restore edit rights on the workbook\n"
                    "- no file review needed"
                ),
                "mentions": [],
                "meeting_id": None,
                "file_refs": ["permissions.xlsx"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.STATUS_NUDGE:
            return {
                "suffix": family.value,
                "team_name": "Operations",
                "channel_name": "project-updates",
                "chat_or_channel": "channel",
                "sender_employee_id": fallback_sender_id,
                "body": (
                    "Status update:\n"
                    "- nudging the tracker row\n"
                    "- no customer-facing work is blocked"
                ),
                "mentions": [],
                "meeting_id": None,
                "file_refs": ["status-tracker.xlsx"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.PARTIAL_HANDOFF:
            return {
                "suffix": family.value,
                "team_name": "Operations",
                "channel_name": "handoffs",
                "chat_or_channel": "channel",
                "sender_employee_id": organizer.id,
                "body": (
                    "Partial handoff:\n"
                    "- moved file cleanup to ops\n"
                    "- left the rest for tomorrow"
                ),
                "mentions": [account_owner.id],
                "meeting_id": None,
                "file_refs": ["handoff-notes.docx"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.IRRELEVANT_FORWARDED_CHAIN:
            return {
                "suffix": family.value,
                "team_name": "Operations",
                "channel_name": "meeting-prep",
                "chat_or_channel": "channel",
                "sender_employee_id": fallback_sender_id,
                "body": (
                    "Document: prior-thread.msg\n"
                    "Forwarded for reference only. No update needed."
                ),
                "mentions": [],
                "meeting_id": None,
                "file_refs": ["prior-thread.msg"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        return {
            "suffix": family.value,
            "team_name": "Operations",
            "channel_name": "ops-bot",
            "chat_or_channel": "channel",
            "sender_employee_id": fallback_sender_id,
            "body": "Automated notice: internal wiki search is degraded. Retry later.",
            "mentions": [],
            "meeting_id": None,
            "file_refs": ["ops-status.docx"],
            "category": noise_category_for_family(family),
            "timestamp": timestamp,
        }

    def _hard_negative_spec_for_family(
        self,
        *,
        family: HardNegativeFamily,
        account: CustomerAccount,
        event: Event,
        opportunity_id: str,
        ticket_id: str,
    ) -> TeamsHardNegativeSpec:
        if family == HardNegativeFamily.EVENT_NO_ATTENDANCE:
            return {
                "suffix": family.value,
                "body": (
                    "Agenda addendum:\n"
                    "- review attendance is still unconfirmed\n"
                    "- no attendee accepted the date hold or declined yet"
                ),
                "file_refs": ["attendance-draft.docx"],
                "category": CommunicationCategory.SCHEDULING_ONLY,
                "timestamp": event.starts_at - timedelta(hours=6),
                "linked_opportunity_id": None,
                "linked_ticket_id": None,
                "provenance": self._weak_provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    explanation=hard_negative_explanation(family),
                ),
            }
        if family == HardNegativeFamily.BUDGET_NON_BUYING:
            return {
                "suffix": family.value,
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
                "provenance": self._weak_provenance(
                    object_type=ProvenanceObjectType.OPPORTUNITY,
                    object_id=opportunity_id,
                    explanation=hard_negative_explanation(family),
                ),
            }
        if family == HardNegativeFamily.COMPLAINT_NON_PRODUCT:
            return {
                "suffix": family.value,
                "body": (
                    "Document: weather-plan.docx\n"
                    "Travel complaint may move the onsite review check-in by 30 minutes."
                ),
                "file_refs": ["weather-plan.docx"],
                "category": CommunicationCategory.FYI_FORWARD,
                "timestamp": event.starts_at - timedelta(hours=5),
                "linked_opportunity_id": None,
                "linked_ticket_id": None,
                "provenance": self._weak_provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    explanation=hard_negative_explanation(family),
                ),
            }
        if family == HardNegativeFamily.ESCALATION_NO_BLOCKER:
            return {
                "suffix": family.value,
                "body": (
                    "Status update:\n"
                    "- sounds escalated for schedule visibility\n"
                    "- there is not a live blocker in validation or export"
                ),
                "file_refs": ["status-tracker.xlsx"],
                "category": CommunicationCategory.STATUS_UPDATES,
                "timestamp": event.starts_at - timedelta(hours=4, minutes=45),
                "linked_opportunity_id": None,
                "linked_ticket_id": ticket_id,
                "provenance": self._weak_provenance(
                    object_type=ProvenanceObjectType.TICKET,
                    object_id=ticket_id,
                    explanation=hard_negative_explanation(family),
                ),
            }
        if family == HardNegativeFamily.EXECUTIVE_VISIBILITY_ONLY:
            return {
                "suffix": family.value,
                "body": (
                    "Document: exec-summary.docx\n"
                    "Adding the VP for visibility only. No executive sponsor is engaged on "
                    "owners or next steps."
                ),
                "file_refs": ["exec-summary.docx"],
                "category": CommunicationCategory.FYI_FORWARD,
                "timestamp": event.starts_at - timedelta(hours=4, minutes=15),
                "linked_opportunity_id": opportunity_id,
                "linked_ticket_id": None,
                "provenance": self._weak_provenance(
                    object_type=ProvenanceObjectType.ACCOUNT,
                    object_id=account.id,
                    explanation=hard_negative_explanation(family),
                ),
            }
        return {
            "suffix": family.value,
            "body": (
                "Action log:\n"
                "- follow up mentioned for the review owners\n"
                "- no required action owner or due date"
            ),
            "file_refs": ["action-log.docx"],
            "category": CommunicationCategory.LOW_SIGNAL_CHECKIN,
            "timestamp": event.starts_at - timedelta(hours=4),
            "linked_opportunity_id": None,
            "linked_ticket_id": None,
            "provenance": self._weak_provenance(
                object_type=ProvenanceObjectType.EVENT,
                object_id=event.id,
                explanation=hard_negative_explanation(family),
            ),
        }

    def _weak_provenance(
        self,
        *,
        object_type: ProvenanceObjectType,
        object_id: str,
        explanation: str,
    ) -> LabelProvenance:
        return LabelProvenance(
            object_type=object_type,
            object_id=object_id,
            strength=ProvenanceStrength.WEAK,
            explanation=explanation,
        )

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
