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
        scenario = EventScenarioResolver(self.enterprise).primary_account_event()
        account = scenario.account
        event = scenario.event
        opportunity = scenario.opportunity
        ticket = scenario.ticket
        organizer = scenario.organizer
        account_owner = scenario.account_owner
        support_owner = scenario.support_owner
        primary_contact = scenario.primary_contact
        if opportunity is None:
            raise ValueError("slack rendering requires an account opportunity")
        if ticket is None:
            raise ValueError("slack rendering requires an account-linked ticket")

        relevant_messages = self._build_relevant_messages(
            account=account,
            event=event,
            opportunity_id=opportunity.id,
            ticket_id=ticket.id,
            organizer=organizer,
            account_owner=account_owner,
            support_owner=support_owner,
            primary_contact_name=primary_contact.first_name,
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
            thread_id=self._thread_id(f"{event.id}:hard-negatives"),
            parent_message_id=None,
            sender_employee_id=organizer.id,
        )

        attendance_messages = self._build_attendance_messages(
            account=account,
            event=event,
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
        opportunity_id: str,
        ticket_id: str,
        organizer: Employee,
        account_owner: Employee,
        support_owner: Employee,
        primary_contact_name: str,
    ) -> list[SlackRecord]:
        profile = self.context.language_profile
        thread_id = self._thread_id(event.id)
        account_channel_name = f"#acct-{_slugify(account.name)}"
        account_channel_id = self._channel_id(account_channel_name)
        product_reference = profile.product_reference(self.enterprise.products[0].name)
        event_provenance = LabelProvenance(
            object_type=ProvenanceObjectType.EVENT,
            object_id=event.id,
            explanation=(
                f"Customer timing is still open on account {account.id} "
                "before the scheduled review."
            ),
        )
        blocker_provenance = LabelProvenance(
            object_type=ProvenanceObjectType.TICKET,
            object_id=ticket_id,
            explanation=(
                f"Open export fix is constraining the customer review tied to "
                f"event {event.id}."
            ),
        )

        root = SlackRecord(
            slack_message_id=self._message_id("relevant-root"),
            channel_id=account_channel_id,
            channel_name=account_channel_name,
            thread_id=thread_id,
            parent_message_id=None,
            timestamp=event.starts_at - timedelta(hours=6),
            sender_employee_id=account_owner.id,
            body=(
                f"{profile.slack_marker}: {primary_contact_name} still needs timing "
                f"before Friday on {product_reference} or this slips a week."
            ),
            mentions=[organizer.id],
            reactions=[],
            attachments=[],
            linked_account_id=account.id,
            linked_opportunity_id=opportunity_id,
            linked_event_id=event.id,
            linked_ticket_id=None,
            primary_category=CommunicationCategory.FOLLOW_UP,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.FOLLOW_UP,
                is_relevant=True,
                provenance=event_provenance,
            ),
            provenance=event_provenance,
            source_system="slack",
        )
        blocker_reply = SlackRecord(
            slack_message_id=self._message_id("relevant-reply"),
            channel_id=account_channel_id,
            channel_name=account_channel_name,
            thread_id=thread_id,
            parent_message_id=root.slack_message_id,
            timestamp=root.timestamp + timedelta(minutes=22),
            sender_employee_id=support_owner.id,
            body=(
                f"{profile.team_alias} still has the export fix in test. "
                "I can send the date hold right after it clears."
            ),
            mentions=[account_owner.id],
            reactions=[":eyes:"],
            attachments=[],
            linked_account_id=account.id,
            linked_opportunity_id=None,
            linked_event_id=event.id,
            linked_ticket_id=ticket_id,
            primary_category=CommunicationCategory.BLOCKER,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.BLOCKER,
                is_relevant=True,
                provenance=blocker_provenance,
            ),
            provenance=blocker_provenance,
            source_system="slack",
        )
        partial_nested_reply = SlackRecord(
            slack_message_id=self._message_id("relevant-partial-nested"),
            channel_id=account_channel_id,
            channel_name=account_channel_name,
            thread_id=thread_id,
            parent_message_id=blocker_reply.slack_message_id,
            timestamp=blocker_reply.timestamp + timedelta(minutes=11),
            sender_employee_id=organizer.id,
            body="On it. Holding Friday.",
            mentions=[],
            reactions=[":thumbsup:"],
            attachments=[],
            linked_account_id=account.id,
            linked_opportunity_id=None,
            linked_event_id=event.id,
            linked_ticket_id=ticket_id,
            primary_category=CommunicationCategory.STATUS_UPDATES,
            is_relevant=False,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.STATUS_UPDATES,
                is_relevant=False,
                provenance=None,
            ),
            provenance=None,
            source_system="slack",
        )
        clarification_question = SlackRecord(
            slack_message_id=self._message_id("relevant-clarification-question"),
            channel_id=account_channel_id,
            channel_name=account_channel_name,
            thread_id=thread_id,
            parent_message_id=blocker_reply.slack_message_id,
            timestamp=partial_nested_reply.timestamp + timedelta(minutes=7),
            sender_employee_id=account_owner.id,
            body="ETA?",
            mentions=[support_owner.id],
            reactions=[":thinking_face:"],
            attachments=[],
            linked_account_id=account.id,
            linked_opportunity_id=None,
            linked_event_id=event.id,
            linked_ticket_id=ticket_id,
            primary_category=CommunicationCategory.STATUS_UPDATES,
            is_relevant=False,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.STATUS_UPDATES,
                is_relevant=False,
                provenance=None,
            ),
            provenance=None,
            source_system="slack",
        )
        clarification_answer = SlackRecord(
            slack_message_id=self._message_id("relevant-clarification-answer"),
            channel_id=account_channel_id,
            channel_name=account_channel_name,
            thread_id=thread_id,
            parent_message_id=clarification_question.slack_message_id,
            timestamp=clarification_question.timestamp + timedelta(minutes=5),
            sender_employee_id=support_owner.id,
            body="sgtm, 2pm.",
            mentions=[account_owner.id],
            reactions=[":white_check_mark:"],
            attachments=[],
            linked_account_id=account.id,
            linked_opportunity_id=None,
            linked_event_id=event.id,
            linked_ticket_id=ticket_id,
            primary_category=CommunicationCategory.STATUS_UPDATES,
            is_relevant=False,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.STATUS_UPDATES,
                is_relevant=False,
                provenance=None,
            ),
            provenance=None,
            source_system="slack",
        )
        branch_follow_up = SlackRecord(
            slack_message_id=self._message_id("relevant-branch-follow-up"),
            channel_id=account_channel_id,
            channel_name=account_channel_name,
            thread_id=thread_id,
            parent_message_id=root.slack_message_id,
            timestamp=clarification_answer.timestamp + timedelta(hours=2, minutes=25),
            sender_employee_id=organizer.id,
            body=(
                "keeping the same review plan. still need one owner on the resend before "
                "we send the recap"
            ),
            mentions=[support_owner.id],
            reactions=[],
            attachments=[],
            linked_account_id=account.id,
            linked_opportunity_id=opportunity_id,
            linked_event_id=event.id,
            linked_ticket_id=None,
            primary_category=CommunicationCategory.FOLLOW_UP,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.FOLLOW_UP,
                is_relevant=True,
                provenance=event_provenance,
            ),
            provenance=event_provenance,
            source_system="slack",
        )
        reopened_follow_up = SlackRecord(
            slack_message_id=self._message_id("relevant-reopened"),
            channel_id=account_channel_id,
            channel_name=account_channel_name,
            thread_id=thread_id,
            parent_message_id=root.slack_message_id,
            timestamp=branch_follow_up.timestamp + timedelta(days=2, hours=1),
            sender_employee_id=account_owner.id,
            body=(
                "reopening this thread since the Friday slot is live again. same review, "
                "same account, still waiting on final owner confirmation"
            ),
            mentions=[organizer.id],
            reactions=[":hourglass_flowing_sand:"],
            attachments=[],
            linked_account_id=account.id,
            linked_opportunity_id=opportunity_id,
            linked_event_id=event.id,
            linked_ticket_id=None,
            primary_category=CommunicationCategory.FOLLOW_UP,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.FOLLOW_UP,
                is_relevant=True,
                provenance=event_provenance,
            ),
            provenance=event_provenance,
            source_system="slack",
        )
        resolution_ack = SlackRecord(
            slack_message_id=self._message_id("relevant-resolution"),
            channel_id=account_channel_id,
            channel_name=account_channel_name,
            thread_id=thread_id,
            parent_message_id=reopened_follow_up.slack_message_id,
            timestamp=reopened_follow_up.timestamp + timedelta(minutes=17),
            sender_employee_id=support_owner.id,
            body="Fix is clear. recap next.",
            mentions=[account_owner.id],
            reactions=[":white_check_mark:"],
            attachments=[],
            linked_account_id=account.id,
            linked_opportunity_id=opportunity_id,
            linked_event_id=event.id,
            linked_ticket_id=ticket_id,
            primary_category=CommunicationCategory.FOLLOW_UP,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.FOLLOW_UP,
                is_relevant=True,
                provenance=event_provenance,
            ),
            provenance=event_provenance,
            source_system="slack",
        )
        return [
            root,
            blocker_reply,
            partial_nested_reply,
            clarification_question,
            clarification_answer,
            branch_follow_up,
            reopened_follow_up,
            resolution_ack,
        ]

    def _build_attendance_messages(
        self,
        *,
        account: CustomerAccount,
        event: Event,
    ) -> list[SlackRecord]:
        lifecycle = EventAttendanceBuilder(
            context=self.context,
            enterprise=self.enterprise,
        ).build(event_id=event.id)
        contacts_by_id = {
            contact.id: contact for contact in self.enterprise.contacts
        }
        attended_contact = contacts_by_id[lifecycle.attended_contact_ids[0]]
        no_show_contact = contacts_by_id[lifecycle.no_show_contact_ids[0]]
        channel_name = f"#acct-{_slugify(account.name)}"
        event_provenance = LabelProvenance(
            object_type=ProvenanceObjectType.EVENT,
            object_id=event.id,
            explanation=(
                "Attendance is grounded in invite, RSVP, reminder, and event roll call."
            ),
        )

        return [
            SlackRecord(
                slack_message_id=self._message_id("attendance-coordination"),
                channel_id=self._channel_id(channel_name),
                channel_name=channel_name,
                thread_id=self._thread_id(f"{event.id}:attendance"),
                parent_message_id=None,
                timestamp=lifecycle.internal_coordination_at,
                sender_employee_id=event.organizer_employee_id,
                body=(
                    f"{attended_contact.first_name} is already in on the bridge. "
                    f"{no_show_contact.first_name} never made it, so keep the recap light."
                ),
                mentions=[],
                reactions=[":spiral_calendar_pad:"],
                attachments=[],
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
                source_system="slack",
            )
        ]

    def _build_noise_messages(
        self,
        *,
        account: CustomerAccount,
        organizer: Employee,
        account_owner: Employee,
        event: Event,
    ) -> list[SlackRecord]:
        noise_count = max(3, int(round(8 * self.context.noise_ratio)))
        families = NoiseEngine(self.context).plan(source_system="slack", count=noise_count)

        noise_messages: list[SlackRecord] = []
        for index, family in enumerate(families):
            spec = self._noise_spec_for_family(
                family=family,
                index=index,
                account=account,
                organizer=organizer,
                account_owner=account_owner,
                event=event,
            )
            noise_messages.append(
                SlackRecord(
                    slack_message_id=self._message_id(f"{spec['suffix']}:{index}"),
                    channel_id=self._channel_id(spec["channel_name"]),
                    channel_name=spec["channel_name"],
                    thread_id=None,
                    parent_message_id=None,
                    timestamp=spec["timestamp"],
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
        parent_message_id: str | None,
        sender_employee_id: str,
    ) -> list[SlackRecord]:
        total = hard_negative_count(
            template_count=len(HardNegativeFamily),
            ratio=self.context.hard_negative_ratio,
        )
        families = HardNegativeEngine(self.context).plan(count=total)
        channel_name = f"#acct-{_slugify(account.name)}"
        channel_id = self._channel_id(channel_name)
        messages: list[SlackRecord] = []

        for index, family in enumerate(families):
            spec = self._hard_negative_spec_for_family(
                family=family,
                account=account,
                event=event,
                opportunity_id=opportunity_id,
                ticket_id=ticket_id,
            )
            messages.append(
                SlackRecord(
                    slack_message_id=self._message_id(
                        f"hard-negative-{spec['suffix']}-{index}"
                    ),
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

    def _noise_spec_for_family(
        self,
        *,
        family: NoiseFamily,
        index: int,
        account: CustomerAccount,
        organizer: Employee,
        account_owner: Employee,
        event: Event,
    ) -> NoiseMessageSpec:
        employees = self.enterprise.employees
        fallback_sender_id = employees[index % len(employees)].id
        timestamp = event.starts_at - timedelta(days=3) + timedelta(minutes=17 * index)

        if family == NoiseFamily.VAGUE_FOLLOW_UP:
            return {
                "suffix": family.value,
                "channel_name": "#deal-desk",
                "sender_employee_id": organizer.id,
                "body": "circling back next week once the notes settle",
                "mentions": [account_owner.id],
                "reactions": [],
                "attachments": [],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.DOCUMENT_REVIEW_PING:
            return {
                "suffix": family.value,
                "channel_name": "#deal-desk",
                "sender_employee_id": account_owner.id,
                "body": f"can someone skim the deck notes for {account.name} before archive?",
                "mentions": [organizer.id],
                "reactions": [],
                "attachments": ["deck-link.url"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.OWNERSHIP_AMBIGUITY:
            return {
                "suffix": family.value,
                "channel_name": "#deal-desk",
                "sender_employee_id": organizer.id,
                "body": "not sure who owns the resend, leaving it here for whoever has it",
                "mentions": [],
                "reactions": [":thinking_face:"],
                "attachments": [],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.SOCIAL_BANTER:
            return {
                "suffix": family.value,
                "channel_name": "#watercooler",
                "sender_employee_id": fallback_sender_id,
                "body": "coffee run after standup or are we all stuck on calls again?",
                "mentions": [],
                "reactions": [":coffee:"],
                "attachments": [],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.COMPLIANCE_ADMIN_REMINDER:
            return {
                "suffix": family.value,
                "channel_name": "#sales-admin",
                "sender_employee_id": fallback_sender_id,
                "body": "annual policy ack closes Friday. no exception requests in slack please.",
                "mentions": [],
                "reactions": [],
                "attachments": ["policy-ack.pdf"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.ACCESS_REQUEST:
            return {
                "suffix": family.value,
                "channel_name": "#it-help",
                "sender_employee_id": fallback_sender_id,
                "body": "need folder access restored for the archived workbook",
                "mentions": [],
                "reactions": [],
                "attachments": ["access-request.txt"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.EMPTY_CHECK_IN:
            return {
                "suffix": family.value,
                "channel_name": "#watercooler",
                "sender_employee_id": fallback_sender_id,
                "body": "any updates?",
                "mentions": [],
                "reactions": [":wave:"],
                "attachments": [],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.STATUS_NUDGE:
            return {
                "suffix": family.value,
                "channel_name": "#ops-bot",
                "sender_employee_id": fallback_sender_id,
                "body": "gentle nudge to close the checklist row when you get a minute",
                "mentions": [],
                "reactions": [],
                "attachments": ["checklist.csv"],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.PARTIAL_HANDOFF:
            return {
                "suffix": family.value,
                "channel_name": "#deal-desk",
                "sender_employee_id": organizer.id,
                "body": (
                    "handing this thread to ops for the file cleanup. "
                    "no customer action on it."
                ),
                "mentions": [account_owner.id],
                "reactions": [],
                "attachments": [],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.TOOL_OUTAGE_CHATTER:
            return {
                "suffix": family.value,
                "channel_name": "#ops-bot",
                "sender_employee_id": fallback_sender_id,
                "body": "wiki search is flaky again. saving links locally for now.",
                "mentions": [],
                "reactions": [":warning:"],
                "attachments": [],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        if family == NoiseFamily.RECRUITING_HR_CHATTER:
            return {
                "suffix": family.value,
                "channel_name": "#people-ops",
                "sender_employee_id": fallback_sender_id,
                "body": "onsite panel grid moved. someone please swap the interview room.",
                "mentions": [],
                "reactions": [],
                "attachments": [],
                "category": noise_category_for_family(family),
                "timestamp": timestamp,
            }
        return {
            "suffix": family.value,
            "channel_name": "#sales-admin",
            "sender_employee_id": fallback_sender_id,
            "body": "procurement form is missing the internal routing code only",
            "mentions": [],
            "reactions": [],
            "attachments": ["routing-code.txt"],
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
    ) -> HardNegativeMessageSpec:
        if family == HardNegativeFamily.EVENT_NO_ATTENDANCE:
            return {
                "suffix": family.value,
                "body": (
                    "review timing and the date hold are in thread, but no attendee accepted "
                    "or declined yet"
                ),
                "category": CommunicationCategory.SCHEDULING_ONLY,
                "timestamp": event.starts_at - timedelta(hours=8),
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
                    f"budget line for {account.name} is hotel, date hold meals, and review "
                    "travel, not customer spend"
                ),
                "category": CommunicationCategory.STATUS_UPDATES,
                "timestamp": event.starts_at - timedelta(hours=7, minutes=30),
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
                    "weather complaint from the travel desk may shift the review timing by a "
                    "day, nothing product-related"
                ),
                "category": CommunicationCategory.FYI_FORWARD,
                "timestamp": event.starts_at - timedelta(hours=7),
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
                    "calling the review date hold escalated for calendar cleanup, but there is "
                    "not a live blocker or export fix blocking it"
                ),
                "category": CommunicationCategory.STATUS_UPDATES,
                "timestamp": event.starts_at - timedelta(hours=6, minutes=45),
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
                    "looping the vp in on review timing for visibility only. no decision-maker "
                    "is engaged on the date hold"
                ),
                "category": CommunicationCategory.FYI_FORWARD,
                "timestamp": event.starts_at - timedelta(hours=6, minutes=15),
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
                "follow up note for the review owners is here, but no required action owner "
                "or date hold is actually assigned"
            ),
            "category": CommunicationCategory.LOW_SIGNAL_CHECKIN,
            "timestamp": event.starts_at - timedelta(hours=6),
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
