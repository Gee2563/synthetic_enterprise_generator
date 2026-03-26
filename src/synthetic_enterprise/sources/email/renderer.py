from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypedDict

from synthetic_enterprise.contracts.records.email import EmailRecord
from synthetic_enterprise.domain import (
    Contact,
    CustomerAccount,
    Employee,
    EnterpriseGraph,
    Event,
    EventScenarioResolver,
    Opportunity,
    TicketIssue,
)
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.hard_negatives import hard_negative_count
from synthetic_enterprise.labeling.grounding import (
    LabelProvenance,
    ProvenanceObjectType,
    ProvenanceStrength,
    build_relevance_reason,
)
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory, CommunicationTaxonomy


class RenderedEmailContent(TypedDict):
    subject: str
    body: str
    attachments: list[str]
    cc: list[str]
    opportunity_id: str | None
    ticket_id: str | None
    provenance: LabelProvenance | None


class HardNegativeEmailContent(RenderedEmailContent):
    category: CommunicationCategory


@dataclass(slots=True)
class EmailRenderer:
    """Render one deterministic email row at a time from simulated events."""

    context: GeneratorContext
    enterprise: EnterpriseGraph

    def generate_messages(self, *, event_id: str) -> list[EmailRecord]:
        messages = [
            self.render_from_event(
                event_id=event_id,
                primary_category=CommunicationCategory.SCHEDULING_ONLY,
                message_index_in_thread=0,
            ),
            self.render_from_event(
                event_id=event_id,
                primary_category=CommunicationCategory.FOLLOW_UP,
                message_index_in_thread=1,
            ),
        ]
        hard_negative_total = hard_negative_count(
            template_count=4,
            ratio=self.context.hard_negative_ratio,
        )

        for template_index in range(hard_negative_total):
            messages.append(
                self._render_hard_negative_from_event(
                    event_id=event_id,
                    template_index=template_index,
                    message_index_in_thread=template_index + 2,
                )
            )

        return sorted(messages, key=lambda message: message.message_index_in_thread)

    def render_from_event(
        self,
        *,
        event_id: str,
        primary_category: CommunicationCategory,
        message_index_in_thread: int = 0,
    ) -> EmailRecord:
        scenario = EventScenarioResolver(self.enterprise).for_event(event_id)
        event = scenario.event
        taxonomy = CommunicationTaxonomy(primary_category=primary_category)
        account = scenario.account
        recipients = list(scenario.attendee_contacts)
        if not recipients:
            raise ValueError("event emails require at least one attendee contact")

        organizer = scenario.organizer
        opportunity = scenario.opportunity
        ticket = scenario.ticket
        thread_id = self._thread_id(event.id)
        timestamp = self._timestamp_for(event, taxonomy.primary_category, message_index_in_thread)
        content = self._render_content(
            event=event,
            account=account,
            organizer=organizer,
            recipients=recipients,
            opportunity=opportunity,
            ticket=ticket,
            category=taxonomy.primary_category,
        )

        return EmailRecord(
            email_id=self._email_id(event.id, taxonomy.primary_category, message_index_in_thread),
            thread_id=thread_id,
            message_index_in_thread=message_index_in_thread,
            timestamp=timestamp,
            sender_employee_id=organizer.id,
            sender_contact_id=None,
            to=[contact.id for contact in recipients],
            cc=content["cc"],
            bcc=[],
            subject=content["subject"],
            body=content["body"],
            attachments=content["attachments"],
            account_id=account.id,
            opportunity_id=content["opportunity_id"],
            event_id=event.id,
            ticket_id=content["ticket_id"],
            primary_category=taxonomy.primary_category,
            is_relevant=taxonomy.is_relevant,
            relevance_reason=build_relevance_reason(
                primary_category=taxonomy.primary_category,
                is_relevant=taxonomy.is_relevant,
                provenance=content["provenance"],
            ),
            provenance=content["provenance"],
            source_system="email",
        )

    def _render_hard_negative_from_event(
        self,
        *,
        event_id: str,
        template_index: int,
        message_index_in_thread: int,
    ) -> EmailRecord:
        scenario = EventScenarioResolver(self.enterprise).for_event(event_id)
        event = scenario.event
        account = scenario.account
        recipients = list(scenario.attendee_contacts)
        organizer = scenario.organizer
        opportunity = scenario.opportunity
        ticket = scenario.ticket
        content = self._hard_negative_content(
            event=event,
            account=account,
            organizer=organizer,
            opportunity=opportunity,
            ticket=ticket,
            template_index=template_index,
        )

        return EmailRecord(
            email_id=self._email_id(
                event.id,
                content["category"],
                message_index_in_thread,
            ),
            thread_id=self._thread_id(event.id),
            message_index_in_thread=message_index_in_thread,
            timestamp=(
                event.starts_at
                - timedelta(hours=12)
                + timedelta(minutes=20 * template_index)
            ),
            sender_employee_id=organizer.id,
            sender_contact_id=None,
            to=[contact.id for contact in recipients],
            cc=content["cc"],
            bcc=[],
            subject=content["subject"],
            body=content["body"],
            attachments=content["attachments"],
            account_id=account.id,
            opportunity_id=content["opportunity_id"],
            event_id=event.id,
            ticket_id=content["ticket_id"],
            primary_category=content["category"],
            is_relevant=False,
            relevance_reason=build_relevance_reason(
                primary_category=content["category"],
                is_relevant=False,
                provenance=content["provenance"],
            ),
            provenance=content["provenance"],
            source_system="email",
        )

    def _thread_id(self, event_id: str) -> str:
        thread_seed = self.context.derive_seed(f"email-thread:{event_id}")
        return f"thread_{thread_seed:016x}"

    def _email_id(
        self,
        event_id: str,
        category: CommunicationCategory,
        message_index_in_thread: int,
    ) -> str:
        email_seed = self.context.derive_seed(
            f"email:{event_id}:{category.value}:{message_index_in_thread}"
        )
        return f"email_{email_seed:016x}"

    def _timestamp_for(
        self,
        event: Event,
        category: CommunicationCategory,
        message_index_in_thread: int,
    ) -> datetime:
        if category == CommunicationCategory.SCHEDULING_ONLY:
            base = event.starts_at - timedelta(days=2)
            return base + timedelta(minutes=30 * message_index_in_thread)

        base = event.ends_at + timedelta(hours=1)
        return base + timedelta(hours=message_index_in_thread)

    def _render_content(
        self,
        *,
        event: Event,
        account: CustomerAccount,
        organizer: Employee,
        recipients: list[Contact],
        opportunity: Opportunity | None,
        ticket: TicketIssue | None,
        category: CommunicationCategory,
    ) -> RenderedEmailContent:
        if category == CommunicationCategory.SCHEDULING_ONLY:
            return {
                "subject": f"Time update for {event.title}",
                "body": (
                    f"Hi team, moving {event.title} for {account.name} to "
                    f"{event.starts_at:%A %H:%M %Z}. Please confirm the calendar still works."
                ),
                "attachments": ["calendar.ics"],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": None,
            }

        if category != CommunicationCategory.FOLLOW_UP:
            raise ValueError(
                "basic email renderer currently supports follow_up and scheduling_only"
            )

        opportunity_id = opportunity.id if opportunity is not None else None
        ticket_id = ticket.id if ticket is not None else None
        next_step_text = "Please reply with owners for the next step by Friday."
        if opportunity is not None:
            next_step_text = (
                f"We should keep the {opportunity.stage} motion moving with the current plan. "
                "Please reply with owners for the next step by Friday."
            )

        issue_text = ""
        if ticket is not None:
            issue_text = f" We also need to close the open issue on {ticket.summary}"

        return {
            "subject": f"Follow up from {event.title} with {account.name}",
            "body": (
                f"Hi {recipients[0].first_name}, following our {event.title} with {account.name}, "
                f"{next_step_text}{issue_text}"
            ),
            "attachments": ["action-items.txt"],
            "cc": [account.owner_employee_id],
            "opportunity_id": opportunity_id,
            "ticket_id": ticket_id,
            "provenance": LabelProvenance(
                object_type=ProvenanceObjectType.EVENT,
                object_id=event.id,
                explanation=(
                    f"Customer review for account {account.id} ended with outstanding "
                    f"owners from organizer {organizer.id}."
                ),
            ),
        }

    def _hard_negative_content(
        self,
        *,
        event: Event,
        account: CustomerAccount,
        organizer: Employee,
        opportunity: Opportunity | None,
        ticket: TicketIssue | None,
        template_index: int,
    ) -> HardNegativeEmailContent:
        templates: list[HardNegativeEmailContent] = [
            {
                "category": CommunicationCategory.SCHEDULING_ONLY,
                "subject": f"Next week review timing for {account.name}",
                "body": (
                    f"Let's discuss the {event.title.lower()} logistics next week after travel "
                    "is rebooked. No customer decision is needed in this thread."
                ),
                "attachments": ["travel-note.txt"],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "Event timing is mentioned without attendance or follow-up "
                        "commitment."
                    ),
                ),
            },
            {
                "category": CommunicationCategory.FYI_FORWARD,
                "subject": f"Fwd: budget worksheet for {account.name} review",
                "body": (
                    f"Forwarding the travel and room budget worksheet for {event.title.lower()}. "
                    "This is internal planning only and not a commercial update."
                ),
                "attachments": ["review-budget.xlsx"],
                "cc": [organizer.id],
                "opportunity_id": opportunity.id if opportunity is not None else None,
                "ticket_id": None,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.OPPORTUNITY,
                    object_id=opportunity.id if opportunity is not None else event.id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "Budget wording refers to travel logistics rather than "
                        "buying intent."
                    ),
                )
                if opportunity is not None
                else LabelProvenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "Budget wording refers to travel logistics rather than "
                        "buying intent."
                    ),
                ),
            },
            {
                "category": CommunicationCategory.STATUS_UPDATES,
                "subject": f"Weather note for {event.title}",
                "body": (
                    "Travel complaints may shift arrival times for the quarterly review, "
                    "but there is no product issue to resolve."
                ),
                "attachments": [],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "The complaint is about weather and travel rather than "
                        "customer pain."
                    ),
                ),
            },
            {
                "category": CommunicationCategory.DUPLICATE_SUMMARY,
                "subject": f"Reference deck for {event.title}",
                "body": (
                    "Sharing the export FAQ and prior review deck for reference only. "
                    "There is no new customer action in this note."
                ),
                "attachments": ["review-reference.pdf"],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": ticket.id if ticket is not None else None,
                "provenance": LabelProvenance(
                    object_type=ProvenanceObjectType.TICKET,
                    object_id=ticket.id if ticket is not None else event.id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "Keyword overlap comes from forwarded reference material "
                        "without an active blocker."
                    ),
                )
                if ticket is not None
                else LabelProvenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    strength=ProvenanceStrength.WEAK,
                    explanation=(
                        "Keyword overlap comes from forwarded reference material "
                        "without an active blocker."
                    ),
                ),
            },
        ]

        return templates[template_index % len(templates)]
