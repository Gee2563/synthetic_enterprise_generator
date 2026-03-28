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
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory, CommunicationTaxonomy
from synthetic_enterprise.sources.email.templates import (
    EmailAudience,
    EmailDetailLevel,
    EmailSeniority,
    EmailStyleEngine,
    EmailStyleRequest,
    EmailSubjectMode,
)


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
        scenario = EventScenarioResolver(self.enterprise).for_event(event_id)
        messages = self._build_thread_messages(event_id=event_id)
        messages.extend(self._build_attendance_messages(event_id=event_id))
        logistics_email = self.render_from_event(
            event_id=event_id,
            primary_category=CommunicationCategory.SCHEDULING_ONLY,
            message_index_in_thread=0,
        ).model_copy(
            update={
                "thread_id": self._thread_id(f"{event_id}:logistics"),
                "message_index_in_thread": 0,
            }
        )
        messages.append(logistics_email)
        noise_families = NoiseEngine(self.context).plan(
            source_system="email",
            count=max(3, int(round(5 * self.context.noise_ratio))),
        )
        next_message_index = 6

        for noise_family in noise_families:
            messages.append(
                self._render_noise_from_event(
                    event_id=event_id,
                    family=noise_family,
                    message_index_in_thread=next_message_index,
                )
            )
            next_message_index += 1

        hard_negative_total = hard_negative_count(
            template_count=len(HardNegativeFamily),
            ratio=self.context.hard_negative_ratio,
        )
        hard_negative_families = HardNegativeEngine(self.context).plan(count=hard_negative_total)

        for template_index, hard_negative_family in enumerate(hard_negative_families):
            messages.append(
                self._render_hard_negative_from_event(
                    event_id=event_id,
                    thread_id=self._thread_id(
                        f"{scenario.event.id}:hard-negative:{hard_negative_family.value}"
                    ),
                    family=hard_negative_family,
                    message_index_in_thread=0,
                    email_namespace_index=next_message_index + template_index,
                )
            )

        return sorted(messages, key=lambda message: (message.timestamp, message.email_id))

    def _build_attendance_messages(self, *, event_id: str) -> list[EmailRecord]:
        scenario = EventScenarioResolver(self.enterprise).for_event(event_id)
        lifecycle = EventAttendanceBuilder(
            context=self.context,
            enterprise=self.enterprise,
        ).build(event_id=event_id)
        account = scenario.account
        organizer = scenario.organizer
        contacts_by_id = {
            contact.id: contact for contact in scenario.account_contacts
        }
        attended_contact = contacts_by_id[lifecycle.attended_contact_ids[0]]
        no_show_contact = contacts_by_id[lifecycle.no_show_contact_ids[0]]
        thread_id = self._thread_id(f"{event_id}:attendance")
        invite_subject = f"RSVP requested for {scenario.event.title} with {account.name}"
        event_provenance = LabelProvenance(
            object_type=ProvenanceObjectType.EVENT,
            object_id=scenario.event.id,
            explanation=(
                "Attendance is grounded in the invite, RSVP, reminder, and post-event "
                "follow-up trail."
            ),
        )

        return [
            EmailRecord(
                email_id=self._email_id(
                    event_id,
                    CommunicationCategory.SCHEDULING_ONLY,
                    100,
                ),
                thread_id=thread_id,
                message_index_in_thread=0,
                timestamp=lifecycle.rsvp_requested_at,
                sender_employee_id=organizer.id,
                sender_contact_id=None,
                to=[participant.contact_id for participant in lifecycle.participants],
                cc=[],
                bcc=[],
                subject=invite_subject,
                body=(
                    f"Please RSVP for {scenario.event.title} with {account.name}. "
                    "Reply with accept, tentative, or decline so we can lock the attendee list."
                ),
                attachments=["calendar.ics"],
                account_id=account.id,
                opportunity_id=None,
                event_id=scenario.event.id,
                ticket_id=None,
                primary_category=CommunicationCategory.SCHEDULING_ONLY,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.SCHEDULING_ONLY,
                    is_relevant=False,
                    provenance=None,
                ),
                provenance=None,
                source_system="email",
            ),
            EmailRecord(
                email_id=self._email_id(
                    event_id,
                    CommunicationCategory.SCHEDULING_ONLY,
                    101,
                ),
                thread_id=thread_id,
                message_index_in_thread=1,
                timestamp=lifecycle.reminder_sent_at,
                sender_employee_id=organizer.id,
                sender_contact_id=None,
                to=[participant.contact_id for participant in lifecycle.participants],
                cc=[],
                bcc=[],
                subject=f"Re: {invite_subject}",
                body=(
                    f"Reminder: please confirm whether you can make {scenario.event.title}. "
                    "We are finalizing the attendee roll call today."
                ),
                attachments=[],
                account_id=account.id,
                opportunity_id=None,
                event_id=scenario.event.id,
                ticket_id=None,
                primary_category=CommunicationCategory.SCHEDULING_ONLY,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.SCHEDULING_ONLY,
                    is_relevant=False,
                    provenance=None,
                ),
                provenance=None,
                source_system="email",
            ),
            EmailRecord(
                email_id=self._email_id(
                    event_id,
                    CommunicationCategory.EVENT_ATTENDANCE,
                    102,
                ),
                thread_id=thread_id,
                message_index_in_thread=2,
                timestamp=lifecycle.post_event_follow_up_at,
                sender_employee_id=organizer.id,
                sender_contact_id=None,
                to=[attended_contact.id],
                cc=[scenario.account_owner.id],
                bcc=[],
                subject=f"Thanks for joining {scenario.event.title}",
                body=(
                    f"Thanks for joining {scenario.event.title}. "
                    f"We saw {attended_contact.first_name} "
                    "join live, and I will send the follow up notes next."
                ),
                attachments=["meeting-notes.docx"],
                account_id=account.id,
                opportunity_id=None,
                event_id=scenario.event.id,
                ticket_id=None,
                primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                    is_relevant=True,
                    provenance=event_provenance,
                ),
                provenance=event_provenance,
                source_system="email",
            ),
            EmailRecord(
                email_id=self._email_id(
                    event_id,
                    CommunicationCategory.EVENT_ATTENDANCE,
                    103,
                ),
                thread_id=thread_id,
                message_index_in_thread=3,
                timestamp=lifecycle.post_event_follow_up_at + timedelta(minutes=30),
                sender_employee_id=organizer.id,
                sender_contact_id=None,
                to=[no_show_contact.id],
                cc=[scenario.account_owner.id],
                bcc=[],
                subject=f"Sorry we missed you at {scenario.event.title}",
                body=(
                    f"We missed {no_show_contact.first_name} at {scenario.event.title}. "
                    "I am sending the recap and would still like a quick follow up on whether "
                    "you want the recording."
                ),
                attachments=["meeting-recap.pdf"],
                account_id=account.id,
                opportunity_id=None,
                event_id=scenario.event.id,
                ticket_id=None,
                primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                    is_relevant=True,
                    provenance=event_provenance,
                ),
                provenance=event_provenance,
                source_system="email",
            ),
        ]

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
            thread_id=thread_id,
            message_index_in_thread=message_index_in_thread,
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

    def _render_noise_from_event(
        self,
        *,
        event_id: str,
        family: NoiseFamily,
        message_index_in_thread: int,
    ) -> EmailRecord:
        scenario = EventScenarioResolver(self.enterprise).for_event(event_id)
        event = scenario.event
        account = scenario.account
        recipients = list(scenario.attendee_contacts)
        organizer = scenario.organizer
        content = self._noise_content(
            family=family,
            event=event,
            account=account,
            organizer=organizer,
        )
        thread_id = self._thread_id(f"{event.id}:{family.value}:{message_index_in_thread}")

        return EmailRecord(
            email_id=self._email_id(
                event.id,
                content["category"],
                message_index_in_thread,
            ),
            thread_id=thread_id,
            message_index_in_thread=0,
            timestamp=(
                event.starts_at
                - timedelta(days=4)
                + timedelta(minutes=29 * message_index_in_thread)
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
            opportunity_id=None,
            event_id=event.id,
            ticket_id=None,
            primary_category=content["category"],
            is_relevant=False,
            relevance_reason=build_relevance_reason(
                primary_category=content["category"],
                is_relevant=False,
                provenance=None,
            ),
            provenance=None,
            source_system="email",
        )

    def _render_hard_negative_from_event(
        self,
        *,
        event_id: str,
        thread_id: str,
        family: HardNegativeFamily,
        message_index_in_thread: int,
        email_namespace_index: int,
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
            family=family,
        )

        return EmailRecord(
            email_id=self._email_id(
                event.id,
                content["category"],
                email_namespace_index,
            ),
            thread_id=thread_id,
            message_index_in_thread=message_index_in_thread,
            timestamp=(
                event.starts_at
                - timedelta(hours=12)
                + timedelta(minutes=20 * email_namespace_index)
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

    def _build_thread_messages(self, *, event_id: str) -> list[EmailRecord]:
        scenario = EventScenarioResolver(self.enterprise).for_event(event_id)
        event = scenario.event
        account = scenario.account
        organizer = scenario.organizer
        account_owner = scenario.account_owner
        support_owner = scenario.support_owner
        primary_contact = scenario.primary_contact
        opportunity = scenario.opportunity
        ticket = scenario.ticket
        thread_id = self._thread_id(event.id)
        subject = f"Follow up from {event.title} with {account.name}"
        root = self.render_from_event(
            event_id=event.id,
            primary_category=CommunicationCategory.FOLLOW_UP,
            message_index_in_thread=0,
        )
        if opportunity is None:
            raise ValueError("email thread rendering requires an account opportunity")

        support_status = EmailRecord(
            email_id=self._email_id(event.id, CommunicationCategory.STATUS_UPDATES, 1),
            thread_id=thread_id,
            message_index_in_thread=1,
            timestamp=root.timestamp + timedelta(hours=2),
            sender_employee_id=support_owner.id,
            sender_contact_id=None,
            to=[organizer.id],
            cc=[account_owner.id],
            bcc=[],
            subject=f"FW: Export validation for {event.title} / {account.name}",
            body=(
                f"Forwarding the internal validation chain for {account.name}. "
                "The export validation is still in progress and no customer reply is needed yet."
                f"\n\nRegards,\n{support_owner.first_name} {support_owner.last_name}"
                f"\n{support_owner.title}\nDisclaimer: internal prep only."
            ),
            attachments=["validation-checklist.xlsx"],
            account_id=account.id,
            opportunity_id=opportunity.id,
            event_id=event.id,
            ticket_id=ticket.id if ticket is not None else None,
            primary_category=CommunicationCategory.STATUS_UPDATES,
            is_relevant=False,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.STATUS_UPDATES,
                is_relevant=False,
                provenance=None,
            ),
            provenance=None,
            source_system="email",
        )
        partial_reply = EmailRecord(
            email_id=self._email_id(event.id, CommunicationCategory.LOW_SIGNAL_CHECKIN, 2),
            thread_id=thread_id,
            message_index_in_thread=2,
            timestamp=support_status.timestamp + timedelta(hours=3),
            sender_employee_id=None,
            sender_contact_id=primary_contact.id,
            to=[organizer.id],
            cc=[],
            bcc=[],
            subject=f"Re: {subject}",
            body="Friday works. Owners pending.",
            attachments=[],
            account_id=account.id,
            opportunity_id=opportunity.id,
            event_id=event.id,
            ticket_id=ticket.id if ticket is not None else None,
            primary_category=CommunicationCategory.LOW_SIGNAL_CHECKIN,
            is_relevant=False,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.LOW_SIGNAL_CHECKIN,
                is_relevant=False,
                provenance=None,
            ),
            provenance=None,
            source_system="email",
        )
        repeated_follow_up = EmailRecord(
            email_id=self._email_id(event.id, CommunicationCategory.FOLLOW_UP, 3),
            thread_id=thread_id,
            message_index_in_thread=3,
            timestamp=partial_reply.timestamp + timedelta(days=2),
            sender_employee_id=organizer.id,
            sender_contact_id=None,
            to=[contact.id for contact in scenario.attendee_contacts],
            cc=[account_owner.id, support_owner.id],
            bcc=[],
            subject=f"Re: Updated owners for {event.title} / {account.name}",
            body=(
                f"Checking back on owners for {event.title} with {account.name}. "
                "The current plan is unchanged, and we still need one confirmed next step."
            ),
            attachments=["action-items.txt"],
            account_id=account.id,
            opportunity_id=opportunity.id,
            event_id=event.id,
            ticket_id=ticket.id if ticket is not None else None,
            primary_category=CommunicationCategory.FOLLOW_UP,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.FOLLOW_UP,
                is_relevant=True,
                provenance=self._event_provenance(account_id=account.id, event_id=event.id),
            ),
            provenance=self._event_provenance(account_id=account.id, event_id=event.id),
            source_system="email",
        )
        dropped_thread_nudge = EmailRecord(
            email_id=self._email_id(event.id, CommunicationCategory.STATUS_UPDATES, 4),
            thread_id=thread_id,
            message_index_in_thread=4,
            timestamp=repeated_follow_up.timestamp + timedelta(hours=4),
            sender_employee_id=account_owner.id,
            sender_contact_id=None,
            to=[organizer.id],
            cc=[support_owner.id],
            bcc=[],
            subject=f"Re: {subject}",
            body=(
                "Keeping this thread warm for Friday planning. No new commitment yet, just a "
                "quick nudge on the same owners."
            ),
            attachments=[],
            account_id=account.id,
            opportunity_id=opportunity.id,
            event_id=event.id,
            ticket_id=ticket.id if ticket is not None else None,
            primary_category=CommunicationCategory.STATUS_UPDATES,
            is_relevant=False,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.STATUS_UPDATES,
                is_relevant=False,
                provenance=None,
            ),
            provenance=None,
            source_system="email",
        )
        resolution_ack = EmailRecord(
            email_id=self._email_id(event.id, CommunicationCategory.FOLLOW_UP, 5),
            thread_id=thread_id,
            message_index_in_thread=5,
            timestamp=dropped_thread_nudge.timestamp + timedelta(days=2),
            sender_employee_id=None,
            sender_contact_id=primary_contact.id,
            to=[organizer.id],
            cc=[account_owner.id],
            bcc=[],
            subject=f"Re: Final timing confirmed for {account.name}",
            body=(
                f"Thanks. Friday is fine now for {account.name}, and we can use the same plan "
                "you outlined in the follow up."
            ),
            attachments=[],
            account_id=account.id,
            opportunity_id=opportunity.id,
            event_id=event.id,
            ticket_id=ticket.id if ticket is not None else None,
            primary_category=CommunicationCategory.FOLLOW_UP,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.FOLLOW_UP,
                is_relevant=True,
                provenance=self._event_provenance(account_id=account.id, event_id=event.id),
            ),
            provenance=self._event_provenance(account_id=account.id, event_id=event.id),
            source_system="email",
        )
        return [
            root,
            support_status,
            partial_reply,
            repeated_follow_up,
            dropped_thread_nudge,
            resolution_ack,
        ]

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
        thread_id: str,
        message_index_in_thread: int,
    ) -> RenderedEmailContent:
        profile = self.context.language_profile
        product_name = self.enterprise.products[0].name
        product_reference = profile.product_reference(product_name)
        style_engine = EmailStyleEngine(context=self.context)
        sender_name = f"{organizer.first_name} {organizer.last_name}"
        seniority = (
            EmailSeniority.EXECUTIVE
            if "chief" in organizer.title.lower() or "vp" in organizer.title.lower()
            else EmailSeniority.IC
        )

        if category == CommunicationCategory.SCHEDULING_ONLY:
            styled = style_engine.render(
                EmailStyleRequest(
                    thread_id=thread_id,
                    message_index_in_thread=message_index_in_thread,
                    base_subject=f"Time update for {event.title} [{profile.abbreviation}]",
                    sender_name=sender_name,
                    sender_role=organizer.title,
                    recipient_name=recipients[0].first_name,
                    subject_mode=EmailSubjectMode.NEW,
                    audience=EmailAudience.EXTERNAL_CUSTOMER,
                    seniority=seniority,
                    detail_level=EmailDetailLevel.DETAILED,
                    include_signature=False,
                    context_lines=(
                        f"{account.name} is now set for {event.starts_at:%A %H:%M %Z}.",
                        f"Please keep the {profile.meeting_language} on the calendar.",
                    ),
                    action_ask="Please confirm the calendar still works.",
                    disclaimer_line="This note is only about calendar logistics.",
                )
            )
            return {
                "subject": styled.subject,
                "body": styled.body,
                "attachments": ["calendar.ics"],
                "cc": list(styled.cc),
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
            issue_text = f"The open issue on {ticket.summary} still needs closure."

        styled = style_engine.render(
            EmailStyleRequest(
                thread_id=thread_id,
                message_index_in_thread=message_index_in_thread,
                base_subject=f"Follow up from {event.title} with {account.name}",
                sender_name=sender_name,
                sender_role=organizer.title,
                recipient_name=recipients[0].first_name,
                subject_mode=(
                    EmailSubjectMode.REPLY
                    if message_index_in_thread > 0
                    else EmailSubjectMode.NEW
                ),
                audience=EmailAudience.EXTERNAL_CUSTOMER,
                seniority=seniority,
                detail_level=EmailDetailLevel.DETAILED,
                include_signature=False,
                context_lines=(
                    (
                        f"Our {profile.email_marker} for {event.title} with {account.name} "
                        f"is tracking {product_reference}."
                    ),
                    (
                        (
                            f"We should keep the {opportunity.stage} motion moving "
                            "with the current plan."
                        )
                        if opportunity is not None
                        else f"We should keep the timing aligned for {account.name}."
                    ),
                ),
                prior_thread_summary=(
                    f"We already reviewed {event.title} with {account.name}."
                ),
                objection_line=issue_text or None,
                action_ask=next_step_text,
            )
        )

        return {
            "subject": styled.subject,
            "body": styled.body,
            "attachments": ["action-items.txt"],
            "cc": [account.owner_employee_id, *list(styled.cc)],
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
        family: HardNegativeFamily,
    ) -> HardNegativeEmailContent:
        if family == HardNegativeFamily.EVENT_NO_ATTENDANCE:
            return {
                "category": CommunicationCategory.SCHEDULING_ONLY,
                "subject": f"Follow up on {event.title} attendance for {account.name}",
                "body": (
                    f"We can keep the {event.title.lower()} Friday plan in place for "
                    f"{account.name}, but no RSVP, attendee status, owner, or next step "
                    "changed in this thread and no reply is required today."
                ),
                "attachments": ["travel-note.txt"],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": self._weak_provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    explanation=hard_negative_explanation(family),
                ),
            }
        if family == HardNegativeFamily.BUDGET_NON_BUYING:
            object_type = (
                ProvenanceObjectType.OPPORTUNITY
                if opportunity is not None
                else ProvenanceObjectType.EVENT
            )
            object_id = opportunity.id if opportunity is not None else event.id
            return {
                "category": CommunicationCategory.FYI_FORWARD,
                "subject": f"Fwd: budget follow up for {event.title} with {account.name}",
                "body": (
                    f"The budget line for {account.name} covers the {event.title.lower()} room "
                    "block, travel plan, and catering only. It is not evidence of a buying "
                    "signal, owner change, or new next step, so no reply is needed."
                ),
                "attachments": ["review-budget.xlsx"],
                "cc": [organizer.id],
                "opportunity_id": opportunity.id if opportunity is not None else None,
                "ticket_id": None,
                "provenance": self._weak_provenance(
                    object_type=object_type,
                    object_id=object_id,
                    explanation=hard_negative_explanation(family),
                ),
            }
        if family == HardNegativeFamily.COMPLAINT_NON_PRODUCT:
            return {
                "category": CommunicationCategory.STATUS_UPDATES,
                "subject": f"Complaint note for {event.title} travel timing",
                "body": (
                    f"The complaint here is about weather delays and hotel timing for "
                    f"{event.title} with {account.name}, not a product pain point or next "
                    "step blocker, even though the current plan and Friday timing may shift."
                ),
                "attachments": [],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": self._weak_provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    explanation=hard_negative_explanation(family),
                ),
            }
        if family == HardNegativeFamily.ESCALATION_NO_BLOCKER:
            object_type = (
                ProvenanceObjectType.TICKET
                if ticket is not None
                else ProvenanceObjectType.EVENT
            )
            object_id = ticket.id if ticket is not None else event.id
            return {
                "category": CommunicationCategory.STATUS_UPDATES,
                "subject": f"Escalation wording on {event.title} plan",
                "body": (
                    f"Marking the {event.title.lower()} plan as urgent for calendar cleanup "
                    "only. The thread sounds escalated and still mentions the export issue, "
                    "but there is not a live blocker, owner change, or required reply to "
                    "resolve."
                ),
                "attachments": [],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": ticket.id if ticket is not None else None,
                "provenance": self._weak_provenance(
                    object_type=object_type,
                    object_id=object_id,
                    explanation=hard_negative_explanation(family),
                ),
            }
        if family == HardNegativeFamily.EXECUTIVE_VISIBILITY_ONLY:
            return {
                "category": CommunicationCategory.FYI_FORWARD,
                "subject": f"VP visibility on {event.title} follow up",
                "body": (
                    f"Adding the VP to the {event.title.lower()} follow up for {account.name} "
                    "for visibility only. No executive sponsor engagement, owner approval, or "
                    "decision-maker action is requested, and the current plan stays the same."
                ),
                "attachments": [],
                "cc": [organizer.id],
                "opportunity_id": opportunity.id if opportunity is not None else None,
                "ticket_id": None,
                "provenance": self._weak_provenance(
                    object_type=ProvenanceObjectType.ACCOUNT,
                    object_id=account.id,
                    explanation=hard_negative_explanation(family),
                ),
            }
        return {
            "category": CommunicationCategory.LOW_SIGNAL_CHECKIN,
            "subject": f"Follow up note on {event.title} next step for {account.name}",
                "body": (
                    f"This follow up repeats the {event.title.lower()} plan, owners, and next "
                    f"step for {account.name}, but there is no required action, owner, due date, "
                    "or committed Friday reply in this note even though the current plan is "
                    "restated."
                ),
            "attachments": [],
            "cc": [],
            "opportunity_id": None,
            "ticket_id": None,
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

    def _event_provenance(self, *, account_id: str, event_id: str) -> LabelProvenance:
        return LabelProvenance(
            object_type=ProvenanceObjectType.EVENT,
            object_id=event_id,
            explanation=(
                f"Customer review for account {account_id} still has an active follow-up "
                "thread with outstanding owners."
            ),
        )

    def _noise_content(
        self,
        *,
        family: NoiseFamily,
        event: Event,
        account: CustomerAccount,
        organizer: Employee,
    ) -> HardNegativeEmailContent:
        recipient_hint = organizer.first_name

        if family == NoiseFamily.VAGUE_FOLLOW_UP:
            return {
                "category": noise_category_for_family(family),
                "subject": f"Quick follow-up on {event.title}",
                "body": (
                    f"Hi {recipient_hint}, circling back on the thread for {account.name}. "
                    "Nothing new to close today, but keep next week in mind."
                ),
                "attachments": [],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": None,
            }
        if family == NoiseFamily.DOCUMENT_REVIEW_PING:
            return {
                "category": noise_category_for_family(family),
                "subject": f"Please skim the notes for {account.name}",
                "body": (
                    "Sharing the draft notes deck for a quick proofread before archive. "
                    "No customer-facing update is attached to this note."
                ),
                "attachments": ["notes-draft.docx"],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": None,
            }
        if family == NoiseFamily.AGENDA_COORDINATION:
            return {
                "category": noise_category_for_family(family),
                "subject": f"Agenda hold for {event.title}",
                "body": (
                    "Keeping time open for agenda setup and room coordination only. "
                    "Please leave the calendar placeholder in place."
                ),
                "attachments": ["agenda-draft.txt"],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": None,
            }
        if family == NoiseFamily.INTERNAL_FYI_SUMMARY:
            return {
                "category": noise_category_for_family(family),
                "subject": f"FYI summary for {account.name}",
                "body": (
                    "Forwarding the internal summary from ops. Folder names are updated and "
                    "the archive pass is complete. No reply needed."
                ),
                "attachments": ["ops-summary.msg"],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": None,
            }
        if family == NoiseFamily.DUPLICATE_REMINDER:
            return {
                "category": noise_category_for_family(family),
                "subject": f"Reminder: same note for {account.name}",
                "body": (
                    "Repeating the earlier reminder to rename the shared folder before the "
                    "weekly archive job runs."
                ),
                "attachments": [],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": None,
            }
        if family == NoiseFamily.OWNERSHIP_AMBIGUITY:
            return {
                "category": noise_category_for_family(family),
                "subject": f"Owner still unclear for {account.name} admin task",
                "body": (
                    "I am not sure who owns the cleanup pass on the shared notes. "
                    "Parking this here until the right owner picks it up."
                ),
                "attachments": [],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": None,
            }
        if family == NoiseFamily.COMPLIANCE_ADMIN_REMINDER:
            return {
                "category": noise_category_for_family(family),
                "subject": "Annual compliance reminder",
                "body": (
                    "Please complete the annual acknowledgment form and upload the PDF by "
                    "Friday. This is an internal admin reminder only."
                ),
                "attachments": ["policy-ack.pdf"],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": None,
            }
        if family == NoiseFamily.TRAVEL_LOGISTICS:
            return {
                "category": noise_category_for_family(family),
                "subject": f"Travel timing for {event.title}",
                "body": (
                    "Hotel and shuttle timing changed for the onsite portion of the visit. "
                    "This note is only for travel coordination."
                ),
                "attachments": ["travel-logistics.txt"],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": None,
            }
        if family == NoiseFamily.EMPTY_CHECK_IN:
            return {
                "category": noise_category_for_family(family),
                "subject": f"Checking in on {account.name}",
                "body": "Just checking in. No immediate action needed from this note.",
                "attachments": [],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": None,
            }
        if family == NoiseFamily.IRRELEVANT_FORWARDED_CHAIN:
            return {
                "category": noise_category_for_family(family),
                "subject": f"Fwd: old thread for {event.title}",
                "body": (
                    "Forwarding an older internal chain for reference only. There is no "
                    "new ask in this message."
                ),
                "attachments": ["old-thread.msg"],
                "cc": [],
                "opportunity_id": None,
                "ticket_id": None,
                "provenance": None,
            }
        return {
            "category": noise_category_for_family(family),
            "subject": f"Procurement routing note for {account.name}",
            "body": (
                "The internal procurement form is missing a routing code and tax field. "
                "This does not change customer scope or timing."
            ),
            "attachments": ["routing-form.pdf"],
            "cc": [],
            "opportunity_id": None,
            "ticket_id": None,
            "provenance": None,
        }
