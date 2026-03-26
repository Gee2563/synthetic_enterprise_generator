from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import TypeAlias

from synthetic_enterprise.contracts.records.email import EmailRecord
from synthetic_enterprise.contracts.records.salesforce import (
    SalesforceObjectType,
    SalesforceRecord,
)
from synthetic_enterprise.contracts.records.slack import SlackRecord
from synthetic_enterprise.contracts.records.teams import TeamsRecord
from synthetic_enterprise.domain import (
    Contact,
    CustomerAccount,
    Employee,
    EnterpriseGraph,
    Event,
    Opportunity,
    TicketIssue,
)
from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.labeling.grounding import (
    LabelProvenance,
    ProvenanceObjectType,
    build_relevance_reason,
)
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory

CrossSystemRecord: TypeAlias = EmailRecord | SlackRecord | TeamsRecord | SalesforceRecord


class CrossSystemEventBundle(EnterpriseModel):
    """Coordinated multi-system rendering for a single business event."""

    event_id: str
    account_id: str
    email_records: list[EmailRecord]
    slack_records: list[SlackRecord]
    teams_records: list[TeamsRecord]
    salesforce_records: list[SalesforceRecord]

    @property
    def all_records(self) -> list[CrossSystemRecord]:
        return [
            *self.email_records,
            *self.slack_records,
            *self.teams_records,
            *self.salesforce_records,
        ]

    @property
    def rendered_system_count(self) -> int:
        return sum(
            bool(records)
            for records in (
                self.email_records,
                self.slack_records,
                self.teams_records,
                self.salesforce_records,
            )
        )


@dataclass(slots=True)
class CrossSystemRenderer:
    """Render one event consistently across email, Slack, Teams, and Salesforce."""

    context: GeneratorContext
    enterprise: EnterpriseGraph

    def render_all_events(self) -> list[CrossSystemEventBundle]:
        events = sorted(
            (
                event
                for event in self.enterprise.events
                if event.account_id is not None
            ),
            key=lambda event: (event.starts_at, event.id),
        )
        selected_count = _selected_event_count(
            total_events=len(events),
            ratio=self.context.cross_system_ratio,
        )

        return [
            self.render_event_bundle(event.id)
            for event in events[:selected_count]
        ]

    def render_event_bundle(self, event_id: str) -> CrossSystemEventBundle:
        event = self._event_by_id[event_id]
        if event.account_id is None:
            raise ValueError("cross-system rendering requires an account-linked event")

        account = self._account_by_id[event.account_id]
        contacts = self._contacts_for_account(account.id)
        organizer = self._employee_by_id[event.organizer_employee_id]
        account_owner = self._employee_by_id[account.owner_employee_id]
        opportunity = self._first_opportunity_for_account(account.id)
        ticket = self._first_ticket_for_account(account.id)
        support_owner = organizer
        if ticket is not None and ticket.owner_employee_id is not None:
            support_owner = self._employee_by_id[ticket.owner_employee_id]

        return CrossSystemEventBundle(
            event_id=event.id,
            account_id=account.id,
            email_records=self._email_records(
                event=event,
                account=account,
                contacts=contacts,
                organizer=organizer,
                opportunity=opportunity,
                ticket=ticket,
            ),
            slack_records=self._slack_records(
                event=event,
                account=account,
                organizer=organizer,
                opportunity=opportunity,
                ticket=ticket,
            ),
            teams_records=self._teams_records(
                event=event,
                account=account,
                organizer=organizer,
                account_owner=account_owner,
                support_owner=support_owner,
                opportunity=opportunity,
                ticket=ticket,
            ),
            salesforce_records=self._salesforce_records(
                event=event,
                account=account,
                contacts=contacts,
                owner=account_owner,
                opportunity=opportunity,
                ticket=ticket,
            ),
        )

    @property
    def _event_by_id(self) -> dict[str, Event]:
        return {event.id: event for event in self.enterprise.events}

    @property
    def _account_by_id(self) -> dict[str, CustomerAccount]:
        return {account.id: account for account in self.enterprise.customer_accounts}

    @property
    def _employee_by_id(self) -> dict[str, Employee]:
        return {employee.id: employee for employee in self.enterprise.employees}

    def _contacts_for_account(self, account_id: str) -> list[Contact]:
        return [
            contact
            for contact in self.enterprise.contacts
            if contact.account_id == account_id
        ]

    def _first_opportunity_for_account(self, account_id: str) -> Opportunity | None:
        return next(
            (
                opportunity
                for opportunity in self.enterprise.opportunities
                if opportunity.account_id == account_id
            ),
            None,
        )

    def _first_ticket_for_account(self, account_id: str) -> TicketIssue | None:
        return next(
            (
                ticket
                for ticket in self.enterprise.ticket_issues
                if ticket.account_id == account_id
            ),
            None,
        )

    def _email_records(
        self,
        *,
        event: Event,
        account: CustomerAccount,
        contacts: list[Contact],
        organizer: Employee,
        opportunity: Opportunity | None,
        ticket: TicketIssue | None,
    ) -> list[EmailRecord]:
        invite_provenance = LabelProvenance(
            object_type=ProvenanceObjectType.EVENT,
            object_id=event.id,
            explanation="Confirmed attendee list is attached to the customer review.",
        )
        follow_up_object_type = ProvenanceObjectType.EVENT
        follow_up_object_id = event.id
        follow_up_explanation = "Post-event owners still need confirmation."
        follow_up_category = CommunicationCategory.FOLLOW_UP

        if ticket is not None:
            follow_up_object_type = ProvenanceObjectType.TICKET
            follow_up_object_id = ticket.id
            follow_up_explanation = (
                f"Open issue '{ticket.summary}' is constraining the next step."
            )
            follow_up_category = CommunicationCategory.BLOCKER

        attendee_names = ", ".join(
            f"{contact.first_name} {contact.last_name}"
            for contact in contacts[:2]
        )

        return [
            EmailRecord(
                email_id=self._id("email", f"{event.id}:invite"),
                thread_id=self._id("thread", f"{event.id}:email-thread"),
                message_index_in_thread=0,
                timestamp=event.starts_at - timedelta(days=2),
                sender_employee_id=organizer.id,
                sender_contact_id=None,
                to=[contact.id for contact in contacts[:2]],
                cc=[],
                bcc=[],
                subject=f"Confirmed attendees for {event.title}",
                body=(
                    f"Confirmed attendees for {event.title} at {account.name}: "
                    f"{attendee_names}."
                ),
                attachments=["invite.ics"],
                account_id=account.id,
                opportunity_id=opportunity.id if opportunity is not None else None,
                event_id=event.id,
                ticket_id=None,
                primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                    is_relevant=True,
                    provenance=invite_provenance,
                ),
                provenance=invite_provenance,
                source_system="email",
            ),
            EmailRecord(
                email_id=self._id("email", f"{event.id}:follow-up"),
                thread_id=self._id("thread", f"{event.id}:email-thread"),
                message_index_in_thread=1,
                timestamp=event.ends_at + timedelta(hours=2),
                sender_employee_id=organizer.id,
                sender_contact_id=None,
                to=[contact.id for contact in contacts[:2]],
                cc=[account.owner_employee_id],
                bcc=[],
                subject=f"Next steps after {event.title}",
                body=(
                    f"Following {event.title} with {account.name}, we still need "
                    "owners and dates recorded."
                ),
                attachments=["action-items.txt"],
                account_id=account.id,
                opportunity_id=opportunity.id if opportunity is not None else None,
                event_id=event.id,
                ticket_id=ticket.id if ticket is not None else None,
                primary_category=follow_up_category,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=follow_up_category,
                    is_relevant=True,
                    provenance=LabelProvenance(
                        object_type=follow_up_object_type,
                        object_id=follow_up_object_id,
                        explanation=follow_up_explanation,
                    ),
                ),
                provenance=LabelProvenance(
                    object_type=follow_up_object_type,
                    object_id=follow_up_object_id,
                    explanation=follow_up_explanation,
                ),
                source_system="email",
            ),
        ]

    def _slack_records(
        self,
        *,
        event: Event,
        account: CustomerAccount,
        organizer: Employee,
        opportunity: Opportunity | None,
        ticket: TicketIssue | None,
    ) -> list[SlackRecord]:
        channel_name = f"#acct-{_slugify(account.name)}"
        thread_id = self._id("slack_thread", f"{event.id}:slack-thread")
        root_category = CommunicationCategory.FOLLOW_UP
        root_opportunity_id = None
        root_provenance = LabelProvenance(
            object_type=ProvenanceObjectType.EVENT,
            object_id=event.id,
            explanation="Owners still need to be named after the customer review.",
        )
        root_body = "Need owners and dates locked before the review recap goes out."

        if opportunity is not None:
            root_category = CommunicationCategory.BUYING_SIGNAL
            root_opportunity_id = opportunity.id
            root_provenance = LabelProvenance(
                object_type=ProvenanceObjectType.OPPORTUNITY,
                object_id=opportunity.id,
                explanation=(
                    f"Opportunity is still active in stage {opportunity.stage} "
                    "after the review."
                ),
            )
            root_body = (
                f"{account.name} sounds open to keeping the {opportunity.stage} motion "
                "moving after the review."
            )

        records = [
            SlackRecord(
                slack_message_id=self._id("slack_message", f"{event.id}:slack-root"),
                channel_id=self._id("channel", f"{event.id}:slack-channel"),
                channel_name=channel_name,
                thread_id=thread_id,
                parent_message_id=None,
                timestamp=event.starts_at - timedelta(hours=5),
                sender_employee_id=organizer.id,
                body=root_body,
                mentions=[],
                reactions=[":eyes:"],
                attachments=[],
                linked_account_id=account.id,
                linked_opportunity_id=root_opportunity_id,
                linked_event_id=event.id,
                linked_ticket_id=None,
                primary_category=root_category,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=root_category,
                    is_relevant=True,
                    provenance=root_provenance,
                ),
                provenance=root_provenance,
                source_system="slack",
            )
        ]

        if ticket is not None:
            blocker_provenance = LabelProvenance(
                object_type=ProvenanceObjectType.TICKET,
                object_id=ticket.id,
                explanation=(
                    f"Open issue '{ticket.summary}' is constraining the customer "
                    "follow-up."
                ),
            )
            records.append(
                SlackRecord(
                    slack_message_id=self._id(
                        "slack_message",
                        f"{event.id}:slack-blocker",
                    ),
                    channel_id=self._id("channel", f"{event.id}:slack-channel"),
                    channel_name=channel_name,
                    thread_id=thread_id,
                    parent_message_id=records[0].slack_message_id,
                    timestamp=event.starts_at - timedelta(hours=4, minutes=20),
                    sender_employee_id=organizer.id,
                    body="Export fix is still open, so I would not promise the date yet.",
                    mentions=[],
                    reactions=[],
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
                        provenance=blocker_provenance,
                    ),
                    provenance=blocker_provenance,
                    source_system="slack",
                )
            )

        return records

    def _teams_records(
        self,
        *,
        event: Event,
        account: CustomerAccount,
        organizer: Employee,
        account_owner: Employee,
        support_owner: Employee,
        opportunity: Opportunity | None,
        ticket: TicketIssue | None,
    ) -> list[TeamsRecord]:
        team_id = self._id("team", f"{event.id}:teams-team")
        channel_id = self._id("channel", f"{event.id}:teams-channel")
        thread_id = self._id("teams_thread", f"{event.id}:teams-thread")
        meeting_id = self._id("meeting", f"{event.id}:teams-meeting")
        coordination_provenance = LabelProvenance(
            object_type=ProvenanceObjectType.EVENT,
            object_id=event.id,
            explanation="Meeting coordination reflects confirmed participation for the review.",
        )
        linked_opportunity_id = opportunity.id if opportunity is not None else None

        records = [
            TeamsRecord(
                teams_message_id=self._id("teams_message", f"{event.id}:teams-root"),
                team_id=team_id,
                channel_id=channel_id,
                chat_or_channel="channel",
                thread_id=thread_id,
                timestamp=event.starts_at - timedelta(hours=3),
                sender_employee_id=organizer.id,
                body=(
                    "Agenda:\n"
                    f"- confirm attendee roll call for {account.name}\n"
                    "- finalize meeting notes owner\n"
                    "Document: attendee-plan.docx"
                ),
                mentions=[account_owner.id, support_owner.id],
                meeting_id=meeting_id,
                file_refs=["attendee-plan.docx", "meeting-notes.docx"],
                linked_account_id=account.id,
                linked_opportunity_id=linked_opportunity_id,
                linked_event_id=event.id,
                linked_ticket_id=None,
                primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                    is_relevant=True,
                    provenance=coordination_provenance,
                ),
                provenance=coordination_provenance,
                source_system="teams",
            )
        ]

        if ticket is not None:
            blocker_provenance = LabelProvenance(
                object_type=ProvenanceObjectType.TICKET,
                object_id=ticket.id,
                explanation="Support validation work is still gating the promised date.",
            )
            records.append(
                TeamsRecord(
                    teams_message_id=self._id(
                        "teams_message",
                        f"{event.id}:teams-blocker",
                    ),
                    team_id=team_id,
                    channel_id=channel_id,
                    chat_or_channel="channel",
                    thread_id=thread_id,
                    timestamp=event.starts_at - timedelta(hours=2, minutes=30),
                    sender_employee_id=support_owner.id,
                    body=(
                        "Status update:\n"
                        "- export validation is still open\n"
                        "- avoid committing to the rollout date in the meeting"
                    ),
                    mentions=[organizer.id],
                    meeting_id=meeting_id,
                    file_refs=["validation-checklist.xlsx"],
                    linked_account_id=account.id,
                    linked_opportunity_id=None,
                    linked_event_id=event.id,
                    linked_ticket_id=ticket.id,
                    primary_category=CommunicationCategory.BLOCKER,
                    is_relevant=True,
                    relevance_reason=build_relevance_reason(
                        primary_category=CommunicationCategory.BLOCKER,
                        is_relevant=True,
                        provenance=blocker_provenance,
                    ),
                    provenance=blocker_provenance,
                    source_system="teams",
                )
            )

        return records

    def _salesforce_records(
        self,
        *,
        event: Event,
        account: CustomerAccount,
        contacts: list[Contact],
        owner: Employee,
        opportunity: Opportunity | None,
        ticket: TicketIssue | None,
    ) -> list[SalesforceRecord]:
        attendee_ids = [contact.id for contact in contacts[:2]]
        campaign_id = self._id("campaign", f"{event.id}:campaign")
        records = [
            SalesforceRecord(
                salesforce_record_id=self._id("sf_record", f"{event.id}:sf-event"),
                object_type=SalesforceObjectType.EVENT,
                record_id=event.id,
                timestamp=event.starts_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                event_id=event.id,
                attendee_contact_ids=attendee_ids,
                subject=event.title,
                text_body="Attendance and participation are captured on the event record.",
                structured_fields={"attendance_count": len(attendee_ids)},
                primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                    is_relevant=True,
                    provenance=LabelProvenance(
                        object_type=ProvenanceObjectType.EVENT,
                        object_id=event.id,
                        explanation="Attendance is recorded directly on the CRM event.",
                    ),
                ),
                provenance=LabelProvenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    explanation="Attendance is recorded directly on the CRM event.",
                ),
                source_system="salesforce",
            ),
            SalesforceRecord(
                salesforce_record_id=self._id("sf_record", f"{event.id}:sf-campaign"),
                object_type=SalesforceObjectType.CAMPAIGN,
                record_id=campaign_id,
                timestamp=event.starts_at - timedelta(days=1),
                owner_employee_id=owner.id,
                campaign_id=campaign_id,
                subject=f"{account.name} review outreach",
                text_body="Campaign groups the outreach and attendance for the review.",
                structured_fields={"status": "In Progress"},
                primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                    is_relevant=True,
                    provenance=LabelProvenance(
                        object_type=ProvenanceObjectType.CAMPAIGN,
                        object_id=campaign_id,
                        explanation="Campaign tracks participation around the customer review.",
                    ),
                ),
                provenance=LabelProvenance(
                    object_type=ProvenanceObjectType.CAMPAIGN,
                    object_id=campaign_id,
                    explanation="Campaign tracks participation around the customer review.",
                ),
                source_system="salesforce",
            ),
        ]

        if attendee_ids:
            records.append(
                SalesforceRecord(
                    salesforce_record_id=self._id(
                        "sf_record",
                        f"{event.id}:sf-campaign-member",
                    ),
                    object_type=SalesforceObjectType.CAMPAIGN_MEMBER,
                    record_id=self._id(
                        "campaign_member",
                        f"{event.id}:campaign-member",
                    ),
                    timestamp=event.starts_at,
                    campaign_id=campaign_id,
                    contact_id=attendee_ids[0],
                    event_id=event.id,
                    structured_fields={"member_status": "Attended"},
                    primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                    is_relevant=True,
                    relevance_reason=build_relevance_reason(
                        primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                        is_relevant=True,
                        provenance=LabelProvenance(
                            object_type=ProvenanceObjectType.EVENT,
                            object_id=event.id,
                            explanation="Contact attendance is linked to the event and campaign.",
                        ),
                    ),
                    provenance=LabelProvenance(
                        object_type=ProvenanceObjectType.EVENT,
                        object_id=event.id,
                        explanation="Contact attendance is linked to the event and campaign.",
                    ),
                    source_system="salesforce",
                )
            )

        task_category = CommunicationCategory.FOLLOW_UP
        task_provenance = LabelProvenance(
            object_type=ProvenanceObjectType.EVENT,
            object_id=event.id,
            explanation="Post-event action items still need ownership.",
        )
        task_case_id = None

        if ticket is not None:
            task_category = CommunicationCategory.BLOCKER
            task_provenance = LabelProvenance(
                object_type=ProvenanceObjectType.TICKET,
                object_id=ticket.id,
                explanation="Open support issue is blocking the promised next step.",
            )
            task_case_id = ticket.id
            records.append(
                SalesforceRecord(
                    salesforce_record_id=self._id("sf_record", f"{event.id}:sf-case"),
                    object_type=SalesforceObjectType.CASE,
                    record_id=ticket.id,
                    timestamp=event.starts_at - timedelta(hours=2),
                    owner_employee_id=owner.id,
                    account_id=account.id,
                    case_id=ticket.id,
                    subject=ticket.summary,
                    text_body="Open case remains active during event follow-up planning.",
                    structured_fields={"status": ticket.status, "severity": ticket.severity},
                    primary_category=CommunicationCategory.BLOCKER,
                    is_relevant=True,
                    relevance_reason=build_relevance_reason(
                        primary_category=CommunicationCategory.BLOCKER,
                        is_relevant=True,
                        provenance=LabelProvenance(
                            object_type=ProvenanceObjectType.TICKET,
                            object_id=ticket.id,
                            explanation="Open support issue is blocking the follow-up plan.",
                        ),
                    ),
                    provenance=LabelProvenance(
                        object_type=ProvenanceObjectType.TICKET,
                        object_id=ticket.id,
                        explanation="Open support issue is blocking the follow-up plan.",
                    ),
                    source_system="salesforce",
                )
            )

        records.append(
            SalesforceRecord(
                salesforce_record_id=self._id("sf_record", f"{event.id}:sf-task"),
                object_type=SalesforceObjectType.TASK,
                record_id=self._id("task", f"{event.id}:task"),
                timestamp=event.ends_at + timedelta(hours=1),
                owner_employee_id=owner.id,
                account_id=account.id,
                opportunity_id=opportunity.id if opportunity is not None else None,
                event_id=event.id,
                case_id=task_case_id,
                subject="Capture event follow-up actions",
                text_body="Document owners and dates from the customer review.",
                structured_fields={"status": "Open"},
                primary_category=task_category,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=task_category,
                    is_relevant=True,
                    provenance=task_provenance,
                ),
                provenance=task_provenance,
                source_system="salesforce",
            )
        )

        if opportunity is not None:
            note_provenance = LabelProvenance(
                object_type=ProvenanceObjectType.OPPORTUNITY,
                object_id=opportunity.id,
                explanation=(
                    f"Opportunity remains active in stage {opportunity.stage} after "
                    "the review."
                ),
            )
            records.append(
                SalesforceRecord(
                    salesforce_record_id=self._id("sf_record", f"{event.id}:sf-note"),
                    object_type=SalesforceObjectType.NOTE,
                    record_id=self._id("note", f"{event.id}:note"),
                    timestamp=event.ends_at + timedelta(minutes=30),
                    owner_employee_id=owner.id,
                    account_id=account.id,
                    opportunity_id=opportunity.id,
                    event_id=event.id,
                    parent_record_id=opportunity.id,
                    subject="Commercial note after review",
                    text_body=(
                        f"The {opportunity.stage} motion is still active following "
                        "the customer review."
                    ),
                    structured_fields={"activity_type": "Call Note"},
                    primary_category=CommunicationCategory.BUYING_SIGNAL,
                    is_relevant=True,
                    relevance_reason=build_relevance_reason(
                        primary_category=CommunicationCategory.BUYING_SIGNAL,
                        is_relevant=True,
                        provenance=note_provenance,
                    ),
                    provenance=note_provenance,
                    source_system="salesforce",
                )
            )

        return records

    def _id(self, prefix: str, namespace: str) -> str:
        derived = self.context.derive_seed(f"cross-system:{namespace}")
        return f"{prefix}_{derived:016x}"


def _selected_event_count(*, total_events: int, ratio: float) -> int:
    if total_events == 0 or ratio <= 0.0:
        return 0

    return min(total_events, max(1, int(round(total_events * ratio))))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "account"
