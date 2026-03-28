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
    EventScenarioContext,
    EventScenarioResolver,
    Opportunity,
    TicketIssue,
)
from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.generation.account_profiles import (
    AccountBehaviorProfile,
    AccountBehaviorProfileType,
    AccountBehaviorResolver,
)
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
        scenario_resolver = EventScenarioResolver(self.enterprise)
        behavior_resolver = AccountBehaviorResolver(
            context=self.context,
            enterprise=self.enterprise,
        )

        bundles: list[CrossSystemEventBundle] = []
        for index, event in enumerate(events[:selected_count]):
            scenario = scenario_resolver.for_event(event.id)
            bundles.append(
                self._apply_account_behavior_profile(
                    bundle=self.render_event_bundle(event.id),
                    scenario=scenario,
                    profile=behavior_resolver.profile_for_account(scenario.account.id),
                    event=event,
                    event_index=index,
                    total_selected=selected_count,
                )
            )
        return bundles

    def render_event_bundle(self, event_id: str) -> CrossSystemEventBundle:
        scenario = EventScenarioResolver(self.enterprise).for_event(event_id)

        return CrossSystemEventBundle(
            event_id=scenario.event.id,
            account_id=scenario.account.id,
            email_records=self._email_records(
                event=scenario.event,
                account=scenario.account,
                contacts=list(scenario.account_contacts),
                organizer=scenario.organizer,
                opportunity=scenario.opportunity,
                ticket=scenario.ticket,
            ),
            slack_records=self._slack_records(
                event=scenario.event,
                account=scenario.account,
                organizer=scenario.organizer,
                opportunity=scenario.opportunity,
                ticket=scenario.ticket,
            ),
            teams_records=self._teams_records(
                event=scenario.event,
                account=scenario.account,
                organizer=scenario.organizer,
                account_owner=scenario.account_owner,
                support_owner=scenario.support_owner,
                opportunity=scenario.opportunity,
                ticket=scenario.ticket,
            ),
            salesforce_records=self._salesforce_records(
                event=scenario.event,
                account=scenario.account,
                contacts=list(scenario.account_contacts),
                owner=scenario.account_owner,
                opportunity=scenario.opportunity,
                ticket=scenario.ticket,
            ),
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
        attendance_timestamp = event.ends_at + timedelta(minutes=20)

        return [
            EmailRecord(
                email_id=self._id("email", f"{event.id}:invite"),
                thread_id=self._id("thread", f"{event.id}:email-thread"),
                message_index_in_thread=0,
                timestamp=attendance_timestamp,
                sender_employee_id=organizer.id,
                sender_contact_id=None,
                to=[contact.id for contact in contacts[:2]],
                cc=[],
                bcc=[],
                subject=f"Attendance recap for {event.title}",
                body=(
                    f"Thanks for joining {event.title}. We saw {attendee_names} "
                    f"joined live for {account.name}."
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

        if len(contacts := self._event_contacts(event=event, account=account)) >= 2:
            attendance_provenance = LabelProvenance(
                object_type=ProvenanceObjectType.EVENT,
                object_id=event.id,
                explanation="Live attendance is being referenced during the customer review.",
            )
            attended_contact = contacts[0]
            no_show_contact = contacts[1]
            records.append(
                SlackRecord(
                    slack_message_id=self._id(
                        "slack_message",
                        f"{event.id}:slack-attendance",
                    ),
                    channel_id=self._id("channel", f"{event.id}:slack-channel"),
                    channel_name=channel_name,
                    thread_id=thread_id,
                    parent_message_id=records[0].slack_message_id,
                    timestamp=event.starts_at + timedelta(minutes=15),
                    sender_employee_id=organizer.id,
                    body=(
                        f"{attended_contact.first_name} is already on the bridge for "
                        f"{event.title}; {no_show_contact.first_name} never made it."
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
                        provenance=attendance_provenance,
                    ),
                    provenance=attendance_provenance,
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

        records.append(
            TeamsRecord(
                teams_message_id=self._id(
                    "teams_message",
                    f"{event.id}:teams-attendance-recap",
                ),
                team_id=team_id,
                channel_id=channel_id,
                chat_or_channel="channel",
                thread_id=thread_id,
                timestamp=event.ends_at + timedelta(minutes=10),
                sender_employee_id=organizer.id,
                body=(
                    "Recap:\n"
                    f"- live attendance confirmed for {account.name}\n"
                    "- one attendee missed the session\n"
                    "- send recording and follow-up notes"
                ),
                mentions=[account_owner.id],
                meeting_id=meeting_id,
                file_refs=["meeting-notes.docx"],
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
        )

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
                    timestamp=event.ends_at + timedelta(hours=1),
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
            if len(attendee_ids) > 1:
                records.append(
                    SalesforceRecord(
                        salesforce_record_id=self._id(
                            "sf_record",
                            f"{event.id}:sf-campaign-member-no-show",
                        ),
                        object_type=SalesforceObjectType.CAMPAIGN_MEMBER,
                        record_id=self._id(
                            "campaign_member",
                            f"{event.id}:campaign-member-no-show",
                        ),
                        timestamp=event.ends_at + timedelta(hours=1, minutes=20),
                        campaign_id=campaign_id,
                        contact_id=attendee_ids[1],
                        event_id=event.id,
                        structured_fields={"member_status": "No Show"},
                        primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                        is_relevant=True,
                        relevance_reason=build_relevance_reason(
                            primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                            is_relevant=True,
                            provenance=LabelProvenance(
                                object_type=ProvenanceObjectType.EVENT,
                                object_id=event.id,
                                explanation=(
                                    "Attendance outcome is linked to the event and campaign."
                                ),
                            ),
                        ),
                        provenance=LabelProvenance(
                            object_type=ProvenanceObjectType.EVENT,
                            object_id=event.id,
                            explanation=(
                                "Attendance outcome is linked to the event and campaign."
                            ),
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
                    timestamp=event.ends_at + timedelta(hours=3),
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

    def _apply_account_behavior_profile(
        self,
        *,
        bundle: CrossSystemEventBundle,
        scenario: EventScenarioContext,
        profile: AccountBehaviorProfile,
        event: Event,
        event_index: int,
        total_selected: int,
    ) -> CrossSystemEventBundle:
        visible_systems = self._visible_systems_for_event(
            event=event,
            event_index=event_index,
            total_selected=total_selected,
            preferred_systems=profile.preferred_systems,
        )
        bundle = bundle.model_copy(
            update={
                "email_records": [
                    *bundle.email_records,
                    *self._profile_email_records(
                        scenario=scenario,
                        profile=profile,
                    ),
                ],
                "slack_records": [
                    *bundle.slack_records,
                    *self._profile_slack_records(
                        scenario=scenario,
                        profile=profile,
                    ),
                ],
                "teams_records": [
                    *bundle.teams_records,
                    *self._profile_teams_records(
                        scenario=scenario,
                        profile=profile,
                    ),
                ],
                "salesforce_records": [
                    *bundle.salesforce_records,
                    *self._profile_salesforce_records(
                        scenario=scenario,
                        profile=profile,
                    ),
                ],
            }
        )

        if len(visible_systems) == 4:
            return bundle

        return bundle.model_copy(
            update={
                "email_records": bundle.email_records if "email" in visible_systems else [],
                "slack_records": bundle.slack_records if "slack" in visible_systems else [],
                "teams_records": bundle.teams_records if "teams" in visible_systems else [],
                "salesforce_records": (
                    bundle.salesforce_records if "salesforce" in visible_systems else []
                ),
            }
        )

    def _visible_systems_for_event(
        self,
        *,
        event: Event,
        event_index: int,
        total_selected: int,
        preferred_systems: tuple[str, ...],
    ) -> tuple[str, ...]:
        del event, event_index, total_selected
        return preferred_systems

    def _profile_email_records(
        self,
        *,
        scenario: EventScenarioContext,
        profile: AccountBehaviorProfile,
    ) -> list[EmailRecord]:
        if profile.profile_type == AccountBehaviorProfileType.HIGHLY_ENGAGED_CHAMPION_LED:
            opportunity = scenario.opportunity
            if opportunity is None:
                return []
            organizer = scenario.organizer
            account = scenario.account
            contacts = list(scenario.account_contacts)[:2]
            provenance = LabelProvenance(
                object_type=ProvenanceObjectType.OPPORTUNITY,
                object_id=opportunity.id,
                explanation="Champion-led account is pushing the commercial motion forward.",
            )
            return [
                EmailRecord(
                    email_id=self._id("email", f"{scenario.event.id}:profile:champion"),
                    thread_id=self._id("thread", f"{scenario.event.id}:email-thread"),
                    message_index_in_thread=2,
                    timestamp=scenario.event.ends_at + timedelta(hours=3),
                    sender_employee_id=organizer.id,
                    sender_contact_id=None,
                    to=[contact.id for contact in contacts],
                    cc=[account.owner_employee_id],
                    bcc=[],
                    subject=f"Expansion path after {scenario.event.title}",
                    body=(
                        f"{account.name} wants to keep the {opportunity.stage} motion moving "
                        "and asked for the next expansion step."
                    ),
                    attachments=[],
                    account_id=account.id,
                    opportunity_id=opportunity.id,
                    event_id=scenario.event.id,
                    ticket_id=None,
                    primary_category=CommunicationCategory.BUYING_SIGNAL,
                    is_relevant=True,
                    relevance_reason=build_relevance_reason(
                        primary_category=CommunicationCategory.BUYING_SIGNAL,
                        is_relevant=True,
                        provenance=provenance,
                    ),
                    provenance=provenance,
                    source_system="email",
                )
            ]

        if profile.profile_type == AccountBehaviorProfileType.EXECUTIVE_SPONSORED_EXPANSION:
            organizer = scenario.organizer
            account = scenario.account
            contacts = list(scenario.account_contacts)[:1]
            provenance = LabelProvenance(
                object_type=ProvenanceObjectType.ACCOUNT,
                object_id=account.id,
                explanation="Executive sponsor review is already active on this account.",
            )
            return [
                EmailRecord(
                    email_id=self._id("email", f"{scenario.event.id}:profile:executive"),
                    thread_id=self._id("thread", f"{scenario.event.id}:email-thread"),
                    message_index_in_thread=2,
                    timestamp=scenario.event.ends_at + timedelta(hours=4),
                    sender_employee_id=organizer.id,
                    sender_contact_id=None,
                    to=[contact.id for contact in contacts],
                    cc=[account.owner_employee_id],
                    bcc=[],
                    subject=f"Executive sponsor follow-up for {account.name}",
                    body=(
                        f"{account.name} asked to bring the executive sponsor into the next "
                        "commercial review."
                    ),
                    attachments=[],
                    account_id=account.id,
                    opportunity_id=None,
                    event_id=scenario.event.id,
                    ticket_id=None,
                    primary_category=CommunicationCategory.DECISION_MAKER_SIGNAL,
                    is_relevant=True,
                    relevance_reason=build_relevance_reason(
                        primary_category=CommunicationCategory.DECISION_MAKER_SIGNAL,
                        is_relevant=True,
                        provenance=provenance,
                    ),
                    provenance=provenance,
                    source_system="email",
                )
            ]

        return []

    def _profile_slack_records(
        self,
        *,
        scenario: EventScenarioContext,
        profile: AccountBehaviorProfile,
    ) -> list[SlackRecord]:
        if profile.profile_type != AccountBehaviorProfileType.SUPPORT_INTENSIVE_UNHAPPY:
            return []

        ticket = scenario.ticket
        if ticket is None:
            return []
        organizer = scenario.organizer
        account = scenario.account
        provenance = LabelProvenance(
            object_type=ProvenanceObjectType.TICKET,
            object_id=ticket.id,
            explanation="Support-intensive account is escalating the unresolved ticket internally.",
        )
        return [
            SlackRecord(
                slack_message_id=self._id(
                    "slack_message",
                    f"{scenario.event.id}:profile:support-escalation",
                ),
                channel_id=self._id("channel", f"{scenario.event.id}:slack-channel"),
                channel_name=f"#acct-{_slugify(account.name)}",
                thread_id=self._id("slack_thread", f"{scenario.event.id}:slack-thread"),
                parent_message_id=None,
                timestamp=scenario.event.ends_at + timedelta(minutes=45),
                sender_employee_id=organizer.id,
                body=(
                    "Escalating this internally because the customer still sees the issue "
                    "and wants a named owner."
                ),
                mentions=[],
                reactions=[":rotating_light:"],
                attachments=[],
                linked_account_id=account.id,
                linked_opportunity_id=None,
                linked_event_id=scenario.event.id,
                linked_ticket_id=ticket.id,
                primary_category=CommunicationCategory.ESCALATION,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.ESCALATION,
                    is_relevant=True,
                    provenance=provenance,
                ),
                provenance=provenance,
                source_system="slack",
            )
        ]

    def _profile_teams_records(
        self,
        *,
        scenario: EventScenarioContext,
        profile: AccountBehaviorProfile,
    ) -> list[TeamsRecord]:
        if profile.profile_type != AccountBehaviorProfileType.HIGHLY_ENGAGED_CHAMPION_LED:
            return []

        account = scenario.account
        organizer = scenario.organizer
        account_owner = scenario.account_owner
        opportunity = scenario.opportunity
        if opportunity is None:
            return []
        provenance = LabelProvenance(
            object_type=ProvenanceObjectType.OPPORTUNITY,
            object_id=opportunity.id,
            explanation="Champion-led account is coordinating next-step expansion planning.",
        )
        return [
            TeamsRecord(
                teams_message_id=self._id(
                    "teams_message",
                    f"{scenario.event.id}:profile:champion",
                ),
                team_id=self._id("team", f"{scenario.event.id}:teams-team"),
                channel_id=self._id("channel", f"{scenario.event.id}:teams-channel"),
                chat_or_channel="channel",
                thread_id=self._id("teams_thread", f"{scenario.event.id}:teams-thread"),
                timestamp=scenario.event.ends_at + timedelta(hours=2, minutes=30),
                sender_employee_id=organizer.id,
                body=(
                    "Task list:\n"
                    "- capture the expansion owner\n"
                    "- schedule the next champion check-in"
                ),
                mentions=[account_owner.id],
                meeting_id=self._id("meeting", f"{scenario.event.id}:teams-meeting"),
                file_refs=["expansion-plan.docx"],
                linked_account_id=account.id,
                linked_opportunity_id=opportunity.id,
                linked_event_id=scenario.event.id,
                linked_ticket_id=None,
                primary_category=CommunicationCategory.FOLLOW_UP,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.FOLLOW_UP,
                    is_relevant=True,
                    provenance=provenance,
                ),
                provenance=provenance,
                source_system="teams",
            )
        ]

    def _profile_salesforce_records(
        self,
        *,
        scenario: EventScenarioContext,
        profile: AccountBehaviorProfile,
    ) -> list[SalesforceRecord]:
        if profile.profile_type != AccountBehaviorProfileType.HIGHLY_ENGAGED_CHAMPION_LED:
            return []

        opportunity = scenario.opportunity
        account = scenario.account
        account_owner = scenario.account_owner
        if opportunity is None:
            return []
        provenance = LabelProvenance(
            object_type=ProvenanceObjectType.OPPORTUNITY,
            object_id=opportunity.id,
            explanation="Champion-led account is logging fresh expansion interest.",
        )
        return [
            SalesforceRecord(
                salesforce_record_id=self._id(
                    "sf_record",
                    f"{scenario.event.id}:profile:champion-note",
                ),
                object_type=SalesforceObjectType.NOTE,
                record_id=self._id("note", f"{scenario.event.id}:profile:champion-note"),
                timestamp=scenario.event.ends_at + timedelta(hours=4),
                owner_employee_id=account_owner.id,
                account_id=account.id,
                opportunity_id=opportunity.id,
                event_id=scenario.event.id,
                parent_record_id=opportunity.id,
                subject="Champion follow-up note",
                text_body="Champion asked for the expansion plan and next owner update.",
                structured_fields={"activity_type": "Call Note"},
                primary_category=CommunicationCategory.BUYING_SIGNAL,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.BUYING_SIGNAL,
                    is_relevant=True,
                    provenance=provenance,
                ),
                provenance=provenance,
                source_system="salesforce",
            )
        ]

    def _event_contacts(
        self,
        *,
        event: Event,
        account: CustomerAccount,
    ) -> list[Contact]:
        event_contact_ids = set(event.attendee_contact_ids)
        account_contacts = [
            contact
            for contact in self.enterprise.contacts
            if contact.account_id == account.id
        ]
        matched_contacts = [
            contact
            for contact in account_contacts
            if contact.id in event_contact_ids
        ]
        return matched_contacts or account_contacts

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
