from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from synthetic_enterprise.contracts.records.salesforce import (
    SalesforceObjectType,
    SalesforceOpportunityStage,
    SalesforceRecord,
    StructuredFieldValue,
)
from synthetic_enterprise.domain import (
    Contact,
    CustomerAccount,
    Employee,
    EnterpriseGraph,
    Event,
    EventScenarioResolver,
    Opportunity,
)
from synthetic_enterprise.generation.attendance import (
    EventAttendanceBuilder,
    EventAttendanceLifecycle,
)
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


@dataclass(slots=True)
class SalesforceRenderer:
    """Render a small deterministic Salesforce CRM dataset from simulated enterprise state."""

    context: GeneratorContext
    enterprise: EnterpriseGraph

    def generate_records(self) -> list[SalesforceRecord]:
        scenario = EventScenarioResolver(self.enterprise).primary_account_event()
        account = scenario.account
        contacts = list(scenario.account_contacts)
        opportunity = scenario.opportunity
        event = scenario.event
        case = scenario.ticket
        owner = scenario.account_owner
        if opportunity is None:
            raise ValueError("salesforce rendering requires an account opportunity")
        if case is None:
            raise ValueError("salesforce rendering requires an account-linked ticket")
        campaign_id = self._campaign_id("customer-review")
        lead_id = self._lead_id("stale-webinar-lead")
        stale_opportunity_id = self._opportunity_id("stale-opportunity")
        follow_up_task_id = self._task_id("follow-up")
        duplicate_task_id = self._task_id("duplicate-follow-up")
        relevant_note_id = self._note_id("meeting-note")
        noisy_note_id = self._note_id("low-value-note")
        campaign_member_contact_id = self._campaign_member_id("contact-member")
        campaign_member_no_show_contact_id = self._campaign_member_id("contact-member-no-show")
        campaign_member_lead_id = self._campaign_member_id("lead-member")
        attendance_lifecycle = EventAttendanceBuilder(
            context=self.context,
            enterprise=self.enterprise,
        ).build(event_id=event.id)

        records: list[SalesforceRecord] = []
        records.extend(self._account_records(account=account, owner=owner))
        records.extend(self._contact_records(account=account, contacts=contacts, owner=owner))
        records.append(self._lead_record(lead_id=lead_id, owner=owner))
        records.extend(
            self._opportunity_records(
                account=account,
                opportunity=opportunity,
                stale_opportunity_id=stale_opportunity_id,
                owner=owner,
            )
        )
        records.extend(
            self._task_records(
                account=account,
                opportunity_id=opportunity.id,
                event=event,
                case_id=case.id,
                owner=owner,
                follow_up_task_id=follow_up_task_id,
                duplicate_task_id=duplicate_task_id,
            )
        )
        records.append(
            self._event_record(
                account=account,
                event=event,
                owner=owner,
                contacts=contacts,
                attendance_lifecycle=attendance_lifecycle,
            )
        )
        records.append(self._case_record(account=account, case_id=case.id, owner=owner))
        records.append(
            self._campaign_record(
                account=account,
                campaign_id=campaign_id,
                owner=owner,
            )
        )
        records.extend(
            self._campaign_member_records(
                campaign_id=campaign_id,
                contact_id=contacts[0].id,
                no_show_contact_id=contacts[1].id if len(contacts) > 1 else None,
                lead_id=lead_id,
                event_id=event.id,
                contact_member_id=campaign_member_contact_id,
                no_show_contact_member_id=campaign_member_no_show_contact_id,
                lead_member_id=campaign_member_lead_id,
                attendance_lifecycle=attendance_lifecycle,
            )
        )
        records.extend(
            self._note_records(
                account=account,
                opportunity_id=opportunity.id,
                event_id=event.id,
                owner=owner,
                relevant_note_id=relevant_note_id,
                noisy_note_id=noisy_note_id,
            )
        )
        records.extend(
            self._noise_records(
                account=account,
                event_id=event.id,
                owner=owner,
            )
        )
        records.extend(
            self._hard_negative_records(
                account=account,
                opportunity_id=opportunity.id,
                event_id=event.id,
                case_id=case.id,
                owner=owner,
            )
        )

        return self._apply_messy_data(
            records=records,
            account=account,
            contacts=contacts,
            opportunity=opportunity,
            event=event,
            case_id=case.id,
            follow_up_task_id=follow_up_task_id,
            duplicate_task_id=duplicate_task_id,
            relevant_note_id=relevant_note_id,
            noisy_note_id=noisy_note_id,
            owner=owner,
        )

    def _apply_messy_data(
        self,
        *,
        records: list[SalesforceRecord],
        account: CustomerAccount,
        contacts: list[Contact],
        opportunity: Opportunity,
        event: Event,
        case_id: str,
        follow_up_task_id: str,
        duplicate_task_id: str,
        relevant_note_id: str,
        noisy_note_id: str,
        owner: Employee,
    ) -> list[SalesforceRecord]:
        if self.context.messiness_rate <= 0.0:
            return records

        candidate_names = [
            "account_stale_owner",
            "contact_partial_primary",
            "contact_partial_secondary",
            "opportunity_stage_outdated",
            "follow_up_task_gap",
            "event_attendance_incomplete",
            "case_reopened",
            "relevant_note_delayed",
            "noisy_note_contradiction",
            "duplicate_task_low_quality",
        ]
        if len(contacts) < 2:
            candidate_names.remove("contact_partial_secondary")

        target_count = min(
            len(candidate_names),
            max(0, round(len(records) * self.context.messiness_rate)),
        )
        if target_count == 0:
            return records

        selected = set(candidate_names[:target_count])
        records_by_id = {record.record_id: record for record in records}

        if "account_stale_owner" in selected:
            account_record = records_by_id[account.id]
            account_fields = dict(account_record.structured_fields)
            account_fields["stale_owner_snapshot_days"] = 45
            records_by_id[account.id] = account_record.model_copy(
                update={
                    "structured_fields": account_fields,
                    "text_body": (
                        f"{account.name} account created for {account.industry}. "
                        "Owner snapshot may be stale."
                    ),
                }
            )

        if "contact_partial_primary" in selected:
            primary_contact = contacts[0]
            contact_record = records_by_id[primary_contact.id]
            records_by_id[primary_contact.id] = contact_record.model_copy(
                update={
                    "structured_fields": {
                        "title": primary_contact.title,
                        "missing_fields": ["email", "phone"],
                    },
                    "text_body": (
                        f"{primary_contact.title} at {account.name}. "
                        "CRM contact info incomplete."
                    ),
                }
            )

        if "contact_partial_secondary" in selected and len(contacts) > 1:
            secondary_contact = contacts[1]
            contact_record = records_by_id[secondary_contact.id]
            records_by_id[secondary_contact.id] = contact_record.model_copy(
                update={
                    "structured_fields": {
                        "email": secondary_contact.email,
                        "title": secondary_contact.title,
                        "missing_fields": ["mobile_phone"],
                    },
                }
            )

        if "opportunity_stage_outdated" in selected:
            opportunity_record = records_by_id[opportunity.id]
            opportunity_fields = dict(opportunity_record.structured_fields)
            opportunity_fields.update(
                {
                    "crm_stage_snapshot": "discovery",
                    "stage_sync_delay_days": 21,
                }
            )
            records_by_id[opportunity.id] = opportunity_record.model_copy(
                update={
                    "structured_fields": opportunity_fields,
                    "text_body": (
                        "Commercial evaluation is still active and tied to next-step "
                        "ownership, but the CRM stage snapshot is lagging."
                    ),
                }
            )

        if "follow_up_task_gap" in selected:
            follow_up_task = records_by_id[follow_up_task_id]
            task_fields = dict(follow_up_task.structured_fields)
            task_fields["follow_up_capture_state"] = "missing_in_queue"
            records_by_id[follow_up_task_id] = follow_up_task.model_copy(
                update={
                    "structured_fields": task_fields,
                    "text_body": (
                        "Customer asked for owners and target dates by Friday, but the "
                        "follow-up queue entry was logged late."
                    ),
                }
            )

        if "event_attendance_incomplete" in selected:
            event_record = records_by_id[event.id]
            attendee_ids = list(event_record.attendee_contact_ids)
            captured_ids = attendee_ids[:1]
            event_fields = dict(event_record.structured_fields)
            event_fields.update(
                {
                    "attendance_reconciliation_pending": True,
                    "captured_attendance_count": len(captured_ids),
                    "expected_attendance_count": len(attendee_ids),
                }
            )
            records_by_id[event.id] = event_record.model_copy(
                update={
                    "attendee_contact_ids": captured_ids,
                    "structured_fields": event_fields,
                    "text_body": (
                        "Customer review held with attendance capture pending reconciliation."
                    ),
                }
            )

        if "case_reopened" in selected:
            case_record = records_by_id[case_id]
            case_fields = dict(case_record.structured_fields)
            case_fields.update({"status": "Reopened", "reopened_after_days": 5})
            records_by_id[case_id] = case_record.model_copy(
                update={
                    "status": "Reopened",
                    "structured_fields": case_fields,
                    "text_body": (
                        "Support case reopened after customer retest and is still tracked "
                        "alongside the account plan."
                    ),
                }
            )

        if "relevant_note_delayed" in selected:
            note_record = records_by_id[relevant_note_id]
            note_fields = dict(note_record.structured_fields)
            note_fields["late_entry_days"] = 3
            records_by_id[relevant_note_id] = note_record.model_copy(
                update={
                    "timestamp": note_record.timestamp + timedelta(days=3),
                    "structured_fields": note_fields,
                    "text_body": (
                        f"{note_record.text_body} Entered after the notes backlog was cleared."
                    ),
                }
            )

        if "noisy_note_contradiction" in selected:
            noisy_note = records_by_id[noisy_note_id]
            note_fields = dict(noisy_note.structured_fields)
            note_fields["contradicts_record_id"] = follow_up_task_id
            records_by_id[noisy_note_id] = noisy_note.model_copy(
                update={
                    "structured_fields": note_fields,
                    "text_body": (
                        "Internal note says the customer already confirmed owners, but the "
                        "open follow-up task is still unreconciled."
                    ),
                }
            )

        if "duplicate_task_low_quality" in selected:
            duplicate_task = records_by_id[duplicate_task_id]
            duplicate_fields = dict(duplicate_task.structured_fields)
            duplicate_fields["follow_up_capture_state"] = "duplicate_only"
            records_by_id[duplicate_task_id] = duplicate_task.model_copy(
                update={
                    "structured_fields": duplicate_fields,
                    "text_body": "Duplicate reminder copied again with no new detail.",
                }
            )

        return [records_by_id[record.record_id] for record in records]

    def _account_records(
        self,
        *,
        account: CustomerAccount,
        owner: Employee,
    ) -> list[SalesforceRecord]:
        return [
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("account"),
                object_type=SalesforceObjectType.ACCOUNT,
                record_id=account.id,
                timestamp=account.created_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                subject=account.name,
                text_body=f"{account.name} account created for {account.industry}.",
                structured_fields={
                    "industry": account.industry,
                    "owner_employee_id": owner.id,
                },
                primary_category=CommunicationCategory.STATUS_UPDATES,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.STATUS_UPDATES,
                    is_relevant=False,
                    provenance=None,
                ),
                provenance=None,
                source_system="salesforce",
            )
        ]

    def _contact_records(
        self,
        *,
        account: CustomerAccount,
        contacts: list[Contact],
        owner: Employee,
    ) -> list[SalesforceRecord]:
        return [
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id(f"contact-{index}"),
                object_type=SalesforceObjectType.CONTACT,
                record_id=contact.id,
                timestamp=contact.created_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                contact_id=contact.id,
                subject=f"{contact.first_name} {contact.last_name}",
                text_body=f"{contact.title} at {account.name}",
                structured_fields={"email": contact.email, "title": contact.title},
                primary_category=CommunicationCategory.STATUS_UPDATES,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.STATUS_UPDATES,
                    is_relevant=False,
                    provenance=None,
                ),
                provenance=None,
                source_system="salesforce",
            )
            for index, contact in enumerate(contacts)
        ]

    def _lead_record(self, *, lead_id: str, owner: Employee) -> SalesforceRecord:
        return SalesforceRecord(
            salesforce_record_id=self._sf_record_id("lead"),
            object_type=SalesforceObjectType.LEAD,
            record_id=lead_id,
            timestamp=self.context.date_range.start_at,
            owner_employee_id=owner.id,
            lead_id=lead_id,
            subject="Webinar lead from Q1 list",
            text_body="No meaningful follow-up since initial import.",
            structured_fields={"status": "Open - Not Contacted", "source": "Webinar List"},
            primary_category=CommunicationCategory.IRRELEVANT_MARKETING,
            is_relevant=False,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.IRRELEVANT_MARKETING,
                is_relevant=False,
                provenance=None,
            ),
            provenance=None,
            source_system="salesforce",
        )

    def _opportunity_records(
        self,
        *,
        account: CustomerAccount,
        opportunity: Opportunity,
        stale_opportunity_id: str,
        owner: Employee,
    ) -> list[SalesforceRecord]:
        return [
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("opportunity-active"),
                object_type=SalesforceObjectType.OPPORTUNITY,
                record_id=opportunity.id,
                timestamp=opportunity.created_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                opportunity_id=opportunity.id,
                stage=SalesforceOpportunityStage(opportunity.stage),
                subject=f"{account.name} expansion",
                text_body=(
                    "Commercial evaluation is still active and tied to next-step "
                    "ownership."
                ),
                structured_fields={
                    "amount": opportunity.amount,
                    "product_ids": opportunity.product_ids,
                },
                primary_category=CommunicationCategory.BUYING_SIGNAL,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.BUYING_SIGNAL,
                    is_relevant=True,
                    provenance=self._provenance(
                        object_type=ProvenanceObjectType.OPPORTUNITY,
                        object_id=opportunity.id,
                        explanation=(
                            f"Opportunity remains active in stage {opportunity.stage}."
                        ),
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.OPPORTUNITY,
                    object_id=opportunity.id,
                    explanation=f"Opportunity remains active in stage {opportunity.stage}.",
                ),
                source_system="salesforce",
            ),
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("opportunity-stale"),
                object_type=SalesforceObjectType.OPPORTUNITY,
                record_id=stale_opportunity_id,
                timestamp=opportunity.created_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                opportunity_id=stale_opportunity_id,
                stage=SalesforceOpportunityStage.QUALIFICATION,
                subject=f"{account.name} legacy add-on",
                text_body="No update in 60 days. Left open for historical reference.",
                structured_fields={"stale_days": 60, "amount": 15000.0},
                primary_category=CommunicationCategory.STATUS_UPDATES,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.STATUS_UPDATES,
                    is_relevant=False,
                    provenance=None,
                ),
                provenance=None,
                source_system="salesforce",
            ),
        ]

    def _noise_records(
        self,
        *,
        account: CustomerAccount,
        event_id: str,
        owner: Employee,
    ) -> list[SalesforceRecord]:
        families = NoiseEngine(self.context).plan(
            source_system="salesforce",
            count=max(4, int(round(6 * self.context.noise_ratio))),
        )
        records: list[SalesforceRecord] = []

        for index, family in enumerate(families):
            timestamp = self.context.date_range.start_at + timedelta(days=index)
            if family in {
                NoiseFamily.DOCUMENT_REVIEW_PING,
                NoiseFamily.VAGUE_FOLLOW_UP,
                NoiseFamily.DUPLICATE_REMINDER,
                NoiseFamily.OWNERSHIP_AMBIGUITY,
                NoiseFamily.PARTIAL_HANDOFF,
            }:
                records.append(
                    SalesforceRecord(
                        salesforce_record_id=self._sf_record_id(f"noise-task-{family.value}-{index}"),
                        object_type=SalesforceObjectType.TASK,
                        record_id=self._task_id(f"noise-task-{family.value}-{index}"),
                        timestamp=timestamp,
                        owner_employee_id=owner.id,
                        account_id=account.id,
                        event_id=event_id,
                        subject=self._noise_subject(family, account.name),
                        text_body=self._noise_text_body(family, account.name),
                        structured_fields=self._noise_structured_fields(
                            family=family,
                            owner=owner,
                        ),
                        primary_category=noise_category_for_family(family),
                        is_relevant=False,
                        relevance_reason=build_relevance_reason(
                            primary_category=noise_category_for_family(family),
                            is_relevant=False,
                            provenance=None,
                        ),
                        provenance=None,
                        source_system="salesforce",
                    )
                )
                continue

            records.append(
                SalesforceRecord(
                    salesforce_record_id=self._sf_record_id(f"noise-note-{family.value}-{index}"),
                    object_type=SalesforceObjectType.NOTE,
                    record_id=self._note_id(f"noise-note-{family.value}-{index}"),
                    timestamp=timestamp,
                    owner_employee_id=owner.id,
                    account_id=account.id,
                    parent_record_id=account.id,
                    subject=self._noise_subject(family, account.name),
                    text_body=self._noise_text_body(family, account.name),
                    structured_fields={"activity_type": "Internal Note"},
                    primary_category=noise_category_for_family(family),
                    is_relevant=False,
                    relevance_reason=build_relevance_reason(
                        primary_category=noise_category_for_family(family),
                        is_relevant=False,
                        provenance=None,
                    ),
                    provenance=None,
                    source_system="salesforce",
                )
            )

        return records

    def _task_records(
        self,
        *,
        account: CustomerAccount,
        opportunity_id: str,
        event: Event,
        case_id: str,
        owner: Employee,
        follow_up_task_id: str,
        duplicate_task_id: str,
    ) -> list[SalesforceRecord]:
        return [
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("task-follow-up"),
                object_type=SalesforceObjectType.TASK,
                record_id=follow_up_task_id,
                timestamp=event.ends_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                opportunity_id=opportunity_id,
                event_id=event.id,
                subject="Send next-step recap after customer review",
                text_body="Customer asked for owners and target dates by Friday.",
                structured_fields={"priority": "High", "status": "Open"},
                primary_category=CommunicationCategory.FOLLOW_UP,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.FOLLOW_UP,
                    is_relevant=True,
                    provenance=self._provenance(
                        object_type=ProvenanceObjectType.EVENT,
                        object_id=event.id,
                        explanation=(
                            f"Customer review still needs owners and due dates for "
                            f"opportunity {opportunity_id}."
                        ),
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    explanation=(
                        f"Customer review still needs owners and due dates for "
                        f"opportunity {opportunity_id}."
                    ),
                ),
                source_system="salesforce",
            ),
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("task-duplicate"),
                object_type=SalesforceObjectType.TASK,
                record_id=duplicate_task_id,
                timestamp=event.ends_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                case_id=case_id,
                parent_record_id=follow_up_task_id,
                subject="Send next-step recap after customer review",
                text_body="Duplicate reminder copied from earlier task.",
                structured_fields={
                    "priority": "Low",
                    "status": "Open",
                    "duplicate_of": follow_up_task_id,
                },
                primary_category=CommunicationCategory.DUPLICATE_SUMMARY,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.DUPLICATE_SUMMARY,
                    is_relevant=False,
                    provenance=None,
                ),
                provenance=None,
                source_system="salesforce",
            ),
        ]

    def _event_record(
        self,
        *,
        account: CustomerAccount,
        event: Event,
        owner: Employee,
        contacts: list[Contact],
        attendance_lifecycle: EventAttendanceLifecycle,
    ) -> SalesforceRecord:
        attendee_ids = [contact.id for contact in contacts[:2]]
        return SalesforceRecord(
            salesforce_record_id=self._sf_record_id("event"),
            object_type=SalesforceObjectType.EVENT,
            record_id=event.id,
            timestamp=event.starts_at,
            owner_employee_id=owner.id,
            account_id=account.id,
            event_id=event.id,
            attendee_contact_ids=attendee_ids,
            subject=event.title,
            text_body=(
                "Invite sent, RSVP requested, reminder sent, and attendance capture "
                "updated after the customer review."
            ),
            structured_fields={
                "invite_sent_at": attendance_lifecycle.invite_sent_at.isoformat(),
                "rsvp_requested_at": attendance_lifecycle.rsvp_requested_at.isoformat(),
                "reminder_sent_at": attendance_lifecycle.reminder_sent_at.isoformat(),
                "attendance_count": len(attendance_lifecycle.attended_contact_ids),
                "no_show_count": len(attendance_lifecycle.no_show_contact_ids),
                "event_type": event.event_type,
            },
            primary_category=CommunicationCategory.EVENT_ATTENDANCE,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                is_relevant=True,
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event.id,
                    explanation="Attendee participation is captured on the event record.",
                ),
            ),
            provenance=self._provenance(
                object_type=ProvenanceObjectType.EVENT,
                object_id=event.id,
                explanation="Attendee participation is captured on the event record.",
            ),
            source_system="salesforce",
        )

    def _case_record(
        self,
        *,
        account: CustomerAccount,
        case_id: str,
        owner: Employee,
    ) -> SalesforceRecord:
        return SalesforceRecord(
            salesforce_record_id=self._sf_record_id("case"),
            object_type=SalesforceObjectType.CASE,
            record_id=case_id,
            timestamp=self.context.date_range.start_at,
            owner_employee_id=owner.id,
            account_id=account.id,
            case_id=case_id,
            subject="Dashboard export timeout",
            text_body="Support case remains active and is tracked alongside the account plan.",
            structured_fields={"status": "Open", "severity": "High"},
            primary_category=CommunicationCategory.BLOCKER,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.BLOCKER,
                is_relevant=True,
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.TICKET,
                    object_id=case_id,
                    explanation=(
                        f"Open export timeout issue is constraining account {account.id}."
                    ),
                ),
            ),
            provenance=self._provenance(
                object_type=ProvenanceObjectType.TICKET,
                object_id=case_id,
                explanation=f"Open export timeout issue is constraining account {account.id}.",
            ),
            source_system="salesforce",
        )

    def _campaign_record(
        self,
        *,
        account: CustomerAccount,
        campaign_id: str,
        owner: Employee,
    ) -> SalesforceRecord:
        return SalesforceRecord(
            salesforce_record_id=self._sf_record_id("campaign"),
            object_type=SalesforceObjectType.CAMPAIGN,
            record_id=campaign_id,
            timestamp=self.context.date_range.start_at,
            owner_employee_id=owner.id,
            campaign_id=campaign_id,
            subject=f"{account.name} customer review outreach",
            text_body="Campaign used to track customer review attendance and prep.",
            structured_fields={"status": "In Progress", "type": "Customer Webinar"},
            primary_category=CommunicationCategory.EVENT_ATTENDANCE,
            is_relevant=True,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                is_relevant=True,
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.CAMPAIGN,
                    object_id=campaign_id,
                    explanation="Campaign tracks attendance for the customer review program.",
                ),
            ),
            provenance=self._provenance(
                object_type=ProvenanceObjectType.CAMPAIGN,
                object_id=campaign_id,
                explanation="Campaign tracks attendance for the customer review program.",
            ),
            source_system="salesforce",
        )

    def _campaign_member_records(
        self,
        *,
        campaign_id: str,
        contact_id: str,
        no_show_contact_id: str | None,
        lead_id: str,
        event_id: str,
        contact_member_id: str,
        no_show_contact_member_id: str,
        lead_member_id: str,
        attendance_lifecycle: EventAttendanceLifecycle,
    ) -> list[SalesforceRecord]:
        records = [
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("campaign-member-contact"),
                object_type=SalesforceObjectType.CAMPAIGN_MEMBER,
                record_id=contact_member_id,
                timestamp=attendance_lifecycle.post_event_follow_up_at,
                campaign_id=campaign_id,
                contact_id=contact_id,
                event_id=event_id,
                structured_fields={"member_status": "Attended"},
                primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                    is_relevant=True,
                    provenance=self._provenance(
                        object_type=ProvenanceObjectType.EVENT,
                        object_id=event_id,
                        explanation="Contact attendance is recorded against the customer review.",
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event_id,
                    explanation="Contact attendance is recorded against the customer review.",
                ),
                source_system="salesforce",
            ),
        ]
        if no_show_contact_id is not None:
            records.append(
                SalesforceRecord(
                    salesforce_record_id=self._sf_record_id("campaign-member-contact-no-show"),
                    object_type=SalesforceObjectType.CAMPAIGN_MEMBER,
                    record_id=no_show_contact_member_id,
                    timestamp=attendance_lifecycle.post_event_follow_up_at,
                    campaign_id=campaign_id,
                    contact_id=no_show_contact_id,
                    event_id=event_id,
                    structured_fields={"member_status": "No Show"},
                    primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                    is_relevant=True,
                    relevance_reason=build_relevance_reason(
                        primary_category=CommunicationCategory.EVENT_ATTENDANCE,
                        is_relevant=True,
                        provenance=self._provenance(
                            object_type=ProvenanceObjectType.EVENT,
                            object_id=event_id,
                            explanation=(
                                "No-show attendance outcome is recorded against the "
                                "customer review."
                            ),
                        ),
                    ),
                    provenance=self._provenance(
                        object_type=ProvenanceObjectType.EVENT,
                        object_id=event_id,
                        explanation=(
                            "No-show attendance outcome is recorded against the "
                            "customer review."
                        ),
                    ),
                    source_system="salesforce",
                )
            )
        records.append(
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("campaign-member-lead"),
                object_type=SalesforceObjectType.CAMPAIGN_MEMBER,
                record_id=lead_member_id,
                campaign_id=campaign_id,
                lead_id=lead_id,
                structured_fields={"member_status": "Sent"},
                primary_category=CommunicationCategory.IRRELEVANT_MARKETING,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.IRRELEVANT_MARKETING,
                    is_relevant=False,
                    provenance=None,
                ),
                provenance=None,
                source_system="salesforce",
                timestamp=self.context.date_range.start_at,
            )
        )
        return records

    def _note_records(
        self,
        *,
        account: CustomerAccount,
        opportunity_id: str,
        event_id: str,
        owner: Employee,
        relevant_note_id: str,
        noisy_note_id: str,
    ) -> list[SalesforceRecord]:
        profile = self.context.language_profile
        product_reference = profile.product_reference(self.enterprise.products[0].name)
        return [
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("note-relevant"),
                object_type=SalesforceObjectType.NOTE,
                record_id=relevant_note_id,
                timestamp=self.context.date_range.start_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                opportunity_id=opportunity_id,
                event_id=event_id,
                parent_record_id=opportunity_id,
                subject=f"Post-meeting notes - {profile.crm_marker}",
                text_body=(
                    f"{profile.crm_marker}: customer asked for revised timing and named "
                    f"owners before renewal planning on {product_reference}."
                ),
                structured_fields={"activity_type": "Call Note"},
                primary_category=CommunicationCategory.DECISION_MAKER_SIGNAL,
                is_relevant=True,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.DECISION_MAKER_SIGNAL,
                    is_relevant=True,
                    provenance=self._provenance(
                        object_type=ProvenanceObjectType.OPPORTUNITY,
                        object_id=opportunity_id,
                        explanation=(
                            "Meeting notes capture named owners ahead of renewal planning."
                        ),
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.OPPORTUNITY,
                    object_id=opportunity_id,
                    explanation="Meeting notes capture named owners ahead of renewal planning.",
                ),
                source_system="salesforce",
            ),
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("note-noisy"),
                object_type=SalesforceObjectType.NOTE,
                record_id=noisy_note_id,
                timestamp=self.context.date_range.start_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                parent_record_id=account.id,
                subject="Workspace reminder",
                text_body="Folder rename pending.",
                structured_fields={"activity_type": "Internal Note"},
                primary_category=CommunicationCategory.ADMIN_OPS,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.ADMIN_OPS,
                    is_relevant=False,
                    provenance=None,
                ),
                provenance=None,
                source_system="salesforce",
            ),
        ]

    def _hard_negative_records(
        self,
        *,
        account: CustomerAccount,
        opportunity_id: str,
        event_id: str,
        case_id: str,
        owner: Employee,
    ) -> list[SalesforceRecord]:
        total = hard_negative_count(
            template_count=len(HardNegativeFamily),
            ratio=self.context.hard_negative_ratio,
        )
        families = HardNegativeEngine(self.context).plan(count=total)
        return [
            self._hard_negative_record_for_family(
                family=family,
                index=index,
                account=account,
                opportunity_id=opportunity_id,
                event_id=event_id,
                case_id=case_id,
                owner=owner,
            )
            for index, family in enumerate(families)
        ]

    def _hard_negative_record_for_family(
        self,
        *,
        family: HardNegativeFamily,
        index: int,
        account: CustomerAccount,
        opportunity_id: str,
        event_id: str,
        case_id: str,
        owner: Employee,
    ) -> SalesforceRecord:
        timestamp = self.context.date_range.start_at + timedelta(hours=index)

        if family == HardNegativeFamily.EVENT_NO_ATTENDANCE:
            return SalesforceRecord(
                salesforce_record_id=self._sf_record_id(f"hard-negative-{family.value}-{index}"),
                object_type=SalesforceObjectType.TASK,
                record_id=self._task_id(f"hard-negative-{family.value}-{index}"),
                timestamp=timestamp,
                owner_employee_id=owner.id,
                account_id=account.id,
                event_id=event_id,
                subject="Review attendance follow-up",
                text_body=(
                    "Discuss the review attendance and next step next week. No attendee "
                    "accepted, declined, or checked in yet."
                ),
                structured_fields={"status": "Open", "priority": "Low"},
                primary_category=CommunicationCategory.SCHEDULING_ONLY,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.SCHEDULING_ONLY,
                    is_relevant=False,
                    provenance=self._provenance(
                        object_type=ProvenanceObjectType.EVENT,
                        object_id=event_id,
                        explanation=hard_negative_explanation(family),
                        strength=ProvenanceStrength.WEAK,
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event_id,
                    explanation=hard_negative_explanation(family),
                    strength=ProvenanceStrength.WEAK,
                ),
                source_system="salesforce",
            )
        if family == HardNegativeFamily.BUDGET_NON_BUYING:
            return SalesforceRecord(
                salesforce_record_id=self._sf_record_id(f"hard-negative-{family.value}-{index}"),
                object_type=SalesforceObjectType.NOTE,
                record_id=self._note_id(f"hard-negative-{family.value}-{index}"),
                timestamp=timestamp,
                owner_employee_id=owner.id,
                account_id=account.id,
                opportunity_id=opportunity_id,
                parent_record_id=opportunity_id,
                subject="Budget note for review logistics",
                text_body=(
                    "Budget line covers room block, travel, and catering for the review, not "
                    "commercial scope."
                ),
                structured_fields={"activity_type": "Internal Note"},
                primary_category=CommunicationCategory.ADMIN_OPS,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.ADMIN_OPS,
                    is_relevant=False,
                    provenance=self._provenance(
                        object_type=ProvenanceObjectType.OPPORTUNITY,
                        object_id=opportunity_id,
                        explanation=hard_negative_explanation(family),
                        strength=ProvenanceStrength.WEAK,
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.OPPORTUNITY,
                    object_id=opportunity_id,
                    explanation=hard_negative_explanation(family),
                    strength=ProvenanceStrength.WEAK,
                ),
                source_system="salesforce",
            )
        if family == HardNegativeFamily.COMPLAINT_NON_PRODUCT:
            return SalesforceRecord(
                salesforce_record_id=self._sf_record_id(f"hard-negative-{family.value}-{index}"),
                object_type=SalesforceObjectType.NOTE,
                record_id=self._note_id(f"hard-negative-{family.value}-{index}"),
                timestamp=timestamp,
                owner_employee_id=owner.id,
                account_id=account.id,
                event_id=event_id,
                subject="Travel complaint for onsite review",
                text_body=(
                    "Weather and shuttle delays may affect arrival for the onsite review. "
                    "This is not a product issue."
                ),
                structured_fields={"activity_type": "Travel Note"},
                primary_category=CommunicationCategory.STATUS_UPDATES,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.STATUS_UPDATES,
                    is_relevant=False,
                    provenance=self._provenance(
                        object_type=ProvenanceObjectType.EVENT,
                        object_id=event_id,
                        explanation=hard_negative_explanation(family),
                        strength=ProvenanceStrength.WEAK,
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event_id,
                    explanation=hard_negative_explanation(family),
                    strength=ProvenanceStrength.WEAK,
                ),
                source_system="salesforce",
            )
        if family == HardNegativeFamily.ESCALATION_NO_BLOCKER:
            return SalesforceRecord(
                salesforce_record_id=self._sf_record_id(f"hard-negative-{family.value}-{index}"),
                object_type=SalesforceObjectType.TASK,
                record_id=self._task_id(f"hard-negative-{family.value}-{index}"),
                timestamp=timestamp,
                owner_employee_id=owner.id,
                account_id=account.id,
                case_id=case_id,
                subject="Escalated schedule wording",
                text_body=(
                    "Flagging the review thread as escalated for scheduling visibility, but "
                    "there is not a live blocker, owner change, or case action."
                ),
                structured_fields={"status": "Open", "priority": "Low"},
                primary_category=CommunicationCategory.STATUS_UPDATES,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.STATUS_UPDATES,
                    is_relevant=False,
                    provenance=self._provenance(
                        object_type=ProvenanceObjectType.TICKET,
                        object_id=case_id,
                        explanation=hard_negative_explanation(family),
                        strength=ProvenanceStrength.WEAK,
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.TICKET,
                    object_id=case_id,
                    explanation=hard_negative_explanation(family),
                    strength=ProvenanceStrength.WEAK,
                ),
                source_system="salesforce",
            )
        if family == HardNegativeFamily.EXECUTIVE_VISIBILITY_ONLY:
            return SalesforceRecord(
                salesforce_record_id=self._sf_record_id(f"hard-negative-{family.value}-{index}"),
                object_type=SalesforceObjectType.NOTE,
                record_id=self._note_id(f"hard-negative-{family.value}-{index}"),
                timestamp=timestamp,
                owner_employee_id=owner.id,
                account_id=account.id,
                parent_record_id=account.id,
                subject="VP visibility note",
                text_body=(
                    "Added the VP for visibility only on the renewal summary and next-step "
                    "owners. This is not an engaged decision-maker signal."
                ),
                structured_fields={"activity_type": "Executive Note"},
                primary_category=CommunicationCategory.FYI_FORWARD,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.FYI_FORWARD,
                    is_relevant=False,
                    provenance=self._provenance(
                        object_type=ProvenanceObjectType.ACCOUNT,
                        object_id=account.id,
                        explanation=hard_negative_explanation(family),
                        strength=ProvenanceStrength.WEAK,
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.ACCOUNT,
                    object_id=account.id,
                    explanation=hard_negative_explanation(family),
                    strength=ProvenanceStrength.WEAK,
                ),
                source_system="salesforce",
            )
        return SalesforceRecord(
            salesforce_record_id=self._sf_record_id(f"hard-negative-{family.value}-{index}"),
            object_type=SalesforceObjectType.TASK,
            record_id=self._task_id(f"hard-negative-{family.value}-{index}"),
            timestamp=timestamp,
            owner_employee_id=owner.id,
            account_id=account.id,
            event_id=event_id,
            subject="Follow up note without action",
            text_body=(
                "Follow up is noted after the review with owners and next-step wording, but "
                "no required action, owner, or due date is recorded."
            ),
            structured_fields={"status": "Open", "priority": "Low"},
            primary_category=CommunicationCategory.LOW_SIGNAL_CHECKIN,
            is_relevant=False,
            relevance_reason=build_relevance_reason(
                primary_category=CommunicationCategory.LOW_SIGNAL_CHECKIN,
                is_relevant=False,
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event_id,
                    explanation=hard_negative_explanation(family),
                    strength=ProvenanceStrength.WEAK,
                ),
            ),
            provenance=self._provenance(
                object_type=ProvenanceObjectType.EVENT,
                object_id=event_id,
                explanation=hard_negative_explanation(family),
                strength=ProvenanceStrength.WEAK,
            ),
            source_system="salesforce",
        )

    def _provenance(
        self,
        *,
        object_type: ProvenanceObjectType,
        object_id: str,
        explanation: str,
        strength: ProvenanceStrength = ProvenanceStrength.STRONG,
    ) -> LabelProvenance:
        return LabelProvenance(
            object_type=object_type,
            object_id=object_id,
            explanation=explanation,
            strength=strength,
        )

    def _noise_subject(self, family: NoiseFamily, account_name: str) -> str:
        if family == NoiseFamily.STALE_CRM_UPDATE:
            return f"Stale CRM cleanup for {account_name}"
        if family == NoiseFamily.PROCUREMENT_ADMIN:
            return f"Procurement routing for {account_name}"
        if family == NoiseFamily.ACCESS_REQUEST:
            return "Access request for shared workspace"
        if family == NoiseFamily.COMPLIANCE_ADMIN_REMINDER:
            return "Admin compliance reminder"
        if family == NoiseFamily.IRRELEVANT_FORWARDED_CHAIN:
            return f"Forwarded internal chain for {account_name}"
        return f"Internal note for {account_name}"

    def _noise_structured_fields(
        self,
        *,
        family: NoiseFamily,
        owner: Employee,
    ) -> dict[str, StructuredFieldValue]:
        fields: dict[str, StructuredFieldValue] = {"status": "Open", "priority": "Low"}
        if family == NoiseFamily.OWNERSHIP_AMBIGUITY:
            pending_owner = next(
                (
                    employee.id
                    for employee in self.enterprise.employees
                    if employee.id != owner.id
                ),
                owner.id,
            )
            fields.update(
                {
                    "previous_owner_employee_id": owner.id,
                    "pending_owner_employee_id": pending_owner,
                }
            )
        return fields

    def _noise_text_body(self, family: NoiseFamily, account_name: str) -> str:
        if family == NoiseFamily.VAGUE_FOLLOW_UP:
            return (
                f"Follow back up with {account_name} next week if the archive thread reopens. "
                "No customer commitment is pending."
            )
        if family == NoiseFamily.DOCUMENT_REVIEW_PING:
            return (
                f"Please skim the workspace notes for {account_name} before filing them away."
            )
        if family == NoiseFamily.DUPLICATE_REMINDER:
            return "Duplicate reminder copied from the prior task for visibility only."
        if family == NoiseFamily.OWNERSHIP_AMBIGUITY:
            return "Owner not assigned for internal cleanup follow-through."
        if family == NoiseFamily.COMPLIANCE_ADMIN_REMINDER:
            return "Reminder to upload the annual acknowledgment PDF."
        if family == NoiseFamily.ACCESS_REQUEST:
            return "Restore edit permissions on the archived workbook."
        if family == NoiseFamily.PARTIAL_HANDOFF:
            return "Partial handoff completed. Remaining admin notes moved to ops."
        if family == NoiseFamily.IRRELEVANT_FORWARDED_CHAIN:
            return "Forwarding an older internal chain for record-keeping only."
        if family == NoiseFamily.PROCUREMENT_ADMIN:
            return (
                "Budget code and supplier form need cleanup. This is procurement admin only."
            )
        return "CRM last-touch date is stale and needs a housekeeping update."

    def _sf_record_id(self, namespace: str) -> str:
        derived = self.context.derive_seed(f"sf-row:{namespace}")
        return f"sf_record_{derived:016x}"

    def _lead_id(self, namespace: str) -> str:
        derived = self.context.derive_seed(f"sf-lead:{namespace}")
        return f"lead_{derived:016x}"

    def _campaign_id(self, namespace: str) -> str:
        derived = self.context.derive_seed(f"sf-campaign:{namespace}")
        return f"campaign_{derived:016x}"

    def _campaign_member_id(self, namespace: str) -> str:
        derived = self.context.derive_seed(f"sf-campaign-member:{namespace}")
        return f"campaign_member_{derived:016x}"

    def _opportunity_id(self, namespace: str) -> str:
        derived = self.context.derive_seed(f"sf-opportunity:{namespace}")
        return f"opportunity_{derived:016x}"

    def _task_id(self, namespace: str) -> str:
        derived = self.context.derive_seed(f"sf-task:{namespace}")
        return f"task_{derived:016x}"

    def _note_id(self, namespace: str) -> str:
        derived = self.context.derive_seed(f"sf-note:{namespace}")
        return f"note_{derived:016x}"
