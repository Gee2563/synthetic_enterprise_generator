from __future__ import annotations

from dataclasses import dataclass

from synthetic_enterprise.contracts.records.salesforce import (
    SalesforceObjectType,
    SalesforceOpportunityStage,
    SalesforceRecord,
)
from synthetic_enterprise.domain import (
    Contact,
    CustomerAccount,
    Employee,
    EnterpriseGraph,
    Event,
    Opportunity,
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


@dataclass(slots=True)
class SalesforceRenderer:
    """Render a small deterministic Salesforce CRM dataset from simulated enterprise state."""

    context: GeneratorContext
    enterprise: EnterpriseGraph

    def generate_records(self) -> list[SalesforceRecord]:
        account = self.enterprise.customer_accounts[0]
        contacts = [
            contact
            for contact in self.enterprise.contacts
            if contact.account_id == account.id
        ]
        opportunity = next(
            current
            for current in self.enterprise.opportunities
            if current.account_id == account.id
        )
        event = next(
            current
            for current in self.enterprise.events
            if current.account_id == account.id
        )
        case = next(
            current
            for current in self.enterprise.ticket_issues
            if current.account_id == account.id
        )
        owner = self._employee_by_id[account.owner_employee_id]
        campaign_id = self._campaign_id("customer-review")
        lead_id = self._lead_id("stale-webinar-lead")
        stale_opportunity_id = self._opportunity_id("stale-opportunity")
        follow_up_task_id = self._task_id("follow-up")
        duplicate_task_id = self._task_id("duplicate-follow-up")
        relevant_note_id = self._note_id("meeting-note")
        noisy_note_id = self._note_id("low-value-note")
        campaign_member_contact_id = self._campaign_member_id("contact-member")
        campaign_member_lead_id = self._campaign_member_id("lead-member")

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
                lead_id=lead_id,
                event_id=event.id,
                contact_member_id=campaign_member_contact_id,
                lead_member_id=campaign_member_lead_id,
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
            self._hard_negative_records(
                account=account,
                opportunity_id=opportunity.id,
                event_id=event.id,
                case_id=case.id,
                owner=owner,
            )
        )

        return records

    @property
    def _employee_by_id(self) -> dict[str, Employee]:
        return {employee.id: employee for employee in self.enterprise.employees}

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
            text_body="Customer review held with documented attendees and follow-up items.",
            structured_fields={
                "attendance_count": len(attendee_ids),
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
        lead_id: str,
        event_id: str,
        contact_member_id: str,
        lead_member_id: str,
    ) -> list[SalesforceRecord]:
        return [
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("campaign-member-contact"),
                object_type=SalesforceObjectType.CAMPAIGN_MEMBER,
                record_id=contact_member_id,
                timestamp=self.context.date_range.start_at,
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
            ),
        ]

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
                subject="Post-meeting notes",
                text_body=(
                    "Customer asked for revised timing and named owners before "
                    "renewal planning."
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
                text_body="Remember to rename the shared folder before archiving older files.",
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
            template_count=4,
            ratio=self.context.hard_negative_ratio,
        )
        templates = [
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("hard-negative-review-timing"),
                object_type=SalesforceObjectType.TASK,
                record_id=self._task_id("hard-negative-review-timing"),
                timestamp=self.context.date_range.start_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                event_id=event_id,
                subject="Discuss review logistics next week",
                text_body=(
                    "Move the customer review discussion to next week after hotel "
                    "rebooking. No attendee status changed."
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
                        explanation=(
                            "Review timing is mentioned without attendance or "
                            "follow-up action."
                        ),
                        strength=ProvenanceStrength.WEAK,
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event_id,
                    explanation=(
                        "Review timing is mentioned without attendance or "
                        "follow-up action."
                    ),
                    strength=ProvenanceStrength.WEAK,
                ),
                source_system="salesforce",
            ),
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("hard-negative-budget-note"),
                object_type=SalesforceObjectType.NOTE,
                record_id=self._note_id("hard-negative-budget-note"),
                timestamp=self.context.date_range.start_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                opportunity_id=opportunity_id,
                parent_record_id=opportunity_id,
                subject="Budget note for review logistics",
                text_body=(
                    "Budget line covers room block and catering for the review, not "
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
                        explanation=(
                            "Budget wording is logistical and not evidence of "
                            "buying intent."
                        ),
                        strength=ProvenanceStrength.WEAK,
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.OPPORTUNITY,
                    object_id=opportunity_id,
                    explanation="Budget wording is logistical and not evidence of buying intent.",
                    strength=ProvenanceStrength.WEAK,
                ),
                source_system="salesforce",
            ),
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("hard-negative-travel-note"),
                object_type=SalesforceObjectType.NOTE,
                record_id=self._note_id("hard-negative-travel-note"),
                timestamp=self.context.date_range.start_at,
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
                        explanation=(
                            "Complaint concerns travel conditions rather than "
                            "customer pain."
                        ),
                        strength=ProvenanceStrength.WEAK,
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.EVENT,
                    object_id=event_id,
                    explanation="Complaint concerns travel conditions rather than customer pain.",
                    strength=ProvenanceStrength.WEAK,
                ),
                source_system="salesforce",
            ),
            SalesforceRecord(
                salesforce_record_id=self._sf_record_id("hard-negative-export-fwd"),
                object_type=SalesforceObjectType.TASK,
                record_id=self._task_id("hard-negative-export-fwd"),
                timestamp=self.context.date_range.start_at,
                owner_employee_id=owner.id,
                account_id=account.id,
                case_id=case_id,
                subject="Forward export FAQ for prep",
                text_body=(
                    "Forward the prior export FAQ to the internal prep folder. "
                    "No case action is required."
                ),
                structured_fields={"status": "Completed", "priority": "Low"},
                primary_category=CommunicationCategory.FYI_FORWARD,
                is_relevant=False,
                relevance_reason=build_relevance_reason(
                    primary_category=CommunicationCategory.FYI_FORWARD,
                    is_relevant=False,
                    provenance=self._provenance(
                        object_type=ProvenanceObjectType.TICKET,
                        object_id=case_id,
                        explanation=(
                            "Export keyword appears in reference material without "
                            "an active blocker."
                        ),
                        strength=ProvenanceStrength.WEAK,
                    ),
                ),
                provenance=self._provenance(
                    object_type=ProvenanceObjectType.TICKET,
                    object_id=case_id,
                    explanation=(
                        "Export keyword appears in reference material without an "
                        "active blocker."
                    ),
                    strength=ProvenanceStrength.WEAK,
                ),
                source_system="salesforce",
            ),
        ]

        return templates[:total]

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
