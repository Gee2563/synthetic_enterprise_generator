from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeVar

from pydantic import Field, model_validator

from synthetic_enterprise.domain.accounts import CustomerAccount
from synthetic_enterprise.domain.activities import CRMActivity, MessageEnvelope
from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.domain.company import Company, Department
from synthetic_enterprise.domain.events import Campaign, Event
from synthetic_enterprise.domain.issues import TicketIssue
from synthetic_enterprise.domain.opportunities import Opportunity
from synthetic_enterprise.domain.people import Contact, Employee
from synthetic_enterprise.domain.product import Product


class EnterpriseGraph(EnterpriseModel):
    """Validated collection of enterprise entities with explicit relationships."""

    companies: list[Company] = Field(default_factory=list)
    departments: list[Department] = Field(default_factory=list)
    employees: list[Employee] = Field(default_factory=list)
    customer_accounts: list[CustomerAccount] = Field(default_factory=list)
    contacts: list[Contact] = Field(default_factory=list)
    opportunities: list[Opportunity] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    products: list[Product] = Field(default_factory=list)
    ticket_issues: list[TicketIssue] = Field(default_factory=list)
    campaigns: list[Campaign] = Field(default_factory=list)
    message_envelopes: list[MessageEnvelope] = Field(default_factory=list)
    crm_activities: list[CRMActivity] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_relationships(self) -> EnterpriseGraph:
        company_ids = _index(self.companies)
        department_ids = _index(self.departments)
        employee_ids = _index(self.employees)
        account_ids = _index(self.customer_accounts)
        contact_ids = _index(self.contacts)
        opportunity_ids = _index(self.opportunities)
        product_ids = _index(self.products)
        campaign_ids = _index(self.campaigns)
        message_ids = _index(self.message_envelopes)

        _ensure_unique_entity_ids(self)

        for department in self.departments:
            _require_reference("department.company_id", department.company_id, company_ids)
            if department.parent_department_id is not None:
                _require_reference(
                    "department.parent_department_id",
                    department.parent_department_id,
                    department_ids,
                )
            if department.leader_employee_id is not None:
                _require_reference(
                    "department.leader_employee_id",
                    department.leader_employee_id,
                    employee_ids,
                )

        for employee in self.employees:
            _require_reference("employee.company_id", employee.company_id, company_ids)
            _require_reference("employee.department_id", employee.department_id, department_ids)
            if employee.manager_employee_id is not None:
                _require_reference(
                    "employee.manager_employee_id",
                    employee.manager_employee_id,
                    employee_ids,
                )
            department = department_ids[employee.department_id]
            if department.company_id != employee.company_id:
                raise ValueError("employee department must belong to the same company")

        for account in self.customer_accounts:
            _require_reference("customer_account.company_id", account.company_id, company_ids)
            _require_reference(
                "customer_account.owner_employee_id",
                account.owner_employee_id,
                employee_ids,
            )
            owner = employee_ids[account.owner_employee_id]
            if owner.company_id != account.company_id:
                raise ValueError("customer account owner must belong to the owning company")

        for contact in self.contacts:
            _require_reference("contact.account_id", contact.account_id, account_ids)

        for product in self.products:
            _require_reference("product.company_id", product.company_id, company_ids)

        for opportunity in self.opportunities:
            _require_reference("opportunity.account_id", opportunity.account_id, account_ids)
            _require_reference(
                "opportunity.owner_employee_id",
                opportunity.owner_employee_id,
                employee_ids,
            )
            _require_reference(
                "opportunity.primary_contact_id",
                opportunity.primary_contact_id,
                contact_ids,
            )
            _require_many("opportunity.product_ids", opportunity.product_ids, product_ids)

        for event in self.events:
            _require_reference("event.company_id", event.company_id, company_ids)
            if event.account_id is not None:
                _require_reference("event.account_id", event.account_id, account_ids)
            _require_reference(
                "event.organizer_employee_id",
                event.organizer_employee_id,
                employee_ids,
            )
            _require_many("event.attendee_contact_ids", event.attendee_contact_ids, contact_ids)

        for issue in self.ticket_issues:
            _require_reference("ticket_issue.account_id", issue.account_id, account_ids)
            _require_reference(
                "ticket_issue.opened_by_contact_id",
                issue.opened_by_contact_id,
                contact_ids,
            )
            if issue.owner_employee_id is not None:
                _require_reference(
                    "ticket_issue.owner_employee_id",
                    issue.owner_employee_id,
                    employee_ids,
                )
            if issue.related_product_id is not None:
                _require_reference(
                    "ticket_issue.related_product_id",
                    issue.related_product_id,
                    product_ids,
                )

        for campaign in self.campaigns:
            _require_reference("campaign.company_id", campaign.company_id, company_ids)
            _require_reference(
                "campaign.owner_employee_id",
                campaign.owner_employee_id,
                employee_ids,
            )
            _require_many("campaign.target_account_ids", campaign.target_account_ids, account_ids)
            _require_many("campaign.product_ids", campaign.product_ids, product_ids)

        for envelope in self.message_envelopes:
            _require_reference("message_envelope.company_id", envelope.company_id, company_ids)
            if envelope.sender_employee_id is not None:
                _require_reference(
                    "message_envelope.sender_employee_id",
                    envelope.sender_employee_id,
                    employee_ids,
                )
            if envelope.sender_contact_id is not None:
                _require_reference(
                    "message_envelope.sender_contact_id",
                    envelope.sender_contact_id,
                    contact_ids,
                )
            _require_many(
                "message_envelope.recipient_employee_ids",
                envelope.recipient_employee_ids,
                employee_ids,
            )
            _require_many(
                "message_envelope.recipient_contact_ids",
                envelope.recipient_contact_ids,
                contact_ids,
            )
            if envelope.related_account_id is not None:
                _require_reference(
                    "message_envelope.related_account_id",
                    envelope.related_account_id,
                    account_ids,
                )
            if envelope.related_opportunity_id is not None:
                _require_reference(
                    "message_envelope.related_opportunity_id",
                    envelope.related_opportunity_id,
                    opportunity_ids,
                )

        for activity in self.crm_activities:
            _require_reference("crm_activity.company_id", activity.company_id, company_ids)
            _require_reference("crm_activity.account_id", activity.account_id, account_ids)
            _require_reference("crm_activity.employee_id", activity.employee_id, employee_ids)
            if activity.contact_id is not None:
                _require_reference("crm_activity.contact_id", activity.contact_id, contact_ids)
            if activity.opportunity_id is not None:
                _require_reference(
                    "crm_activity.opportunity_id",
                    activity.opportunity_id,
                    opportunity_ids,
                )
            if activity.campaign_id is not None:
                _require_reference("crm_activity.campaign_id", activity.campaign_id, campaign_ids)
            if activity.message_id is not None:
                _require_reference("crm_activity.message_id", activity.message_id, message_ids)

        return self


class HasId(Protocol):
    id: str


EntityT = TypeVar("EntityT", bound=HasId)


def _index(entities: Sequence[EntityT]) -> dict[str, EntityT]:
    return {getattr(entity, "id"): entity for entity in entities}


def _require_reference(field_name: str, reference_id: str, entities: Mapping[str, object]) -> None:
    if reference_id not in entities:
        raise ValueError(f"{field_name} references unknown id {reference_id!r}")


def _require_many(
    field_name: str,
    reference_ids: list[str],
    entities: Mapping[str, object],
) -> None:
    for reference_id in reference_ids:
        _require_reference(field_name, reference_id, entities)


def _ensure_unique_entity_ids(graph: EnterpriseGraph) -> None:
    entity_groups: tuple[Sequence[HasId], ...] = (
        graph.companies,
        graph.departments,
        graph.employees,
        graph.customer_accounts,
        graph.contacts,
        graph.opportunities,
        graph.events,
        graph.products,
        graph.ticket_issues,
        graph.campaigns,
        graph.message_envelopes,
        graph.crm_activities,
    )
    all_ids = [
        entity.id
        for entity_group in entity_groups
        for entity in entity_group
    ]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("entity ids must be globally unique within an enterprise graph")
