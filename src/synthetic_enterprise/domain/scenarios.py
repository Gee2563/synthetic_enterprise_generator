from __future__ import annotations

from dataclasses import dataclass

from synthetic_enterprise.domain.accounts import CustomerAccount
from synthetic_enterprise.domain.company import Company
from synthetic_enterprise.domain.enterprise import EnterpriseGraph
from synthetic_enterprise.domain.events import Event
from synthetic_enterprise.domain.issues import TicketIssue
from synthetic_enterprise.domain.opportunities import Opportunity
from synthetic_enterprise.domain.people import Contact, Employee


@dataclass(frozen=True, slots=True)
class ScenarioRule:
    """Placeholder scenario domain rule."""

    kind: str


@dataclass(frozen=True, slots=True)
class EventScenarioContext:
    """Resolved business context for one account-linked event."""

    company: Company
    account: CustomerAccount
    event: Event
    organizer: Employee
    account_owner: Employee
    support_owner: Employee
    primary_contact: Contact
    account_contacts: tuple[Contact, ...]
    attendee_contacts: tuple[Contact, ...]
    opportunity: Opportunity | None
    ticket: TicketIssue | None


@dataclass(frozen=True, slots=True)
class EventScenarioResolver:
    """Resolve reusable business context from the enterprise graph."""

    enterprise: EnterpriseGraph

    def primary_account_event(self) -> EventScenarioContext:
        if not self.enterprise.customer_accounts:
            raise ValueError("enterprise must include at least one customer account")

        account = self.enterprise.customer_accounts[0]
        event = next(
            (
                current_event
                for current_event in self.enterprise.events
                if current_event.account_id == account.id
            ),
            None,
        )
        if event is None:
            raise ValueError("enterprise must include an account-linked event")
        return self.for_event(event.id)

    def for_event(self, event_id: str) -> EventScenarioContext:
        event = next(
            (
                current_event
                for current_event in self.enterprise.events
                if current_event.id == event_id
            ),
            None,
        )
        if event is None:
            raise ValueError(f"unknown event id {event_id!r}")
        if event.account_id is None:
            raise ValueError("event context requires an account-linked event")

        account = next(
            (
                current_account
                for current_account in self.enterprise.customer_accounts
                if current_account.id == event.account_id
            ),
            None,
        )
        if account is None:
            raise ValueError(f"unknown account id {event.account_id!r} for event {event.id!r}")

        company = next(
            (
                current_company
                for current_company in self.enterprise.companies
                if current_company.id == account.company_id
            ),
            None,
        )
        if company is None:
            raise ValueError(
                f"unknown company id {account.company_id!r} for account {account.id!r}"
            )

        organizer = _employee_by_id(
            employees=self.enterprise.employees,
            employee_id=event.organizer_employee_id,
        )
        account_owner = _employee_by_id(
            employees=self.enterprise.employees,
            employee_id=account.owner_employee_id,
        )
        account_contacts = tuple(
            contact
            for contact in self.enterprise.contacts
            if contact.account_id == account.id
        )
        if not account_contacts:
            raise ValueError(f"account {account.id!r} has no contacts")

        attendee_contacts = tuple(
            _contact_by_id(
                contacts=self.enterprise.contacts,
                contact_id=contact_id,
            )
            for contact_id in event.attendee_contact_ids
        )
        primary_contact = account_contacts[0]
        opportunity = next(
            (
                current_opportunity
                for current_opportunity in self.enterprise.opportunities
                if current_opportunity.account_id == account.id
            ),
            None,
        )
        ticket = next(
            (
                current_ticket
                for current_ticket in self.enterprise.ticket_issues
                if current_ticket.account_id == account.id
            ),
            None,
        )
        support_owner = organizer
        if ticket is not None and ticket.owner_employee_id is not None:
            support_owner = _employee_by_id(
                employees=self.enterprise.employees,
                employee_id=ticket.owner_employee_id,
            )

        return EventScenarioContext(
            company=company,
            account=account,
            event=event,
            organizer=organizer,
            account_owner=account_owner,
            support_owner=support_owner,
            primary_contact=primary_contact,
            account_contacts=account_contacts,
            attendee_contacts=attendee_contacts,
            opportunity=opportunity,
            ticket=ticket,
        )


def _employee_by_id(*, employees: list[Employee], employee_id: str) -> Employee:
    employee = next(
        (candidate for candidate in employees if candidate.id == employee_id),
        None,
    )
    if employee is None:
        raise ValueError(f"unknown employee id {employee_id!r}")
    return employee


def _contact_by_id(*, contacts: list[Contact], contact_id: str) -> Contact:
    contact = next(
        (candidate for candidate in contacts if candidate.id == contact_id),
        None,
    )
    if contact is None:
        raise ValueError(f"unknown contact id {contact_id!r}")
    return contact
