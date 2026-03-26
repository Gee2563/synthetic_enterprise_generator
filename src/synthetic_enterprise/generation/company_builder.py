from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from synthetic_enterprise.domain import (
    Company,
    Contact,
    CustomerAccount,
    Department,
    Employee,
    EnterpriseGraph,
    Event,
    Opportunity,
    Product,
    TicketIssue,
)
from synthetic_enterprise.generation.context import GeneratorContext


@dataclass(frozen=True, slots=True)
class _RoleBlueprint:
    title: str
    department_key: str
    manager_title: str | None


ROLE_BLUEPRINTS: tuple[_RoleBlueprint, ...] = (
    _RoleBlueprint("Chief Executive Officer", "leadership", None),
    _RoleBlueprint("VP Sales", "sales", "Chief Executive Officer"),
    _RoleBlueprint("Account Executive", "sales", "VP Sales"),
    _RoleBlueprint("VP Customer Success", "customer_success", "Chief Executive Officer"),
    _RoleBlueprint("Customer Success Manager", "customer_success", "VP Customer Success"),
    _RoleBlueprint("Support Engineer", "support", "VP Customer Success"),
)

EXTRA_ROLE_BLUEPRINTS: tuple[_RoleBlueprint, ...] = (
    _RoleBlueprint("Account Executive", "sales", "VP Sales"),
    _RoleBlueprint("Customer Success Manager", "customer_success", "VP Customer Success"),
    _RoleBlueprint("Support Engineer", "support", "VP Customer Success"),
)

DEPARTMENT_NAMES: dict[str, str] = {
    "leadership": "Leadership",
    "sales": "Sales",
    "customer_success": "Customer Success",
    "support": "Support",
}

DEPARTMENT_PARENTS: dict[str, str | None] = {
    "leadership": None,
    "sales": "leadership",
    "customer_success": "leadership",
    "support": "customer_success",
}

PRODUCT_CATALOG: tuple[tuple[str, str, str], ...] = (
    ("Signal Cloud", "SIG-CLOUD", "platform"),
    ("Assist Desk", "AST-DESK", "support"),
)

ACCOUNT_INDUSTRIES: tuple[str, ...] = (
    "Retail",
    "Healthcare",
    "Financial Services",
    "Manufacturing",
)

ISSUE_SUMMARIES: tuple[str, ...] = (
    "Dashboard exports are timing out during monthly reporting.",
    "Users need faster resolution on integration errors.",
)


class CompanyBuilder:
    """Builds the smallest plausible deterministic enterprise state."""

    def build(
        self,
        context: GeneratorContext,
        *,
        account_count: int = 2,
    ) -> EnterpriseGraph:
        if account_count <= 0:
            raise ValueError("account_count must be greater than zero")

        employee_count = max(context.company_size.min_employees, len(ROLE_BLUEPRINTS))
        if employee_count > context.company_size.max_employees:
            raise ValueError("company_size config does not allow the minimum simulator role set")

        company = self._build_company(context)
        departments = self._build_departments(context, company.id)
        employees = self._build_employees(context, company, departments, employee_count)
        departments = self._assign_department_leaders(departments, employees)
        products = self._build_products(context, company.id)
        accounts = self._build_accounts(
            context,
            company.id,
            employees,
            account_count=account_count,
        )
        contacts = self._build_contacts(context, accounts)
        opportunities = self._build_opportunities(context, accounts, contacts, products, employees)
        events = self._build_events(context, company.id, accounts, contacts, employees)
        tickets = self._build_tickets(context, accounts, contacts, products, employees)

        return EnterpriseGraph(
            companies=[company],
            departments=departments,
            employees=employees,
            customer_accounts=accounts,
            contacts=contacts,
            opportunities=opportunities,
            events=events,
            products=products,
            ticket_issues=tickets,
        )

    def _build_company(self, context: GeneratorContext) -> Company:
        company_name = f"{context.faker.company()} Systems"
        return Company.from_seed(
            seed=context.derive_seed("company"),
            name=company_name,
            domain=f"{_slugify(company_name)}.example",
            timezone=context.timezone,
        )

    def _build_departments(self, context: GeneratorContext, company_id: str) -> list[Department]:
        departments: list[Department] = []

        for department_key, department_name in DEPARTMENT_NAMES.items():
            parent_key = DEPARTMENT_PARENTS[department_key]
            parent_department_id = None
            if parent_key is not None:
                parent_department_id = next(
                    department.id
                    for department in departments
                    if department.name == DEPARTMENT_NAMES[parent_key]
                )

            departments.append(
                Department.from_seed(
                    seed=context.derive_seed(f"department:{department_key}"),
                    company_id=company_id,
                    name=department_name,
                    parent_department_id=parent_department_id,
                )
            )

        return departments

    def _build_employees(
        self,
        context: GeneratorContext,
        company: Company,
        departments: list[Department],
        employee_count: int,
    ) -> list[Employee]:
        employees: list[Employee] = []
        department_id_by_key = {
            department_key: next(
                department.id
                for department in departments
                if department.name == DEPARTMENT_NAMES[department_key]
            )
            for department_key in DEPARTMENT_NAMES
        }
        employee_id_by_title: dict[str, str] = {}
        used_emails: set[str] = set()
        blueprints = list(ROLE_BLUEPRINTS)

        while len(blueprints) < employee_count:
            blueprint_index = (len(blueprints) - len(ROLE_BLUEPRINTS)) % len(
                EXTRA_ROLE_BLUEPRINTS
            )
            blueprints.append(EXTRA_ROLE_BLUEPRINTS[blueprint_index])

        for index, blueprint in enumerate(blueprints):
            first_name = context.faker.first_name()
            last_name = context.faker.last_name()
            manager_employee_id = None
            if blueprint.manager_title is not None:
                manager_employee_id = employee_id_by_title[blueprint.manager_title]

            email = _unique_email(
                first_name=first_name,
                last_name=last_name,
                domain=company.domain,
                used_emails=used_emails,
            )
            employee = Employee.from_seed(
                seed=context.derive_seed(f"employee:{index}:{blueprint.title}"),
                company_id=company.id,
                department_id=department_id_by_key[blueprint.department_key],
                email=email,
                first_name=first_name,
                last_name=last_name,
                title=blueprint.title,
                manager_employee_id=manager_employee_id,
            )
            employees.append(employee)
            employee_id_by_title.setdefault(blueprint.title, employee.id)

        return employees

    def _assign_department_leaders(
        self,
        departments: list[Department],
        employees: list[Employee],
    ) -> list[Department]:
        first_employee_by_title = {
            title: _employee_by_title(employees, title)
            for title in {
                "Chief Executive Officer",
                "VP Sales",
                "VP Customer Success",
                "Support Engineer",
            }
        }
        leader_by_department = {
            "Leadership": first_employee_by_title["Chief Executive Officer"].id,
            "Sales": first_employee_by_title["VP Sales"].id,
            "Customer Success": first_employee_by_title["VP Customer Success"].id,
            "Support": first_employee_by_title["Support Engineer"].id,
        }

        return [
            department.model_copy(
                update={"leader_employee_id": leader_by_department[department.name]}
            )
            for department in departments
        ]

    def _build_products(self, context: GeneratorContext, company_id: str) -> list[Product]:
        products: list[Product] = []

        for index, (name, sku, family) in enumerate(PRODUCT_CATALOG):
            products.append(
                Product.from_seed(
                    seed=context.derive_seed(f"product:{index}"),
                    company_id=company_id,
                    name=name,
                    sku=sku,
                    family=family,
                )
            )

        return products

    def _build_accounts(
        self,
        context: GeneratorContext,
        company_id: str,
        employees: list[Employee],
        *,
        account_count: int,
    ) -> list[CustomerAccount]:
        account_owner = _employee_by_title(employees, "Account Executive")
        accounts: list[CustomerAccount] = []

        for index in range(account_count):
            account_name = context.faker.company()
            accounts.append(
                CustomerAccount.from_seed(
                    seed=context.derive_seed(f"account:{index}"),
                    company_id=company_id,
                    name=account_name,
                    industry=ACCOUNT_INDUSTRIES[index % len(ACCOUNT_INDUSTRIES)],
                    owner_employee_id=account_owner.id,
                )
            )

        return accounts

    def _build_contacts(
        self,
        context: GeneratorContext,
        accounts: list[CustomerAccount],
    ) -> list[Contact]:
        contacts: list[Contact] = []
        titles = ("VP Operations", "Director of IT")

        for account_index, account in enumerate(accounts):
            account_domain = f"{_slugify(account.name)}.example"

            for contact_index, title in enumerate(titles):
                first_name = context.faker.first_name()
                last_name = context.faker.last_name()
                contacts.append(
                    Contact.from_seed(
                        seed=context.derive_seed(f"contact:{account_index}:{contact_index}"),
                        account_id=account.id,
                        first_name=first_name,
                        last_name=last_name,
                        email=f"{_email_local_part(first_name, last_name)}@{account_domain}",
                        title=title,
                    )
                )

        return contacts

    def _build_opportunities(
        self,
        context: GeneratorContext,
        accounts: list[CustomerAccount],
        contacts: list[Contact],
        products: list[Product],
        employees: list[Employee],
    ) -> list[Opportunity]:
        account_owner = _employee_by_title(employees, "Account Executive")
        contacts_by_account = _group_contacts_by_account(contacts)
        opportunities: list[Opportunity] = []
        stages = ("evaluation", "proposal")

        for index, account in enumerate(accounts):
            primary_contact = contacts_by_account[account.id][0]
            product = products[index % len(products)]
            amount = float(50000 + (index * 25000))
            opportunities.append(
                Opportunity.from_seed(
                    seed=context.derive_seed(f"opportunity:{index}"),
                    account_id=account.id,
                    owner_employee_id=account_owner.id,
                    primary_contact_id=primary_contact.id,
                    product_ids=[product.id],
                    stage=stages[index % len(stages)],
                    amount=amount,
                )
            )

        return opportunities

    def _build_events(
        self,
        context: GeneratorContext,
        company_id: str,
        accounts: list[CustomerAccount],
        contacts: list[Contact],
        employees: list[Employee],
    ) -> list[Event]:
        contacts_by_account = _group_contacts_by_account(contacts)
        organizer = _employee_by_title(employees, "Customer Success Manager")
        events: list[Event] = []

        for index, account in enumerate(accounts):
            starts_at, ends_at = _event_window(context)
            attendees = contacts_by_account[account.id][:2]
            events.append(
                Event.from_seed(
                    seed=context.derive_seed(f"event:{index}"),
                    company_id=company_id,
                    account_id=account.id,
                    organizer_employee_id=organizer.id,
                    attendee_contact_ids=[contact.id for contact in attendees],
                    title=f"Quarterly review with {account.name}",
                    event_type="customer_meeting",
                    starts_at=starts_at,
                    ends_at=ends_at,
                )
            )

        return events

    def _build_tickets(
        self,
        context: GeneratorContext,
        accounts: list[CustomerAccount],
        contacts: list[Contact],
        products: list[Product],
        employees: list[Employee],
    ) -> list[TicketIssue]:
        support_owner = _employee_by_title(employees, "Support Engineer")
        contacts_by_account = _group_contacts_by_account(contacts)
        tickets: list[TicketIssue] = []

        for index, account in enumerate(accounts):
            related_product_id = None
            if index == 0:
                related_product_id = products[0].id

            tickets.append(
                TicketIssue.from_seed(
                    seed=context.derive_seed(f"ticket:{index}"),
                    account_id=account.id,
                    opened_by_contact_id=contacts_by_account[account.id][0].id,
                    owner_employee_id=support_owner.id,
                    related_product_id=related_product_id,
                    status="open" if index == 0 else "in_progress",
                    severity="high" if index == 0 else "medium",
                    summary=ISSUE_SUMMARIES[index % len(ISSUE_SUMMARIES)],
                )
            )

        return tickets


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "enterprise"


def _email_local_part(first_name: str, last_name: str) -> str:
    return f"{_slugify(first_name)}.{_slugify(last_name)}"


def _unique_email(
    first_name: str,
    last_name: str,
    domain: str,
    used_emails: set[str],
) -> str:
    local_part = _email_local_part(first_name, last_name)
    candidate = f"{local_part}@{domain}"
    suffix = 2

    while candidate in used_emails:
        candidate = f"{local_part}{suffix}@{domain}"
        suffix += 1

    used_emails.add(candidate)
    return candidate


def _group_contacts_by_account(contacts: list[Contact]) -> dict[str, list[Contact]]:
    contacts_by_account: dict[str, list[Contact]] = {}

    for contact in contacts:
        contacts_by_account.setdefault(contact.account_id, []).append(contact)

    return contacts_by_account


def _employee_by_title(employees: list[Employee], title: str) -> Employee:
    return next(employee for employee in employees if employee.title == title)


def _event_window(context: GeneratorContext) -> tuple[datetime, datetime]:
    duration = timedelta(minutes=60)
    start_at = context.date_range.start_at
    end_at = context.date_range.end_at
    total_seconds = int((end_at - start_at - duration).total_seconds())

    if total_seconds <= 0:
        event_start = start_at.astimezone(context.config.tzinfo)
        return event_start, (event_start + duration).astimezone(context.config.tzinfo)

    offset_seconds = context.rng.randint(0, total_seconds)
    event_start = (start_at + timedelta(seconds=offset_seconds)).astimezone(context.config.tzinfo)
    return event_start, (event_start + duration).astimezone(context.config.tzinfo)
