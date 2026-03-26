from __future__ import annotations

from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext


def build_graph() -> tuple[GeneratorContext, CompanyBuilder]:
    context = GeneratorContext(
        seed=2026,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
        ),
    )
    builder = CompanyBuilder()
    return context, builder


def test_builder_generates_minimal_enterprise_state() -> None:
    context, builder = build_graph()
    graph = builder.build(context)

    assert len(graph.companies) == 1
    assert len(graph.departments) >= 2
    assert len(graph.employees) >= 6
    assert len(graph.customer_accounts) >= 2
    assert len(graph.contacts) >= len(graph.customer_accounts)
    assert len(graph.products) >= 1
    assert len(graph.events) >= 1
    assert len(graph.opportunities) >= 1
    assert len(graph.ticket_issues) >= 1


def test_generated_ids_are_unique() -> None:
    context, builder = build_graph()
    graph = builder.build(context)

    entity_ids = [entity.id for entity in graph.companies]
    entity_ids.extend(entity.id for entity in graph.departments)
    entity_ids.extend(entity.id for entity in graph.employees)
    entity_ids.extend(entity.id for entity in graph.customer_accounts)
    entity_ids.extend(entity.id for entity in graph.contacts)
    entity_ids.extend(entity.id for entity in graph.products)
    entity_ids.extend(entity.id for entity in graph.events)
    entity_ids.extend(entity.id for entity in graph.opportunities)
    entity_ids.extend(entity.id for entity in graph.ticket_issues)

    assert len(entity_ids) == len(set(entity_ids))


def test_all_employees_belong_to_departments() -> None:
    context, builder = build_graph()
    graph = builder.build(context)
    department_ids = {department.id for department in graph.departments}

    assert all(employee.department_id in department_ids for employee in graph.employees)


def test_managers_exist_where_expected() -> None:
    context, builder = build_graph()
    graph = builder.build(context)
    employee_ids = {employee.id for employee in graph.employees}

    executives = {"Chief Executive Officer"}

    for employee in graph.employees:
        if employee.title in executives:
            assert employee.manager_employee_id is None
        else:
            assert employee.manager_employee_id in employee_ids


def test_opportunities_are_linked_to_valid_accounts() -> None:
    context, builder = build_graph()
    graph = builder.build(context)
    account_ids = {account.id for account in graph.customer_accounts}

    assert graph.opportunities
    assert all(opportunity.account_id in account_ids for opportunity in graph.opportunities)


def test_contacts_belong_to_valid_accounts() -> None:
    context, builder = build_graph()
    graph = builder.build(context)
    account_ids = {account.id for account in graph.customer_accounts}

    assert all(contact.account_id in account_ids for contact in graph.contacts)


def test_events_have_plausible_attendee_pools() -> None:
    context, builder = build_graph()
    graph = builder.build(context)

    contacts_by_account = {
        account.id: {contact.id for contact in graph.contacts if contact.account_id == account.id}
        for account in graph.customer_accounts
    }

    assert graph.events

    for event in graph.events:
        if event.account_id is None:
            continue
        valid_attendees = contacts_by_account[event.account_id]
        assert event.attendee_contact_ids
        assert set(event.attendee_contact_ids).issubset(valid_attendees)


def test_tickets_map_to_accounts_and_products_when_present() -> None:
    context, builder = build_graph()
    graph = builder.build(context)
    account_ids = {account.id for account in graph.customer_accounts}
    product_ids = {product.id for product in graph.products}

    assert graph.ticket_issues

    for ticket in graph.ticket_issues:
        assert ticket.account_id in account_ids
        if ticket.related_product_id is not None:
            assert ticket.related_product_id in product_ids
