from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from synthetic_enterprise.domain import (
    Campaign,
    Company,
    Contact,
    CRMActivity,
    CustomerAccount,
    Department,
    Employee,
    EnterpriseGraph,
    Event,
    MessageEnvelope,
    Opportunity,
    Product,
    TicketIssue,
)


def build_company_graph() -> EnterpriseGraph:
    company = Company(
        id="company_001",
        seed=101,
        name="Northwind Systems",
        domain="northwind.example",
        timezone="UTC",
        created_at=datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc),
    )
    department = Department(
        id="department_001",
        seed=102,
        company_id=company.id,
        name="Revenue",
        created_at=datetime(2026, 1, 10, 9, 5, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 10, 9, 5, tzinfo=timezone.utc),
    )
    employee = Employee(
        id="employee_001",
        seed=103,
        company_id=company.id,
        department_id=department.id,
        email="owner@northwind.example",
        first_name="Avery",
        last_name="Stone",
        title="Account Executive",
        created_at=datetime(2026, 1, 10, 9, 10, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 10, 9, 10, tzinfo=timezone.utc),
    )
    account = CustomerAccount(
        id="account_001",
        seed=104,
        company_id=company.id,
        name="Contoso Retail",
        industry="Retail",
        owner_employee_id=employee.id,
        created_at=datetime(2026, 1, 11, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 11, 10, 0, tzinfo=timezone.utc),
    )
    contact = Contact(
        id="contact_001",
        seed=105,
        account_id=account.id,
        first_name="Jordan",
        last_name="Lee",
        email="jordan.lee@contoso.example",
        title="VP Operations",
        created_at=datetime(2026, 1, 11, 10, 5, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 11, 10, 5, tzinfo=timezone.utc),
    )
    product = Product(
        id="product_001",
        seed=106,
        company_id=company.id,
        name="Signal Cloud",
        sku="SIG-CLOUD",
        family="platform",
        created_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
    )
    opportunity = Opportunity(
        id="opportunity_001",
        seed=107,
        account_id=account.id,
        owner_employee_id=employee.id,
        primary_contact_id=contact.id,
        product_ids=[product.id],
        stage="evaluation",
        amount=125000.0,
        created_at=datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc),
    )
    event = Event(
        id="event_001",
        seed=108,
        company_id=company.id,
        account_id=account.id,
        organizer_employee_id=employee.id,
        attendee_contact_ids=[contact.id],
        title="Contoso Platform Workshop",
        event_type="customer_meeting",
        starts_at=datetime(2026, 1, 20, 15, 0, tzinfo=timezone.utc),
        ends_at=datetime(2026, 1, 20, 16, 0, tzinfo=timezone.utc),
        created_at=datetime(2026, 1, 12, 8, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 12, 8, 0, tzinfo=timezone.utc),
    )
    issue = TicketIssue(
        id="issue_001",
        seed=109,
        account_id=account.id,
        opened_by_contact_id=contact.id,
        owner_employee_id=employee.id,
        related_product_id=product.id,
        status="open",
        severity="high",
        summary="Dashboard export is timing out",
        created_at=datetime(2026, 1, 18, 11, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 18, 11, 0, tzinfo=timezone.utc),
    )
    campaign = Campaign(
        id="campaign_001",
        seed=110,
        company_id=company.id,
        owner_employee_id=employee.id,
        name="Q1 Expansion",
        channel="email",
        target_account_ids=[account.id],
        product_ids=[product.id],
        created_at=datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc),
    )
    message = MessageEnvelope(
        id="message_001",
        seed=111,
        company_id=company.id,
        source="email",
        subject="Need resolution before renewal",
        body_text="We need a plan for the timeout issue before renewal.",
        sender_employee_id=employee.id,
        recipient_contact_ids=[contact.id],
        related_account_id=account.id,
        related_opportunity_id=opportunity.id,
        sent_at=datetime(2026, 1, 18, 12, 30, tzinfo=timezone.utc),
        created_at=datetime(2026, 1, 18, 12, 30, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 18, 12, 30, tzinfo=timezone.utc),
    )
    activity = CRMActivity(
        id="activity_001",
        seed=112,
        company_id=company.id,
        account_id=account.id,
        employee_id=employee.id,
        contact_id=contact.id,
        opportunity_id=opportunity.id,
        campaign_id=campaign.id,
        message_id=message.id,
        activity_type="follow_up",
        occurred_at=datetime(2026, 1, 18, 13, 0, tzinfo=timezone.utc),
        created_at=datetime(2026, 1, 18, 13, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 18, 13, 0, tzinfo=timezone.utc),
    )

    return EnterpriseGraph(
        companies=[company],
        departments=[department],
        employees=[employee],
        customer_accounts=[account],
        contacts=[contact],
        opportunities=[opportunity],
        events=[event],
        products=[product],
        ticket_issues=[issue],
        campaigns=[campaign],
        message_envelopes=[message],
        crm_activities=[activity],
    )


def test_schema_validation_accepts_valid_enterprise_graph() -> None:
    graph = build_company_graph()

    assert graph.companies[0].timezone == "UTC"
    assert graph.events[0].starts_at.tzinfo is not None


def test_relationship_integrity_is_enforced() -> None:
    valid_graph = build_company_graph()
    broken_message = valid_graph.message_envelopes[0].model_copy(
        update={"related_account_id": "account_missing"}
    )

    with pytest.raises(ValidationError):
        EnterpriseGraph(
            companies=valid_graph.companies,
            departments=valid_graph.departments,
            employees=valid_graph.employees,
            customer_accounts=valid_graph.customer_accounts,
            contacts=valid_graph.contacts,
            opportunities=valid_graph.opportunities,
            events=valid_graph.events,
            products=valid_graph.products,
            ticket_issues=valid_graph.ticket_issues,
            campaigns=valid_graph.campaigns,
            message_envelopes=[broken_message],
            crm_activities=valid_graph.crm_activities,
        )


def test_serialization_round_trip_preserves_entity_data() -> None:
    graph = build_company_graph()
    payload = graph.model_dump(mode="json")
    restored = EnterpriseGraph.model_validate(payload)
    json_payload = graph.model_dump_json()
    restored_from_json = EnterpriseGraph.model_validate_json(json_payload)

    assert restored == graph
    assert restored_from_json == graph


def test_invalid_records_raise_validation_errors() -> None:
    with pytest.raises(ValidationError):
        Company(
            id="company_001",
            seed=1,
            name="",
            domain="invalid-domain",
            timezone="UTC",
            created_at=datetime(2026, 1, 10, 9, 0),
            updated_at=datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc),
        )

    with pytest.raises(ValidationError):
        Employee(
            id="employee_001",
            seed=1,
            company_id="company_001",
            department_id="department_001",
            email="not-an-email",
            first_name="Avery",
            last_name="Stone",
            title="AE",
            created_at=datetime(2026, 1, 10, 9, 10, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 10, 9, 10, tzinfo=timezone.utc),
        )


def test_seeded_generation_is_deterministic() -> None:
    first = Product.from_seed(
        seed=500,
        company_id="company_001",
        name="Signal Cloud",
        sku="SIG-CLOUD",
        family="platform",
    )
    second = Product.from_seed(
        seed=500,
        company_id="company_001",
        name="Signal Cloud",
        sku="SIG-CLOUD",
        family="platform",
    )

    assert first.id == second.id
    assert first.created_at == second.created_at
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
