from __future__ import annotations

import pytest

from synthetic_enterprise.domain.scenarios import EventScenarioResolver
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext


def build_resolver() -> EventScenarioResolver:
    context = GeneratorContext(
        seed=7070,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
        ),
    )
    enterprise = CompanyBuilder().build(context)
    return EventScenarioResolver(enterprise=enterprise)


def test_primary_account_event_context_links_related_entities() -> None:
    resolver = build_resolver()
    scenario = resolver.primary_account_event()

    assert scenario.event.account_id == scenario.account.id
    assert scenario.primary_contact.account_id == scenario.account.id
    assert scenario.account_owner.id == scenario.account.owner_employee_id
    assert all(contact.account_id == scenario.account.id for contact in scenario.account_contacts)
    assert all(contact.account_id == scenario.account.id for contact in scenario.attendee_contacts)

    if scenario.opportunity is not None:
        assert scenario.opportunity.account_id == scenario.account.id
    if scenario.ticket is not None:
        assert scenario.ticket.account_id == scenario.account.id

    assert resolver.for_event(scenario.event.id) == scenario


def test_support_owner_falls_back_to_organizer_when_ticket_has_no_owner() -> None:
    resolver = build_resolver()
    baseline = resolver.primary_account_event()
    assert baseline.ticket is not None

    enterprise_without_ticket_owner = resolver.enterprise.model_copy(
        update={
            "ticket_issues": [
                ticket.model_copy(update={"owner_employee_id": None})
                if ticket.id == baseline.ticket.id
                else ticket
                for ticket in resolver.enterprise.ticket_issues
            ]
        }
    )

    scenario = EventScenarioResolver(
        enterprise=enterprise_without_ticket_owner
    ).primary_account_event()

    assert scenario.ticket is not None
    assert scenario.ticket.owner_employee_id is None
    assert scenario.support_owner.id == scenario.organizer.id


def test_account_linked_event_is_required() -> None:
    resolver = build_resolver()
    baseline = resolver.primary_account_event()
    enterprise_without_account_link = resolver.enterprise.model_copy(
        update={
            "events": [
                event.model_copy(update={"account_id": None})
                if event.id == baseline.event.id
                else event
                for event in resolver.enterprise.events
            ]
        }
    )

    with pytest.raises(ValueError, match="account-linked event"):
        EventScenarioResolver(enterprise=enterprise_without_account_link).for_event(
            baseline.event.id
        )
