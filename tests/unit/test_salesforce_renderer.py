from __future__ import annotations

import pandas as pd

from synthetic_enterprise.contracts.records.salesforce import (
    SalesforceObjectType,
    SalesforceRecord,
)
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.sources.salesforce.renderer import SalesforceRenderer


def build_renderer(noise_ratio: float = 0.8, seed: int = 6060) -> SalesforceRenderer:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
            noise_ratio=noise_ratio,
        ),
    )
    graph = CompanyBuilder().build(context)
    return SalesforceRenderer(context=context, enterprise=graph)


def test_object_relationships_are_valid() -> None:
    renderer = build_renderer()
    records = renderer.generate_records()
    ids_by_type = _ids_by_type(records)

    for record in records:
        if record.account_id is not None:
            assert record.account_id in ids_by_type[SalesforceObjectType.ACCOUNT]
        if record.contact_id is not None:
            assert record.contact_id in ids_by_type[SalesforceObjectType.CONTACT]
        if record.lead_id is not None:
            assert record.lead_id in ids_by_type[SalesforceObjectType.LEAD]
        if record.opportunity_id is not None:
            assert record.opportunity_id in ids_by_type[SalesforceObjectType.OPPORTUNITY]
        if record.event_id is not None:
            assert record.event_id in ids_by_type[SalesforceObjectType.EVENT]
        if record.case_id is not None:
            assert record.case_id in ids_by_type[SalesforceObjectType.CASE]
        if record.campaign_id is not None:
            assert record.campaign_id in ids_by_type[SalesforceObjectType.CAMPAIGN]
        if record.parent_record_id is not None:
            assert record.parent_record_id in _all_record_ids(records)


def test_opportunity_stages_are_valid() -> None:
    renderer = build_renderer()
    records = renderer.generate_records()
    stages = {
        record.stage
        for record in records
        if record.object_type == SalesforceObjectType.OPPORTUNITY
    }

    assert stages
    assert stages.issubset(
        {
            "qualification",
            "discovery",
            "evaluation",
            "proposal",
            "closed_won",
            "closed_lost",
        }
    )


def test_account_contact_integrity_holds() -> None:
    renderer = build_renderer()
    records = renderer.generate_records()
    account_ids = _ids_by_type(records)[SalesforceObjectType.ACCOUNT]
    contact_records = [
        record for record in records if record.object_type == SalesforceObjectType.CONTACT
    ]

    assert contact_records
    assert all(record.account_id in account_ids for record in contact_records)


def test_campaign_members_map_to_contacts_or_leads() -> None:
    renderer = build_renderer()
    records = renderer.generate_records()
    campaign_ids = _ids_by_type(records)[SalesforceObjectType.CAMPAIGN]
    contact_ids = _ids_by_type(records)[SalesforceObjectType.CONTACT]
    lead_ids = _ids_by_type(records)[SalesforceObjectType.LEAD]
    member_records = [
        record for record in records if record.object_type == SalesforceObjectType.CAMPAIGN_MEMBER
    ]

    assert member_records
    for record in member_records:
        assert record.campaign_id in campaign_ids
        assert (record.contact_id is None) != (record.lead_id is None)
        if record.contact_id is not None:
            assert record.contact_id in contact_ids
        if record.lead_id is not None:
            assert record.lead_id in lead_ids


def test_event_attendance_can_be_inferred_from_crm_records() -> None:
    renderer = build_renderer()
    records = renderer.generate_records()
    attended_members = [
        record
        for record in records
        if record.object_type == SalesforceObjectType.CAMPAIGN_MEMBER
        and record.structured_fields.get("member_status") == "Attended"
    ]
    event_records = [
        record
        for record in records
        if record.object_type == SalesforceObjectType.EVENT
        and record.attendee_contact_ids
    ]

    assert attended_members
    assert event_records


def test_notes_and_tasks_can_be_relevant_or_noisy() -> None:
    renderer = build_renderer()
    records = renderer.generate_records()
    notes_and_tasks = [
        record
        for record in records
        if record.object_type in {SalesforceObjectType.NOTE, SalesforceObjectType.TASK}
    ]

    assert notes_and_tasks
    assert any(record.is_relevant for record in notes_and_tasks)
    assert any(not record.is_relevant for record in notes_and_tasks)


def test_structured_tables_export_cleanly() -> None:
    renderer = build_renderer()
    rows = [record.to_dataframe_row() for record in renderer.generate_records()]
    frame = pd.DataFrame(rows)

    assert frame["source_system"].eq("salesforce").all()
    assert frame["object_type"].isin(
        {object_type.value for object_type in SalesforceObjectType}
    ).all()
    assert "structured_fields" in frame.columns


def _ids_by_type(records: list[SalesforceRecord]) -> dict[SalesforceObjectType, set[str]]:
    ids: dict[SalesforceObjectType, set[str]] = {
        object_type: set() for object_type in SalesforceObjectType
    }
    for record in records:
        ids[record.object_type].add(record.record_id)
    return ids


def _all_record_ids(records: list[SalesforceRecord]) -> set[str]:
    return {record.record_id for record in records}
