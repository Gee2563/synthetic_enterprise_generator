from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.evaluation.quality_report import DatasetQualityEvaluator
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import DatasetTargets, GenerationPipeline


def build_enterprise() -> EnterpriseGraph:
    context = GeneratorContext(
        seed=20260330,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=6, max_employees=8),
        ),
    )
    return CompanyBuilder().build(context, account_count=2)


def build_toy_rows(enterprise: EnterpriseGraph) -> dict[str, list[dict[str, object]]]:
    account = enterprise.customer_accounts[0]
    opportunity = enterprise.opportunities[0]
    event = enterprise.events[0]
    ticket = enterprise.ticket_issues[0]
    contact = enterprise.contacts[0]
    employee_ids = [employee.id for employee in enterprise.employees[:3]]

    return {
        "email": [
            {
                "source_system": "email",
                "email_id": "email_a",
                "thread_id": "thread_a",
                "message_index_in_thread": 0,
                "sender_employee_id": employee_ids[0],
                "to": [contact.id],
                "cc": [employee_ids[1]],
                "bcc": [],
                "subject": "Next steps after review",
                "body": "Need owners for the quarterly review and export fix.",
                "account_id": account.id,
                "opportunity_id": opportunity.id,
                "event_id": event.id,
                "ticket_id": ticket.id,
                "primary_category": "follow_up",
                "is_relevant": True,
                "relevance_reason": "toy relevant row",
                "provenance": {
                    "object_type": "event",
                    "object_id": event.id,
                    "strength": "strong",
                    "explanation": "Customer review needs assigned owners.",
                },
            },
            {
                "source_system": "email",
                "email_id": "email_b",
                "thread_id": "thread_a",
                "message_index_in_thread": 1,
                "sender_employee_id": employee_ids[0],
                "to": [contact.id],
                "cc": [],
                "bcc": [],
                "subject": "Let's discuss next week",
                "body": "Moving the review time only.",
                "account_id": account.id,
                "event_id": event.id,
                "primary_category": "scheduling_only",
                "is_relevant": False,
                "relevance_reason": "toy noise row",
                "provenance": None,
            },
        ],
        "slack": [
            {
                "source_system": "slack",
                "slack_message_id": "slack_message_a",
                "channel_id": "channel_a",
                "channel_name": "#watercooler",
                "thread_id": None,
                "parent_message_id": None,
                "sender_employee_id": employee_ids[2],
                "body": "coffee after standup",
                "mentions": [],
                "reactions": [],
                "attachments": [],
                "linked_account_id": None,
                "linked_opportunity_id": None,
                "linked_event_id": None,
                "linked_ticket_id": None,
                "primary_category": "social_chatter",
                "is_relevant": False,
                "relevance_reason": "toy chatter row",
                "provenance": None,
            }
        ],
        "teams": [
            {
                "source_system": "teams",
                "teams_message_id": "teams_message_a",
                "team_id": "team_a",
                "channel_id": "channel_b",
                "chat_or_channel": "channel",
                "thread_id": None,
                "meeting_id": "meeting_a",
                "sender_employee_id": employee_ids[1],
                "body": "Budget review covers room block, not customer spend.",
                "mentions": [employee_ids[0]],
                "file_refs": ["budget.xlsx"],
                "linked_account_id": account.id,
                "linked_opportunity_id": opportunity.id,
                "linked_event_id": event.id,
                "linked_ticket_id": None,
                "primary_category": "status_updates",
                "is_relevant": False,
                "relevance_reason": "toy hard negative row",
                "provenance": {
                    "object_type": "opportunity",
                    "object_id": opportunity.id,
                    "strength": "weak",
                    "explanation": "Budget wording is logistical rather than commercial.",
                },
            }
        ],
        "salesforce": [
            {
                "source_system": "salesforce",
                "salesforce_record_id": "sf_record_a",
                "object_type": "Event",
                "record_id": event.id,
                "owner_employee_id": employee_ids[0],
                "account_id": account.id,
                "event_id": event.id,
                "attendee_contact_ids": [contact.id],
                "subject": "Quarterly review",
                "text_body": "Attendance captured on the CRM event.",
                "primary_category": "event_attendance",
                "is_relevant": True,
                "relevance_reason": "toy attendance row",
                "provenance": {
                    "object_type": "event",
                    "object_id": event.id,
                    "strength": "strong",
                    "explanation": "Attendance is captured on the CRM event.",
                },
            }
        ],
    }


def test_quality_metrics_compute_on_toy_dataset() -> None:
    enterprise = build_enterprise()
    rows_by_source = build_toy_rows(enterprise)
    report = DatasetQualityEvaluator(enterprise=enterprise).evaluate(rows_by_source)

    assert report.total_rows == 5
    assert report.source_row_counts == {
        "email": 2,
        "slack": 1,
        "teams": 1,
        "salesforce": 1,
    }
    assert report.relevance_rate == pytest.approx(0.4)
    assert report.category_distribution["follow_up"] == pytest.approx(0.2)
    assert report.category_distribution["event_attendance"] == pytest.approx(0.2)
    assert report.noise_category_distribution["scheduling_only"] == pytest.approx(1 / 3)
    assert report.noise_category_distribution["social_chatter"] == pytest.approx(1 / 3)
    assert report.noise_category_distribution["status_updates"] == pytest.approx(1 / 3)
    assert report.thread_depth_distribution == {"1": 2, "2": 1}
    assert report.average_message_length_by_source["email"] > 0.0
    assert report.cross_system_linkage_rate > 0.5
    assert report.hard_negative_rate == pytest.approx(0.2)
    assert report.duplicate_rate == pytest.approx(0.0)
    assert report.lexical_diversity > 0.5
    assert report.entity_coverage.overall_rate > 0.0


@pytest.mark.parametrize(
    ("rows_by_source", "message"),
    [
        (
            {"email": [{"source_system": "email", "is_relevant": True}]},
            "primary_category",
        ),
        (
            {
                "zoom": [
                    {
                        "source_system": "zoom",
                        "primary_category": "follow_up",
                        "is_relevant": True,
                    }
                ]
            },
            "unsupported source",
        ),
    ],
)
def test_quality_evaluator_invalid_inputs_fail_clearly(
    rows_by_source: dict[str, list[dict[str, object]]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DatasetQualityEvaluator().evaluate(rows_by_source)


def test_quality_report_structure_is_stable() -> None:
    enterprise = build_enterprise()
    report = DatasetQualityEvaluator(enterprise=enterprise).evaluate(build_toy_rows(enterprise))

    assert list(report.to_dict()) == [
        "total_rows",
        "source_row_counts",
        "relevance_rate",
        "category_distribution",
        "noise_category_distribution",
        "thread_depth_distribution",
        "average_message_length_by_source",
        "cross_system_linkage_rate",
        "hard_negative_rate",
        "duplicate_rate",
        "lexical_diversity",
        "entity_coverage",
    ]
    assert list(report.entity_coverage.to_dict()) == ["overall_rate", "by_entity_type"]


def test_quality_metrics_distinguish_degenerate_datasets_from_realistic_ones(
    tmp_path: Path,
) -> None:
    context = GeneratorContext(
        seed=20260331,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=30, max_employees=30),
            noise_ratio=0.85,
            hard_negative_ratio=0.25,
            cross_system_ratio=0.5,
        ),
    )
    realistic = GenerationPipeline(context=context).generate_dataset(
        targets=DatasetTargets(
            account_count=20,
            email_count=80,
            slack_count=80,
            teams_count=80,
            salesforce_count=80,
            chunk_size=40,
        ),
        destination_root=tmp_path,
    )
    realistic_rows = {
        "email": [record.to_dict() for record in realistic.email_records],
        "slack": [record.to_dict() for record in realistic.slack_records],
        "teams": [record.to_dict() for record in realistic.teams_records],
        "salesforce": [record.to_dict() for record in realistic.salesforce_records],
    }
    realistic_report = DatasetQualityEvaluator(enterprise=realistic.enterprise).evaluate(
        realistic_rows
    )

    repeated_row = realistic.email_records[0].to_dict()
    degenerate_rows: dict[str, list[dict[str, object]]] = {"email": []}
    for index in range(80):
        row = deepcopy(repeated_row)
        row["email_id"] = f"email_duplicate_{index}"
        degenerate_rows["email"].append(row)

    degenerate_report = DatasetQualityEvaluator(enterprise=realistic.enterprise).evaluate(
        degenerate_rows
    )

    assert degenerate_report.duplicate_rate > realistic_report.duplicate_rate
    assert degenerate_report.lexical_diversity < realistic_report.lexical_diversity
    assert degenerate_report.cross_system_linkage_rate < realistic_report.cross_system_linkage_rate
    assert (
        degenerate_report.entity_coverage.overall_rate
        < realistic_report.entity_coverage.overall_rate
    )
