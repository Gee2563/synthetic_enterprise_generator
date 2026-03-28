from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.evaluation.quality_report import (
    SUPPORTED_SOURCES,
    _business_entity_ids,
    _entity_references,
    _extract_text,
    _row_to_dict,
)
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import (
    DatasetTargets,
    GeneratedDataset,
    GenerationPipeline,
)

RowLike: TypeAlias = Mapping[str, Any] | EnterpriseModel

PHASE1_REGRESSION_TARGETS = DatasetTargets(
    account_count=10,
    email_count=60,
    slack_count=60,
    teams_count=60,
    salesforce_count=60,
    chunk_size=20,
)


class RegressionSnapshot(EnterpriseModel):
    """Compact deterministic fingerprint for a small regression fixture."""

    total_rows: int
    row_counts: dict[str, int]
    digests_by_source: dict[str, str]
    overall_digest: str


@dataclass(slots=True)
class RegressionFixture:
    """Generated regression dataset with normalized row payloads."""

    context: GeneratorContext
    dataset: GeneratedDataset
    rows_by_source: dict[str, list[dict[str, object]]]


def build_regression_context(*, seed: int = 20260430) -> GeneratorContext:
    return GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=18, max_employees=18),
            noise_ratio=0.85,
            hard_negative_ratio=0.25,
            cross_system_ratio=0.5,
        ),
    )


def build_regression_fixture(
    *,
    destination_root: Path,
    seed: int = 20260430,
) -> RegressionFixture:
    context = build_regression_context(seed=seed)
    dataset = GenerationPipeline(context=context).generate_dataset(
        targets=PHASE1_REGRESSION_TARGETS,
        destination_root=destination_root,
    )
    return RegressionFixture(
        context=context,
        dataset=dataset,
        rows_by_source=_rows_by_source(dataset),
    )


def build_regression_snapshot(
    rows_by_source: Mapping[str, Sequence[RowLike]],
) -> RegressionSnapshot:
    unexpected_sources = set(rows_by_source) - set(SUPPORTED_SOURCES)
    if unexpected_sources:
        raise ValueError(f"unsupported sources for regression snapshot: {unexpected_sources}")

    row_counts = {
        source_name: len(rows_by_source.get(source_name, ()))
        for source_name in SUPPORTED_SOURCES
    }
    digests_by_source = {
        source_name: _source_digest(
            source_name=source_name,
            rows=rows_by_source.get(source_name, ()),
        )
        for source_name in SUPPORTED_SOURCES
    }
    overall_digest = _stable_digest(
        {
            "row_counts": row_counts,
            "digests_by_source": digests_by_source,
        }
    )

    return RegressionSnapshot(
        total_rows=sum(row_counts.values()),
        row_counts=row_counts,
        digests_by_source=digests_by_source,
        overall_digest=overall_digest,
    )


def collect_reference_errors(
    *,
    enterprise: EnterpriseGraph,
    rows_by_source: Mapping[str, Sequence[RowLike]],
) -> list[str]:
    known_entity_ids = {
        "employees": {employee.id for employee in enterprise.employees},
        "accounts": {account.id for account in enterprise.customer_accounts},
        "contacts": {contact.id for contact in enterprise.contacts},
        "opportunities": {opportunity.id for opportunity in enterprise.opportunities},
        "events": {event.id for event in enterprise.events},
        "tickets": {ticket.id for ticket in enterprise.ticket_issues},
        "campaigns": {campaign.id for campaign in enterprise.campaigns},
    }
    errors: list[str] = []

    for source_name, rows in rows_by_source.items():
        for row in rows:
            payload = _row_to_dict(row)
            row_id = _source_row_id(source_name, payload)

            for entity_type, entity_ids in _validated_entity_references(
                source_name=source_name,
                payload=payload,
            ).items():
                allowed_ids = known_entity_ids[entity_type]
                for entity_id in sorted(entity_ids):
                    if entity_id not in allowed_ids:
                        errors.append(
                            f"{source_name}:{row_id} broken "
                            f"{entity_type[:-1]} reference {entity_id}"
                        )

            provenance = payload.get("provenance")
            if not isinstance(provenance, Mapping):
                continue

            object_type = provenance.get("object_type")
            object_id = provenance.get("object_id")
            if not isinstance(object_type, str) or not isinstance(object_id, str):
                errors.append(f"{source_name}:{row_id} invalid provenance payload")
                continue

            if object_type == "account" and object_id not in known_entity_ids["accounts"]:
                errors.append(f"{source_name}:{row_id} broken provenance account {object_id}")
            elif object_type == "contact" and object_id not in known_entity_ids["contacts"]:
                errors.append(f"{source_name}:{row_id} broken provenance contact {object_id}")
            elif (
                object_type == "opportunity"
                and object_id not in known_entity_ids["opportunities"]
            ):
                errors.append(
                    f"{source_name}:{row_id} broken provenance opportunity {object_id}"
                )
            elif object_type == "event" and object_id not in known_entity_ids["events"]:
                errors.append(f"{source_name}:{row_id} broken provenance event {object_id}")
            elif object_type == "ticket" and object_id not in known_entity_ids["tickets"]:
                errors.append(f"{source_name}:{row_id} broken provenance ticket {object_id}")
            elif object_type == "campaign" and object_id not in known_entity_ids["campaigns"]:
                continue

    return sorted(errors)


def _rows_by_source(dataset: GeneratedDataset) -> dict[str, list[dict[str, object]]]:
    return {
        "email": [record.to_dict() for record in dataset.email_records],
        "slack": [record.to_dict() for record in dataset.slack_records],
        "teams": [record.to_dict() for record in dataset.teams_records],
        "salesforce": [record.to_dict() for record in dataset.salesforce_records],
    }


def _iter_row_payloads(
    rows_by_source: Mapping[str, Sequence[RowLike]],
) -> list[dict[str, Any]]:
    return [
        _row_to_dict(row)
        for rows in rows_by_source.values()
        for row in rows
    ]


def _validated_entity_references(
    *,
    source_name: str,
    payload: Mapping[str, Any],
) -> dict[str, set[str]]:
    references = _entity_references(payload)
    if source_name != "salesforce":
        return references

    return {
        "employees": references["employees"],
        "accounts": references["accounts"],
        "contacts": references["contacts"],
        "opportunities": set(),
        "events": references["events"],
        "tickets": references["tickets"],
        "campaigns": set(),
    }


def _source_digest(
    *,
    source_name: str,
    rows: Sequence[RowLike],
) -> str:
    projections = [
        _snapshot_projection(source_name=source_name, payload=_row_to_dict(row))
        for row in rows
    ]
    projections.sort(key=lambda item: str(item["row_id"]))
    return _stable_digest(projections)


def _snapshot_projection(
    *,
    source_name: str,
    payload: Mapping[str, Any],
) -> dict[str, object]:
    business_entity_ids = sorted(_business_entity_ids(source_name, payload))
    entity_references = _entity_references(payload)
    provenance = payload.get("provenance")

    return {
        "row_id": _source_row_id(source_name, payload),
        "source_system": source_name,
        "timestamp": payload.get("timestamp"),
        "thread_key": _thread_key(source_name, payload),
        "primary_category": payload.get("primary_category"),
        "is_relevant": payload.get("is_relevant"),
        "business_entity_ids": business_entity_ids,
        "entity_references": {
            entity_type: sorted(entity_ids)
            for entity_type, entity_ids in entity_references.items()
            if entity_ids
        },
        "routing": _routing_projection(source_name, payload),
        "provenance": _provenance_projection(provenance),
        "text_digest": _stable_digest(_extract_text(source_name, payload)),
    }


def _source_row_id(source_name: str, payload: Mapping[str, Any]) -> str:
    if source_name == "email":
        return str(payload["email_id"])
    if source_name == "slack":
        return str(payload["slack_message_id"])
    if source_name == "teams":
        return str(payload["teams_message_id"])
    return str(payload["salesforce_record_id"])


def _thread_key(source_name: str, payload: Mapping[str, Any]) -> str | None:
    if source_name == "email":
        return str(payload.get("thread_id"))
    if source_name == "slack":
        value = payload.get("thread_id") or payload.get("slack_message_id")
        return str(value) if value is not None else None
    if source_name == "teams":
        value = payload.get("thread_id") or payload.get("teams_message_id")
        return str(value) if value is not None else None
    return None


def _routing_projection(
    source_name: str,
    payload: Mapping[str, Any],
) -> dict[str, object]:
    field_names = {
        "email": (
            "thread_id",
            "message_index_in_thread",
            "sender_employee_id",
            "sender_contact_id",
        ),
        "slack": (
            "channel_id",
            "channel_name",
            "thread_id",
            "parent_message_id",
            "sender_employee_id",
        ),
        "teams": (
            "team_id",
            "channel_id",
            "chat_or_channel",
            "thread_id",
            "meeting_id",
            "sender_employee_id",
        ),
        "salesforce": (
            "object_type",
            "record_id",
            "parent_record_id",
            "owner_employee_id",
            "stage",
            "status",
        ),
    }[source_name]
    return {
        field_name: payload[field_name]
        for field_name in field_names
        if payload.get(field_name) is not None
    }


def _provenance_projection(provenance: object) -> dict[str, object] | None:
    if not isinstance(provenance, Mapping):
        return None
    return {
        key: provenance.get(key)
        for key in ("object_type", "object_id", "strength")
        if provenance.get(key) is not None
    }


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
