from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from pydantic import Field

from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory, NoiseCategory

SUPPORTED_SOURCES: tuple[str, ...] = ("email", "slack", "teams", "salesforce")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

RowLike: TypeAlias = Mapping[str, Any] | EnterpriseModel


class EntityCoverageReport(EnterpriseModel):
    """Coverage of enterprise entities referenced in the dataset."""

    overall_rate: float = Field(ge=0.0, le=1.0)
    by_entity_type: dict[str, float] = Field(default_factory=dict)


class QualityReport(EnterpriseModel):
    """Stable dataset quality summary."""

    total_rows: int = Field(ge=0)
    source_row_counts: dict[str, int] = Field(default_factory=dict)
    relevance_rate: float = Field(ge=0.0, le=1.0)
    category_distribution: dict[str, float] = Field(default_factory=dict)
    noise_category_distribution: dict[str, float] = Field(default_factory=dict)
    thread_depth_distribution: dict[str, int] = Field(default_factory=dict)
    average_message_length_by_source: dict[str, float] = Field(default_factory=dict)
    cross_system_linkage_rate: float = Field(ge=0.0, le=1.0)
    hard_negative_rate: float = Field(ge=0.0, le=1.0)
    duplicate_rate: float = Field(ge=0.0, le=1.0)
    lexical_diversity: float = Field(ge=0.0, le=1.0)
    entity_coverage: EntityCoverageReport


@dataclass(slots=True)
class _NormalizedRow:
    source_system: str
    primary_category: CommunicationCategory
    is_relevant: bool
    text: str
    thread_key: str | None
    business_entity_ids: set[str]
    entity_references: dict[str, set[str]]
    provenance_strength: str | None


@dataclass(slots=True)
class DatasetQualityEvaluator:
    """Compute quality metrics across mixed enterprise source rows."""

    enterprise: EnterpriseGraph | None = None

    def evaluate(
        self,
        rows_by_source: Mapping[str, Sequence[RowLike]],
    ) -> QualityReport:
        normalized_rows: list[_NormalizedRow] = []

        for source_name, rows in rows_by_source.items():
            if source_name not in SUPPORTED_SOURCES:
                raise ValueError(f"unsupported source {source_name!r}")

            for row_index, row in enumerate(rows):
                normalized_rows.append(
                    self._normalize_row(
                        source_name=source_name,
                        row=row,
                        row_index=row_index,
                    )
                )

        total_rows = len(normalized_rows)
        source_row_counts = {
            source_name: len(rows_by_source.get(source_name, ()))
            for source_name in SUPPORTED_SOURCES
        }
        category_distribution = self._category_distribution(normalized_rows, total_rows)
        noise_category_distribution = self._noise_category_distribution(
            normalized_rows,
        )

        return QualityReport(
            total_rows=total_rows,
            source_row_counts=source_row_counts,
            relevance_rate=_safe_rate(
                numerator=sum(row.is_relevant for row in normalized_rows),
                denominator=total_rows,
            ),
            category_distribution=category_distribution,
            noise_category_distribution=noise_category_distribution,
            thread_depth_distribution=self._thread_depth_distribution(normalized_rows),
            average_message_length_by_source=self._average_message_length_by_source(
                normalized_rows
            ),
            cross_system_linkage_rate=self._cross_system_linkage_rate(normalized_rows),
            hard_negative_rate=_safe_rate(
                numerator=sum(
                    (not row.is_relevant) and row.provenance_strength == "weak"
                    for row in normalized_rows
                ),
                denominator=total_rows,
            ),
            duplicate_rate=self._duplicate_rate(normalized_rows),
            lexical_diversity=self._lexical_diversity(normalized_rows),
            entity_coverage=self._entity_coverage(normalized_rows),
        )

    def _normalize_row(
        self,
        *,
        source_name: str,
        row: RowLike,
        row_index: int,
    ) -> _NormalizedRow:
        payload = _row_to_dict(row)
        actual_source = _required_value(payload, "source_system")
        if not isinstance(actual_source, str):
            raise ValueError("source_system must be a string")
        if actual_source not in SUPPORTED_SOURCES:
            raise ValueError(f"unsupported source {actual_source!r}")
        if actual_source != source_name:
            raise ValueError(
                "row source_system does not match the source bucket "
                f"{source_name!r}: {actual_source!r}"
            )

        category_value = _required_value(payload, "primary_category")
        try:
            category = CommunicationCategory(str(category_value))
        except ValueError as exc:
            raise ValueError(f"invalid primary_category {category_value!r}") from exc

        is_relevant = payload.get("is_relevant")
        if not isinstance(is_relevant, bool):
            raise ValueError("is_relevant must be a bool")

        return _NormalizedRow(
            source_system=actual_source,
            primary_category=category,
            is_relevant=is_relevant,
            text=_extract_text(actual_source, payload),
            thread_key=_extract_thread_key(actual_source, payload, row_index),
            business_entity_ids=_business_entity_ids(actual_source, payload),
            entity_references=_entity_references(payload),
            provenance_strength=_provenance_strength(payload.get("provenance")),
        )

    def _category_distribution(
        self,
        rows: Sequence[_NormalizedRow],
        total_rows: int,
    ) -> dict[str, float]:
        counts = Counter(row.primary_category.value for row in rows)
        return {
            category.value: _safe_rate(counts[category.value], total_rows)
            for category in CommunicationCategory
        }

    def _noise_category_distribution(
        self,
        rows: Sequence[_NormalizedRow],
    ) -> dict[str, float]:
        noise_rows = [row for row in rows if not row.is_relevant]
        noise_total = len(noise_rows)
        counts = Counter(row.primary_category.value for row in noise_rows)
        return {
            category.value: _safe_rate(counts[category.value], noise_total)
            for category in NoiseCategory
        }

    def _thread_depth_distribution(self, rows: Sequence[_NormalizedRow]) -> dict[str, int]:
        thread_sizes = Counter(
            row.thread_key for row in rows if row.thread_key is not None
        )
        depth_counts = Counter(thread_sizes.values())
        return {
            str(depth): depth_counts[depth]
            for depth in sorted(depth_counts)
        }

    def _average_message_length_by_source(
        self,
        rows: Sequence[_NormalizedRow],
    ) -> dict[str, float]:
        token_lengths_by_source: dict[str, list[int]] = {
            source_name: []
            for source_name in SUPPORTED_SOURCES
        }
        for row in rows:
            token_lengths_by_source[row.source_system].append(len(_tokenize(row.text)))

        return {
            source_name: (
                sum(lengths) / len(lengths) if lengths else 0.0
            )
            for source_name, lengths in token_lengths_by_source.items()
        }

    def _cross_system_linkage_rate(self, rows: Sequence[_NormalizedRow]) -> float:
        entity_sources: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            for entity_id in row.business_entity_ids:
                entity_sources[entity_id].add(row.source_system)

        cross_system_entities = {
            entity_id
            for entity_id, sources in entity_sources.items()
            if len(sources) > 1
        }
        linked_row_count = sum(
            bool(row.business_entity_ids & cross_system_entities)
            for row in rows
        )
        return _safe_rate(linked_row_count, len(rows))

    def _duplicate_rate(self, rows: Sequence[_NormalizedRow]) -> float:
        if not rows:
            return 0.0

        fingerprints = {
            (row.source_system, row.primary_category.value, _normalize_text(row.text))
            for row in rows
        }
        duplicate_rows = len(rows) - len(fingerprints)
        return _safe_rate(duplicate_rows, len(rows))

    def _lexical_diversity(self, rows: Sequence[_NormalizedRow]) -> float:
        tokens = [
            token
            for row in rows
            for token in _tokenize(row.text)
        ]
        if not tokens:
            return 0.0
        return _safe_rate(len(set(tokens)), len(tokens))

    def _entity_coverage(self, rows: Sequence[_NormalizedRow]) -> EntityCoverageReport:
        coverage_keys = (
            "employees",
            "accounts",
            "contacts",
            "opportunities",
            "events",
            "tickets",
            "campaigns",
        )
        referenced_by_type: dict[str, set[str]] = {
            key: set()
            for key in coverage_keys
        }
        for row in rows:
            for entity_type, entity_ids in row.entity_references.items():
                referenced_by_type[entity_type].update(entity_ids)

        if self.enterprise is None:
            return EntityCoverageReport(
                overall_rate=0.0,
                by_entity_type={key: 0.0 for key in coverage_keys},
            )

        totals = {
            "employees": len(self.enterprise.employees),
            "accounts": len(self.enterprise.customer_accounts),
            "contacts": len(self.enterprise.contacts),
            "opportunities": len(self.enterprise.opportunities),
            "events": len(self.enterprise.events),
            "tickets": len(self.enterprise.ticket_issues),
            "campaigns": len(self.enterprise.campaigns),
        }
        by_entity_type = {
            key: _safe_rate(len(referenced_by_type[key]), totals[key])
            for key in coverage_keys
        }
        available_total = sum(total for total in totals.values() if total > 0)
        referenced_total = sum(
            min(len(referenced_by_type[key]), totals[key])
            for key in coverage_keys
            if totals[key] > 0
        )
        return EntityCoverageReport(
            overall_rate=_safe_rate(referenced_total, available_total),
            by_entity_type=by_entity_type,
        )


def _row_to_dict(row: RowLike) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    return row.to_dict()


def _required_value(payload: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in payload:
        raise ValueError(f"row missing required field '{field_name}'")
    return payload[field_name]


def _extract_text(source_name: str, payload: Mapping[str, Any]) -> str:
    if source_name == "email":
        return _join_text(payload.get("subject"), payload.get("body"))
    if source_name == "salesforce":
        return _join_text(payload.get("subject"), payload.get("text_body"))
    return _join_text(payload.get("body"))


def _extract_thread_key(
    source_name: str,
    payload: Mapping[str, Any],
    row_index: int,
) -> str | None:
    if source_name == "salesforce":
        return None
    if source_name == "email":
        return str(_required_value(payload, "thread_id"))
    if source_name == "slack":
        return str(payload.get("thread_id") or _required_value(payload, "slack_message_id"))
    if source_name == "teams":
        return str(payload.get("thread_id") or _required_value(payload, "teams_message_id"))
    return f"{source_name}:{row_index}"


def _business_entity_ids(source_name: str, payload: Mapping[str, Any]) -> set[str]:
    field_names = {
        "email": ("account_id", "opportunity_id", "event_id", "ticket_id"),
        "slack": (
            "linked_account_id",
            "linked_opportunity_id",
            "linked_event_id",
            "linked_ticket_id",
        ),
        "teams": (
            "linked_account_id",
            "linked_opportunity_id",
            "linked_event_id",
            "linked_ticket_id",
        ),
        "salesforce": ("account_id", "opportunity_id", "event_id", "case_id", "campaign_id"),
    }[source_name]
    return {
        str(payload[field_name])
        for field_name in field_names
        if payload.get(field_name) is not None
    }


def _entity_references(payload: Mapping[str, Any]) -> dict[str, set[str]]:
    references: dict[str, set[str]] = {
        "employees": set(),
        "accounts": set(),
        "contacts": set(),
        "opportunities": set(),
        "events": set(),
        "tickets": set(),
        "campaigns": set(),
    }

    references["employees"].update(
        entity_id
        for entity_id in (
            payload.get("sender_employee_id"),
            payload.get("owner_employee_id"),
        )
        if isinstance(entity_id, str)
    )
    references["employees"].update(
        entity_id
        for entity_id in _coerce_list(payload.get("mentions"))
        if entity_id.startswith("employee_")
    )
    references["employees"].update(
        entity_id
        for field_name in ("to", "cc", "bcc")
        for entity_id in _coerce_list(payload.get(field_name))
        if entity_id.startswith("employee_")
    )

    references["contacts"].update(
        entity_id
        for entity_id in (
            payload.get("sender_contact_id"),
            payload.get("contact_id"),
        )
        if isinstance(entity_id, str)
    )
    references["contacts"].update(
        entity_id
        for field_name in ("to", "cc", "bcc", "attendee_contact_ids")
        for entity_id in _coerce_list(payload.get(field_name))
        if entity_id.startswith("contact_")
    )

    references["accounts"].update(
        entity_id
        for entity_id in (
            payload.get("account_id"),
            payload.get("linked_account_id"),
        )
        if isinstance(entity_id, str)
    )
    references["opportunities"].update(
        entity_id
        for entity_id in (
            payload.get("opportunity_id"),
            payload.get("linked_opportunity_id"),
        )
        if isinstance(entity_id, str)
    )
    references["events"].update(
        entity_id
        for entity_id in (
            payload.get("event_id"),
            payload.get("linked_event_id"),
        )
        if isinstance(entity_id, str)
    )
    references["tickets"].update(
        entity_id
        for entity_id in (
            payload.get("ticket_id"),
            payload.get("linked_ticket_id"),
            payload.get("case_id"),
        )
        if isinstance(entity_id, str)
    )
    references["campaigns"].update(
        entity_id
        for entity_id in (payload.get("campaign_id"),)
        if isinstance(entity_id, str)
    )

    return references


def _provenance_strength(value: object) -> str | None:
    mapping = _coerce_mapping(value, field_name="provenance")
    if mapping is None:
        return None
    strength = mapping.get("strength")
    if strength is None:
        return None
    return str(strength).lower()


def _coerce_mapping(value: object, *, field_name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, EnterpriseModel):
        return value.to_dict()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be a mapping or JSON object string") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{field_name} JSON must decode to an object")
        return parsed
    raise ValueError(f"{field_name} must be a mapping")


def _coerce_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [value]
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return [value]
    return [str(value)]


def _join_text(*values: object) -> str:
    return " ".join(
        str(value).strip()
        for value in values
        if isinstance(value, str) and value.strip()
    )


def _normalize_text(value: str) -> str:
    return " ".join(_tokenize(value))


def _tokenize(value: str) -> list[str]:
    return TOKEN_PATTERN.findall(value.lower())


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
