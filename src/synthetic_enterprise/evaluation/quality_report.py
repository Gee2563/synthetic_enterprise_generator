from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeAlias

from pydantic import Field

from synthetic_enterprise.contracts.scenario import ScenarioFamily
from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.generation.account_profiles import (
    AccountBehaviorProfileType,
    AccountBehaviorResolver,
)
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.scenario_engine import ScenarioEngine
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory, NoiseCategory

SUPPORTED_SOURCES: tuple[str, ...] = ("email", "slack", "teams", "salesforce")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TEMPORAL_LAG_BUCKETS: tuple[str, ...] = ("0-6h", "6-24h", "1-3d", ">3d")

RowLike: TypeAlias = Mapping[str, Any] | EnterpriseModel

QUALITY_METRIC_DEFINITIONS: dict[str, str] = {
    "total_rows": "Total number of rows evaluated across all supported sources.",
    "source_row_counts": "Per-source row counts after normalization.",
    "relevance_rate": "Share of rows labeled relevant.",
    "category_distribution": "Normalized distribution across all communication categories.",
    "noise_category_distribution": "Normalized distribution across non-relevant noise categories.",
    "thread_depth_distribution": "Histogram of thread sizes across message-based sources.",
    "average_message_length_by_source": "Average token length of rendered text per source.",
    "cross_system_linkage_rate": (
        "Share of rows linked to business entities that appear in multiple sources."
    ),
    "hard_negative_rate": "Share of rows that are non-relevant with weak provenance.",
    "messy_data_rate": "Share of rows exhibiting messy-data markers.",
    "duplicate_rate": "Overall normalized duplicate rate.",
    "duplicate_rate_by_source": "Per-source duplicate rate.",
    "lexical_diversity": "Overall unique-token ratio.",
    "lexical_diversity_by_source": "Unique-token ratio for each source.",
    "lexical_diversity_by_company": "Unique-token ratio for each company_id present.",
    "scenario_coverage": "Coverage of deterministic scenario families evidenced by the dataset.",
    "source_style_separation_proxy": "Average pairwise token-set distance between sources.",
    "hard_negative_difficulty_proxy": (
        "Average lexical overlap between hard negatives and relevant rows."
    ),
    "noise_family_entropy": "Normalized entropy of the noise-category mix.",
    "temporal_lag_distribution": (
        "Lag bucket histogram across linked entities with multiple timestamps."
    ),
    "account_profile_coverage": (
        "Coverage of deterministic account behavior profiles referenced by the dataset."
    ),
    "entity_coverage": "Coverage of enterprise entities referenced in the dataset.",
}


class EntityCoverageReport(EnterpriseModel):
    """Coverage of enterprise entities referenced in the dataset."""

    overall_rate: float = Field(ge=0.0, le=1.0)
    by_entity_type: dict[str, float] = Field(default_factory=dict)


class ScenarioCoverageReport(EnterpriseModel):
    """Coverage of scenario families evidenced by linked dataset rows."""

    overall_rate: float = Field(ge=0.0, le=1.0)
    by_family: dict[str, float] = Field(default_factory=dict)


class AccountProfileCoverageReport(EnterpriseModel):
    """Coverage of deterministic account behavior profiles."""

    overall_rate: float = Field(ge=0.0, le=1.0)
    by_profile: dict[str, float] = Field(default_factory=dict)


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
    messy_data_rate: float = Field(ge=0.0, le=1.0)
    duplicate_rate: float = Field(ge=0.0, le=1.0)
    duplicate_rate_by_source: dict[str, float] = Field(default_factory=dict)
    lexical_diversity: float = Field(ge=0.0, le=1.0)
    lexical_diversity_by_source: dict[str, float] = Field(default_factory=dict)
    lexical_diversity_by_company: dict[str, float] = Field(default_factory=dict)
    scenario_coverage: ScenarioCoverageReport
    source_style_separation_proxy: float = Field(ge=0.0, le=1.0)
    hard_negative_difficulty_proxy: float = Field(ge=0.0, le=1.0)
    noise_family_entropy: float = Field(ge=0.0, le=1.0)
    temporal_lag_distribution: dict[str, int] = Field(default_factory=dict)
    account_profile_coverage: AccountProfileCoverageReport
    entity_coverage: EntityCoverageReport


@dataclass(slots=True)
class _NormalizedRow:
    source_system: str
    primary_category: CommunicationCategory
    is_relevant: bool
    text: str
    thread_key: str | None
    timestamp: datetime | None
    company_id: str | None
    business_entity_ids: set[str]
    entity_references: dict[str, set[str]]
    provenance_strength: str | None
    is_messy: bool


@dataclass(slots=True)
class DatasetQualityEvaluator:
    """Compute quality metrics across mixed enterprise source rows."""

    enterprise: EnterpriseGraph | None = None
    context: GeneratorContext | None = None

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
            messy_data_rate=_safe_rate(
                numerator=sum(row.is_messy for row in normalized_rows),
                denominator=total_rows,
            ),
            duplicate_rate=self._duplicate_rate(normalized_rows),
            duplicate_rate_by_source=self._duplicate_rate_by_source(normalized_rows),
            lexical_diversity=self._lexical_diversity(normalized_rows),
            lexical_diversity_by_source=self._lexical_diversity_by_source(normalized_rows),
            lexical_diversity_by_company=self._lexical_diversity_by_company(normalized_rows),
            scenario_coverage=self._scenario_coverage(normalized_rows),
            source_style_separation_proxy=self._source_style_separation_proxy(
                normalized_rows
            ),
            hard_negative_difficulty_proxy=self._hard_negative_difficulty_proxy(
                normalized_rows
            ),
            noise_family_entropy=self._noise_family_entropy(normalized_rows),
            temporal_lag_distribution=self._temporal_lag_distribution(normalized_rows),
            account_profile_coverage=self._account_profile_coverage(normalized_rows),
            entity_coverage=self._entity_coverage(normalized_rows),
        )

    def evaluate_by_company(
        self,
        rows_by_source: Mapping[str, Sequence[RowLike]],
    ) -> dict[str, QualityReport]:
        rows_grouped_by_company: dict[str, dict[str, list[dict[str, object]]]] = {}

        for source_name, rows in rows_by_source.items():
            if source_name not in SUPPORTED_SOURCES:
                raise ValueError(f"unsupported source {source_name!r}")

            for row in rows:
                payload = _row_to_dict(row)
                company_id = payload.get("company_id")
                if not isinstance(company_id, str):
                    raise ValueError("multi-company evaluation rows must include company_id")
                rows_grouped_by_company.setdefault(
                    company_id,
                    {supported_source: [] for supported_source in SUPPORTED_SOURCES},
                )[source_name].append(payload)

        reports: dict[str, QualityReport] = {}
        for company_id, company_rows in sorted(rows_grouped_by_company.items()):
            company_enterprise = self._enterprise_for_company(company_id)
            reports[company_id] = DatasetQualityEvaluator(
                enterprise=company_enterprise,
                context=self.context,
            ).evaluate(company_rows)

        return reports

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
            timestamp=_optional_timestamp(payload.get("timestamp")),
            company_id=_optional_company_id(payload.get("company_id")),
            business_entity_ids=_business_entity_ids(actual_source, payload),
            entity_references=_entity_references(payload),
            provenance_strength=_provenance_strength(payload.get("provenance")),
            is_messy=_is_messy_row(actual_source, payload),
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

    def _duplicate_rate_by_source(
        self,
        rows: Sequence[_NormalizedRow],
    ) -> dict[str, float]:
        return {
            source_name: self._duplicate_rate(
                [row for row in rows if row.source_system == source_name]
            )
            for source_name in SUPPORTED_SOURCES
        }

    def _lexical_diversity_by_source(
        self,
        rows: Sequence[_NormalizedRow],
    ) -> dict[str, float]:
        return {
            source_name: self._lexical_diversity(
                [row for row in rows if row.source_system == source_name]
            )
            for source_name in SUPPORTED_SOURCES
        }

    def _lexical_diversity_by_company(
        self,
        rows: Sequence[_NormalizedRow],
    ) -> dict[str, float]:
        company_ids = sorted(
            {
                row.company_id
                for row in rows
                if row.company_id is not None
            }
        )
        return {
            company_id: self._lexical_diversity(
                [row for row in rows if row.company_id == company_id]
            )
            for company_id in company_ids
        }

    def _scenario_coverage(
        self,
        rows: Sequence[_NormalizedRow],
    ) -> ScenarioCoverageReport:
        if self.enterprise is None or self.context is None or len(self.enterprise.companies) != 1:
            return ScenarioCoverageReport(
                overall_rate=0.0,
                by_family={family.value: 0.0 for family in ScenarioFamily},
            )

        scenarios = ScenarioEngine(context=self.context).build_for_enterprise(self.enterprise)
        covered_by_family: Counter[str] = Counter()
        totals_by_family: Counter[str] = Counter(
            scenario.kind.value if hasattr(scenario.kind, "value") else str(scenario.kind)
            for scenario in scenarios
        )
        covered_count = 0

        for scenario in scenarios:
            family = scenario.kind.value if hasattr(scenario.kind, "value") else str(scenario.kind)
            scenario_ids = {
                *scenario.account_ids,
                *scenario.opportunity_ids,
                *scenario.event_ids,
                *scenario.ticket_ids,
            }
            if any(
                row.primary_category in set(scenario.expected_labels)
                and bool(row.business_entity_ids & scenario_ids)
                for row in rows
            ):
                covered_count += 1
                covered_by_family[family] += 1

        return ScenarioCoverageReport(
            overall_rate=_safe_rate(covered_count, len(scenarios)),
            by_family={
                family.value: _safe_rate(
                    covered_by_family[family.value],
                    totals_by_family[family.value],
                )
                for family in ScenarioFamily
            },
        )

    def _source_style_separation_proxy(
        self,
        rows: Sequence[_NormalizedRow],
    ) -> float:
        token_sets = {
            source_name: {
                token
                for row in rows
                if row.source_system == source_name
                for token in _tokenize(row.text)
            }
            for source_name in SUPPORTED_SOURCES
        }
        present_sources = [
            source_name
            for source_name, tokens in token_sets.items()
            if tokens
        ]
        if len(present_sources) < 2:
            return 0.0

        distances: list[float] = []
        for index, source_name in enumerate(present_sources):
            for other_source in present_sources[index + 1 :]:
                distances.append(
                    _token_distance(
                        token_sets[source_name],
                        token_sets[other_source],
                    )
                )

        return sum(distances) / len(distances)

    def _hard_negative_difficulty_proxy(
        self,
        rows: Sequence[_NormalizedRow],
    ) -> float:
        hard_negative_rows = [
            row
            for row in rows
            if (not row.is_relevant) and row.provenance_strength == "weak"
        ]
        relevant_token_sets = [
            set(_tokenize(row.text))
            for row in rows
            if row.is_relevant and row.text
        ]
        if not hard_negative_rows or not relevant_token_sets:
            return 0.0

        scores: list[float] = []
        for row in hard_negative_rows:
            row_tokens = set(_tokenize(row.text))
            if not row_tokens:
                scores.append(0.0)
                continue
            scores.append(
                max(
                    _token_similarity(row_tokens, relevant_tokens)
                    for relevant_tokens in relevant_token_sets
                )
            )

        return sum(scores) / len(scores)

    def _noise_family_entropy(
        self,
        rows: Sequence[_NormalizedRow],
    ) -> float:
        counts = [
            count
            for count in Counter(
                row.primary_category.value
                for row in rows
                if not row.is_relevant
            ).values()
            if count > 0
        ]
        if len(counts) <= 1:
            return 0.0

        total = sum(counts)
        entropy = -sum((count / total) * math.log(count / total) for count in counts)
        return entropy / math.log(len(counts))

    def _temporal_lag_distribution(
        self,
        rows: Sequence[_NormalizedRow],
    ) -> dict[str, int]:
        distribution = {bucket: 0 for bucket in TEMPORAL_LAG_BUCKETS}
        timestamps_by_entity: dict[str, list[datetime]] = defaultdict(list)

        for row in rows:
            if row.timestamp is None:
                continue
            for entity_id in row.business_entity_ids:
                timestamps_by_entity[entity_id].append(row.timestamp)

        for timestamps in timestamps_by_entity.values():
            if len(timestamps) < 2:
                continue
            lag_seconds = (max(timestamps) - min(timestamps)).total_seconds()
            if lag_seconds <= 6 * 60 * 60:
                distribution["0-6h"] += 1
            elif lag_seconds <= 24 * 60 * 60:
                distribution["6-24h"] += 1
            elif lag_seconds <= 3 * 24 * 60 * 60:
                distribution["1-3d"] += 1
            else:
                distribution[">3d"] += 1

        return distribution

    def _account_profile_coverage(
        self,
        rows: Sequence[_NormalizedRow],
    ) -> AccountProfileCoverageReport:
        if self.enterprise is None or self.context is None or len(self.enterprise.companies) != 1:
            return AccountProfileCoverageReport(
                overall_rate=0.0,
                by_profile={
                    profile_type.value: 0.0
                    for profile_type in AccountBehaviorProfileType
                },
            )

        resolver = AccountBehaviorResolver(context=self.context, enterprise=self.enterprise)
        referenced_account_ids = {
            account_id
            for row in rows
            for account_id in row.entity_references["accounts"]
        }
        accounts_by_profile: defaultdict[str, set[str]] = defaultdict(set)
        covered_by_profile: defaultdict[str, set[str]] = defaultdict(set)

        for account in self.enterprise.customer_accounts:
            profile_value = resolver.profile_for_account(account.id).profile_type.value
            accounts_by_profile[profile_value].add(account.id)
            if account.id in referenced_account_ids:
                covered_by_profile[profile_value].add(account.id)

        return AccountProfileCoverageReport(
            overall_rate=_safe_rate(
                len(referenced_account_ids),
                len(self.enterprise.customer_accounts),
            ),
            by_profile={
                profile_type.value: _safe_rate(
                    len(covered_by_profile[profile_type.value]),
                    len(accounts_by_profile[profile_type.value]),
                )
                for profile_type in AccountBehaviorProfileType
            },
        )

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

    def _enterprise_for_company(self, company_id: str) -> EnterpriseGraph | None:
        if self.enterprise is None:
            return None

        account_ids = {
            account.id
            for account in self.enterprise.customer_accounts
            if account.company_id == company_id
        }
        opportunity_ids = {
            opportunity.id
            for opportunity in self.enterprise.opportunities
            if opportunity.account_id in account_ids
        }
        event_ids = {
            event.id
            for event in self.enterprise.events
            if event.company_id == company_id
        }
        ticket_ids = {
            ticket.id
            for ticket in self.enterprise.ticket_issues
            if ticket.account_id in account_ids
        }
        campaign_ids = {
            campaign.id
            for campaign in self.enterprise.campaigns
            if campaign.company_id == company_id
        }

        return self.enterprise.model_copy(
            update={
                "companies": [
                    company
                    for company in self.enterprise.companies
                    if company.id == company_id
                ],
                "departments": [
                    department
                    for department in self.enterprise.departments
                    if department.company_id == company_id
                ],
                "employees": [
                    employee
                    for employee in self.enterprise.employees
                    if employee.company_id == company_id
                ],
                "customer_accounts": [
                    account
                    for account in self.enterprise.customer_accounts
                    if account.company_id == company_id
                ],
                "contacts": [
                    contact
                    for contact in self.enterprise.contacts
                    if contact.account_id in account_ids
                ],
                "opportunities": [
                    opportunity
                    for opportunity in self.enterprise.opportunities
                    if opportunity.id in opportunity_ids
                ],
                "events": [
                    event
                    for event in self.enterprise.events
                    if event.id in event_ids
                ],
                "products": [
                    product
                    for product in self.enterprise.products
                    if product.company_id == company_id
                ],
                "ticket_issues": [
                    ticket
                    for ticket in self.enterprise.ticket_issues
                    if ticket.id in ticket_ids
                ],
                "campaigns": [
                    campaign
                    for campaign in self.enterprise.campaigns
                    if campaign.id in campaign_ids
                ],
                "message_envelopes": [
                    envelope
                    for envelope in self.enterprise.message_envelopes
                    if envelope.company_id == company_id
                ],
                "crm_activities": [
                    activity
                    for activity in self.enterprise.crm_activities
                    if activity.company_id == company_id
                ],
            }
        )


def _row_to_dict(row: RowLike) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    return row.to_dict()


def _required_value(payload: Mapping[str, Any], field_name: str) -> Any:
    if field_name not in payload:
        raise ValueError(f"row missing required field '{field_name}'")
    return payload[field_name]


def _optional_company_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("company_id must be a string when provided")
    return value


def _optional_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be an ISO datetime string") from exc
    raise ValueError("timestamp must be an ISO datetime string or datetime")


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


def _is_messy_row(source_name: str, payload: Mapping[str, Any]) -> bool:
    if source_name != "salesforce":
        return False

    if str(payload.get("status") or "").lower() == "reopened":
        return True

    structured_fields = _coerce_mapping(
        payload.get("structured_fields"),
        field_name="structured_fields",
    )
    if structured_fields is None:
        return False

    messy_keys = {
        "missing_fields",
        "late_entry_days",
        "contradicts_record_id",
        "stage_sync_delay_days",
        "stale_owner_snapshot_days",
        "attendance_reconciliation_pending",
        "follow_up_capture_state",
    }
    return any(key in structured_fields for key in messy_keys)


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


def _token_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _token_distance(left: set[str], right: set[str]) -> float:
    return 1.0 - _token_similarity(left, right)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
