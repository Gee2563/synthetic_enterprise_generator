from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from pydantic import Field, model_validator

from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.domain.base import EnterpriseModel, stable_entity_id
from synthetic_enterprise.evaluation.quality_report import (
    SUPPORTED_SOURCES,
    _business_entity_ids,
    _entity_references,
    _provenance_strength,
    _required_value,
    _row_to_dict,
)
from synthetic_enterprise.labeling.taxonomy import RELEVANT_CATEGORIES, CommunicationCategory
from synthetic_enterprise.storage.jsonl_writer import write_jsonl
from synthetic_enterprise.storage.parquet_writer import rows_to_dataframe, write_parquet

RowLike: TypeAlias = Mapping[str, Any] | EnterpriseModel


class GoldSetConfig(EnterpriseModel):
    """Sampling configuration for the benchmark gold subset."""

    relevant_categories: tuple[CommunicationCategory, ...]
    samples_per_relevant_category: int = Field(ge=1)
    hard_negative_count: int = Field(default=0, ge=0)
    min_cross_system_examples: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_categories(self) -> GoldSetConfig:
        if not self.relevant_categories:
            raise ValueError("relevant_categories must not be empty")
        if len(set(self.relevant_categories)) != len(self.relevant_categories):
            raise ValueError("relevant_categories must be unique")
        invalid = [
            category.value
            for category in self.relevant_categories
            if category not in RELEVANT_CATEGORIES
        ]
        if invalid:
            raise ValueError(
                f"relevant_categories must only contain relevant labels: {invalid}"
            )
        return self


class GoldExample(EnterpriseModel):
    """One benchmark-ready gold example row."""

    gold_example_id: str
    source_system: str
    source_row_id: str
    primary_category: CommunicationCategory
    is_relevant: bool
    is_hard_negative: bool = False
    rationale: str = Field(min_length=1)
    cross_system_group_id: str | None = None
    referenced_entity_ids: tuple[str, ...] = Field(default_factory=tuple)
    row_data: dict[str, Any] = Field(default_factory=dict)

    def to_export_row(self) -> dict[str, object]:
        return {
            **self.row_data,
            "gold_example_id": self.gold_example_id,
            "gold_source_row_id": self.source_row_id,
            "gold_rationale": self.rationale,
            "gold_is_hard_negative": self.is_hard_negative,
            "gold_cross_system_group_id": self.cross_system_group_id,
            "gold_referenced_entity_ids": list(self.referenced_entity_ids),
        }


class GoldSet(EnterpriseModel):
    """Balanced evaluation subset with traceable benchmark labels."""

    config: GoldSetConfig
    examples: list[GoldExample] = Field(default_factory=list)


@dataclass(slots=True)
class _GoldCandidate:
    source_system: str
    source_row_id: str
    primary_category: CommunicationCategory
    is_relevant: bool
    is_hard_negative: bool
    rationale: str
    cross_system_group_id: str | None
    referenced_entity_ids: tuple[str, ...]
    row_data: dict[str, Any]
    reference_error: str | None = None

    @property
    def selection_key(self) -> tuple[int, str, str]:
        return (
            0 if self.cross_system_group_id is not None else 1,
            self.source_system,
            self.source_row_id,
        )


@dataclass(slots=True)
class GoldSetBuilder:
    """Create a benchmark-oriented gold subset from generated enterprise rows."""

    enterprise: EnterpriseGraph | None = None

    def create(
        self,
        *,
        rows_by_source: Mapping[str, Sequence[RowLike]],
        config: GoldSetConfig,
    ) -> GoldSet:
        candidates = self._normalize_candidates(rows_by_source)
        relevant_candidates = [
            candidate
            for candidate in candidates
            if candidate.is_relevant
            and not candidate.is_hard_negative
        ]
        relevant_by_category: dict[CommunicationCategory, list[_GoldCandidate]] = defaultdict(list)
        for candidate in relevant_candidates:
            relevant_by_category[candidate.primary_category].append(candidate)

        selected: list[_GoldCandidate] = []
        selected_keys: set[tuple[str, str]] = set()

        for category in config.relevant_categories:
            available = sorted(
                (
                    candidate
                    for candidate in relevant_by_category[category]
                    if candidate.reference_error is None
                ),
                key=lambda item: item.selection_key,
            )
            if len(available) < config.samples_per_relevant_category:
                broken_reference = next(
                    (
                        candidate.reference_error
                        for candidate in relevant_by_category[category]
                        if candidate.reference_error is not None
                    ),
                    None,
                )
                if broken_reference is not None:
                    raise ValueError(broken_reference)
                raise ValueError(
                    f"not enough clean candidates for category {category.value}"
                )
            chosen = available[:config.samples_per_relevant_category]
            selected.extend(chosen)
            selected_keys.update(
                (candidate.source_system, candidate.source_row_id)
                for candidate in chosen
            )

        hard_negative_candidates = sorted(
            (
                candidate
                for candidate in candidates
                if candidate.is_hard_negative
                and candidate.reference_error is None
                and (candidate.source_system, candidate.source_row_id) not in selected_keys
            ),
            key=lambda item: item.selection_key,
        )
        if len(hard_negative_candidates) < config.hard_negative_count:
            raise ValueError("not enough hard-negative candidates for the requested gold set")

        selected.extend(hard_negative_candidates[:config.hard_negative_count])

        cross_system_count = sum(
            candidate.cross_system_group_id is not None
            for candidate in selected
        )
        if cross_system_count < config.min_cross_system_examples:
            raise ValueError("not enough cross-system linked candidates for the requested gold set")

        examples = [
            GoldExample(
                gold_example_id=stable_entity_id(
                    "gold_example",
                    0,
                    f"{candidate.source_system}:{candidate.source_row_id}",
                ),
                source_system=candidate.source_system,
                source_row_id=candidate.source_row_id,
                primary_category=candidate.primary_category,
                is_relevant=candidate.is_relevant,
                is_hard_negative=candidate.is_hard_negative,
                rationale=candidate.rationale,
                cross_system_group_id=candidate.cross_system_group_id,
                referenced_entity_ids=candidate.referenced_entity_ids,
                row_data=candidate.row_data,
            )
            for candidate in selected
        ]

        return GoldSet(config=config, examples=examples)

    def export(
        self,
        *,
        gold_set: GoldSet,
        destination_root: Path,
    ) -> dict[str, Path]:
        destination_root.mkdir(parents=True, exist_ok=True)
        export_rows = [example.to_export_row() for example in gold_set.examples]
        parquet_path = destination_root / "gold_set.parquet"
        jsonl_path = destination_root / "gold_set.jsonl"

        write_parquet(rows_to_dataframe(export_rows), parquet_path)
        write_jsonl(export_rows, jsonl_path)

        return {
            "parquet": parquet_path,
            "jsonl": jsonl_path,
        }

    def _normalize_candidates(
        self,
        rows_by_source: Mapping[str, Sequence[RowLike]],
    ) -> list[_GoldCandidate]:
        row_payloads: list[dict[str, Any]] = []
        for source_name, rows in rows_by_source.items():
            if source_name not in SUPPORTED_SOURCES:
                raise ValueError(f"unsupported source {source_name!r}")
            for row in rows:
                payload = _row_to_dict(row)
                actual_source = _required_value(payload, "source_system")
                if actual_source != source_name:
                    raise ValueError(
                        "row source_system does not match the source bucket "
                        f"{source_name!r}: {actual_source!r}"
                    )
                row_payloads.append(payload)

        cross_system_group_ids = self._cross_system_group_ids(row_payloads)
        candidates: list[_GoldCandidate] = []

        for payload in row_payloads:
            category = CommunicationCategory(str(_required_value(payload, "primary_category")))
            is_relevant = _required_value(payload, "is_relevant")
            if not isinstance(is_relevant, bool):
                raise ValueError("is_relevant must be a bool")

            row_id = self._source_row_id(payload)
            entity_refs = _entity_references(payload)
            reference_error = self._reference_error(entity_refs)
            business_ids = _business_entity_ids(str(payload["source_system"]), payload)
            rationale = self._rationale(payload=payload, is_relevant=is_relevant)
            provenance_strength = _provenance_strength(payload.get("provenance"))
            cross_system_group_id = next(
                (
                    entity_id
                    for entity_id in sorted(business_ids)
                    if entity_id in cross_system_group_ids
                ),
                None,
            )

            candidates.append(
                _GoldCandidate(
                    source_system=str(payload["source_system"]),
                    source_row_id=row_id,
                    primary_category=category,
                    is_relevant=is_relevant,
                    is_hard_negative=(not is_relevant) and provenance_strength == "weak",
                    rationale=rationale,
                    cross_system_group_id=cross_system_group_id,
                    referenced_entity_ids=tuple(
                        sorted(
                            {
                                entity_id
                                for entity_ids in entity_refs.values()
                                for entity_id in entity_ids
                            }
                        )
                    ),
                    row_data=payload,
                    reference_error=reference_error,
                )
            )

        return candidates

    def _cross_system_group_ids(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> set[str]:
        sources_by_entity: dict[str, set[str]] = defaultdict(set)
        for payload in rows:
            source_system = str(payload["source_system"])
            for entity_id in _business_entity_ids(source_system, payload):
                sources_by_entity[entity_id].add(source_system)
        return {
            entity_id
            for entity_id, sources in sources_by_entity.items()
            if len(sources) > 1
        }

    def _validate_references(self, entity_refs: Mapping[str, set[str]]) -> None:
        reference_error = self._reference_error(entity_refs)
        if reference_error is not None:
            raise ValueError(reference_error)

    def _reference_error(self, entity_refs: Mapping[str, set[str]]) -> str | None:
        if self.enterprise is None:
            return None

        valid_ids = {
            "employees": {employee.id for employee in self.enterprise.employees},
            "accounts": {account.id for account in self.enterprise.customer_accounts},
            "contacts": {contact.id for contact in self.enterprise.contacts},
            "opportunities": {opportunity.id for opportunity in self.enterprise.opportunities},
            "events": {event.id for event in self.enterprise.events},
            "tickets": {ticket.id for ticket in self.enterprise.ticket_issues},
            "campaigns": {campaign.id for campaign in self.enterprise.campaigns},
        }

        for entity_type, referenced_ids in entity_refs.items():
            missing_ids = sorted(referenced_ids - valid_ids[entity_type])
            if missing_ids:
                return f"broken reference for {entity_type}: unknown ids {missing_ids}"

        return None

    def _source_row_id(self, payload: Mapping[str, Any]) -> str:
        source_name = str(payload["source_system"])
        id_field = {
            "email": "email_id",
            "slack": "slack_message_id",
            "teams": "teams_message_id",
            "salesforce": "salesforce_record_id",
        }[source_name]
        return str(_required_value(payload, id_field))

    def _rationale(self, *, payload: Mapping[str, Any], is_relevant: bool) -> str:
        raw_reason = str(_required_value(payload, "relevance_reason")).strip()
        if not raw_reason:
            raise ValueError("gold-set rows require a non-empty relevance_reason")
        provenance_strength = _provenance_strength(payload.get("provenance"))

        if is_relevant:
            return f"Relevant because {raw_reason}"
        if provenance_strength == "weak":
            return f"Hard negative because {raw_reason}"
        return f"Non-relevant because {raw_reason}"
