from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TypeAlias

from pydantic import Field

from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.domain.base import EnterpriseModel, stable_entity_id
from synthetic_enterprise.evaluation.gold_set import GoldSet
from synthetic_enterprise.evaluation.quality_report import (
    SUPPORTED_SOURCES,
    _business_entity_ids,
    _required_value,
    _row_to_dict,
)
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory

RowLike: TypeAlias = Mapping[str, Any] | EnterpriseModel
SourceFilter: TypeAlias = set[str] | None
CategoryFilter: TypeAlias = set[CommunicationCategory] | None
MESSAGE_SOURCES = frozenset({"email", "slack", "teams"})


class RowClassificationSample(EnterpriseModel):
    sample_id: str
    source_system: str
    source_row_id: str
    text: str
    label: str
    primary_category: CommunicationCategory
    is_relevant: bool
    target_object_type: str | None = None
    target_object_id: str | None = None
    rationale: str | None = None


class RowClassificationDataset(EnterpriseModel):
    samples: list[RowClassificationSample] = Field(default_factory=list)


class MessageClassificationSample(EnterpriseModel):
    sample_id: str
    source_system: str
    source_row_id: str
    thread_id: str | None = None
    text: str
    label: str
    primary_category: CommunicationCategory
    is_relevant: bool
    target_object_type: str | None = None
    target_object_id: str | None = None


class MessageClassificationDataset(EnterpriseModel):
    samples: list[MessageClassificationSample] = Field(default_factory=list)


class RationaleSpan(EnterpriseModel):
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text: str = Field(min_length=1)


class RationaleExtractionSample(EnterpriseModel):
    sample_id: str
    source_system: str
    source_row_id: str
    text: str
    label: str
    primary_category: CommunicationCategory
    target_object_type: str | None = None
    target_object_id: str | None = None
    rationale: str = Field(min_length=1)
    rationale_spans: list[RationaleSpan] = Field(default_factory=list)


class RationaleExtractionDataset(EnterpriseModel):
    samples: list[RationaleExtractionSample] = Field(default_factory=list)


class ThreadMessage(EnterpriseModel):
    source_row_id: str
    message_order: int = Field(ge=0)
    timestamp: str
    text: str
    primary_category: CommunicationCategory
    is_relevant: bool


class GroupedThreadClassificationSample(EnterpriseModel):
    sample_id: str
    source_system: str
    thread_id: str
    label: str
    primary_category: CommunicationCategory
    is_relevant: bool
    messages: list[ThreadMessage] = Field(default_factory=list)


class GroupedThreadClassificationDataset(EnterpriseModel):
    samples: list[GroupedThreadClassificationSample] = Field(default_factory=list)


class CrossDocumentLinkSample(EnterpriseModel):
    sample_id: str
    left_source_system: str
    left_row_id: str
    right_source_system: str
    right_row_id: str
    left_primary_category: CommunicationCategory
    right_primary_category: CommunicationCategory
    shared_object_type: str | None = None
    shared_object_id: str | None = None
    is_linked: bool


class CrossDocumentLinkingDataset(EnterpriseModel):
    samples: list[CrossDocumentLinkSample] = Field(default_factory=list)


@dataclass(slots=True)
class _NormalizedTrainingRow:
    source_system: str
    source_row_id: str
    text: str
    primary_category: CommunicationCategory
    is_relevant: bool
    rationale: str | None
    thread_id: str | None
    message_order: int
    timestamp: datetime
    target_object_type: str | None
    target_object_id: str | None
    business_entity_ids: set[str]


@dataclass(slots=True)
class TrainingFormatExporter:
    """Build typed training datasets from generated rows and gold examples."""

    enterprise: EnterpriseGraph | None = None

    def source_row_id(self, row: RowLike) -> str:
        return self._row_id(_row_to_dict(row))

    def build_row_classification(
        self,
        *,
        rows_by_source: Mapping[str, Sequence[RowLike]],
        source_filter: SourceFilter = None,
        category_filter: CategoryFilter = None,
    ) -> RowClassificationDataset:
        rows = self._filtered_rows(
            rows_by_source=rows_by_source,
            source_filter=source_filter,
            category_filter=category_filter,
        )
        return RowClassificationDataset(
            samples=[
                RowClassificationSample(
                    sample_id=stable_entity_id(
                        "row_classification",
                        0,
                        f"{row.source_system}:{row.source_row_id}",
                    ),
                    source_system=row.source_system,
                    source_row_id=row.source_row_id,
                    text=row.text,
                    label=row.primary_category.value,
                    primary_category=row.primary_category,
                    is_relevant=row.is_relevant,
                    target_object_type=row.target_object_type,
                    target_object_id=row.target_object_id,
                    rationale=row.rationale,
                )
                for row in rows
            ]
        )

    def build_message_classification(
        self,
        *,
        rows_by_source: Mapping[str, Sequence[RowLike]],
        source_filter: SourceFilter = None,
        category_filter: CategoryFilter = None,
    ) -> MessageClassificationDataset:
        effective_sources = source_filter or set(MESSAGE_SOURCES)
        rows = self._filtered_rows(
            rows_by_source=rows_by_source,
            source_filter={source for source in effective_sources if source in MESSAGE_SOURCES},
            category_filter=category_filter,
        )
        return MessageClassificationDataset(
            samples=[
                MessageClassificationSample(
                    sample_id=stable_entity_id(
                        "message_classification",
                        0,
                        f"{row.source_system}:{row.source_row_id}",
                    ),
                    source_system=row.source_system,
                    source_row_id=row.source_row_id,
                    thread_id=row.thread_id,
                    text=row.text,
                    label=row.primary_category.value,
                    primary_category=row.primary_category,
                    is_relevant=row.is_relevant,
                    target_object_type=row.target_object_type,
                    target_object_id=row.target_object_id,
                )
                for row in rows
            ]
        )

    def build_rationale_extraction(
        self,
        *,
        gold_set: GoldSet,
        source_filter: SourceFilter = None,
        category_filter: CategoryFilter = None,
        require_rationale: bool = True,
    ) -> RationaleExtractionDataset:
        filtered_examples = [
            example
            for example in gold_set.examples
            if self._source_allowed(example.source_system, source_filter)
            and self._category_allowed(example.primary_category, category_filter)
        ]
        samples: list[RationaleExtractionSample] = []

        for example in filtered_examples:
            rationale = example.rationale.strip()
            if require_rationale and not rationale:
                raise ValueError("rationale is required for rationale extraction samples")

            row_payload = example.row_data
            text = self._text_for_row(example.source_system, row_payload)
            target_object_type, target_object_id = self._target_from_provenance(row_payload)
            spans: list[RationaleSpan] = []
            if text:
                spans.append(
                    RationaleSpan(
                        start_char=0,
                        end_char=len(text),
                        text=text,
                    )
                )

            samples.append(
                RationaleExtractionSample(
                    sample_id=stable_entity_id(
                        "rationale_extraction",
                        0,
                        f"{example.source_system}:{example.source_row_id}",
                    ),
                    source_system=example.source_system,
                    source_row_id=example.source_row_id,
                    text=text,
                    label=example.primary_category.value,
                    primary_category=example.primary_category,
                    target_object_type=target_object_type,
                    target_object_id=target_object_id,
                    rationale=rationale,
                    rationale_spans=spans,
                )
            )

        return RationaleExtractionDataset(samples=samples)

    def build_grouped_thread_classification(
        self,
        *,
        rows_by_source: Mapping[str, Sequence[RowLike]],
        source_filter: SourceFilter = None,
        category_filter: CategoryFilter = None,
    ) -> GroupedThreadClassificationDataset:
        effective_sources = source_filter or set(MESSAGE_SOURCES)
        rows = self._filtered_rows(
            rows_by_source=rows_by_source,
            source_filter={source for source in effective_sources if source in MESSAGE_SOURCES},
            category_filter=None,
        )
        rows_by_thread: dict[tuple[str, str], list[_NormalizedTrainingRow]] = defaultdict(list)

        for row in rows:
            thread_id = row.thread_id or row.source_row_id
            rows_by_thread[(row.source_system, thread_id)].append(row)

        samples: list[GroupedThreadClassificationSample] = []
        for (source_system, thread_id), thread_rows in sorted(rows_by_thread.items()):
            ordered_rows = sorted(
                thread_rows,
                key=lambda row: (row.message_order, row.timestamp, row.source_row_id),
            )
            primary_row = next((row for row in ordered_rows if row.is_relevant), ordered_rows[0])
            if not self._category_allowed(primary_row.primary_category, category_filter):
                continue

            samples.append(
                GroupedThreadClassificationSample(
                    sample_id=stable_entity_id(
                        "thread_classification",
                        0,
                        f"{source_system}:{thread_id}",
                    ),
                    source_system=source_system,
                    thread_id=thread_id,
                    label=primary_row.primary_category.value,
                    primary_category=primary_row.primary_category,
                    is_relevant=any(row.is_relevant for row in ordered_rows),
                    messages=[
                        ThreadMessage(
                            source_row_id=row.source_row_id,
                            message_order=index,
                            timestamp=row.timestamp.isoformat(),
                            text=row.text,
                            primary_category=row.primary_category,
                            is_relevant=row.is_relevant,
                        )
                        for index, row in enumerate(ordered_rows)
                    ],
                )
            )

        return GroupedThreadClassificationDataset(samples=samples)

    def build_cross_document_linking(
        self,
        *,
        rows_by_source: Mapping[str, Sequence[RowLike]],
        source_filter: SourceFilter = None,
        category_filter: CategoryFilter = None,
    ) -> CrossDocumentLinkingDataset:
        rows = self._filtered_rows(
            rows_by_source=rows_by_source,
            source_filter=source_filter,
            category_filter=category_filter,
        )
        positive_pairs: list[CrossDocumentLinkSample] = []
        negative_pairs: list[CrossDocumentLinkSample] = []

        for index, left_row in enumerate(rows):
            for right_row in rows[index + 1:]:
                if left_row.source_system == right_row.source_system:
                    continue

                shared_ids = sorted(left_row.business_entity_ids & right_row.business_entity_ids)
                if shared_ids:
                    shared_id = shared_ids[0]
                    positive_pairs.append(
                        CrossDocumentLinkSample(
                            sample_id=stable_entity_id(
                                "cross_document_link",
                                0,
                                f"{left_row.source_row_id}:{right_row.source_row_id}:{shared_id}",
                            ),
                            left_source_system=left_row.source_system,
                            left_row_id=left_row.source_row_id,
                            right_source_system=right_row.source_system,
                            right_row_id=right_row.source_row_id,
                            left_primary_category=left_row.primary_category,
                            right_primary_category=right_row.primary_category,
                            shared_object_type=self._object_type_from_id(shared_id),
                            shared_object_id=shared_id,
                            is_linked=True,
                        )
                    )
                elif len(negative_pairs) < max(len(positive_pairs), 1):
                    negative_pairs.append(
                        CrossDocumentLinkSample(
                            sample_id=stable_entity_id(
                                "cross_document_link",
                                0,
                                f"{left_row.source_row_id}:{right_row.source_row_id}:negative",
                            ),
                            left_source_system=left_row.source_system,
                            left_row_id=left_row.source_row_id,
                            right_source_system=right_row.source_system,
                            right_row_id=right_row.source_row_id,
                            left_primary_category=left_row.primary_category,
                            right_primary_category=right_row.primary_category,
                            shared_object_type=None,
                            shared_object_id=None,
                            is_linked=False,
                        )
                    )

        return CrossDocumentLinkingDataset(samples=[*positive_pairs, *negative_pairs])

    def _filtered_rows(
        self,
        *,
        rows_by_source: Mapping[str, Sequence[RowLike]],
        source_filter: SourceFilter,
        category_filter: CategoryFilter,
    ) -> list[_NormalizedTrainingRow]:
        normalized_rows: list[_NormalizedTrainingRow] = []

        for source_name, rows in rows_by_source.items():
            if source_name not in SUPPORTED_SOURCES:
                raise ValueError(f"unsupported source {source_name!r}")
            if not self._source_allowed(source_name, source_filter):
                continue

            for row in rows:
                normalized = self._normalize_row(source_name=source_name, row=row)
                if self._category_allowed(normalized.primary_category, category_filter):
                    normalized_rows.append(normalized)

        return normalized_rows

    def _normalize_row(
        self,
        *,
        source_name: str,
        row: RowLike,
    ) -> _NormalizedTrainingRow:
        payload = _row_to_dict(row)
        actual_source = str(_required_value(payload, "source_system"))
        if actual_source != source_name:
            raise ValueError(
                "row source_system does not match the source bucket "
                f"{source_name!r}: {actual_source!r}"
            )

        category = CommunicationCategory(str(_required_value(payload, "primary_category")))
        is_relevant = _required_value(payload, "is_relevant")
        if not isinstance(is_relevant, bool):
            raise ValueError("is_relevant must be a bool")

        target_object_type, target_object_id = self._target_from_provenance(payload)

        return _NormalizedTrainingRow(
            source_system=source_name,
            source_row_id=self._row_id(payload),
            text=self._text_for_row(source_name, payload),
            primary_category=category,
            is_relevant=is_relevant,
            rationale=(
                str(payload.get("relevance_reason"))
                if payload.get("relevance_reason")
                else None
            ),
            thread_id=self._thread_id(source_name, payload),
            message_order=self._message_order(source_name, payload),
            timestamp=self._timestamp(payload),
            target_object_type=target_object_type,
            target_object_id=target_object_id,
            business_entity_ids=_business_entity_ids(source_name, payload),
        )

    def _row_id(self, payload: Mapping[str, Any]) -> str:
        source_name = str(payload["source_system"])
        field_name = {
            "email": "email_id",
            "slack": "slack_message_id",
            "teams": "teams_message_id",
            "salesforce": "salesforce_record_id",
        }[source_name]
        return str(_required_value(payload, field_name))

    def _text_for_row(self, source_name: str, payload: Mapping[str, Any]) -> str:
        if source_name == "email":
            return self._join_text(payload.get("subject"), payload.get("body"))
        if source_name == "salesforce":
            return self._join_text(payload.get("subject"), payload.get("text_body"))
        return self._join_text(payload.get("body"))

    def _thread_id(self, source_name: str, payload: Mapping[str, Any]) -> str | None:
        if source_name == "email":
            return str(_required_value(payload, "thread_id"))
        if source_name in {"slack", "teams"}:
            thread_value = payload.get("thread_id")
            if thread_value is not None:
                return str(thread_value)
            return self._row_id(payload)
        return None

    def _message_order(self, source_name: str, payload: Mapping[str, Any]) -> int:
        if source_name == "email":
            return int(payload.get("message_index_in_thread", 0))
        return 0

    def _timestamp(self, payload: Mapping[str, Any]) -> datetime:
        raw_timestamp = str(_required_value(payload, "timestamp"))
        if raw_timestamp.endswith("Z"):
            raw_timestamp = f"{raw_timestamp[:-1]}+00:00"
        return datetime.fromisoformat(raw_timestamp)

    def _target_from_provenance(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[str | None, str | None]:
        provenance = payload.get("provenance")
        if not isinstance(provenance, Mapping):
            return None, None
        object_type = provenance.get("object_type")
        object_id = provenance.get("object_id")
        return (
            str(object_type) if object_type is not None else None,
            str(object_id) if object_id is not None else None,
        )

    def _object_type_from_id(self, entity_id: str) -> str:
        for prefix, object_type in (
            ("account_", "account"),
            ("contact_", "contact"),
            ("opportunity_", "opportunity"),
            ("event_", "event"),
            ("ticket_issue_", "ticket"),
            ("campaign_", "campaign"),
        ):
            if entity_id.startswith(prefix):
                return object_type
        return "entity"

    def _source_allowed(self, source_name: str, source_filter: SourceFilter) -> bool:
        return source_filter is None or source_name in source_filter

    def _category_allowed(
        self,
        category: CommunicationCategory,
        category_filter: CategoryFilter,
    ) -> bool:
        return category_filter is None or category in category_filter

    def _join_text(self, *values: object) -> str:
        return " ".join(
            str(value).strip()
            for value in values
            if isinstance(value, str) and value.strip()
        )
