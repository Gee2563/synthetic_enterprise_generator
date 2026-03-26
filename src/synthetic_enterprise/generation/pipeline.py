from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar

from synthetic_enterprise.contracts.manifest import DatasetManifest
from synthetic_enterprise.contracts.records.email import EmailRecord
from synthetic_enterprise.contracts.records.salesforce import SalesforceRecord
from synthetic_enterprise.contracts.records.slack import SlackRecord
from synthetic_enterprise.contracts.records.teams import TeamsRecord
from synthetic_enterprise.domain import EnterpriseGraph, Event
from synthetic_enterprise.generation.company_builder import CompanyBuilder
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.cross_system import CrossSystemEventBundle, CrossSystemRenderer
from synthetic_enterprise.sources.email.renderer import EmailRenderer
from synthetic_enterprise.sources.salesforce.renderer import SalesforceRenderer
from synthetic_enterprise.sources.slack.renderer import SlackRenderer
from synthetic_enterprise.sources.teams.renderer import TeamsRenderer
from synthetic_enterprise.storage.parquet_writer import (
    write_source_dataset,
    write_source_dataset_streaming,
)

DatasetSourceName: TypeAlias = Literal["email", "slack", "teams", "salesforce"]
DatasetRecord: TypeAlias = EmailRecord | SlackRecord | TeamsRecord | SalesforceRecord
SOURCE_NAMES: tuple[DatasetSourceName, ...] = ("email", "slack", "teams", "salesforce")
ItemT = TypeVar("ItemT")


@dataclass(frozen=True, slots=True)
class DatasetTargets:
    """Target entity and row counts for a small end-to-end dataset."""

    account_count: int = 20
    email_count: int = 200
    slack_count: int = 200
    teams_count: int = 200
    salesforce_count: int = 200
    chunk_size: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "account_count",
            "email_count",
            "slack_count",
            "teams_count",
            "salesforce_count",
        ):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be greater than zero")

        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero when provided")

    def rows_for_source(self, source_name: DatasetSourceName) -> int:
        return {
            "email": self.email_count,
            "slack": self.slack_count,
            "teams": self.teams_count,
            "salesforce": self.salesforce_count,
        }[source_name]


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    """Deterministic generation plan for one source chunk."""

    source_name: DatasetSourceName
    chunk_index: int
    row_count: int
    seed: int


@dataclass(slots=True)
class GeneratedDataset:
    """Combined enterprise state, rendered records, and export manifests."""

    enterprise: EnterpriseGraph
    cross_system_bundles: list[CrossSystemEventBundle]
    email_records: list[EmailRecord]
    slack_records: list[SlackRecord]
    teams_records: list[TeamsRecord]
    salesforce_records: list[SalesforceRecord]
    manifests: dict[str, DatasetManifest]


@dataclass(slots=True)
class StreamedGeneratedDataset:
    """Streaming generation result without retaining full record lists."""

    enterprise: EnterpriseGraph
    cross_system_bundles: list[CrossSystemEventBundle]
    manifests: dict[str, DatasetManifest]
    chunk_plans: dict[str, list[ChunkPlan]]


@dataclass(slots=True)
class GenerationPipeline:
    """End-to-end orchestration for deterministic multi-source datasets."""

    context: GeneratorContext
    builder: CompanyBuilder = field(default_factory=CompanyBuilder)

    def plan_source_chunks(
        self,
        *,
        source_name: DatasetSourceName,
        total_rows: int,
        generation_chunk_size: int,
    ) -> list[ChunkPlan]:
        if total_rows <= 0:
            raise ValueError("total_rows must be greater than zero")
        if generation_chunk_size <= 0:
            raise ValueError("generation_chunk_size must be greater than zero")

        plans: list[ChunkPlan] = []
        remaining_rows = total_rows
        chunk_index = 0

        while remaining_rows > 0:
            row_count = min(generation_chunk_size, remaining_rows)
            plans.append(
                ChunkPlan(
                    source_name=source_name,
                    chunk_index=chunk_index,
                    row_count=row_count,
                    seed=self.context.derive_seed(
                        f"stream:{source_name}:chunk:{chunk_index}"
                    ),
                )
            )
            remaining_rows -= row_count
            chunk_index += 1

        return plans

    def generate_dataset(
        self,
        *,
        targets: DatasetTargets,
        destination_root: Path,
        write_csv: bool = False,
    ) -> GeneratedDataset:
        enterprise = self.builder.build(
            self.context,
            account_count=targets.account_count,
        )
        bundles = self._selected_cross_system_bundles(enterprise=enterprise, targets=targets)

        email_records = self._build_email_records(
            enterprise=enterprise,
            targets=targets,
            bundles=bundles,
        )
        slack_records = self._build_slack_records(
            enterprise=enterprise,
            targets=targets,
            bundles=bundles,
        )
        teams_records = self._build_teams_records(
            enterprise=enterprise,
            targets=targets,
            bundles=bundles,
        )
        salesforce_records = self._build_salesforce_records(
            enterprise=enterprise,
            targets=targets,
            bundles=bundles,
        )

        manifests = {
            "email": write_source_dataset(
                rows=email_records,
                destination_root=destination_root,
                source_name="email",
                seed=self.context.seed,
                config=self.context.config,
                chunk_size=targets.chunk_size,
                write_csv=write_csv,
            ),
            "slack": write_source_dataset(
                rows=slack_records,
                destination_root=destination_root,
                source_name="slack",
                seed=self.context.seed,
                config=self.context.config,
                chunk_size=targets.chunk_size,
                write_csv=write_csv,
            ),
            "teams": write_source_dataset(
                rows=teams_records,
                destination_root=destination_root,
                source_name="teams",
                seed=self.context.seed,
                config=self.context.config,
                chunk_size=targets.chunk_size,
                write_csv=write_csv,
            ),
            "salesforce": write_source_dataset(
                rows=salesforce_records,
                destination_root=destination_root,
                source_name="salesforce",
                seed=self.context.seed,
                config=self.context.config,
                chunk_size=targets.chunk_size,
                write_csv=write_csv,
            ),
        }

        return GeneratedDataset(
            enterprise=enterprise,
            cross_system_bundles=bundles,
            email_records=email_records,
            slack_records=slack_records,
            teams_records=teams_records,
            salesforce_records=salesforce_records,
            manifests=manifests,
        )

    def generate_dataset_streaming(
        self,
        *,
        targets: DatasetTargets,
        destination_root: Path,
        generation_chunk_size: int,
        write_csv: bool = False,
        logger: logging.Logger | None = None,
    ) -> StreamedGeneratedDataset:
        if generation_chunk_size <= 0:
            raise ValueError("generation_chunk_size must be greater than zero")

        progress_logger = logger or logging.getLogger(__name__)
        enterprise = self.builder.build(
            self.context,
            account_count=targets.account_count,
        )
        bundles = self._selected_cross_system_bundles(enterprise=enterprise, targets=targets)
        manifests: dict[str, DatasetManifest] = {}
        chunk_plans: dict[str, list[ChunkPlan]] = {}

        for source_name in SOURCE_NAMES:
            plans = self.plan_source_chunks(
                source_name=source_name,
                total_rows=targets.rows_for_source(source_name),
                generation_chunk_size=generation_chunk_size,
            )
            chunk_plans[source_name] = plans
            progress_logger.info(
                "Starting source %s with %s rows across %s chunks",
                source_name,
                targets.rows_for_source(source_name),
                len(plans),
            )
            manifests[source_name] = write_source_dataset_streaming(
                row_chunks=self._iter_source_chunks(
                    source_name=source_name,
                    enterprise=enterprise,
                    bundles=bundles,
                    chunk_plans=plans,
                    logger=progress_logger,
                ),
                destination_root=destination_root,
                source_name=source_name,
                seed=self.context.seed,
                config=self.context.config,
                write_csv=write_csv,
            )
            progress_logger.info(
                "Completed source %s with %s rows across %s chunks",
                source_name,
                manifests[source_name].row_count,
                manifests[source_name].chunk_count,
            )

        return StreamedGeneratedDataset(
            enterprise=enterprise,
            cross_system_bundles=bundles,
            manifests=manifests,
            chunk_plans=chunk_plans,
        )

    def _iter_source_chunks(
        self,
        *,
        source_name: DatasetSourceName,
        enterprise: EnterpriseGraph,
        bundles: list[CrossSystemEventBundle],
        chunk_plans: Sequence[ChunkPlan],
        logger: logging.Logger,
    ) -> Iterator[list[DatasetRecord]]:
        buffered_relevant_rows = self._source_records_from_bundles(
            source_name=source_name,
            bundles=bundles,
        )
        relevant_index = 0
        batch_iteration = 0

        for plan in chunk_plans:
            logger.info(
                "Generating %s chunk %s/%s with seed %s",
                source_name,
                plan.chunk_index + 1,
                len(chunk_plans),
                plan.seed,
            )
            chunk_rows: list[DatasetRecord] = []

            while len(chunk_rows) < plan.row_count and relevant_index < len(buffered_relevant_rows):
                chunk_rows.append(buffered_relevant_rows[relevant_index])
                relevant_index += 1

            batch_index = 0
            while len(chunk_rows) < plan.row_count:
                candidate_rows = self._source_noise_rows(
                    source_name=source_name,
                    enterprise=enterprise,
                    batch_iteration=batch_iteration,
                    chunk_seed=plan.seed,
                    batch_index=batch_index,
                )
                remaining = plan.row_count - len(chunk_rows)
                chunk_rows.extend(candidate_rows[:remaining])
                batch_iteration += 1
                batch_index += 1

            yield chunk_rows

    def _selected_cross_system_bundles(
        self,
        *,
        enterprise: EnterpriseGraph,
        targets: DatasetTargets,
    ) -> list[CrossSystemEventBundle]:
        renderer = CrossSystemRenderer(
            context=self._variant_context("cross-system"),
            enterprise=enterprise,
        )
        candidate_bundles = renderer.render_all_events()
        relevant_budget = int(
            round(
                (
                    targets.email_count
                    + targets.slack_count
                    + targets.teams_count
                    + targets.salesforce_count
                )
                * (1.0 - self.context.noise_ratio)
            )
        )
        selected: list[CrossSystemEventBundle] = []
        used_relevant = 0

        for bundle in candidate_bundles:
            bundle_relevant = sum(record.is_relevant for record in bundle.all_records)
            if selected and used_relevant + bundle_relevant > relevant_budget:
                break
            selected.append(bundle)
            used_relevant += bundle_relevant

        return selected

    def _build_email_records(
        self,
        *,
        enterprise: EnterpriseGraph,
        targets: DatasetTargets,
        bundles: list[CrossSystemEventBundle],
    ) -> list[EmailRecord]:
        records = [
            record
            for bundle in bundles
            for record in bundle.email_records
        ]
        events = self._account_linked_events(enterprise)
        iteration = 0

        while len(records) < targets.email_count:
            event = events[iteration % len(events)]
            renderer = EmailRenderer(
                context=self._variant_context(f"email-noise:{iteration}"),
                enterprise=enterprise,
            )
            noise_rows = [
                record
                for record in renderer.generate_messages(event_id=event.id)
                if not record.is_relevant
            ]
            records.extend(noise_rows)
            iteration += 1

        return sorted(
            records[:targets.email_count],
            key=lambda record: (record.timestamp, record.email_id),
        )

    def _build_slack_records(
        self,
        *,
        enterprise: EnterpriseGraph,
        targets: DatasetTargets,
        bundles: list[CrossSystemEventBundle],
    ) -> list[SlackRecord]:
        records = [
            record
            for bundle in bundles
            for record in bundle.slack_records
        ]
        account_ids = [account.id for account in enterprise.customer_accounts]
        iteration = 0

        while len(records) < targets.slack_count:
            account_id = account_ids[iteration % len(account_ids)]
            renderer = SlackRenderer(
                context=self._noise_only_context(f"slack-noise:{iteration}"),
                enterprise=self._enterprise_for_account(enterprise, account_id),
            )
            noise_rows = [
                record
                for record in renderer.generate_messages()
                if not record.is_relevant
            ]
            records.extend(noise_rows)
            iteration += 1

        return sorted(
            records[:targets.slack_count],
            key=lambda record: (record.timestamp, record.slack_message_id),
        )

    def _build_teams_records(
        self,
        *,
        enterprise: EnterpriseGraph,
        targets: DatasetTargets,
        bundles: list[CrossSystemEventBundle],
    ) -> list[TeamsRecord]:
        records = [
            record
            for bundle in bundles
            for record in bundle.teams_records
        ]
        account_ids = [account.id for account in enterprise.customer_accounts]
        iteration = 0

        while len(records) < targets.teams_count:
            account_id = account_ids[iteration % len(account_ids)]
            renderer = TeamsRenderer(
                context=self._noise_only_context(f"teams-noise:{iteration}"),
                enterprise=self._enterprise_for_account(enterprise, account_id),
            )
            noise_rows = [
                record
                for record in renderer.generate_messages()
                if not record.is_relevant
            ]
            records.extend(noise_rows)
            iteration += 1

        return sorted(
            records[:targets.teams_count],
            key=lambda record: (record.timestamp, record.teams_message_id),
        )

    def _build_salesforce_records(
        self,
        *,
        enterprise: EnterpriseGraph,
        targets: DatasetTargets,
        bundles: list[CrossSystemEventBundle],
    ) -> list[SalesforceRecord]:
        records = [
            record
            for bundle in bundles
            for record in bundle.salesforce_records
        ]
        account_ids = [account.id for account in enterprise.customer_accounts]
        iteration = 0

        while len(records) < targets.salesforce_count:
            account_id = account_ids[iteration % len(account_ids)]
            renderer = SalesforceRenderer(
                context=self._variant_context(f"salesforce-noise:{iteration}"),
                enterprise=self._enterprise_for_account(enterprise, account_id),
            )
            noise_rows = [
                record
                for record in renderer.generate_records()
                if not record.is_relevant
            ]
            records.extend(noise_rows)
            iteration += 1

        return sorted(
            records[:targets.salesforce_count],
            key=lambda record: (record.timestamp, record.salesforce_record_id),
        )

    def _source_records_from_bundles(
        self,
        *,
        source_name: DatasetSourceName,
        bundles: Sequence[CrossSystemEventBundle],
    ) -> list[DatasetRecord]:
        if source_name == "email":
            return [record for bundle in bundles for record in bundle.email_records]
        if source_name == "slack":
            return [record for bundle in bundles for record in bundle.slack_records]
        if source_name == "teams":
            return [record for bundle in bundles for record in bundle.teams_records]
        return [record for bundle in bundles for record in bundle.salesforce_records]

    def _source_noise_rows(
        self,
        *,
        source_name: DatasetSourceName,
        enterprise: EnterpriseGraph,
        batch_iteration: int,
        chunk_seed: int,
        batch_index: int,
    ) -> list[DatasetRecord]:
        batch_seed = GeneratorContext(seed=chunk_seed, config=self.context.config).derive_seed(
            f"batch:{source_name}:{batch_index}"
        )

        if source_name == "email":
            event = self._account_linked_events(enterprise)[
                batch_iteration % len(self._account_linked_events(enterprise))
            ]
            email_renderer = EmailRenderer(
                context=self._context_from_seed(batch_seed),
                enterprise=enterprise,
            )
            return [
                record
                for record in email_renderer.generate_messages(event_id=event.id)
                if not record.is_relevant
            ]

        account_id = enterprise.customer_accounts[
            batch_iteration % len(enterprise.customer_accounts)
        ].id
        account_enterprise = self._enterprise_for_account(enterprise, account_id)

        if source_name == "slack":
            slack_renderer = SlackRenderer(
                context=self._context_from_seed(batch_seed, noise_only=True),
                enterprise=account_enterprise,
            )
            return [
                record
                for record in slack_renderer.generate_messages()
                if not record.is_relevant
            ]

        if source_name == "teams":
            teams_renderer = TeamsRenderer(
                context=self._context_from_seed(batch_seed, noise_only=True),
                enterprise=account_enterprise,
            )
            return [
                record
                for record in teams_renderer.generate_messages()
                if not record.is_relevant
            ]

        salesforce_renderer = SalesforceRenderer(
            context=self._context_from_seed(batch_seed),
            enterprise=account_enterprise,
        )
        return [
            record
            for record in salesforce_renderer.generate_records()
            if not record.is_relevant
        ]

    def _variant_context(self, namespace: str) -> GeneratorContext:
        return self._context_from_seed(
            self.context.derive_seed(f"pipeline:{namespace}"),
        )

    def _noise_only_context(self, namespace: str) -> GeneratorContext:
        return self._context_from_seed(
            self.context.derive_seed(f"pipeline:{namespace}"),
            noise_only=True,
        )

    def _context_from_seed(self, seed: int, *, noise_only: bool = False) -> GeneratorContext:
        config = self.context.config
        if noise_only:
            config = replace(
                self.context.config,
                noise_ratio=1.0,
                hard_negative_ratio=0.0,
            )
        return GeneratorContext(seed=seed, config=config)

    def _account_linked_events(self, enterprise: EnterpriseGraph) -> list[Event]:
        return [event for event in enterprise.events if event.account_id is not None]

    def _enterprise_for_account(
        self,
        enterprise: EnterpriseGraph,
        account_id: str,
    ) -> EnterpriseGraph:
        return enterprise.model_copy(
            update={
                "customer_accounts": self._account_first(
                    enterprise.customer_accounts,
                    account_id=account_id,
                    relation=lambda account: account.id,
                ),
                "contacts": self._account_first(
                    enterprise.contacts,
                    account_id=account_id,
                    relation=lambda contact: contact.account_id,
                ),
                "opportunities": self._account_first(
                    enterprise.opportunities,
                    account_id=account_id,
                    relation=lambda opportunity: opportunity.account_id,
                ),
                "events": self._account_first(
                    enterprise.events,
                    account_id=account_id,
                    relation=lambda event: event.account_id,
                ),
                "ticket_issues": self._account_first(
                    enterprise.ticket_issues,
                    account_id=account_id,
                    relation=lambda ticket: ticket.account_id,
                ),
            }
        )

    @staticmethod
    def _account_first(
        items: list[ItemT],
        *,
        account_id: str,
        relation: Callable[[ItemT], str | None],
    ) -> list[ItemT]:
        matching = [item for item in items if relation(item) == account_id]
        remaining = [item for item in items if relation(item) != account_id]
        return [*matching, *remaining]
