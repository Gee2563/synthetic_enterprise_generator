from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

from synthetic_enterprise.app.settings import load_config
from synthetic_enterprise.domain import EnterpriseGraph
from synthetic_enterprise.evaluation import (
    DatasetQualityEvaluator,
    GoldSetBuilder,
    GoldSetConfig,
)
from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import (
    SOURCE_NAMES,
    DatasetSourceName,
    DatasetTargets,
    GenerationPipeline,
)
from synthetic_enterprise.labeling.taxonomy import CommunicationCategory
from synthetic_enterprise.storage.parquet_writer import (
    write_source_dataset,
    write_source_dataset_streaming,
)

CommandHandler = Callable[[argparse.Namespace], int]
LOGGER = logging.getLogger(__name__)
DEFAULT_GENERATION_CHUNK_SIZE = 10_000


def build_parser() -> argparse.ArgumentParser:
    """Build the project CLI parser."""

    app_config = load_config()
    parser = argparse.ArgumentParser(prog="synthetic-enterprise")
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_company_parser = subparsers.add_parser("simulate-company")
    _add_generation_args(simulate_company_parser, app_config=app_config)
    simulate_company_parser.set_defaults(handler=_handle_simulate_company)

    generate_email_parser = subparsers.add_parser("generate-email")
    _add_generation_args(generate_email_parser, app_config=app_config)
    generate_email_parser.add_argument("--email-count", type=int, default=200)
    generate_email_parser.set_defaults(handler=_handle_generate_email)

    generate_slack_parser = subparsers.add_parser("generate-slack")
    _add_generation_args(generate_slack_parser, app_config=app_config)
    generate_slack_parser.add_argument("--slack-count", type=int, default=200)
    generate_slack_parser.set_defaults(handler=_handle_generate_slack)

    generate_teams_parser = subparsers.add_parser("generate-teams")
    _add_generation_args(generate_teams_parser, app_config=app_config)
    generate_teams_parser.add_argument("--teams-count", type=int, default=200)
    generate_teams_parser.set_defaults(handler=_handle_generate_teams)

    generate_crm_parser = subparsers.add_parser("generate-crm")
    _add_generation_args(generate_crm_parser, app_config=app_config)
    generate_crm_parser.add_argument("--crm-count", type=int, default=200)
    generate_crm_parser.set_defaults(handler=_handle_generate_crm)

    generate_all_parser = subparsers.add_parser("generate-all")
    _add_generation_args(generate_all_parser, app_config=app_config)
    generate_all_parser.add_argument("--email-count", type=int, default=200)
    generate_all_parser.add_argument("--slack-count", type=int, default=200)
    generate_all_parser.add_argument("--teams-count", type=int, default=200)
    generate_all_parser.add_argument("--crm-count", type=int, default=200)
    generate_all_parser.set_defaults(handler=_handle_generate_all)

    quality_report_parser = subparsers.add_parser("quality-report")
    quality_report_parser.add_argument("--input-root", type=Path, required=True)
    quality_report_parser.add_argument("--output-path", type=Path, required=True)
    quality_report_parser.set_defaults(handler=_handle_quality_report)

    build_gold_set_parser = subparsers.add_parser("build-gold-set")
    build_gold_set_parser.add_argument("--input-root", type=Path, required=True)
    build_gold_set_parser.add_argument("--output-root", type=Path, required=True)
    build_gold_set_parser.add_argument(
        "--relevant-category",
        action="append",
        dest="relevant_categories",
        choices=sorted(category.value for category in CommunicationCategory),
        required=True,
    )
    build_gold_set_parser.add_argument(
        "--samples-per-relevant-category",
        type=int,
        default=1,
    )
    build_gold_set_parser.add_argument("--hard-negative-count", type=int, default=0)
    build_gold_set_parser.add_argument("--min-cross-system-examples", type=int, default=1)
    build_gold_set_parser.set_defaults(handler=_handle_build_gold_set)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process-style exit code."""

    parser = build_parser()

    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        handler = cast(CommandHandler, args.handler)
        return handler(args)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _add_generation_args(
    parser: argparse.ArgumentParser,
    *,
    app_config: Any,
) -> None:
    parser.add_argument("--seed", type=int, default=app_config.default_seed)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--account-count", type=int, default=20)
    parser.add_argument("--min-employees", type=int, default=25)
    parser.add_argument("--max-employees", type=int, default=250)
    parser.add_argument("--locale", default=app_config.default_locale)
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--noise-ratio", type=float, default=0.8)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.25)
    parser.add_argument("--cross-system-ratio", type=float, default=0.5)
    parser.add_argument("--verbosity-ratio", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--generation-chunk-size", type=int, default=DEFAULT_GENERATION_CHUNK_SIZE)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--write-csv", action="store_true")


def _handle_simulate_company(args: argparse.Namespace) -> int:
    pipeline = GenerationPipeline(context=_generator_context_from_args(args))
    enterprise = pipeline.builder.build(
        pipeline.context,
        account_count=int(args.account_count),
    )
    _write_company_state(enterprise=enterprise, destination_root=Path(args.output_root))
    return 0


def _handle_generate_email(args: argparse.Namespace) -> int:
    return _generate_single_source(
        args=args,
        source_name="email",
        row_count=int(args.email_count),
    )


def _handle_generate_slack(args: argparse.Namespace) -> int:
    return _generate_single_source(
        args=args,
        source_name="slack",
        row_count=int(args.slack_count),
    )


def _handle_generate_teams(args: argparse.Namespace) -> int:
    return _generate_single_source(
        args=args,
        source_name="teams",
        row_count=int(args.teams_count),
    )


def _handle_generate_crm(args: argparse.Namespace) -> int:
    return _generate_single_source(
        args=args,
        source_name="salesforce",
        row_count=int(args.crm_count),
    )


def _handle_generate_all(args: argparse.Namespace) -> int:
    pipeline = GenerationPipeline(context=_generator_context_from_args(args))
    targets = DatasetTargets(
        account_count=int(args.account_count),
        email_count=int(args.email_count),
        slack_count=int(args.slack_count),
        teams_count=int(args.teams_count),
        salesforce_count=int(args.crm_count),
        chunk_size=args.chunk_size,
    )
    output_root = Path(args.output_root)

    if args.stream:
        streamed_dataset = pipeline.generate_dataset_streaming(
            targets=targets,
            destination_root=output_root,
            generation_chunk_size=int(args.generation_chunk_size),
            write_csv=bool(args.write_csv),
            logger=LOGGER,
        )
        enterprise = streamed_dataset.enterprise
    else:
        dataset = pipeline.generate_dataset(
            targets=targets,
            destination_root=output_root,
            write_csv=bool(args.write_csv),
        )
        enterprise = dataset.enterprise

    _write_company_state(enterprise=enterprise, destination_root=output_root)
    return 0


def _handle_quality_report(args: argparse.Namespace) -> int:
    rows_by_source = _load_rows_by_source(Path(args.input_root))
    enterprise = _load_company_state(Path(args.input_root), required=False)
    report = DatasetQualityEvaluator(enterprise=enterprise).evaluate(rows_by_source)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return 0


def _handle_build_gold_set(args: argparse.Namespace) -> int:
    input_root = Path(args.input_root)
    enterprise = _load_company_state(input_root, required=True)
    rows_by_source = _load_rows_by_source(input_root)
    gold_set = GoldSetBuilder(enterprise=enterprise).create(
        rows_by_source=rows_by_source,
        config=GoldSetConfig(
            relevant_categories=tuple(
                CommunicationCategory(category)
                for category in args.relevant_categories
            ),
            samples_per_relevant_category=int(args.samples_per_relevant_category),
            hard_negative_count=int(args.hard_negative_count),
            min_cross_system_examples=int(args.min_cross_system_examples),
        ),
    )
    GoldSetBuilder(enterprise=enterprise).export(
        gold_set=gold_set,
        destination_root=Path(args.output_root),
    )
    return 0


def _generate_single_source(
    *,
    args: argparse.Namespace,
    source_name: DatasetSourceName,
    row_count: int,
) -> int:
    pipeline = GenerationPipeline(context=_generator_context_from_args(args))
    output_root = Path(args.output_root)
    targets = _single_source_targets(
        source_name=source_name,
        row_count=row_count,
        account_count=int(args.account_count),
        chunk_size=args.chunk_size,
    )
    enterprise = pipeline.build_enterprise(account_count=targets.account_count)
    bundles = pipeline.select_cross_system_bundles(enterprise=enterprise, targets=targets)

    if args.stream:
        chunk_plans = pipeline.plan_source_chunks(
            source_name=source_name,
            total_rows=row_count,
            generation_chunk_size=int(args.generation_chunk_size),
        )
        write_source_dataset_streaming(
            row_chunks=pipeline._iter_source_chunks(
                source_name=source_name,
                enterprise=enterprise,
                bundles=bundles,
                chunk_plans=chunk_plans,
                logger=LOGGER,
            ),
            destination_root=output_root,
            source_name=source_name,
            seed=pipeline.context.seed,
            config=pipeline.context.config,
            write_csv=bool(args.write_csv),
        )
    else:
        rows = pipeline.build_source_records(
            source_name=source_name,
            enterprise=enterprise,
            targets=targets,
            bundles=bundles,
        )
        write_source_dataset(
            rows=rows,
            destination_root=output_root,
            source_name=source_name,
            seed=pipeline.context.seed,
            config=pipeline.context.config,
            chunk_size=args.chunk_size,
            write_csv=bool(args.write_csv),
        )

    _write_company_state(enterprise=enterprise, destination_root=output_root)
    return 0


def _single_source_targets(
    *,
    source_name: DatasetSourceName,
    row_count: int,
    account_count: int,
    chunk_size: int | None,
) -> DatasetTargets:
    source_counts = {name: max(row_count, 1) for name in SOURCE_NAMES}
    source_counts[source_name] = row_count
    return DatasetTargets(
        account_count=account_count,
        email_count=source_counts["email"],
        slack_count=source_counts["slack"],
        teams_count=source_counts["teams"],
        salesforce_count=source_counts["salesforce"],
        chunk_size=chunk_size,
    )


def _generator_context_from_args(args: argparse.Namespace) -> GeneratorContext:
    config = GeneratorConfig(
        company_size=CompanySizeConfig(
            min_employees=int(args.min_employees),
            max_employees=int(args.max_employees),
        ),
        locale=str(args.locale),
        timezone=str(args.timezone),
        noise_ratio=float(args.noise_ratio),
        hard_negative_ratio=float(args.hard_negative_ratio),
        cross_system_ratio=float(args.cross_system_ratio),
        verbosity_ratio=float(args.verbosity_ratio),
    )
    return GeneratorContext(seed=int(args.seed), config=config)


def _write_company_state(
    *,
    enterprise: EnterpriseGraph,
    destination_root: Path,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    (destination_root / "company_state.json").write_text(
        enterprise.to_json(),
        encoding="utf-8",
    )


def _load_company_state(input_root: Path, *, required: bool) -> EnterpriseGraph | None:
    company_state_path = input_root / "company_state.json"
    if not company_state_path.exists():
        if required:
            raise FileNotFoundError(f"missing company state at {company_state_path}")
        return None
    return cast(
        EnterpriseGraph,
        EnterpriseGraph.from_json(company_state_path.read_text(encoding="utf-8")),
    )


def _load_rows_by_source(input_root: Path) -> dict[str, list[dict[str, object]]]:
    if not input_root.exists():
        raise FileNotFoundError(f"input root does not exist: {input_root}")

    rows_by_source: dict[str, list[dict[str, object]]] = {}

    for partition_path in sorted(input_root.glob("source=*")):
        if not partition_path.is_dir():
            continue

        _, _, source_name = partition_path.name.partition("=")
        parquet_root = partition_path / "parquet"
        parquet_files = sorted(parquet_root.glob("*.parquet"))
        if not parquet_files:
            continue

        frames = [pd.read_parquet(parquet_file) for parquet_file in parquet_files]
        frame = pd.concat(frames, ignore_index=True)
        normalized_frame = frame.astype(object).where(pd.notna(frame), None)
        rows_by_source[source_name] = [
            {str(key): value for key, value in row.items()}
            for row in normalized_frame.to_dict(orient="records")
        ]

    if not rows_by_source:
        raise FileNotFoundError(f"no parquet source exports found under {input_root}")

    return rows_by_source


if __name__ == "__main__":
    raise SystemExit(main())
