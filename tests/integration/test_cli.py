from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthetic_enterprise.app.cli import build_parser, main


def test_cli_commands_parse_args_correctly(tmp_path: Path) -> None:
    parser = build_parser()
    output_root = tmp_path / "out"
    input_root = tmp_path / "input"

    command_specs = [
        (
            [
                "simulate-company",
                "--seed",
                "7",
                "--output-root",
                str(output_root),
                "--account-count",
                "5",
            ],
            "simulate-company",
        ),
        (
            [
                "generate-email",
                "--seed",
                "7",
                "--output-root",
                str(output_root),
                "--email-count",
                "12",
            ],
            "generate-email",
        ),
        (
            [
                "generate-slack",
                "--seed",
                "7",
                "--output-root",
                str(output_root),
                "--slack-count",
                "12",
            ],
            "generate-slack",
        ),
        (
            [
                "generate-teams",
                "--seed",
                "7",
                "--output-root",
                str(output_root),
                "--teams-count",
                "12",
            ],
            "generate-teams",
        ),
        (
            [
                "generate-crm",
                "--seed",
                "7",
                "--output-root",
                str(output_root),
                "--crm-count",
                "12",
            ],
            "generate-crm",
        ),
        (
            [
                "generate-all",
                "--seed",
                "7",
                "--output-root",
                str(output_root),
                "--email-count",
                "12",
                "--slack-count",
                "12",
                "--teams-count",
                "12",
                "--crm-count",
                "12",
            ],
            "generate-all",
        ),
        (
            [
                "quality-report",
                "--input-root",
                str(input_root),
                "--output-path",
                str(tmp_path / "quality_report.json"),
            ],
            "quality-report",
        ),
        (
            [
                "build-gold-set",
                "--input-root",
                str(input_root),
                "--output-root",
                str(tmp_path / "gold"),
                "--relevant-category",
                "buying_signal",
                "--relevant-category",
                "event_attendance",
                "--relevant-category",
                "blocker",
            ],
            "build-gold-set",
        ),
    ]

    for argv, command_name in command_specs:
        parsed = parser.parse_args(argv)
        assert parsed.command == command_name


def test_cli_invalid_configs_fail_with_clear_errors(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    exit_code = main(
        [
            "generate-email",
            "--seed",
            "11",
            "--output-root",
            str(tmp_path / "bad"),
            "--email-count",
            "0",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "email_count must be greater than zero" in captured.err


def test_cli_seed_and_output_path_are_respected_and_manifests_are_written(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "email-dataset"
    exit_code = main(
        [
            "generate-email",
            "--seed",
            "123",
            "--output-root",
            str(output_root),
            "--email-count",
            "16",
            "--account-count",
            "6",
        ]
    )

    manifest_path = output_root / "source=email" / "manifest.json"
    company_state_path = output_root / "company_state.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    company_payload = json.loads(company_state_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert manifest_path.exists()
    assert company_state_path.exists()
    assert payload["seed"] == 123
    assert payload["row_count"] == 16
    assert len(company_payload["companies"]) == 1


def test_cli_chunked_generation_works_and_manifests_are_written(tmp_path: Path) -> None:
    output_root = tmp_path / "all-dataset"
    exit_code = main(
        [
            "generate-all",
            "--seed",
            "88",
            "--output-root",
            str(output_root),
            "--email-count",
            "60",
            "--slack-count",
            "60",
            "--teams-count",
            "60",
            "--crm-count",
            "60",
            "--chunk-size",
            "20",
            "--stream",
            "--generation-chunk-size",
            "20",
        ]
    )

    assert exit_code == 0

    for source_name in ("email", "slack", "teams", "salesforce"):
        manifest_path = output_root / f"source={source_name}" / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        parquet_root = output_root / f"source={source_name}" / "parquet"
        parquet_files = sorted(parquet_root.glob("*.parquet"))

        assert manifest_path.exists()
        assert payload["chunk_count"] == 3
        assert len(parquet_files) == 3


def test_cli_quality_report_and_gold_set_commands(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    quality_report_path = tmp_path / "quality_report.json"
    gold_root = tmp_path / "gold"

    generate_exit_code = main(
        [
            "generate-all",
            "--seed",
            "44",
            "--output-root",
            str(dataset_root),
            "--email-count",
            "60",
            "--slack-count",
            "60",
            "--teams-count",
            "60",
            "--crm-count",
            "60",
            "--chunk-size",
            "20",
        ]
    )
    quality_exit_code = main(
        [
            "quality-report",
            "--input-root",
            str(dataset_root),
            "--output-path",
            str(quality_report_path),
        ]
    )
    gold_exit_code = main(
        [
            "build-gold-set",
            "--input-root",
            str(dataset_root),
            "--output-root",
            str(gold_root),
            "--relevant-category",
            "buying_signal",
            "--relevant-category",
            "event_attendance",
            "--relevant-category",
            "blocker",
            "--samples-per-relevant-category",
            "1",
            "--hard-negative-count",
            "2",
            "--min-cross-system-examples",
            "1",
        ]
    )

    quality_payload = json.loads(quality_report_path.read_text(encoding="utf-8"))

    assert generate_exit_code == 0
    assert quality_exit_code == 0
    assert gold_exit_code == 0
    assert quality_report_path.exists()
    assert "relevance_rate" in quality_payload
    assert (gold_root / "gold_set.parquet").exists()
    assert (gold_root / "gold_set.jsonl").exists()
