from pathlib import Path


def test_package_structure_is_valid() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "synthetic_enterprise"

    expected_paths = [
        "__init__.py",
        "app/__init__.py",
        "app/settings.py",
        "contracts/__init__.py",
        "contracts/company_state.py",
        "contracts/records/__init__.py",
        "contracts/records/email.py",
        "contracts/records/salesforce.py",
        "contracts/records/slack.py",
        "contracts/records/teams.py",
        "domain/__init__.py",
        "domain/company.py",
        "evaluation/__init__.py",
        "evaluation/gold_set.py",
        "evaluation/quality_report.py",
        "generation/__init__.py",
        "generation/config.py",
        "generation/context.py",
        "generation/cross_system.py",
        "generation/hard_negatives.py",
        "generation/pipeline.py",
        "labeling/__init__.py",
        "labeling/grounding.py",
        "labeling/relevance.py",
        "labeling/taxonomy.py",
        "sources/__init__.py",
        "sources/email/__init__.py",
        "sources/email/generator.py",
        "sources/salesforce/__init__.py",
        "sources/salesforce/renderer.py",
        "sources/slack/__init__.py",
        "sources/slack/renderer.py",
        "sources/teams/__init__.py",
        "sources/teams/renderer.py",
        "storage/__init__.py",
        "storage/parquet_writer.py",
    ]

    for relative_path in expected_paths:
        assert (package_root / relative_path).exists(), relative_path
