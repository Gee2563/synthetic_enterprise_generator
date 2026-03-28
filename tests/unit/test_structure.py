from pathlib import Path


def test_package_structure_is_valid() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "synthetic_enterprise"

    expected_paths = [
        "__init__.py",
        "app/__init__.py",
        "app/cli.py",
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
        "domain/scenarios.py",
        "evaluation/__init__.py",
        "evaluation/comparison.py",
        "evaluation/gold_set.py",
        "evaluation/quality_report.py",
        "evaluation/regression.py",
        "evaluation/training_formats.py",
        "generation/__init__.py",
        "generation/config.py",
        "generation/context.py",
        "generation/account_profiles.py",
        "generation/cross_system.py",
        "generation/hard_negatives.py",
        "generation/language_profiles.py",
        "generation/personas.py",
        "generation/pipeline.py",
        "labeling/__init__.py",
        "labeling/grounding.py",
        "labeling/relevance.py",
        "labeling/taxonomy.py",
        "sources/__init__.py",
        "sources/email/__init__.py",
        "sources/email/composition.py",
        "sources/email/generator.py",
        "sources/email/paraphrases.py",
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
