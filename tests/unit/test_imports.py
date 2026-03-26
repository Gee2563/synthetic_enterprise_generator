from importlib import import_module

import synthetic_enterprise

EXPECTED_MODULES = [
    "synthetic_enterprise.app.settings",
    "synthetic_enterprise.contracts.company_state",
    "synthetic_enterprise.contracts.records.email",
    "synthetic_enterprise.contracts.records.salesforce",
    "synthetic_enterprise.contracts.records.slack",
    "synthetic_enterprise.contracts.records.teams",
    "synthetic_enterprise.domain.company",
    "synthetic_enterprise.evaluation.gold_set",
    "synthetic_enterprise.evaluation.quality_report",
    "synthetic_enterprise.generation.config",
    "synthetic_enterprise.generation.context",
    "synthetic_enterprise.generation.cross_system",
    "synthetic_enterprise.generation.hard_negatives",
    "synthetic_enterprise.generation.pipeline",
    "synthetic_enterprise.labeling.grounding",
    "synthetic_enterprise.labeling.relevance",
    "synthetic_enterprise.labeling.taxonomy",
    "synthetic_enterprise.sources.email.generator",
    "synthetic_enterprise.sources.email.templates",
    "synthetic_enterprise.sources.salesforce.renderer",
    "synthetic_enterprise.sources.slack.renderer",
    "synthetic_enterprise.sources.teams.renderer",
    "synthetic_enterprise.storage.parquet_writer",
]


def test_package_imports_work() -> None:
    assert synthetic_enterprise.__version__ == "0.1.0"

    for module_name in EXPECTED_MODULES:
        module = import_module(module_name)
        assert module is not None
