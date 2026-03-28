"""Dataset quality evaluation utilities."""

from synthetic_enterprise.evaluation.comparison import (
    PhaseMetricDelta,
    PhaseMetricsComparator,
    PhaseMetricsComparisonReport,
    PhaseMetricThresholds,
)
from synthetic_enterprise.evaluation.gold_set import (
    GoldExample,
    GoldSet,
    GoldSetBuilder,
    GoldSetConfig,
    GoldSetV2Config,
)
from synthetic_enterprise.evaluation.quality_report import (
    DatasetQualityEvaluator,
    EntityCoverageReport,
    QualityReport,
)
from synthetic_enterprise.evaluation.regression import (
    PHASE1_REGRESSION_TARGETS,
    RegressionFixture,
    RegressionSnapshot,
    build_regression_context,
    build_regression_fixture,
    build_regression_snapshot,
    collect_reference_errors,
)
from synthetic_enterprise.evaluation.training_formats import (
    CrossDocumentLinkingDataset,
    CrossDocumentLinkSample,
    GroupedThreadClassificationDataset,
    GroupedThreadClassificationSample,
    MessageClassificationDataset,
    MessageClassificationSample,
    RationaleExtractionDataset,
    RationaleExtractionSample,
    RowClassificationDataset,
    RowClassificationSample,
    ThreadMessage,
    TrainingFormatExporter,
)

__all__ = [
    "PhaseMetricDelta",
    "PhaseMetricThresholds",
    "PhaseMetricsComparator",
    "PhaseMetricsComparisonReport",
    "CrossDocumentLinkSample",
    "CrossDocumentLinkingDataset",
    "DatasetQualityEvaluator",
    "EntityCoverageReport",
    "GoldExample",
    "GoldSet",
    "GoldSetBuilder",
    "GoldSetConfig",
    "GoldSetV2Config",
    "GroupedThreadClassificationDataset",
    "GroupedThreadClassificationSample",
    "MessageClassificationDataset",
    "MessageClassificationSample",
    "PHASE1_REGRESSION_TARGETS",
    "QualityReport",
    "RegressionFixture",
    "RegressionSnapshot",
    "RationaleExtractionDataset",
    "RationaleExtractionSample",
    "RowClassificationDataset",
    "RowClassificationSample",
    "ThreadMessage",
    "TrainingFormatExporter",
    "build_regression_context",
    "build_regression_fixture",
    "build_regression_snapshot",
    "collect_reference_errors",
]
