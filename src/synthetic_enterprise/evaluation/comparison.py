from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from synthetic_enterprise.domain.base import EnterpriseModel
from synthetic_enterprise.evaluation.quality_report import QualityReport


class PhaseMetricThresholds(EnterpriseModel):
    """Minimum Phase 2 improvements required over a Phase 1 baseline."""

    min_duplicate_reduction: float = Field(default=0.0, ge=0.0)
    min_lexical_diversity_improvement: float = Field(default=0.0)
    min_scenario_coverage_improvement: float = Field(default=0.0)
    min_noise_entropy_improvement: float = Field(default=0.0)
    min_thread_depth_improvement: float = Field(default=0.0)


class PhaseMetricDelta(EnterpriseModel):
    """Side-by-side comparison for one acceptance metric."""

    baseline_value: float
    phase2_value: float
    delta: float
    threshold: float
    passed: bool


class PhaseMetricsComparisonReport(EnterpriseModel):
    """Stable Phase 1 vs Phase 2 comparison report."""

    baseline_name: str = "phase1"
    phase2_name: str = "phase2"
    duplicate_reduction: PhaseMetricDelta
    lexical_diversity_improvement: PhaseMetricDelta
    scenario_coverage_improvement: PhaseMetricDelta
    noise_entropy_improvement: PhaseMetricDelta
    thread_depth_improvement: PhaseMetricDelta
    regressions: list[str] = Field(default_factory=list)
    passed: bool


@dataclass(slots=True)
class PhaseMetricsComparator:
    """Compare Phase 1 and Phase 2 quality reports against improvement thresholds."""

    thresholds: PhaseMetricThresholds = PhaseMetricThresholds()
    baseline_name: str = "phase1"
    phase2_name: str = "phase2"

    def compare(
        self,
        *,
        phase1_report: QualityReport,
        phase2_report: QualityReport,
    ) -> PhaseMetricsComparisonReport:
        duplicate_reduction = self._decrease_is_better(
            baseline_value=phase1_report.duplicate_rate,
            phase2_value=phase2_report.duplicate_rate,
            threshold=self.thresholds.min_duplicate_reduction,
        )
        lexical_diversity_improvement = self._increase_is_better(
            baseline_value=phase1_report.lexical_diversity,
            phase2_value=phase2_report.lexical_diversity,
            threshold=self.thresholds.min_lexical_diversity_improvement,
        )
        scenario_coverage_improvement = self._increase_is_better(
            baseline_value=phase1_report.scenario_coverage.overall_rate,
            phase2_value=phase2_report.scenario_coverage.overall_rate,
            threshold=self.thresholds.min_scenario_coverage_improvement,
        )
        noise_entropy_improvement = self._increase_is_better(
            baseline_value=phase1_report.noise_family_entropy,
            phase2_value=phase2_report.noise_family_entropy,
            threshold=self.thresholds.min_noise_entropy_improvement,
        )
        thread_depth_improvement = self._increase_is_better(
            baseline_value=_average_thread_depth(phase1_report.thread_depth_distribution),
            phase2_value=_average_thread_depth(phase2_report.thread_depth_distribution),
            threshold=self.thresholds.min_thread_depth_improvement,
        )

        comparisons = {
            "duplicate_reduction": duplicate_reduction,
            "lexical_diversity_improvement": lexical_diversity_improvement,
            "scenario_coverage_improvement": scenario_coverage_improvement,
            "noise_entropy_improvement": noise_entropy_improvement,
            "thread_depth_improvement": thread_depth_improvement,
        }
        regressions = [
            metric_name
            for metric_name, comparison in comparisons.items()
            if not comparison.passed
        ]

        return PhaseMetricsComparisonReport(
            baseline_name=self.baseline_name,
            phase2_name=self.phase2_name,
            duplicate_reduction=duplicate_reduction,
            lexical_diversity_improvement=lexical_diversity_improvement,
            scenario_coverage_improvement=scenario_coverage_improvement,
            noise_entropy_improvement=noise_entropy_improvement,
            thread_depth_improvement=thread_depth_improvement,
            regressions=regressions,
            passed=not regressions,
        )

    def _increase_is_better(
        self,
        *,
        baseline_value: float,
        phase2_value: float,
        threshold: float,
    ) -> PhaseMetricDelta:
        delta = phase2_value - baseline_value
        return PhaseMetricDelta(
            baseline_value=_rounded_metric(baseline_value),
            phase2_value=_rounded_metric(phase2_value),
            delta=_rounded_metric(delta),
            threshold=_rounded_metric(threshold),
            passed=delta >= threshold,
        )

    def _decrease_is_better(
        self,
        *,
        baseline_value: float,
        phase2_value: float,
        threshold: float,
    ) -> PhaseMetricDelta:
        delta = baseline_value - phase2_value
        return PhaseMetricDelta(
            baseline_value=_rounded_metric(baseline_value),
            phase2_value=_rounded_metric(phase2_value),
            delta=_rounded_metric(delta),
            threshold=_rounded_metric(threshold),
            passed=delta >= threshold,
        )


def _average_thread_depth(distribution: dict[str, int]) -> float:
    weighted_sum = 0
    total_count = 0

    for depth_value, count in distribution.items():
        depth = int(depth_value)
        weighted_sum += depth * count
        total_count += count

    if total_count == 0:
        return 0.0

    return weighted_sum / total_count


def _rounded_metric(value: float) -> float:
    return round(value, 6)
