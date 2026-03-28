from __future__ import annotations

from synthetic_enterprise.evaluation.comparison import (
    PhaseMetricsComparator,
    PhaseMetricThresholds,
)
from synthetic_enterprise.evaluation.quality_report import (
    AccountProfileCoverageReport,
    EntityCoverageReport,
    QualityReport,
    ScenarioCoverageReport,
)


def build_report(
    *,
    duplicate_rate: float,
    lexical_diversity: float,
    scenario_coverage: float,
    noise_family_entropy: float,
    thread_depth_distribution: dict[str, int],
) -> QualityReport:
    return QualityReport(
        total_rows=100,
        source_row_counts={
            "email": 25,
            "slack": 25,
            "teams": 25,
            "salesforce": 25,
        },
        relevance_rate=0.15,
        category_distribution={"follow_up": 0.2},
        noise_category_distribution={"status_updates": 0.5},
        thread_depth_distribution=thread_depth_distribution,
        average_message_length_by_source={"email": 12.0},
        cross_system_linkage_rate=0.4,
        hard_negative_rate=0.1,
        messy_data_rate=0.05,
        duplicate_rate=duplicate_rate,
        duplicate_rate_by_source={"email": duplicate_rate},
        lexical_diversity=lexical_diversity,
        lexical_diversity_by_source={"email": lexical_diversity},
        lexical_diversity_by_company={"company_a": lexical_diversity},
        scenario_coverage=ScenarioCoverageReport(
            overall_rate=scenario_coverage,
            by_family={"event_invite_to_attendance": scenario_coverage},
        ),
        source_style_separation_proxy=0.6,
        hard_negative_difficulty_proxy=0.3,
        noise_family_entropy=noise_family_entropy,
        temporal_lag_distribution={"0-6h": 1, "6-24h": 0, "1-3d": 0, ">3d": 0},
        account_profile_coverage=AccountProfileCoverageReport(
            overall_rate=0.2,
            by_profile={"highly_engaged_champion_led": 0.2},
        ),
        entity_coverage=EntityCoverageReport(
            overall_rate=0.3,
            by_entity_type={"accounts": 0.3},
        ),
    )


def test_phase_metric_comparison_works_on_toy_reports() -> None:
    phase1 = build_report(
        duplicate_rate=0.6,
        lexical_diversity=0.08,
        scenario_coverage=0.2,
        noise_family_entropy=0.3,
        thread_depth_distribution={"1": 5, "2": 1},
    )
    phase2 = build_report(
        duplicate_rate=0.3,
        lexical_diversity=0.14,
        scenario_coverage=0.6,
        noise_family_entropy=0.7,
        thread_depth_distribution={"1": 1, "3": 2, "5": 1},
    )

    report = PhaseMetricsComparator().compare(
        phase1_report=phase1,
        phase2_report=phase2,
    )

    assert report.passed is True
    assert report.regressions == []
    assert report.duplicate_reduction.delta == 0.3
    assert report.lexical_diversity_improvement.delta == 0.06
    assert report.scenario_coverage_improvement.delta == 0.4
    assert report.noise_entropy_improvement.delta == 0.4
    assert report.thread_depth_improvement.delta > 0.0


def test_threshold_assertions_are_configurable() -> None:
    phase1 = build_report(
        duplicate_rate=0.4,
        lexical_diversity=0.1,
        scenario_coverage=0.3,
        noise_family_entropy=0.5,
        thread_depth_distribution={"1": 3, "2": 1},
    )
    phase2 = build_report(
        duplicate_rate=0.32,
        lexical_diversity=0.12,
        scenario_coverage=0.36,
        noise_family_entropy=0.57,
        thread_depth_distribution={"1": 1, "2": 2, "3": 1},
    )

    report = PhaseMetricsComparator(
        thresholds=PhaseMetricThresholds(
            min_duplicate_reduction=0.05,
            min_lexical_diversity_improvement=0.01,
            min_scenario_coverage_improvement=0.05,
            min_noise_entropy_improvement=0.05,
            min_thread_depth_improvement=0.1,
        )
    ).compare(phase1_report=phase1, phase2_report=phase2)

    assert report.passed is True
    assert report.duplicate_reduction.threshold == 0.05
    assert report.thread_depth_improvement.threshold == 0.1


def test_regressions_are_clearly_surfaced() -> None:
    phase1 = build_report(
        duplicate_rate=0.35,
        lexical_diversity=0.14,
        scenario_coverage=0.55,
        noise_family_entropy=0.7,
        thread_depth_distribution={"2": 3, "4": 1},
    )
    phase2 = build_report(
        duplicate_rate=0.37,
        lexical_diversity=0.12,
        scenario_coverage=0.4,
        noise_family_entropy=0.6,
        thread_depth_distribution={"1": 4},
    )

    report = PhaseMetricsComparator().compare(
        phase1_report=phase1,
        phase2_report=phase2,
    )

    assert report.passed is False
    assert report.regressions == [
        "duplicate_reduction",
        "lexical_diversity_improvement",
        "scenario_coverage_improvement",
        "noise_entropy_improvement",
        "thread_depth_improvement",
    ]
    assert report.duplicate_reduction.passed is False
    assert report.thread_depth_improvement.passed is False


def test_output_format_is_stable() -> None:
    phase1 = build_report(
        duplicate_rate=0.5,
        lexical_diversity=0.08,
        scenario_coverage=0.1,
        noise_family_entropy=0.2,
        thread_depth_distribution={"1": 4},
    )
    phase2 = build_report(
        duplicate_rate=0.3,
        lexical_diversity=0.16,
        scenario_coverage=0.5,
        noise_family_entropy=0.7,
        thread_depth_distribution={"1": 1, "4": 2},
    )

    report = PhaseMetricsComparator().compare(
        phase1_report=phase1,
        phase2_report=phase2,
    )

    assert list(report.to_dict()) == [
        "baseline_name",
        "phase2_name",
        "duplicate_reduction",
        "lexical_diversity_improvement",
        "scenario_coverage_improvement",
        "noise_entropy_improvement",
        "thread_depth_improvement",
        "regressions",
        "passed",
    ]
    assert list(report.duplicate_reduction.to_dict()) == [
        "baseline_value",
        "phase2_value",
        "delta",
        "threshold",
        "passed",
    ]
