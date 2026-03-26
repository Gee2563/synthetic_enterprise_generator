from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthetic_enterprise.labeling.taxonomy import (
    CommunicationCategory,
    CommunicationTaxonomy,
    NoiseCategory,
    RelevantCategory,
    TaxonomyConfig,
)


def test_taxonomy_labels_are_valid_enum_values() -> None:
    expected_labels = {
        "pain_point",
        "complaint",
        "feature_request",
        "buying_signal",
        "churn_risk",
        "event_attendance",
        "follow_up",
        "blocker",
        "escalation",
        "decision_maker_signal",
        "greetings",
        "status_updates",
        "social_chatter",
        "scheduling_only",
        "fyi_forward",
        "automated_notification",
        "duplicate_summary",
        "low_signal_checkin",
        "irrelevant_marketing",
        "admin_ops",
    }

    assert {category.value for category in CommunicationCategory} == expected_labels
    assert {category.value for category in RelevantCategory}.issubset(expected_labels)
    assert {category.value for category in NoiseCategory}.issubset(expected_labels)


def test_every_generated_record_has_exactly_one_primary_category() -> None:
    taxonomy = CommunicationTaxonomy(primary_category=CommunicationCategory.GREETINGS)

    assert taxonomy.primary_category == CommunicationCategory.GREETINGS

    with pytest.raises(ValidationError):
        CommunicationTaxonomy.model_validate(
            {
                "primary_category": [
                    CommunicationCategory.GREETINGS,
                    CommunicationCategory.STATUS_UPDATES,
                ]
            }
        )


def test_relevance_flag_aligns_with_category_mapping() -> None:
    relevant = CommunicationTaxonomy(primary_category=CommunicationCategory.PAIN_POINT)
    noise = CommunicationTaxonomy(primary_category=CommunicationCategory.ADMIN_OPS)

    assert relevant.is_relevant is True
    assert noise.is_relevant is False

    with pytest.raises(ValidationError):
        CommunicationTaxonomy(
            primary_category=CommunicationCategory.BUYING_SIGNAL,
            is_relevant=False,
        )


def test_noise_ratio_can_be_configured() -> None:
    config = TaxonomyConfig(noise_ratio=0.91)

    assert config.noise_ratio == 0.91
