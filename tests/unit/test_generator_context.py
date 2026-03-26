from __future__ import annotations

from datetime import datetime, timezone

from synthetic_enterprise.generation.config import (
    CompanySizeConfig,
    DateRangeConfig,
    GeneratorConfig,
)
from synthetic_enterprise.generation.context import GeneratorContext


def build_config() -> GeneratorConfig:
    return GeneratorConfig(
        date_range=DateRangeConfig(
            start_at=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            end_at=datetime(2026, 1, 31, 23, 59, tzinfo=timezone.utc),
        ),
        company_size=CompanySizeConfig(min_employees=25, max_employees=250),
        locale="en_GB",
        timezone="Europe/London",
        noise_ratio=0.82,
        verbosity_ratio=0.37,
    )


def test_same_seed_produces_same_outputs() -> None:
    config = build_config()
    first = GeneratorContext(seed=42, config=config)
    second = GeneratorContext(seed=42, config=config)

    assert [first.rng.randint(1, 1000) for _ in range(5)] == [
        second.rng.randint(1, 1000) for _ in range(5)
    ]
    assert [first.faker.name() for _ in range(3)] == [second.faker.name() for _ in range(3)]
    assert [first.random_datetime() for _ in range(3)] == [
        second.random_datetime() for _ in range(3)
    ]


def test_different_seeds_produce_different_outputs() -> None:
    config = build_config()
    first = GeneratorContext(seed=42, config=config)
    second = GeneratorContext(seed=43, config=config)

    assert [first.rng.randint(1, 1000) for _ in range(3)] != [
        second.rng.randint(1, 1000) for _ in range(3)
    ]
    assert first.faker.email() != second.faker.email()


def test_faker_and_random_are_seeded_consistently() -> None:
    config = build_config()
    first = GeneratorContext(seed=77, config=config)
    second = GeneratorContext(seed=77, config=config)

    assert first.seed == second.seed == 77
    assert first.faker_seed == second.faker_seed == 77
    assert first.rng_seed == second.rng_seed == 77
    assert first.rng.random() == second.rng.random()
    assert first.faker.company() == second.faker.company()


def test_date_generation_stays_inside_configured_bounds() -> None:
    config = build_config()
    context = GeneratorContext(seed=99, config=config)

    generated = [context.random_datetime() for _ in range(100)]

    assert all(
        config.date_range.start_at <= value <= config.date_range.end_at
        for value in generated
    )
    assert all(value.tzinfo is not None for value in generated)
