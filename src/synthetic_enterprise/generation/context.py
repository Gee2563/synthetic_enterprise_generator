from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from faker import Faker

from synthetic_enterprise.generation.config import (
    CompanySizeConfig,
    DateRangeConfig,
    GeneratorConfig,
)
from synthetic_enterprise.generation.rng import derive_seed


@dataclass(slots=True)
class GeneratorContext:
    """Deterministic generation context shared across generators."""

    seed: int
    config: GeneratorConfig = field(default_factory=GeneratorConfig)
    locale: str = field(init=False)
    faker_seed: int = field(init=False)
    rng_seed: int = field(init=False)
    faker: Faker = field(init=False, repr=False)
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be greater than or equal to zero")

        self.locale = self.config.locale
        self.faker_seed = self.seed
        self.rng_seed = self.seed
        self.faker = Faker(self.locale)
        self.faker.seed_instance(self.faker_seed)
        self.rng = random.Random(self.rng_seed)

    @property
    def date_range(self) -> DateRangeConfig:
        return self.config.date_range

    @property
    def company_size(self) -> CompanySizeConfig:
        return self.config.company_size

    @property
    def timezone(self) -> str:
        return self.config.timezone

    @property
    def noise_ratio(self) -> float:
        return self.config.noise_ratio

    @property
    def hard_negative_ratio(self) -> float:
        return self.config.hard_negative_ratio

    @property
    def cross_system_ratio(self) -> float:
        return self.config.cross_system_ratio

    @property
    def verbosity_ratio(self) -> float:
        return self.config.verbosity_ratio

    def derive_seed(self, namespace: str) -> int:
        return derive_seed(self.seed, namespace)

    def random_datetime(self) -> datetime:
        start_at = self.date_range.start_at
        end_at = self.date_range.end_at
        total_seconds = int((end_at - start_at).total_seconds())

        if total_seconds <= 0:
            return start_at.astimezone(self.config.tzinfo)

        offset_seconds = self.rng.randint(0, total_seconds)
        return (start_at + timedelta(seconds=offset_seconds)).astimezone(self.config.tzinfo)
