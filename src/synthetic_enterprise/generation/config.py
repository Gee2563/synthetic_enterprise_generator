from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def _default_start_at() -> datetime:
    return datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _default_end_at() -> datetime:
    return datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def _is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


@dataclass(frozen=True, slots=True)
class DateRangeConfig:
    """Configures the valid synthetic timeline window."""

    start_at: datetime = field(default_factory=_default_start_at)
    end_at: datetime = field(default_factory=_default_end_at)

    def __post_init__(self) -> None:
        if not _is_aware(self.start_at):
            raise ValueError("start_at must be timezone-aware")
        if not _is_aware(self.end_at):
            raise ValueError("end_at must be timezone-aware")
        if self.end_at < self.start_at:
            raise ValueError("end_at must be greater than or equal to start_at")


@dataclass(frozen=True, slots=True)
class CompanySizeConfig:
    """Bounds the size of generated companies."""

    min_employees: int = 25
    max_employees: int = 250

    def __post_init__(self) -> None:
        if self.min_employees <= 0:
            raise ValueError("min_employees must be greater than zero")
        if self.max_employees < self.min_employees:
            raise ValueError("max_employees must be greater than or equal to min_employees")


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Minimal configuration object shared by deterministic generators."""

    date_range: DateRangeConfig = field(default_factory=DateRangeConfig)
    company_size: CompanySizeConfig = field(default_factory=CompanySizeConfig)
    locale: str = "en_US"
    timezone: str = "UTC"
    noise_ratio: float = 0.8
    hard_negative_ratio: float = 0.25
    cross_system_ratio: float = 0.5
    verbosity_ratio: float = 0.5

    def __post_init__(self) -> None:
        if not self.locale.strip():
            raise ValueError("locale must not be empty")
        if not 0.0 <= self.noise_ratio <= 1.0:
            raise ValueError("noise_ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.hard_negative_ratio <= 1.0:
            raise ValueError("hard_negative_ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.cross_system_ratio <= 1.0:
            raise ValueError("cross_system_ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.verbosity_ratio <= 1.0:
            raise ValueError("verbosity_ratio must be between 0.0 and 1.0")

        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {self.timezone}") from exc

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)
