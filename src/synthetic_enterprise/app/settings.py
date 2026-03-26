from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Minimal application configuration for the scaffold."""

    package_name: str = "synthetic_enterprise"
    default_seed: int = 0
    default_locale: str = "en_US"
    default_export_format: str = "parquet"


def load_config() -> AppConfig:
    """Load config from environment with stable defaults."""

    return AppConfig(
        default_seed=int(os.getenv("SYNTHETIC_ENTERPRISE_DEFAULT_SEED", "0")),
        default_locale=os.getenv("SYNTHETIC_ENTERPRISE_DEFAULT_LOCALE", "en_US"),
        default_export_format=os.getenv("SYNTHETIC_ENTERPRISE_EXPORT_FORMAT", "parquet"),
    )
