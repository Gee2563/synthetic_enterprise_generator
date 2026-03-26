from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, TypeVar

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

from synthetic_enterprise.generation.rng import derive_seed

ENTITY_ID_PATTERN = r"^[a-z]+(?:_[a-z0-9]+)+$"
SECONDS_PER_YEAR = 365 * 24 * 60 * 60
BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)

ModelT = TypeVar("ModelT", bound="EnterpriseEntity")


def _normalize_seed_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _normalize_seed_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_seed_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_seed_value(item) for item in value]
    return value


def _snake_case(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def stable_entity_id(entity_name: str, seed: int, namespace: str) -> str:
    derived = derive_seed(seed, f"{entity_name}:{namespace}:id")
    return f"{entity_name}_{derived:016x}"


def deterministic_timestamp(seed: int, namespace: str, floor: datetime | None = None) -> datetime:
    offset_seconds = derive_seed(seed, f"{namespace}:offset") % SECONDS_PER_YEAR
    candidate = BASE_TIME + timedelta(seconds=offset_seconds)

    if floor is None or candidate >= floor:
        return candidate

    delta_seconds = derive_seed(seed, f"{namespace}:delta") % (24 * 60 * 60)
    return floor + timedelta(seconds=delta_seconds)


class EnterpriseModel(BaseModel):
    """Base model with strict validation and JSON serialization helpers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Any:
        return cls.model_validate(payload)

    @classmethod
    def from_json(cls, payload: str) -> Any:
        return cls.model_validate_json(payload)


class EnterpriseEntity(EnterpriseModel):
    """Base class for seeded enterprise entities."""

    entity_name: ClassVar[str | None] = None
    id: str = Field(pattern=ENTITY_ID_PATTERN)
    seed: int = Field(ge=0)
    created_at: AwareDatetime
    updated_at: AwareDatetime

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if value != value.lower():
            raise ValueError("entity ids must be lowercase")
        return value

    @model_validator(mode="after")
    def validate_timestamps(self) -> EnterpriseEntity:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        return self

    @classmethod
    def from_seed(cls: type[ModelT], seed: int, **data: Any) -> ModelT:
        entity_name = cls.entity_name or _snake_case(cls.__name__)
        namespace = json.dumps(_normalize_seed_value(data), sort_keys=True, separators=(",", ":"))
        created_at = deterministic_timestamp(seed, f"{entity_name}:{namespace}:created_at")
        updated_at = deterministic_timestamp(
            seed,
            f"{entity_name}:{namespace}:updated_at",
            floor=created_at,
        )
        return cls(
            id=stable_entity_id(entity_name, seed, namespace),
            seed=seed,
            created_at=created_at,
            updated_at=updated_at,
            **data,
        )
