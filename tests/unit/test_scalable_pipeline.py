from __future__ import annotations

from synthetic_enterprise.generation.config import CompanySizeConfig, GeneratorConfig
from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.pipeline import GenerationPipeline


def build_pipeline(seed: int = 20260327) -> GenerationPipeline:
    context = GeneratorContext(
        seed=seed,
        config=GeneratorConfig(
            company_size=CompanySizeConfig(min_employees=30, max_employees=30),
        ),
    )
    return GenerationPipeline(context=context)


def test_chunk_seed_strategy_is_deterministic() -> None:
    pipeline = build_pipeline()

    first_plan = pipeline.plan_source_chunks(
        source_name="email",
        total_rows=1050,
        generation_chunk_size=300,
    )
    second_plan = pipeline.plan_source_chunks(
        source_name="email",
        total_rows=1050,
        generation_chunk_size=300,
    )
    different_seed_plan = build_pipeline(seed=20260328).plan_source_chunks(
        source_name="email",
        total_rows=1050,
        generation_chunk_size=300,
    )

    assert [chunk.seed for chunk in first_plan] == [chunk.seed for chunk in second_plan]
    assert [chunk.row_count for chunk in first_plan] == [300, 300, 300, 150]
    assert [chunk.seed for chunk in first_plan] != [chunk.seed for chunk in different_seed_plan]
