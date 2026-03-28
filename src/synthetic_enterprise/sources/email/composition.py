from __future__ import annotations

from dataclasses import dataclass

from synthetic_enterprise.generation.context import GeneratorContext


@dataclass(frozen=True, slots=True)
class TemplateBlock:
    """One composable text block with interchangeable surface forms."""

    key: str
    variants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TemplatePlan:
    """Deterministic composition plan for one email body."""

    blocks: tuple[TemplateBlock, ...]
    order_variants: tuple[tuple[str, ...], ...]


@dataclass(slots=True)
class CompositionalTemplateEngine:
    """Compose a deterministic surface form from interchangeable blocks."""

    context: GeneratorContext

    def render(self, plan: TemplatePlan) -> list[str]:
        blocks_by_key = {
            block.key: block
            for block in plan.blocks
            if any(variant.strip() for variant in block.variants)
        }
        if not blocks_by_key:
            return []

        order = self._order_variant(plan)
        rendered_keys: set[str] = set()
        lines: list[str] = []

        for key in order:
            block = blocks_by_key.get(key)
            if block is None:
                continue
            lines.append(self._choose_variant(block))
            rendered_keys.add(key)

        for key, block in blocks_by_key.items():
            if key not in rendered_keys:
                lines.append(self._choose_variant(block))

        return lines

    def _order_variant(self, plan: TemplatePlan) -> tuple[str, ...]:
        if not plan.order_variants:
            return tuple(block.key for block in plan.blocks)
        variant_index = (
            self.context.seed + self.context.derive_seed("email-composition:order")
        ) % len(plan.order_variants)
        return plan.order_variants[variant_index]

    def _choose_variant(self, block: TemplateBlock) -> str:
        variant_index = (
            self.context.seed
            + self.context.derive_seed(f"email-composition:block:{block.key}")
        ) % len(block.variants)
        return block.variants[variant_index]
