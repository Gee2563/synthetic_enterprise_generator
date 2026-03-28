from __future__ import annotations

from dataclasses import dataclass
from string import Formatter

from synthetic_enterprise.generation.context import GeneratorContext


@dataclass(frozen=True, slots=True)
class EmailParaphraseBlock:
    """One controlled email paraphrase block with explicit fact placeholders."""

    key: str
    templates: tuple[str, ...]
    facts: dict[str, str]


@dataclass(slots=True)
class SafeEmailParaphraser:
    """Deterministic, fact-preserving paraphrase selection for email blocks."""

    context: GeneratorContext

    def render(self, block: EmailParaphraseBlock) -> str:
        variants = self.render_all(block)
        variant_index = (
            self.context.seed
            + self.context.derive_seed(f"email-paraphrase:{block.key}")
        ) % len(variants)
        return variants[variant_index]

    def render_all(self, block: EmailParaphraseBlock) -> tuple[str, ...]:
        rendered = tuple(self._render_template(block, template) for template in block.templates)
        if not rendered:
            raise ValueError("paraphrase blocks require at least one template")
        return rendered

    def _render_template(self, block: EmailParaphraseBlock, template: str) -> str:
        required_fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name is not None
        }
        missing_fields = required_fields - set(block.facts)
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"missing paraphrase facts for template fields: {missing}")
        return template.format(**block.facts)
