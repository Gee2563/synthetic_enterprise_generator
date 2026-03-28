from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from synthetic_enterprise.generation.context import GeneratorContext
from synthetic_enterprise.generation.personas import (
    PersonaVoiceProfile,
    resolve_persona_voice_profile,
)
from synthetic_enterprise.sources.email.composition import (
    CompositionalTemplateEngine,
    TemplateBlock,
    TemplatePlan,
)
from synthetic_enterprise.sources.email.paraphrases import (
    EmailParaphraseBlock,
    SafeEmailParaphraser,
)

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EmailSubjectMode(str, Enum):
    NEW = "new"
    REPLY = "reply"
    FORWARD = "forward"


class EmailAudience(str, Enum):
    INTERNAL = "internal"
    EXTERNAL_CUSTOMER = "external_customer"


class EmailSeniority(str, Enum):
    EXECUTIVE = "executive"
    IC = "ic"


class EmailDetailLevel(str, Enum):
    SHORT = "short"
    DETAILED = "detailed"


class EmailStyleRequest(BaseModel):
    """Input contract for deterministic style rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: NonEmptyText
    message_index_in_thread: int = Field(ge=0)
    base_subject: NonEmptyText
    sender_name: NonEmptyText
    sender_role: NonEmptyText
    recipient_name: NonEmptyText
    subject_mode: EmailSubjectMode = EmailSubjectMode.NEW
    audience: EmailAudience = EmailAudience.INTERNAL
    seniority: EmailSeniority = EmailSeniority.IC
    detail_level: EmailDetailLevel = EmailDetailLevel.DETAILED
    include_signature: bool = True
    cc: tuple[str, ...] = ()
    bcc: tuple[str, ...] = ()
    context_lines: tuple[NonEmptyText, ...] = ()
    prior_thread_summary: NonEmptyText | None = None
    objection_line: NonEmptyText | None = None
    action_ask: NonEmptyText | None = None
    disclaimer_line: NonEmptyText | None = None


class StyledEmailContent(BaseModel):
    """Rendered email style block with preserved thread metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    message_index_in_thread: int
    subject: str
    body: str
    cc: tuple[str, ...] = ()
    bcc: tuple[str, ...] = ()
    signature_block: str | None = None


@dataclass(slots=True)
class EmailStyleEngine:
    """Deterministic structural email style renderer."""

    context: GeneratorContext

    def render(self, request: EmailStyleRequest) -> StyledEmailContent:
        subject = self._render_subject(request)
        greeting = self._render_greeting(request)
        sections = self._render_sections(request)
        signature = self._render_signature(request)

        body_parts = [greeting]
        if sections:
            body_parts.extend(["", sections[0]])
            if len(sections) > 1:
                body_parts.extend(["", *sections[1:]])
        if signature is not None:
            body_parts.extend(["", signature])

        return StyledEmailContent(
            thread_id=request.thread_id,
            message_index_in_thread=request.message_index_in_thread,
            subject=subject,
            body="\n".join(body_parts),
            cc=request.cc,
            bcc=request.bcc,
            signature_block=signature,
        )

    def _render_subject(self, request: EmailStyleRequest) -> str:
        subject = request.base_subject
        if request.subject_mode == EmailSubjectMode.REPLY:
            return f"Re: {subject}"
        if request.subject_mode == EmailSubjectMode.FORWARD:
            return f"FW: {subject}"
        return subject

    def _render_greeting(self, request: EmailStyleRequest) -> str:
        if request.audience == EmailAudience.EXTERNAL_CUSTOMER:
            return self._pick(
                salt="greeting:external",
                options=(f"Hi {request.recipient_name},", f"Hello {request.recipient_name},"),
            )
        if request.seniority == EmailSeniority.EXECUTIVE:
            return self._pick(salt="greeting:exec", options=("Team,", "All,"))
        return self._pick(salt="greeting:ic", options=("Hi team,", "Hi all,"))

    def _render_opener(self, request: EmailStyleRequest) -> str:
        profile = self._voice_profile(request)
        if request.audience == EmailAudience.EXTERNAL_CUSTOMER:
            return self._pick(
                salt=f"opener:{profile.key}:external",
                options=profile.external_openers,
            )
        return self._pick(
            salt=f"opener:{profile.key}:internal",
            options=profile.internal_openers,
        )

    def _render_sections(self, request: EmailStyleRequest) -> list[str]:
        profile = self._voice_profile(request)
        if request.detail_level == EmailDetailLevel.SHORT:
            sections = [self._render_short_acknowledgement(request)]
            if request.action_ask is not None:
                sections.append(self._render_action_ask(request))
            if request.disclaimer_line is not None:
                sections.append(self._render_disclaimer(request))
            return sections

        plan = self._build_plan(request=request, profile=profile)
        return CompositionalTemplateEngine(context=self.context).render(plan)

    def _render_short_acknowledgement(self, request: EmailStyleRequest) -> str:
        profile = self._voice_profile(request)
        if request.audience == EmailAudience.EXTERNAL_CUSTOMER:
            return self._pick(
                salt=f"ack:{profile.key}:external",
                options=profile.external_short_acks,
            )
        return self._pick(
            salt=f"ack:{profile.key}:internal",
            options=profile.internal_short_acks,
        )

    def _render_default_detail_line(self, request: EmailStyleRequest) -> str:
        if request.audience == EmailAudience.EXTERNAL_CUSTOMER:
            return self._voice_profile(request).external_default_detail
        return self._voice_profile(request).internal_default_detail

    def _render_closing_line(self, request: EmailStyleRequest) -> str:
        profile = self._voice_profile(request)
        if request.audience == EmailAudience.EXTERNAL_CUSTOMER:
            return self._pick(
                salt=f"closing:{profile.key}:external",
                options=profile.external_closings,
            )
        return self._pick(
            salt=f"closing:{profile.key}:internal",
            options=profile.internal_closings,
        )

    def _build_plan(
        self,
        *,
        request: EmailStyleRequest,
        profile: PersonaVoiceProfile,
    ) -> TemplatePlan:
        context_lines = list(request.context_lines) or [self._render_default_detail_line(request)]
        core_keys = [f"core_{index}" for index in range(len(context_lines))]
        ordered_keys = tuple(
            key
            for key in (
                "opening",
                "prior_thread",
                *core_keys,
                "objection",
                "action",
                "closing",
                "disclaimer",
            )
            if key == "opening"
            or (key == "prior_thread" and request.prior_thread_summary is not None)
            or (key.startswith("core_"))
            or (key == "objection" and request.objection_line is not None)
            or (key == "action" and request.action_ask is not None)
            or key == "closing"
            or (key == "disclaimer" and request.disclaimer_line is not None)
        )
        reverse_core_keys = tuple(reversed(core_keys))
        order_variants = (
            ordered_keys,
            tuple(
                key
                for key in (
                    "opening",
                    *reverse_core_keys,
                    "prior_thread",
                    "action",
                    "objection",
                    "closing",
                    "disclaimer",
                )
                if key in ordered_keys
            ),
            tuple(
                key
                for key in (
                    "opening",
                    "objection",
                    "action",
                    *core_keys,
                    "prior_thread",
                    "disclaimer",
                    "closing",
                )
                if key in ordered_keys
            ),
        )
        blocks = [
            TemplateBlock(
                key="opening",
                variants=self._opening_variants(request),
            ),
        ]
        if request.prior_thread_summary is not None:
            blocks.append(
                TemplateBlock(
                    key="prior_thread",
                    variants=self._prior_thread_variants(request.prior_thread_summary),
                )
            )
        for index, line in enumerate(context_lines):
            blocks.append(
                TemplateBlock(
                    key=f"core_{index}",
                    variants=self._core_clause_variants(
                        line=line,
                        prefix=profile.detail_prefixes[index % len(profile.detail_prefixes)],
                    ),
                )
            )
        if request.objection_line is not None:
            blocks.append(
                TemplateBlock(
                    key="objection",
                    variants=self._objection_variants(request.objection_line),
                )
            )
        if request.action_ask is not None:
            blocks.append(
                TemplateBlock(
                    key="action",
                    variants=self._action_variants(request.action_ask),
                )
            )
        blocks.append(
            TemplateBlock(
                key="closing",
                variants=(self._render_closing_line(request),),
            )
        )
        if request.disclaimer_line is not None:
            blocks.append(
                TemplateBlock(
                    key="disclaimer",
                    variants=self._disclaimer_variants(request.disclaimer_line),
                )
            )

        return TemplatePlan(
            blocks=tuple(blocks),
            order_variants=order_variants,
        )

    def _opening_variants(self, request: EmailStyleRequest) -> tuple[str, ...]:
        opener = self._render_opener(request)
        variants = [opener]
        if request.audience == EmailAudience.EXTERNAL_CUSTOMER:
            variants.append(f"{opener} I pulled the key notes together below.")
            variants.append(f"{opener} I captured the main points below.")
        else:
            variants.append(f"{opener} I pulled the key notes together below.")
            variants.append(f"{opener} I captured the main points below.")
        return tuple(dict.fromkeys(variants))

    def _prior_thread_variants(self, summary: str) -> tuple[str, ...]:
        return self._paraphrase_variants(
            key="prior_thread",
            templates=(
                "Prior thread: {summary}",
                "From the earlier thread: {summary}",
                "Carrying forward the earlier thread, {summary_lower}",
            ),
            facts={
                "summary": summary,
                "summary_lower": self._lowercase_first(summary),
            },
        )

    def _core_clause_variants(
        self,
        *,
        line: str,
        prefix: str,
    ) -> tuple[str, ...]:
        return self._paraphrase_variants(
            key=f"core:{prefix}",
            templates=(
                "{prefix} {line}",
                "{prefix} Right now, {line_lower}",
                "{prefix} For this thread, {line_lower}",
            ),
            facts={
                "prefix": prefix,
                "line": line,
                "line_lower": self._lowercase_first(line),
            },
        )

    def _objection_variants(self, line: str) -> tuple[str, ...]:
        return self._paraphrase_variants(
            key="objection",
            templates=(
                "Objection: {line}",
                "Constraint: {line}",
                "Current pushback: {line_lower}",
            ),
            facts={
                "line": line,
                "line_lower": self._lowercase_first(line),
            },
        )

    def _render_action_ask(self, request: EmailStyleRequest) -> str:
        assert request.action_ask is not None
        return self._action_variants(request.action_ask)[0]

    def _action_variants(self, line: str) -> tuple[str, ...]:
        return self._paraphrase_variants(
            key="action",
            templates=(
                "Action ask: {line}",
                "Next step: {line}",
                "Please note: {line_lower}",
            ),
            facts={
                "line": line,
                "line_lower": self._lowercase_first(line),
            },
        )

    def _render_disclaimer(self, request: EmailStyleRequest) -> str:
        assert request.disclaimer_line is not None
        return self._disclaimer_variants(request.disclaimer_line)[0]

    def _disclaimer_variants(self, line: str) -> tuple[str, ...]:
        return self._paraphrase_variants(
            key="disclaimer",
            templates=(
                "Planning note: {line}",
                "Context only: {line}",
                "Disclaimer: {line_lower}",
            ),
            facts={
                "line": line,
                "line_lower": self._lowercase_first(line),
            },
        )

    def _render_signature(self, request: EmailStyleRequest) -> str | None:
        if not request.include_signature:
            return None

        signoff = "Best,"
        if (
            request.audience == EmailAudience.INTERNAL
            and request.seniority == EmailSeniority.EXECUTIVE
        ):
            signoff = "Regards,"
        return f"{signoff}\n{request.sender_name}\n{request.sender_role}"

    def _pick(self, *, salt: str, options: tuple[str, ...]) -> str:
        index = (self.context.seed + self.context.derive_seed(salt)) % len(options)
        return options[index]

    def _voice_profile(self, request: EmailStyleRequest) -> PersonaVoiceProfile:
        return resolve_persona_voice_profile(
            sender_role=request.sender_role,
            seniority=request.seniority.value,
        )

    def _paraphrase_variants(
        self,
        *,
        key: str,
        templates: tuple[str, ...],
        facts: dict[str, str],
    ) -> tuple[str, ...]:
        return SafeEmailParaphraser(context=self.context).render_all(
            EmailParaphraseBlock(
                key=key,
                templates=templates,
                facts=facts,
            )
        )

    def _lowercase_first(self, text: str) -> str:
        if not text:
            return text
        return text[0].lower() + text[1:]
