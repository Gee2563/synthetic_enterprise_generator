from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from synthetic_enterprise.generation.context import GeneratorContext

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
        opener = self._render_opener(request)
        detail_lines = self._render_detail_lines(request)
        signature = self._render_signature(request)

        body_parts = [greeting, "", opener]
        if detail_lines:
            body_parts.extend(["", *detail_lines])
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
        if request.audience == EmailAudience.EXTERNAL_CUSTOMER:
            return self._pick(
                salt="opener:external",
                options=(
                    "Thanks again for the time today.",
                    "Thanks again for the discussion today.",
                ),
            )
        if request.seniority == EmailSeniority.EXECUTIVE:
            return self._pick(
                salt="opener:exec",
                options=(
                    "At a high level, we need to keep this moving.",
                    "At a high level, let's keep this moving.",
                ),
            )
        return self._pick(
            salt="opener:ic",
            options=(
                "I pulled together the details below.",
                "I captured the details below.",
            ),
        )

    def _render_detail_lines(self, request: EmailStyleRequest) -> list[str]:
        if request.detail_level == EmailDetailLevel.SHORT:
            return [self._render_short_acknowledgement(request)]

        lines = list(request.context_lines) or [self._render_default_detail_line(request)]
        if self.context.seed % 2 == 1:
            lines.reverse()

        detailed_lines = [self._prefix_detail_line(index, line) for index, line in enumerate(lines)]
        detailed_lines.append(self._render_closing_line(request))
        return detailed_lines

    def _render_short_acknowledgement(self, request: EmailStyleRequest) -> str:
        if request.audience == EmailAudience.EXTERNAL_CUSTOMER:
            return self._pick(
                salt="ack:external",
                options=(
                    "A quick summary is below, and we're aligned on the next step.",
                    "Sharing the short version here so we stay aligned on timing.",
                ),
            )
        if request.seniority == EmailSeniority.EXECUTIVE:
            return self._pick(
                salt="ack:exec",
                options=(
                    "Please keep the owners clear and keep this moving.",
                    "Please keep timing tight and close the loop quickly.",
                ),
            )
        return self._pick(
            salt="ack:ic",
            options=(
                "Quick note below so we're aligned on next steps.",
                "Sending the short version here to keep everyone aligned.",
            ),
        )

    def _render_default_detail_line(self, request: EmailStyleRequest) -> str:
        if request.audience == EmailAudience.EXTERNAL_CUSTOMER:
            return "We will send a written summary and timing update after the meeting."
        if request.seniority == EmailSeniority.EXECUTIVE:
            return "Please keep the timeline, owners, and open decisions visible."
        return "I've outlined the current tasks, timing, and open questions below."

    def _prefix_detail_line(self, index: int, line: str) -> str:
        prefixes = ("First,", "Also,", "Separately,", "Finally,")
        prefix = prefixes[index % len(prefixes)]
        return f"{prefix} {line}"

    def _render_closing_line(self, request: EmailStyleRequest) -> str:
        if request.audience == EmailAudience.EXTERNAL_CUSTOMER:
            return self._pick(
                salt="closing:external",
                options=(
                    "Please let me know if you'd like us to adjust anything on our side.",
                    "If anything should be revised on our side, please send it over.",
                ),
            )
        if request.seniority == EmailSeniority.EXECUTIVE:
            return self._pick(
                salt="closing:exec",
                options=(
                    "Let's keep the decision path simple from here.",
                    "Let's keep the owners and timing clean from here.",
                ),
            )
        return self._pick(
            salt="closing:ic",
            options=(
                "If I missed anything, reply back and I'll update the summary.",
                "If anything looks off, send it back and I'll adjust the detail.",
            ),
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
