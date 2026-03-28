from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PersonaVoiceProfile:
    """Deterministic voice rules for one sender persona."""

    key: str
    internal_openers: tuple[str, ...]
    external_openers: tuple[str, ...]
    internal_short_acks: tuple[str, ...]
    external_short_acks: tuple[str, ...]
    internal_default_detail: str
    external_default_detail: str
    internal_closings: tuple[str, ...]
    external_closings: tuple[str, ...]
    detail_prefixes: tuple[str, ...]


EXECUTIVE_PROFILE = PersonaVoiceProfile(
    key="executive",
    internal_openers=(
        "At a high level, we need to keep this moving.",
        "At a high level, let's keep this moving.",
    ),
    external_openers=(
        "Thanks again for the time today. At a high level, we should keep this simple.",
        "Thanks again for the discussion today. At a high level, we should keep this simple.",
    ),
    internal_short_acks=(
        "At a high level, please keep the owners clear and keep this moving.",
        "At a high level, please keep timing tight and close the loop quickly.",
    ),
    external_short_acks=(
        "Thanks again for the time today. We will keep the path simple from here.",
        "Thanks again for the discussion today. We will keep the path simple from here.",
    ),
    internal_default_detail="Please keep the timeline, owners, and open decisions visible.",
    external_default_detail="We will keep the next step, owner, and decision path visible.",
    internal_closings=(
        "Let's keep the decision path simple from here.",
        "Let's keep the owners and timing clean from here.",
    ),
    external_closings=(
        "Please let me know if you want us to simplify the next step on our side.",
        "If it helps, we can keep the owners and timing tighter on our side.",
    ),
    detail_prefixes=("Summary:", "Decision:", "Risk:", "Next step:"),
)

SALES_PROFILE = PersonaVoiceProfile(
    key="sales",
    internal_openers=(
        "I pulled together the commercial details below.",
        "I captured the commercial details below.",
    ),
    external_openers=(
        "Thanks again for the discussion today. I want to keep the commercial path clear.",
        "Thanks again for the time today. I want to keep the commercial path clear.",
    ),
    internal_short_acks=(
        "Quick commercial note below so we keep the close plan moving.",
        "Sending the short commercial version here to keep the timeline moving.",
    ),
    external_short_acks=(
        "Thanks again for the discussion today. Sharing the short commercial path here.",
        "Thanks again for the time today. Sharing the short commercial path here.",
    ),
    internal_default_detail="I've outlined the commercial path, timing, and owners below.",
    external_default_detail="I've outlined the commercial path and next-step timing below.",
    internal_closings=(
        "If anything changes on timing, send it over and I'll update the commercial path.",
        "If the timing shifts, send it over and I'll update the close plan.",
    ),
    external_closings=(
        "If anything changes on timing, please send it over and I'll update the commercial path.",
        "If the timing shifts, please send it over and I'll update the close plan.",
    ),
    detail_prefixes=("Commercial:", "Timing:", "Owner:", "Next step:"),
)

CSM_PROFILE = PersonaVoiceProfile(
    key="csm",
    internal_openers=(
        "I pulled together the customer details below.",
        "I captured the customer details below.",
    ),
    external_openers=(
        "Thanks again for the time today. I captured the customer actions below.",
        "Thanks again for the discussion today. I captured the customer actions below.",
    ),
    internal_short_acks=(
        "Quick customer note below so we stay aligned on next steps.",
        "Sending the short customer version here to keep everyone aligned.",
    ),
    external_short_acks=(
        "Thanks again for the time today. Sharing the short customer version here.",
        "Thanks again for the discussion today. Sharing the short customer version here.",
    ),
    internal_default_detail="I've outlined the customer actions, owners, and timing below.",
    external_default_detail="I've outlined the customer actions and timing update below.",
    internal_closings=(
        "If I missed anything, reply back and I'll update the customer summary.",
        "If anything looks off, send it back and I'll adjust the customer summary.",
    ),
    external_closings=(
        "Please let me know if you'd like us to adjust anything on our side.",
        "If anything should be revised on our side, please send it over.",
    ),
    detail_prefixes=("Customer:", "Timing:", "Owner:", "Next step:"),
)

SUPPORT_PROFILE = PersonaVoiceProfile(
    key="support",
    internal_openers=(
        "I pulled together the technical details below.",
        "I captured the technical details below.",
    ),
    external_openers=(
        (
            "Thanks again for the time today. "
            "I know this has been frustrating, and I included the fix details below."
        ),
        (
            "Thanks again for the discussion today. "
            "I know this has been frustrating, and I included the fix details below."
        ),
    ),
    internal_short_acks=(
        "Quick technical summary below so we stay aligned on the fix.",
        "Sending the short technical version here so the fix path is clear.",
    ),
    external_short_acks=(
        "Thanks again for the time today. Sharing the short fix path here.",
        "Thanks again for the discussion today. Sharing the short fix path here.",
    ),
    internal_default_detail="I've outlined the current logs, fix path, and open checks below.",
    external_default_detail="I've outlined the current fix path and timing update below.",
    internal_closings=(
        "If anything in the logs looks off, send it back and I'll adjust the fix path.",
        "If the fix path changes, send it back and I'll update the technical detail.",
    ),
    external_closings=(
        "If anything in the fix path should change on our side, please send it over.",
        "If you'd like us to revise the fix path on our side, please let me know.",
    ),
    detail_prefixes=("Log:", "Impact:", "Fix:", "Next:"),
)

PRODUCT_PROFILE = PersonaVoiceProfile(
    key="product",
    internal_openers=(
        "I pulled together the product details below.",
        "I captured the product details below.",
    ),
    external_openers=(
        "Thanks again for the time today. I summarized the feedback and roadmap view below.",
        "Thanks again for the discussion today. I summarized the feedback and roadmap view below.",
    ),
    internal_short_acks=(
        "Quick product note below so the roadmap tradeoff is clear.",
        "Sending the short product version here so the feedback is clear.",
    ),
    external_short_acks=(
        "Thanks again for the time today. Sharing the short roadmap view here.",
        "Thanks again for the discussion today. Sharing the short roadmap view here.",
    ),
    internal_default_detail=(
        "I've outlined the feedback, roadmap tradeoffs, and open constraints below."
    ),
    external_default_detail="I've outlined the feedback, roadmap view, and timing below.",
    internal_closings=(
        (
            "If anything in the feedback should be revised, "
            "send it back and I'll update the roadmap summary."
        ),
        "If the tradeoffs shift, send it back and I'll adjust the roadmap summary.",
    ),
    external_closings=(
        "Please let me know if any part of the feedback summary should be revised.",
        "If the feedback summary should change on our side, please send it over.",
    ),
    detail_prefixes=("Roadmap:", "Feedback:", "Constraint:", "Next step:"),
)

IMPLEMENTATION_PROFILE = PersonaVoiceProfile(
    key="implementation",
    internal_openers=(
        "I pulled together the implementation details below.",
        "I captured the implementation details below.",
    ),
    external_openers=(
        "Thanks again for the time today. I broke the workstream and cutover steps out below.",
        (
            "Thanks again for the discussion today. "
            "I broke the workstream and cutover steps out below."
        ),
    ),
    internal_short_acks=(
        "Quick implementation note below so the workstream is clear.",
        "Sending the short implementation version here so the cutover path is clear.",
    ),
    external_short_acks=(
        "Thanks again for the time today. Sharing the short implementation path here.",
        "Thanks again for the discussion today. Sharing the short implementation path here.",
    ),
    internal_default_detail=(
        "I've outlined the workstream, dependencies, and milestone timing below."
    ),
    external_default_detail="I've outlined the workstream, milestone timing, and handoffs below.",
    internal_closings=(
        "If the workstream shifts, send it back and I'll update the milestone plan.",
        "If any dependency changes, send it back and I'll update the cutover plan.",
    ),
    external_closings=(
        "Please let me know if any workstream or handoff should change on our side.",
        "If any cutover detail should change on our side, please send it over.",
    ),
    detail_prefixes=("Workstream:", "Dependency:", "Milestone:", "Next step:"),
)

OPERATIONS_PROFILE = PersonaVoiceProfile(
    key="operations",
    internal_openers=(
        "I pulled together the operations details below.",
        "I captured the operations details below.",
    ),
    external_openers=(
        "Thanks again for the time today. I mapped the handoffs and timing below.",
        "Thanks again for the discussion today. I mapped the handoffs and timing below.",
    ),
    internal_short_acks=(
        "Quick ops note below so the handoffs stay clear.",
        "Sending the short ops version here so the timing stays clear.",
    ),
    external_short_acks=(
        "Thanks again for the time today. Sharing the short ops timing here.",
        "Thanks again for the discussion today. Sharing the short ops timing here.",
    ),
    internal_default_detail=(
        "I've outlined the operational handoffs, timing, "
        "and open dependencies below."
    ),
    external_default_detail=(
        "I've outlined the handoffs, timing, "
        "and operational dependencies below."
    ),
    internal_closings=(
        "If any handoff shifts, send it back and I'll update the ops summary.",
        "If the timing moves, send it back and I'll update the ops summary.",
    ),
    external_closings=(
        "Please let me know if any handoff should change on our side.",
        "If the timing should move on our side, please send it over.",
    ),
    detail_prefixes=("Ops:", "Handoff:", "Timing:", "Next step:"),
)

DEFAULT_IC_PROFILE = PersonaVoiceProfile(
    key="default_ic",
    internal_openers=(
        "I pulled together the details below.",
        "I captured the details below.",
    ),
    external_openers=(
        "Thanks again for the time today.",
        "Thanks again for the discussion today.",
    ),
    internal_short_acks=(
        "Quick note below so we're aligned on next steps.",
        "Sending the short version here to keep everyone aligned.",
    ),
    external_short_acks=(
        "A quick summary is below, and we're aligned on the next step.",
        "Sharing the short version here so we stay aligned on timing.",
    ),
    internal_default_detail="I've outlined the current tasks, timing, and open questions below.",
    external_default_detail="We will send a written summary and timing update after the meeting.",
    internal_closings=(
        "If I missed anything, reply back and I'll update the summary.",
        "If anything looks off, send it back and I'll adjust the detail.",
    ),
    external_closings=(
        "Please let me know if you'd like us to adjust anything on our side.",
        "If anything should be revised on our side, please send it over.",
    ),
    detail_prefixes=("First,", "Also,", "Separately,", "Finally,"),
)


def resolve_persona_voice_profile(
    *,
    sender_role: str,
    seniority: str,
) -> PersonaVoiceProfile:
    normalized_role = sender_role.strip().lower()

    if seniority == "executive" or any(
        token in normalized_role for token in ("chief", "vp", "executive sponsor")
    ):
        return EXECUTIVE_PROFILE
    if normalized_role in {"ae", "account executive"} or "sales" in normalized_role:
        return SALES_PROFILE
    if "customer success" in normalized_role or normalized_role == "csm":
        return CSM_PROFILE
    if "support" in normalized_role:
        return SUPPORT_PROFILE
    if "product manager" in normalized_role:
        return PRODUCT_PROFILE
    if "implementation" in normalized_role:
        return IMPLEMENTATION_PROFILE
    if "operations" in normalized_role:
        return OPERATIONS_PROFILE
    return DEFAULT_IC_PROFILE
