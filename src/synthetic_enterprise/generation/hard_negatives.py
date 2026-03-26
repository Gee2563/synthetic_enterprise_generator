from __future__ import annotations


def hard_negative_count(*, template_count: int, ratio: float) -> int:
    """Return the deterministic number of hard-negative rows to emit."""

    if template_count <= 0 or ratio <= 0.0:
        return 0

    return min(template_count, max(0, int(round(template_count * ratio))))
