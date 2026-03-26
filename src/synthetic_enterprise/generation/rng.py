from __future__ import annotations

import hashlib


def derive_seed(seed: int, namespace: str) -> int:
    """Derive stable child seeds without relying on Python's hash()."""

    payload = f"{seed}:{namespace}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)
