"""Deterministic keyed RNG.

Every stochastic choice in world compilation and mock agents draws from a local
``random.Random`` seeded from a string key, so identical keys always reproduce
identical sequences regardless of call order elsewhere. The seed derivation
(first 8 bytes of sha256, big-endian) is a frozen compatibility contract:
changing it invalidates every frozen world library.
"""

from __future__ import annotations

import hashlib
import random


def keyed_rng(*key_parts: object) -> random.Random:
    key = "\x1f".join(str(p) for p in key_parts)
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)
