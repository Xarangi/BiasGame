"""Normative posterior engine.

Given a world's slots and the evidence items revealed so far (each carrying a
log-likelihood ratio toward its ``about_slot``), compute the posterior an ideal
observer holds over "which slot is the correct answer", starting from a uniform
prior. This is the yardstick the BUD measure compares elicited beliefs against:
with no revealed evidence the normative posterior stays uniform, so any
demographic tilt in the subject's beliefs is pure prior/stereotype.
"""

from __future__ import annotations

import math


def posterior(slot_ids: list[str], revealed: list[dict]) -> dict[str, float]:
    """``revealed`` rows need keys ``about_slot`` and ``llr``."""
    log_odds = {s: 0.0 for s in slot_ids}
    for ev in revealed:
        if ev["about_slot"] in log_odds:
            log_odds[ev["about_slot"]] += float(ev["llr"])
    mx = max(log_odds.values())
    weights = {s: math.exp(v - mx) for s, v in log_odds.items()}
    z = sum(weights.values())
    return {s: w / z for s, w in weights.items()}
