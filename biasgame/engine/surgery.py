"""Context surgery (Study 4): sanitized-history decision replay.

At the decision point, a forked copy of the subject replays the entire episode
history with every identity surface form replaced by the neutral slot label
("Person k"), then takes the same final decision. Comparing the original and
sanitized decisions decomposes measured decision bias causally:

- the decision FLIPS under sanitization  -> decision-time identity dependence
  (identity cues in the prompt at the moment of choice drive the outcome);
- the biased decision PERSISTS           -> the bias was already baked into
  the accumulated context (e.g., identity-skewed evidence gathering earlier
  in the trajectory produced a skewed information state).

This checkpoint-replay probe is only expressible in an interactive setting —
a static benchmark has no trajectory to operate on. Restricted to non-dialect
markings: dialect-styled speech cannot be inverted by string substitution.
"""

from __future__ import annotations

import re


def sanitize(text: str, firewall, slot_ids) -> str:
    """Replace every rendered identity surface form with the neutral label.

    Longest rendered name first (mirrors Firewall.parse) so overlapping names
    resolve correctly; also strips first-mention descriptors and bare
    first-name/surname mentions.
    """
    ordered = sorted(slot_ids, key=lambda s: -len(firewall.name_of(s)))
    for slot in ordered:
        name = firewall.name_of(slot)
        neutral = f"Person {slot[1:]}"
        if name == neutral:  # neutral package: already identity-free
            continue
        ident = firewall.identity_of(slot)
        if ident.descriptor:
            text = text.replace(f"{name} ({ident.descriptor})", neutral)
        text = text.replace(name, neutral)
        for part in name.split():
            text = re.sub(rf"\b{re.escape(part)}\b", neutral, text)
    return text
