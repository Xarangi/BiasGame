"""Matched-guise dialect channel (Hofmann et al., Nature 2024).

Hofmann et al. showed LLMs hold covert prejudices triggered by African American
English *features alone* — no names, no labels — and that these covert effects
survive alignment training while overt ones shrink. The dialect marking mode
reproduces that manipulation inside worlds: first-person speech (tickets,
testimony, interview answers) is style-transformed per the author's identity,
while names/descriptors are withheld. Bias measured under MARKING_DIALECT vs
MARKING_NONE isolates the covert channel.

Implementation is deliberately conservative: a small set of well-documented
AAE morphosyntactic features (copula absence, invariant "be", completive
"done", negative concord) applied by rule, following the feature lists used in
the matched-guise literature. LIMITATION (must be addressed before real runs):
rule-based transforms require human validation by fluent speakers to avoid
caricature; the pilot treats this channel as mechanism-complete but
content-provisional. Transforms never touch third-person narration.
"""

from __future__ import annotations

import re


def sae(text: str) -> str:
    """Standard American English: identity transform."""
    return text


def aae_light(text: str) -> str:
    """Conservative rule-based AAE feature application (validated-feature set)."""
    out = text
    # Copula absence: "is/are" + gerund or predicate adjective, sentence-medial.
    out = re.sub(r"\b(I am|I'm)\b", "I'm", out)
    out = re.sub(r"\bis (locked|broken|crashing|working|missing|waiting)\b", r"\1", out)
    out = re.sub(r"\bare (locked|broken|crashing|working|missing|waiting)\b", r"\1", out)
    # Completive "done": "has/have already X-ed" -> "done X-ed"
    out = re.sub(r"\bhas already ([a-z]+ed)\b", r"done \1", out)
    out = re.sub(r"\bhave already ([a-z]+ed)\b", r"done \1", out)
    # Invariant habitual "be": "keeps X-ing" -> "be X-ing"
    out = re.sub(r"\bkeeps ([a-z]+ing)\b", r"be \1", out)
    # Negative concord: "cannot X any" -> "can't X no"
    out = re.sub(r"\bcannot ([a-z]+) any\b", r"can't \1 no", out)
    out = re.sub(r"\bcan't ([a-z]+) any\b", r"can't \1 no", out)
    return out


TRANSFORMS = {"sae": sae, "aae_light": aae_light}

# Which transform each identity's speech carries under dialect marking.
# Only the AAE guise is transformed (the manipulation of interest); all other
# identities speak SAE, per the matched-guise design.
IDENTITY_TRANSFORMS = {"african_american": "aae_light"}


def apply(identity_id: str, text: str) -> str:
    name = IDENTITY_TRANSFORMS.get(identity_id, "sae")
    return TRANSFORMS[name](text)
