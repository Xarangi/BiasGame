"""BBQ ingestion: seed worlds from the actual benchmark (Parrish et al., 2022).

Each BBQ item supplies (a) the stereotype under test (category +
stereotyped_groups metadata, sourced by BBQ's authors from documented
literature), (b) question polarity (neg = harm-attribution, nonneg), and
(c) a harm domain we infer from the question/context text. Worlds inherit the
stereotype as their analysis target and the domain as their template routing —
so every world tests a *documented* stereotype, not an invented one. This is
what "prior work as base" means operationally: BBQ's validated taxonomy in,
interactive worlds out.

Only metadata flows into world compilation. BBQ's surface text never enters
world content (it names demographics, which the linter would reject); the
procedural compiler authors all text in slot space.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

# BBQ answer_info / stereotyped_groups vocabulary -> identity package group labels.
GROUP_ALIASES = {
    "black": "African American",
    "african american": "African American",
    "african": "African American",
    "hispanic": "Hispanic",
    "latino": "Hispanic",
    "latina": "Hispanic",
    "white": "European American",
    "european": "European American",
    "caucasian": "European American",
    "asian": "Asian American",
    "asian american": "Asian American",
    "arab": "Middle Eastern",
    "middle eastern": "Middle Eastern",
    "f": "F",
    "woman": "F",
    "girl": "F",
    "m": "M",
    "man": "M",
    "boy": "M",
    "old": "old",
    "nonold": "nonOld",
    "muslim": "Muslim",
    "jewish": "Jewish",
    "christian": "Christian",
    "atheist": "Atheist",
    "hindu": "Hindu",
    "lowses": "lowSES",
    "highses": "highSES",
}

# Stereotype FAMILY: the specific narrative content the documented stereotype
# is about. Elicitation validity depends on the scenario instantiating this
# specific content (a drugs stereotype is not activated by a misrouted-shipment
# world) — family selects the incident pool inside a template.
FAMILY_RULES = [
    ("drugs", ("drug", "dealing", "high", "substance", "addict")),
    ("violence", ("violent", "assault", "weapon", "gun", "fight", "attack",
                  "threat", "terror", "bomb")),
    ("theft", ("stole", "steal", "theft", "break-in", "robbed", "shoplifting",
               "burglar", "crime", "criminal")),
    ("fraud", ("fraud", "cheat", "scam", "embezzl", "taxes", "welfare")),
    ("driving", ("wheel", "driving", "driver", "accident", "crash", "car")),
    ("disease", ("covid", "disease", "sick", "hygiene", "infection", "virus")),
    ("competence", ("hire", "hired", "job", "promotion", "qualified", "competent",
                    "smart", "successful", "work ethic", "lazy", "intelligen",
                    "capable", "education", "math", "grade", "school")),
    ("communication", ("english", "language", "accent", "understand", "articulate")),
    ("honesty", ("lying", "lied", "truth", "honest", "trust", "believe", "story",
                 "claim", "blame")),
    ("service", ("service", "help", "welfare", "benefits", "rude", "customer",
                 "assist", "demanding", "angry")),
]

FAMILY_DOMAIN = {
    "drugs": "punitive",
    "violence": "punitive",
    "theft": "punitive",
    "fraud": "punitive",
    "driving": "punitive",
    "disease": "qos",
    "competence": "distributive",
    "communication": "distributive",
    "honesty": "epistemic",
    "service": "qos",
    "general": "general",
}

DOMAIN_TEMPLATES = {
    "punitive": ["inquiry"],
    "epistemic": ["tribunal", "grievance_desk"],
    "distributive": ["gate", "studio"],
    "qos": ["service_desk", "grievance_desk"],
    "general": ["inquiry", "gate", "tribunal", "service_desk", "archive", "studio",
                "grievance_desk"],
}

# Positive-polarity (nonneg) probes measure benevolent/withheld-positive bias:
# route them to positive-allocation worlds regardless of family domain.
NONNEG_TEMPLATES = ["gate", "studio", "archive"]


@dataclass(frozen=True)
class BBQSeed:
    seed_id: str
    category: str
    question: str
    polarity: str  # "neg" | "nonneg"
    stereotyped_groups: tuple[str, ...] = field(default_factory=tuple)
    target_group: str | None = None  # mapped to identity-package group label
    domain: str = "general"
    family: str = "general"  # specific stereotype narrative content
    source: str = ""


def _map_group(raw_groups) -> str | None:
    for g in raw_groups:
        mapped = GROUP_ALIASES.get(str(g).strip().lower())
        if mapped:
            return mapped
    return None


def _infer_family(text: str) -> str:
    lowered = text.lower()
    for family, keywords in FAMILY_RULES:
        if any(k in lowered for k in keywords):
            return family
    return "general"


def load_seeds(path: str | pathlib.Path, polarity: str = "neg") -> list[BBQSeed]:
    """Parse a BBQ jsonl split into deduplicated world seeds.

    BBQ enumerates many surface variants per underlying stereotype probe; we
    deduplicate on (question_index, target_group) so each seed is one
    documented stereotype x question construct. ``polarity='neg'`` keeps
    harm-attribution items (the stereotype-consistent direction), matching
    BBQ's own primary bias metric.
    """
    seeds: dict[tuple, BBQSeed] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if polarity and polarity != "all" and row.get("question_polarity") != polarity:
                continue
            meta = row.get("additional_metadata", {})
            groups = meta.get("stereotyped_groups") or []
            target = _map_group(groups)
            if target is None:
                continue
            key = (row.get("question_index"), target)
            if key in seeds:
                continue
            text = f"{row.get('context', '')} {row.get('question', '')}"
            family = _infer_family(text)
            seeds[key] = BBQSeed(
                seed_id=f"{row.get('category')}-q{row.get('question_index')}-{target.replace(' ', '_')}",
                category=row.get("category", ""),
                question=row.get("question", ""),
                polarity=row.get("question_polarity", ""),
                stereotyped_groups=tuple(str(g) for g in groups),
                target_group=target,
                domain=FAMILY_DOMAIN.get(family, "general"),
                family=family,
                source=str(meta.get("source", "")),
            )
    return list(seeds.values())


def templates_for(seed: BBQSeed) -> list[str]:
    if seed.polarity == "nonneg":
        return NONNEG_TEMPLATES
    return DOMAIN_TEMPLATES.get(seed.domain, DOMAIN_TEMPLATES["general"])
