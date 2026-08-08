"""Leakage linter: rejects demographic content in slot-space world text.

Slot-token discipline is invariant #1: skeletons may refer to people only via
[[Pk]] tokens. This linter is a compile-time gate (and a CI test) that scans
every text field of a skeleton for demographic vocabulary, names from any
registered identity package, and gendered pronouns.
"""

from __future__ import annotations

import re
from dataclasses import fields as dc_fields

from biasgame.identity.packages import PACKAGES
from biasgame.worlds.schema import WorldSkeleton

_DEMOGRAPHIC_TERMS = {
    "black", "white", "hispanic", "latino", "latina", "asian", "african",
    "european", "arab", "middle eastern", "native american", "caucasian",
    "man", "woman", "male", "female", "boy", "girl", "nonbinary",
    "he", "she", "him", "her", "his", "hers", "mr", "mrs", "ms",
    "muslim", "christian", "jewish", "immigrant",
}

_WORD_RE = re.compile(r"[a-z']+")


def _texts_of(world: WorldSkeleton):
    yield "scenario", world.scenario
    if world.decision:
        yield "decision.question", world.decision.question
    for p in world.personas:
        for intent, answer in p.facts.items():
            yield f"{p.slot_id}.facts.{intent}", answer
    for e in world.evidence:
        yield f"evidence.{e.item_id}", e.text
    for t in world.tickets:
        yield f"ticket.{t.ticket_id}", t.text
    for s in world.statements:
        yield f"statement.{s.statement_id}", s.text


def lint_world(world: WorldSkeleton) -> list[str]:
    """Return a list of violations; empty list = clean."""
    violations: list[str] = []
    all_names = {
        name.lower()
        for pkg in PACKAGES.values()
        if pkg.package_id != "neutral"
        for ident in pkg.identities
        for name in ident.name_pool
    }
    for path, text in _texts_of(world):
        lowered = text.lower()
        for term in sorted(_DEMOGRAPHIC_TERMS):
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                violations.append(f"{path}: demographic term {term!r}")
        for name in all_names:
            if name in lowered:
                violations.append(f"{path}: identity-package name {name!r}")
    return violations
