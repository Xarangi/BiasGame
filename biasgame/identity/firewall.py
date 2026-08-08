"""The identity firewall: the only code that resolves slot tokens to identities.

``render`` substitutes ``[[Pk]]`` tokens in slot-space text with identity
surface forms (name, plus an explicit descriptor on first mention when the
marking mode asks for it). ``parse`` maps names appearing in subject output
back to slot IDs, so actions re-enter the world in slot space.

Invariant (enforced by tests/test_firewall_leak.py): with the NEUTRAL package,
rendered output is identical across all permutations — the world itself
carries zero identity signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from biasgame.identity.packages import Identity, IdentityPackage
from biasgame.identity.permutations import Permutation

_TOKEN_RE = re.compile(r"\[\[(P\d+)\]\]")

MARKING_NONE = "none"  # names only
MARKING_EXPLICIT = "explicit"  # name + stated descriptor on first mention


@dataclass
class Firewall:
    package: IdentityPackage
    permutation: Permutation
    world_id: str
    marking: str = MARKING_EXPLICIT
    _names: dict[str, str] = field(default_factory=dict)  # slot -> rendered name
    _mentioned: set = field(default_factory=set)

    def __post_init__(self) -> None:
        by_id = {i.identity_id: i for i in self.package.identities}
        for slot, ident_id in self.permutation.assignment:
            if self.package.package_id == "neutral":
                # Neutral names follow the SLOT, not the permuted identity, so a
                # neutral render is permutation-invariant — the leak-test contract.
                self._names[slot] = f"Person {slot[1:]}"
            else:
                ident = by_id[ident_id]
                self._names[slot] = self.package.pick_name(ident, self.world_id, slot)

    def identity_of(self, slot_id: str) -> Identity:
        ident_id = self.permutation.identity_for(slot_id)
        return next(i for i in self.package.identities if i.identity_id == ident_id)

    def name_of(self, slot_id: str) -> str:
        return self._names[slot_id]

    def render(self, text: str) -> str:
        def _sub(match: re.Match) -> str:
            slot = match.group(1)
            name = self._names[slot]
            ident = self.identity_of(slot)
            if (
                self.marking == MARKING_EXPLICIT
                and ident.descriptor
                and slot not in self._mentioned
            ):
                self._mentioned.add(slot)
                return f"{name} ({ident.descriptor})"
            self._mentioned.add(slot)
            return name

        return _TOKEN_RE.sub(_sub, text)

    def parse(self, text: str) -> str | None:
        """Return the slot ID of the first rendered name found in ``text``."""
        lowered = text.lower()
        # Longest-name-first so "Mei-Ling Chen" wins over any shorter overlap.
        for slot, name in sorted(self._names.items(), key=lambda kv: -len(kv[1])):
            if name.lower() in lowered:
                return slot
            surname = name.split()[-1].lower()
            if re.search(rf"\b{re.escape(surname)}\b", lowered):
                return slot
        direct = _TOKEN_RE.search(text)
        if direct:
            return direct.group(1)
        m = re.search(r"\b(P\d)\b", text)
        return m.group(1) if m else None
