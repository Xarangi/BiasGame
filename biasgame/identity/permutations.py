"""Identity-to-slot assignments.

A permutation maps each slot ID to an identity from the world's package. The
default design is a cyclic Latin square over k slots x k identities: across the
k rows, every slot carries every identity exactly once, so identity effects are
fully crossed with slot/fact effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from biasgame.identity.packages import Identity, IdentityPackage


@dataclass(frozen=True)
class Permutation:
    permutation_id: str
    assignment: tuple[tuple[str, str], ...]  # ((slot_id, identity_id), ...)

    def identity_for(self, slot_id: str) -> str:
        for slot, ident in self.assignment:
            if slot == slot_id:
                return ident
        raise KeyError(slot_id)

    def slot_for(self, identity_id: str) -> str:
        for slot, ident in self.assignment:
            if ident == identity_id:
                return slot
        raise KeyError(identity_id)


def latin_square(slot_ids: list[str], package: IdentityPackage) -> list[Permutation]:
    idents: list[Identity] = list(package.identities)[: len(slot_ids)]
    if len(idents) < len(slot_ids):
        raise ValueError(
            f"package {package.package_id} has {len(idents)} identities; "
            f"world needs {len(slot_ids)}"
        )
    rows = []
    k = len(slot_ids)
    for r in range(k):
        assignment = tuple(
            (slot_ids[i], idents[(i + r) % k].identity_id) for i in range(k)
        )
        rows.append(Permutation(permutation_id=f"latin_row_{r}", assignment=assignment))
    return rows
