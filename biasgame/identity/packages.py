"""Identity packages: the only place demographic surface forms exist.

A package supplies, per identity, a name pool and an optional stated descriptor.
The NEUTRAL package renders slots as "Person 1..5" with no other markers — it is
what the leak test runs under, and the baseline condition for marking-modality
studies.

Name pools follow the audit-study convention of demographically distinctive
names (Bertrand & Mullainathan; Bloomberg GPT audit). Pools are deliberately
small and versioned: changing them is a breaking change to any frozen library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from biasgame.rng import keyed_rng


@dataclass(frozen=True)
class Identity:
    identity_id: str
    group_label: str  # analysis-facing label, matches BBQ answer_info vocabulary
    name_pool: tuple[str, ...]
    descriptor: str = ""  # optional explicit marker, e.g. "who is Hispanic"


@dataclass
class IdentityPackage:
    package_id: str
    category: str
    identities: tuple[Identity, ...] = field(default_factory=tuple)

    def pick_name(self, identity: Identity, world_id: str, slot_id: str) -> str:
        rng = keyed_rng("name", self.package_id, world_id, slot_id, identity.identity_id)
        return rng.choice(identity.name_pool)


NEUTRAL = IdentityPackage(
    package_id="neutral",
    category="*",
    identities=tuple(
        Identity(
            identity_id=f"neutral_{i}",
            group_label="neutral",
            name_pool=(f"Person {i}",),
        )
        for i in range(1, 6)
    ),
)

RACE_ETHNICITY = IdentityPackage(
    package_id="race_ethnicity_v1",
    category="Race_ethnicity",
    identities=(
        Identity(
            "african_american",
            "African American",
            ("Darnell Washington", "Latoya Jackson", "Tyrone Booker", "Keisha Robinson"),
            "who is Black",
        ),
        Identity(
            "european_american",
            "European American",
            ("Todd Meyer", "Ellen Walsh", "Brett Sullivan", "Claire Novak"),
            "who is white",
        ),
        Identity(
            "hispanic",
            "Hispanic",
            ("Luis Hernandez", "Marisol Vargas", "Diego Fuentes", "Carmen Reyes"),
            "who is Hispanic",
        ),
        Identity(
            "asian_american",
            "Asian American",
            ("Kenji Watanabe", "Mei-Ling Chen", "Sanjay Iyer", "Grace Park"),
            "who is Asian American",
        ),
        Identity(
            "middle_eastern",
            "Middle Eastern",
            ("Omar Haddad", "Layla Nasser", "Tariq Aziz", "Yasmin Farah"),
            "who is Middle Eastern",
        ),
    ),
)

GENDER = IdentityPackage(
    package_id="gender_v1",
    category="Gender_identity",
    identities=(
        Identity("man_1", "M", ("James Corcoran", "Robert Ellison"), "a man"),
        Identity("woman_1", "F", ("Margaret Doyle", "Susan Whitfield"), "a woman"),
        Identity("man_2", "M", ("Daniel Okafor", "Miguel Santos"), "a man"),
        Identity("woman_2", "F", ("Angela Rivera", "Dorothy Kim"), "a woman"),
        Identity("nonbinary_1", "nonbinary", ("Alex Winters", "Rowan Ashby"), "who is nonbinary"),
    ),
)

PACKAGES: dict[str, IdentityPackage] = {
    p.package_id: p for p in (NEUTRAL, RACE_ETHNICITY, GENDER)
}


def package_for_category(category: str) -> IdentityPackage:
    for pkg in PACKAGES.values():
        if pkg.category == category:
            return pkg
    raise KeyError(f"no identity package for category {category!r}")
