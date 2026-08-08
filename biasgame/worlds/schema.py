"""World skeleton schema — everything here lives in slot space.

The linter (worlds.lint) enforces that no field of a skeleton contains
demographic content; people are referred to only via slot tokens like
``[[P3]]``. Identity is attached exclusively by the firewall at render time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SLOT_TOKEN = "[[{slot}]]"


def slot_token(slot_id: str) -> str:
    return SLOT_TOKEN.format(slot=slot_id)


@dataclass
class EvidenceItem:
    """A discoverable fact with a ground-truth diagnostic value.

    ``llr`` is the log-likelihood ratio this item contributes toward the
    proposition "slot ``about_slot`` is the correct answer to the world's
    decision question". 0.0 = non-diagnostic. Positive = inculpatory /
    supports selecting that slot; negative = exculpatory.

    ``held_by`` + ``intent`` locate the item: it is revealed only when the
    subject asks persona ``held_by`` a question matching ``intent``.
    """

    item_id: str
    about_slot: str
    held_by: str
    intent: str
    text: str
    llr: float = 0.0


@dataclass
class SlotPersona:
    """A person in the world, known only by slot ID.

    ``facts`` maps question intents to answer text (slot-space). Every persona
    answers every listed intent deterministically from this sheet; unknown
    intents get the template's deflection line.
    """

    slot_id: str
    role: str
    facts: dict[str, str] = field(default_factory=dict)


@dataclass
class Ticket:
    """Service-desk work item authored by a slot persona."""

    ticket_id: str
    author_slot: str
    text: str
    difficulty: float  # authored ground truth in [0,1]
    legitimacy: float  # authored ground truth in [0,1]; 1 = clearly valid request
    urgency: float  # authored ground truth in [0,1]


@dataclass
class Statement:
    """Archive-world content attributed to a slot persona."""

    statement_id: str
    author_slot: str
    text: str
    importance: float  # authored ground truth in [0,1]


@dataclass
class DecisionSpec:
    question: str
    allow_insufficient: bool = True


@dataclass
class WorldSkeleton:
    world_id: str
    template: str  # inquiry | gate | tribunal | service_desk | archive
    category: str  # BBQ category, e.g. Race_ethnicity
    ambiguity: str  # "ambig" | "disambig"
    evidence_dose: int  # 0..3, number of diagnostic evidence items (ETG dial)
    ground_truth_slot: str | None  # correct answer slot in disambig worlds
    scenario: str  # slot-space scene-setting text
    decision: DecisionSpec | None
    personas: list[SlotPersona] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    tickets: list[Ticket] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)
    intents: dict[str, dict] = field(default_factory=dict)  # intent -> {hostile: bool, label: str}
    bbq_meta: dict = field(default_factory=dict)

    def slot_ids(self) -> list[str]:
        return [p.slot_id for p in self.personas]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @staticmethod
    def from_json(payload: str) -> "WorldSkeleton":
        raw = json.loads(payload)
        raw["personas"] = [SlotPersona(**p) for p in raw["personas"]]
        raw["evidence"] = [EvidenceItem(**e) for e in raw["evidence"]]
        raw["tickets"] = [Ticket(**t) for t in raw["tickets"]]
        raw["statements"] = [Statement(**s) for s in raw["statements"]]
        raw["decision"] = DecisionSpec(**raw["decision"]) if raw.get("decision") else None
        return WorldSkeleton(**raw)
