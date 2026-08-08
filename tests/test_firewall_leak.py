"""THE leak test (engineering invariant #2).

With the NEUTRAL identity package, a full episode transcript must be
byte-identical across permutations: the world itself carries zero identity
signal, so the only thing that can differ between permutations is the identity
surface layer. Run for every template x a deterministic subject.
"""

import pytest

from biasgame.agents.subjects import UnbiasedMock
from biasgame.engine.episode import RUNNERS
from biasgame.identity.firewall import Firewall
from biasgame.identity.packages import NEUTRAL
from biasgame.identity.permutations import latin_square
from biasgame.worlds.compiler import compile_world

TEMPLATES = ["inquiry", "gate", "tribunal", "service_desk", "archive", "studio", "grievance_desk"]


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("ambiguity,dose", [("ambig", 0), ("disambig", 2)])
def test_neutral_render_is_permutation_invariant(template, ambiguity, dose):
    world = compile_world(template, "Race_ethnicity", 0, ambiguity, dose)
    perms = latin_square(world.slot_ids(), NEUTRAL)
    transcripts = []
    for perm in perms[:3]:
        fw = Firewall(package=NEUTRAL, permutation=perm, world_id=world.world_id)
        log = RUNNERS[template](world, fw, UnbiasedMock(seed=7), "unbiased_mock", 7)
        # Strip permutation-identifying metadata; everything else must match.
        rows = log.to_jsonl().splitlines()
        rows = [r for r in rows if '"kind": "meta"' not in r]
        transcripts.append("\n".join(rows))
    assert transcripts[0] == transcripts[1] == transcripts[2], (
        f"identity leak in template {template!r}: transcripts differ across "
        "permutations under the neutral package"
    )
