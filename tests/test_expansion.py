"""Tests for the full scenario build-out: BBQ ingestion, tribunal/TIG,
studio/DBI, dialect channel, question phrasing/tone, per-world targets."""

from biasgame.agents.subjects import BiasedMock, UnbiasedMock
from biasgame.engine.episode import RUNNERS
from biasgame.identity.dialect import aae_light
from biasgame.identity.firewall import MARKING_DIALECT, Firewall
from biasgame.identity.packages import PACKAGES, RACE_ETHNICITY
from biasgame.identity.permutations import latin_square
from biasgame.scoring.measures import dbi, parse_jsonl, scr, tig
from biasgame.worlds.bbq import load_seeds, templates_for
from biasgame.worlds.compiler import compile_world
from biasgame.worlds.lint import lint_world


def _episode(template, subject, perm_index=0, skeleton=0, bbq_seed=None, **kw):
    world = compile_world(
        template, "Race_ethnicity", skeleton, "ambig", 0, bbq_seed=bbq_seed
    )
    perm = latin_square(world.slot_ids(), RACE_ETHNICITY)[perm_index]
    fw = Firewall(package=RACE_ETHNICITY, permutation=perm, world_id=world.world_id)
    log = RUNNERS[template](world, fw, subject, subject.name, 0, **kw)
    return parse_jsonl(log.to_jsonl())


def test_bbq_seeds_load_and_route():
    seeds = load_seeds("Race_ethnicity.jsonl")
    assert len(seeds) >= 30
    assert all(s.target_group for s in seeds)
    crime = [s for s in seeds if "crime" in s.question.lower()]
    assert crime and all("inquiry" in templates_for(s) for s in crime)


def test_bbq_seeded_world_carries_target_and_passes_lint():
    seeds = load_seeds("Race_ethnicity.jsonl")
    seed = next(s for s in seeds if s.domain == "punitive")
    world = compile_world("inquiry", seed.category, 0, "ambig", 0, bbq_seed=seed)
    assert world.bbq_meta["target_group"] == seed.target_group
    assert lint_world(world) == []


def test_all_templates_lint_clean_across_skeletons():
    for template in ("inquiry", "gate", "tribunal", "service_desk", "archive", "studio"):
        for i in range(6):
            world = compile_world(template, "Race_ethnicity", i, "disambig", 2)
            assert lint_world(world) == [], f"lint violations in {world.world_id}"


def test_all_packages_have_five_identities():
    for pkg in PACKAGES.values():
        assert len(pkg.identities) == 5, pkg.package_id


def test_tribunal_truth_values_balanced():
    world = compile_world("tribunal", "Race_ethnicity", 1)
    truth = world.bbq_meta["testimony_truth"]
    assert sum(truth.values()) == 3 and len(truth) == 5
    assert world.ground_truth_slot in [s for s, t in truth.items() if t]


def test_tig_detects_credence_deflation():
    logs_biased, logs_clean = [], []
    for skeleton in range(3):
        for p in range(5):
            logs_biased.append(
                _episode("tribunal", BiasedMock(0.8, "Race_ethnicity", 1), p, skeleton)
            )
            logs_clean.append(_episode("tribunal", UnbiasedMock(1), p, skeleton))
    b = tig(logs_biased, "African American")
    c = tig(logs_clean, "African American")
    assert b["tig_credence_gap"] < -0.15, b
    assert abs(c["tig_credence_gap"]) < 0.02, c


def test_dbi_detects_delegation_bias():
    logs_biased, logs_clean = [], []
    for skeleton in range(3):
        for p in range(5):
            logs_biased.append(
                _episode("studio", BiasedMock(0.9, "Race_ethnicity", 2), p, skeleton)
            )
            logs_clean.append(_episode("studio", UnbiasedMock(2), p, skeleton))
    b = dbi(logs_biased, "African American")
    c = dbi(logs_clean, "African American")
    assert b["dbi_lead_gap"] < -0.1, b  # target withheld from lead role
    assert b["dbi_recheck_solid_gap"] > 0.3, b  # target's solid work re-checked
    assert abs(c["dbi_lead_gap"]) < 0.05 and abs(c["dbi_recheck_solid_gap"]) < 0.05, c


def test_question_phrasing_tone_gap():
    logs = []
    for skeleton in range(3):
        for p in range(5):
            logs.append(
                _episode(
                    "inquiry",
                    BiasedMock(0.9, "Race_ethnicity", 3),
                    p,
                    skeleton,
                    phrase_questions=True,
                )
            )
    s = scr(logs, "African American")
    assert s.get("scr_tone_gap", 0.0) > 0.2, s


def test_dialect_transform_and_marking():
    assert aae_light("My account has already expired and I cannot get any help") != (
        "My account has already expired and I cannot get any help"
    )
    world = compile_world("service_desk", "Race_ethnicity", 0)
    perm = latin_square(world.slot_ids(), RACE_ETHNICITY)[0]
    fw = Firewall(
        package=RACE_ETHNICITY, permutation=perm, world_id=world.world_id,
        marking=MARKING_DIALECT,
    )
    aae_slot = perm.slot_for("african_american")
    other_slot = next(s for s in world.slot_ids() if s != aae_slot)
    speech = "My badge keeps failing and the door is locked."
    assert fw.style(other_slot, speech) == speech
    assert fw.style(aae_slot, speech) != speech


def test_per_world_target_from_bbq_meta():
    seeds = load_seeds("Race_ethnicity.jsonl")
    seed = next(s for s in seeds if s.target_group == "Hispanic")
    subj = BiasedMock(0.9, "Race_ethnicity", 0, target_group=seed.target_group)
    log = _episode("inquiry", subj, 0, 0, bbq_seed=seed)
    assert log["meta"]["target_group"] == "Hispanic"
