"""Determinism, linting, Bayes engine, measures, and synthetic calibration."""

from biasgame.agents.subjects import BiasedMock, UnbiasedMock
from biasgame.engine.episode import RUNNERS
from biasgame.identity.firewall import Firewall
from biasgame.identity.packages import RACE_ETHNICITY
from biasgame.identity.permutations import latin_square
from biasgame.scoring.bayes import posterior
from biasgame.scoring.measures import abs_score, cpb, edg, parse_jsonl, scr
from biasgame.worlds.compiler import compile_world
from biasgame.worlds.lint import lint_world


def _episode(template, subject, perm_index=0, skeleton=0, ambiguity="ambig", dose=0):
    world = compile_world(template, "Race_ethnicity", skeleton, ambiguity, dose)
    perm = latin_square(world.slot_ids(), RACE_ETHNICITY)[perm_index]
    fw = Firewall(package=RACE_ETHNICITY, permutation=perm, world_id=world.world_id)
    log = RUNNERS[template](world, fw, subject, subject.name, 0)
    return parse_jsonl(log.to_jsonl())


def test_replay_determinism():
    a = _episode("inquiry", UnbiasedMock(seed=3))
    b = _episode("inquiry", UnbiasedMock(seed=3))
    assert a == b


def test_worlds_pass_lint():
    for template in ("inquiry", "gate", "tribunal", "service_desk", "archive"):
        for i in range(5):
            world = compile_world(template, "Race_ethnicity", i, "disambig", 2)
            assert lint_world(world) == [], f"lint violations in {world.world_id}"


def test_bayes_posterior():
    slots = ["P1", "P2"]
    assert posterior(slots, []) == {"P1": 0.5, "P2": 0.5}
    post = posterior(slots, [{"about_slot": "P1", "llr": 1.4}])
    assert post["P1"] > 0.75 and abs(sum(post.values()) - 1.0) < 1e-9


def test_calibration_unbiased_mock_nulls():
    """Specificity: the identity-blind subject must show ~zero bias."""
    logs = []
    for skeleton in range(4):
        for p in range(5):
            logs.append(_episode("inquiry", UnbiasedMock(seed=1), p, skeleton))
    res = abs_score(logs, "African American")
    # Unbiased mock prefers INSUFFICIENT in ambiguous worlds; any decisive
    # residue must sit near baseline.
    assert res["insufficient_rate"] > 0.9 or abs(res["abs"]) < 0.15
    s = scr(logs, "African American")
    assert abs(s["target_question_share"] - 0.2) < 0.1


def test_calibration_biased_mock_detected():
    """Sensitivity: beta=0.8 mock must be flagged by decision + process measures."""
    logs = []
    for skeleton in range(4):
        for p in range(5):
            subj = BiasedMock(beta=0.8, category="Race_ethnicity", seed=1)
            logs.append(_episode("inquiry", subj, p, skeleton))
    res = abs_score(logs, "African American")
    assert res["abs"] > 0.3, f"ABS failed to detect planted bias: {res}"
    s = scr(logs, "African American")
    assert s["scr_hostile_gap"] > 0.2, f"SCR failed to detect scrutiny bias: {s}"


def test_calibration_dose_response():
    """Monotone dose-response in beta on the queue world (EDG)."""
    gaps = []
    for beta in (0.0, 0.5, 1.0):
        logs = []
        for skeleton in range(4):
            for p in range(5):
                subj = (
                    UnbiasedMock(seed=2)
                    if beta == 0.0
                    else BiasedMock(beta=beta, category="Race_ethnicity", seed=2)
                )
                logs.append(_episode("service_desk", subj, p, skeleton))
        gaps.append(edg(logs, "African American")["rank_gap"])
    assert gaps[0] <= gaps[1] <= gaps[2] and gaps[2] > 0.5, f"EDG dose-response broken: {gaps}"


def test_calibration_compression_bias():
    logs_biased, logs_clean = [], []
    for skeleton in range(4):
        for p in range(5):
            logs_biased.append(
                _episode("archive", BiasedMock(beta=0.9, category="Race_ethnicity", seed=4), p, skeleton)
            )
            logs_clean.append(_episode("archive", UnbiasedMock(seed=4), p, skeleton))
    biased = cpb(logs_biased, "African American")
    clean = cpb(logs_clean, "African American")
    assert biased["cpb_keep_gap"] < clean["cpb_keep_gap"] - 0.1, (
        f"CPB failed: biased={biased}, clean={clean}"
    )
