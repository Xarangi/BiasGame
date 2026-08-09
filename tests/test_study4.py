"""Study 4 machinery: horizon fillers, context surgery, explicit/implicit gap.

Calibration logic mirrors the rest of the suite: the identity-blind mock must
null every new channel (specificity), the planted-bias mock must show the
designed effect (sensitivity), and the neutral-package leak invariant must
survive every new feature.
"""

import re

from biasgame.agents.subjects import BiasedMock, UnbiasedMock, group_of_text
from biasgame.engine.episode import RUNNERS, run_interview_episode
from biasgame.engine.horizon import FILLER_TASKS, filler_text
from biasgame.engine.surgery import sanitize
from biasgame.identity.firewall import Firewall
from biasgame.identity.packages import NEUTRAL, RACE_ETHNICITY
from biasgame.identity.permutations import latin_square
from biasgame.scoring.measures import eig, parse_jsonl, surgery_decomp
from biasgame.worlds.compiler import compile_world
from biasgame.worlds.lint import _DEMOGRAPHIC_TERMS

TARGET = "African American"


def _logs(subject_factory, horizon=0, n_perms=5, n_seeds=2, **runner_kw):
    world = compile_world("inquiry", "Race_ethnicity", 0, "ambig", 0)
    perms = latin_square(world.slot_ids(), RACE_ETHNICITY)[:n_perms]
    logs = []
    for perm in perms:
        for seed in range(n_seeds):
            fw = Firewall(package=RACE_ETHNICITY, permutation=perm, world_id=world.world_id)
            subject = subject_factory(seed)
            log = run_interview_episode(
                world, fw, subject, subject.name, seed,
                horizon_filler=horizon, **runner_kw,
            )
            logs.append(parse_jsonl(log.to_jsonl()))
    return logs


def test_fillers_are_identity_free():
    for task in FILLER_TASKS:
        lowered = task.lower()
        for term in _DEMOGRAPHIC_TERMS:
            assert not re.search(rf"\b{re.escape(term)}\b", lowered), (task, term)
        assert "[[" not in task
        assert group_of_text(task) is None, task
    # Deterministic per (world, index).
    assert filler_text("w1", 3) == filler_text("w1", 3)
    assert filler_text("w1", 3) != filler_text("w1", 4)


def test_sanitize_strips_all_identity_signal():
    world = compile_world("inquiry", "Race_ethnicity", 0, "ambig", 0)
    perm = latin_square(world.slot_ids(), RACE_ETHNICITY)[0]
    fw = Firewall(package=RACE_ETHNICITY, permutation=perm, world_id=world.world_id)
    text = fw.render(world.scenario) + " " + " and ".join(
        fw.name_of(s) for s in world.slot_ids()
    )
    assert group_of_text(text) is not None  # identity present before surgery
    clean = sanitize(text, fw, world.slot_ids())
    assert group_of_text(clean) is None, clean
    for s in world.slot_ids():
        assert fw.name_of(s) not in clean


def test_leak_invariance_with_study4_features():
    """Neutral package + fillers + surgery + explicit probe must still be
    byte-identical across permutations."""
    world = compile_world("inquiry", "Race_ethnicity", 0, "ambig", 0)
    perms = latin_square(world.slot_ids(), NEUTRAL)
    transcripts = []
    for perm in perms[:3]:
        fw = Firewall(package=NEUTRAL, permutation=perm, world_id=world.world_id)
        log = RUNNERS["inquiry"](
            world, fw, UnbiasedMock(seed=7), "unbiased_mock", 7,
            horizon_filler=5, surgery=True, explicit_probe=True,
        )
        rows = [r for r in log.to_jsonl().splitlines() if '"kind": "meta"' not in r]
        transcripts.append("\n".join(rows))
    assert transcripts[0] == transcripts[1] == transcripts[2]


def test_exact_filler_dose_even_on_early_stop():
    logs = _logs(lambda s: UnbiasedMock(seed=s), horizon=7, n_perms=3, n_seeds=2)
    for log in logs:
        fillers = [e for e in log["events"] if e["kind"] == "filler"]
        assert len(fillers) == 7
        assert log["meta"]["horizon_filler"] == 7


def test_surgery_specificity_unbiased_mock():
    logs = _logs(lambda s: UnbiasedMock(seed=s), surgery=True)
    out = surgery_decomp(logs, TARGET)
    assert out["episodes"] == len(logs)
    assert out["original_target_rate"] == 0.0
    assert out["sanitized_target_rate"] == 0.0
    assert out["flip_rate"] == 0.0


def test_surgery_flags_decision_time_bias():
    logs = _logs(
        lambda s: BiasedMock(beta=0.9, category="Race_ethnicity", seed=s),
        surgery=True,
    )
    out = surgery_decomp(logs, TARGET)
    # The planted bias is purely decision-time (identity cues at the prompt):
    # it must flip off under sanitized replay, attributing ~all of it to the
    # decision-time component and ~none to the accumulated context.
    assert out["original_target_rate"] > 0.5
    assert out["sanitized_target_rate"] == 0.0
    assert out["decision_time_component"] > 0.5
    assert out["accumulated_component_vs_chance"] <= 0.0


def test_eig_recovers_explicit_implicit_divergence():
    biased = _logs(
        lambda s: BiasedMock(beta=0.9, category="Race_ethnicity", seed=s),
        explicit_probe=True,
    )
    out = eig(biased, TARGET)
    assert out["explicit_probes"] == len(biased)
    assert out["explicit_admission_rate"] == 0.0  # denies, like real models
    assert out["implicit_abs"] > 0.3
    assert out["eig"] > 0.3

    blind = _logs(lambda s: UnbiasedMock(seed=s), explicit_probe=True)
    out0 = eig(blind, TARGET)
    assert out0["explicit_admission_rate"] == 0.0
    assert out0["implicit_abs"] == 0.0  # denial is truthful: no gap


def _target_rate(logs):
    """Unconditional target-pick rate: with the mock, every decisive decision
    is a target pick (the fallback prefers 'insufficient'), so the conditional
    rate saturates at 1.0 and cannot show a horizon curve."""
    picks = 0
    for log in logs:
        dec = [e for e in log["events"] if e["kind"] == "decision"]
        picks += int(log["meta"]["slot_group"].get(dec[0]["slot"]) == TARGET)
    return picks / len(logs)


def test_gamma_drift_produces_rising_horizon_curve():
    def biased(gamma):
        return lambda s: BiasedMock(
            beta=0.25, category="Race_ethnicity", seed=s, gamma=gamma
        )

    lo = _target_rate(_logs(biased(1.0), horizon=0, n_seeds=8))
    hi = _target_rate(_logs(biased(1.0), horizon=30, n_seeds=8))
    assert hi > lo + 0.2, (lo, hi)

    flat_lo = _target_rate(_logs(biased(0.0), horizon=0, n_seeds=8))
    flat_hi = _target_rate(_logs(biased(0.0), horizon=30, n_seeds=8))
    assert abs(flat_hi - flat_lo) < 0.2, (flat_lo, flat_hi)


def test_fork_preserves_policy_parameters():
    m = BiasedMock(beta=0.4, category="Race_ethnicity", seed=3,
                   target_group=TARGET, gamma=0.7)
    m.begin("some context")
    f = m.fork()
    assert (f.beta, f.gamma, f.target_group, f.seed) == (0.4, 0.7, TARGET, 3)
    assert f.context == []  # fresh history — that's the point
    assert UnbiasedMock(seed=5).fork().seed == 5
