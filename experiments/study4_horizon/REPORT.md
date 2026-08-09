# Study 4 — Long-Horizon Agentic Bias: Validation Report

**Study:** `study4_horizon_v1` · 1,080 episodes · BBQ-seeded inquiry worlds
(12 Race_ethnicity probes, ambiguous, explicit marking) · 5 permutations ·
2 seeds · horizon doses **0 / 10 / 30** identity-free filler interludes ·
context surgery on · explicit probe on. Double-run determinism: byte-identical
episodes and report.

This study validates the three Study-4 mechanisms from the experimental design
— the pieces that make BiasGame a *long-horizon* instrument rather than a
one-shot one:

1. **Bias–horizon curves** (H8). Episodes receive 0/10/30 deterministic,
   identity-free filler interludes (lint-clean, no slot tokens — enforced in
   `tests/test_study4.py`), so context length varies while information content
   is constant. Any bias-vs-horizon slope is attributable to context length.
2. **Context surgery** (checkpoint replay). At the decision point a forked
   subject replays the entire history with every identity surface form
   replaced by neutral labels, then decides again. Flips under sanitization =
   decision-time identity dependence; persistence = bias already baked into
   the accumulated information state. Only an interactive environment can
   express this causal probe.
3. **Within-trajectory explicit/implicit gap** (EIG). After acting, the
   subject is asked directly whether demographics influenced its judgments;
   the gap between self-report and behavior operationalizes the FAccT-2025
   explicit/implicit divergence inside a single trajectory.

## Calibration ground truth

`BiasedMock(beta, gamma)`: `gamma` adds context-length drift
(`beta_eff = min(1, beta·(1 + gamma·|context|/10))`) — a synthetic version of
FairMT-Bench-style within-trajectory accumulation. The planted bias is purely
decision-time (it reads identity cues at the prompt), and the mock answers the
explicit probe "NO" regardless of beta (the documented real-model pattern:
implicit bias behind explicit egalitarianism).

## Results

Unconditional target-pick rate by horizon dose (chance would put ~0.2 on the
target if deciding uniformly; the identity-blind policy declines instead):

| subject | h=0 | h=10 | h=30 | curve |
|---|---|---|---|---|
| unbiased_mock | 0.000 | 0.000 | 0.000 | flat at zero (specificity) |
| biased β=0.3, γ=0 | 0.292 | 0.283 | 0.250 | flat (no drift planted) |
| biased β=0.3, γ=1 | 0.625 | 0.942 | 1.000 | **rising** (drift recovered) |

Context-surgery decomposition (pooled): for both biased mocks the sanitized
target rate is **0.000** — 100% of the measured bias lands in the
`decision_time_component` (0.275 / 0.856) and none in the accumulated
component, exactly matching the planted mechanism. The unbiased mock shows
zero flips (surgery specificity).

Explicit/implicit gap: explicit admission rate **0.00** for every subject;
implicit ABS **+0.80** for both biased mocks → **EIG = +0.80** recovered,
while the unbiased mock's denial is truthful (implicit 0, no gap).

Permutation test on target picks: p = 0.0002 for both biased mocks.

## Reading the instrument for real models

- A rising horizon curve with γ-style drift is what FairMT-Bench predicts for
  real assistants; a flat curve falsifies accumulation for that model.
- Surgery separates two remediations that look identical from outcomes alone:
  decision-time bias suggests prompt-level identity masking would work;
  accumulated bias means the trajectory itself (evidence gathering, notes)
  is contaminated and masking at the end is too late.
- EIG > 0 on real models replicates Actions Speak Louder inside one
  trajectory, with the same episodes providing both measurements.

## Notes / limitations

- Surgery is restricted to non-dialect markings (dialect-styled speech cannot
  be inverted by string substitution) and replays the world-side history; the
  subject's own free-text outputs are not part of the replayed context.
- The conditional `target_pick_rate` saturates at 1.0 for the mock (its only
  decisive move is the biased one); `unconditional_target_rate` was added to
  `abs` for exactly this case and is the headline curve statistic above.
