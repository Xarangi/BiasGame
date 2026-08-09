# BiasGame

Counterfactual, trajectory-level measurement of social bias in LLM agents.

Static MCQ benchmarks (BBQ etc.) measure what a model *says* when it knows it is
being tested. BiasGame measures what an agent *does* over a whole task: whom it
scrutinizes, what evidence it demands, whose tickets it deprioritizes, whose
statements survive summarization — each against an exact counterfactual world
where only demographic identity is permuted.

## Core design

- **Frozen worlds** (`biasgame/worlds/`): scenario skeletons authored entirely in
  slot space (`[[P1]]..[[P5]]`) with zero demographic content, compiled
  deterministically from BBQ-seeded parameters. A leakage linter rejects any
  demographic token in world text.
- **Identity firewall** (`biasgame/identity/`): the only code that resolves slot
  tokens to identity surface forms (names / stated descriptors). Each world runs
  under a Latin square of identity-to-slot permutations; behavioral differences
  across permutations are causally attributable to identity alone.
  CI enforces the guarantee: under a neutral package, episode transcripts are
  byte-identical across permutations (`tests/test_firewall_leak.py`).
- **Bayesian yardstick** (`biasgame/scoring/bayes.py`): evidence items carry
  ground-truth log-likelihood ratios, so a normative posterior is computable at
  every turn; bias is deviation from a rational observer, not from a parity heuristic.
- **Seven worlds** (`biasgame/worlds/compiler.py`): `inquiry` (punitive),
  `gate` (distributive), `tribunal` (epistemic), `service_desk` (quality of
  service over a ticket queue), `archive` (representational: compression +
  recall), `studio` (delegation + review), `grievance_desk` (complainant
  role-flip: the identity-marked people report harm instead of being judged).
- **Trajectory measures** (`biasgame/scoring/measures.py`): CFR (counterfactual
  flip rate with a within-permutation stochasticity baseline), ABS (ambiguity
  bias), SCR (scrutiny/hostility allocation + judged tone), ETG
  (evidence-threshold gap), BUD (Bayesian update deviation), EDG (effort /
  queue / exception disparities), CPB (compression bias + misattribution),
  TIG (testimonial-injustice credence gap), DBI (delegation bias), SEV
  (continuous punishment severity), SURGERY (context-surgery decomposition),
  EIG (within-trajectory explicit/implicit gap).
- **Long-horizon machinery** (`biasgame/engine/horizon.py`, `engine/surgery.py`):
  identity-free filler doses produce bias-vs-horizon curves; checkpoint replay
  on identity-sanitized history decomposes bias into decision-time vs.
  accumulated-context components (`configs/study4_horizon.json`,
  `experiments/study4_horizon/REPORT.md`).

## Quick start

```bash
python3 -m pytest tests/            # includes the leak test and calibration suite
python3 -m biasgame.studies.run_study \
    --config configs/pilot_synthetic.json --out experiments/pilot_synthetic
```

The synthetic pilot (2,880 episodes) validates the instrument with known-bias
mock subjects: an identity-blind subject nulls on every measure; planted-bias
subjects (β = 0.3/0.6/0.9) produce monotone dose-responses on all channels.
See `experiments/pilot_synthetic/REPORT.md`.

To run a real model, replace a subject entry in the config with
`{"kind": "llm", "model": "<model-id>"}` and set `BIASGAME_API_KEY`
(and `BIASGAME_BASE_URL` for non-OpenAI endpoints).

## Repository layout

```
biasgame/           the instrument (worlds, identity, engine, agents, scoring, studies)
configs/            study configurations (the experiment matrix)
tests/              leak test, determinism, lint, Bayes, calibration
experiments/        run outputs: episode JSONL, manifest, report
datasets/, src/     legacy prototype (2024) kept for provenance
docs/               design documents
```

## Design documents

- Novelty assessment & literature review (~110 works)
- Experimental design proposal (design space, four worlds, hypotheses H1–H10)
- Implementation plan (phases, budgets, risk register)

`Race_ethnicity.jsonl` is the BBQ Race/ethnicity split (Parrish et al., 2022),
the seed taxonomy for world compilation.
