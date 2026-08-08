# BBQ-Seeded Pilot v2 — Full-Instrument Validation Report

**v2 elicitation upgrades** (rationale: bias magnitude tracks stereotype-content
match; continuous DVs carry the largest audit effect sizes; prior work never
tests the complainant role or intersections):
- **Stereotype-family incident routing**: FAMILY_RULES classify each BBQ seed
  (drugs/violence/theft/fraud/driving/disease/competence/communication/honesty/
  service) and the inquiry world draws its incident from the matching pool —
  a drugs stereotype gets a substances incident, not a misrouted shipment.
  Seed-level unclassified rate now < 35% (was 60%).
- **Grievance desk (complainant role-flip)**: the identity-marked people REPORT
  harm instead of being judged; measures epistemic justice toward complainants
  (uptake order, investigation effort, dismissal) — a channel absent from all
  prior interactive work.
- **Continuous severity DV**: decisive accusations also price consequences
  (0-10); with the pilot's biased mocks, ONLY target-identity parties are ever
  sentenced (mean 7.0/9.0 at beta=0.5/0.9) — the only_target_sentenced flag.
- **Nonneg polarity support**: positive-valence BBQ probes route to
  positive-allocation worlds (gate/studio/archive) to measure withheld-positive
  and benevolent bias.
- **Intersectional metadata**: every episode logs rendered name + perceived
  name gender per slot, enabling race x gender stratification (Wilson &
  Caliskan-style) with no extra runs.

**Study:** `pilot_bbq_v1` · 5,820 episodes · seeded from 24 documented
Race_ethnicity stereotype probes in BBQ (Parrish et al., 2022), deduplicated by
(question_index × stereotyped group) and routed by harm domain (crime items →
inquiry, honesty → tribunal, competence → gate/studio, service → service_desk,
general → all) · 6 world templates · ambig + disambig(dose 2) · 5 permutations ·
2 seeds · per-world analysis targets from BBQ `stereotyped_groups` metadata.

## What this validates beyond the first pilot

Every remaining scenario-development pass, live and tested:

1. **BBQ ingestion** — worlds now inherit their stereotype under test from the
   benchmark's documented metadata (each with BBQ's literature source string
   preserved in `bbq_meta.source`); nothing is invented.
2. **Tribunal restructure** — 5 conflicting testimonies with authored truth
   values (exactly 3 true / 2 false, shuffled), credence probes, and the TIG
   measure: identical content × rotated identity across the Latin square.
3. **Studio world** — delegation + review with authored, balanced output
   quality; DBI measures lead-assignment share and recheck-of-solid-work.
4. **Judge channel** — free-text question phrasing scored for hostility on
   de-identified text (lexicon judge in CI; LLM judge shipped for real runs).
5. **Dialect channel** — Hofmann-style matched-guise transforms behind
   `marking="dialect"` (mechanism tested; content flagged for human validation
   before real use).
6. **Expanded pools** — 12 places, 14 incidents, 15 tickets, 18 archive
   statements, 10 testimony pairs, 8 award types, 10 roles; all lint-clean.

## Results — final verification pass (target = per-world BBQ stereotyped group)

A full verification pass (v2.1) found and fixed: a number-parsing bug that
corrupted probe/severity values shaped like "8/10" or containing digit-bearing
names; missing decision-policy coverage in gate worlds (decisions there were
silently unbiased for all subjects); ETG pooling gate episodes that have no
discoverable evidence; and CFR lacking a stochasticity baseline. Post-fix,
double-run determinism verified byte-identical across all 4,620 episodes.

| subject | ABS | CFR | CFR_excess | within-perm | sev (target) | tone gap | TIG | DBI recheck | EDG denial | CPB | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| unbiased_mock | +0.000 | 0.000 | +0.000 | 0.000 | — | +0.000 | +0.000 | +0.000 | +0.00 | +0.000 | 0.53 |
| biased β=0.5 | +0.620 | 0.727 | +0.260 | 0.467 | 7.0 (only targets) | +0.496 | −0.150 | +0.495 | +0.52 | −0.378 | 0.0002 |
| biased β=0.9 | +0.786 | 0.981 | +0.812 | 0.169 | 9.0 (only targets) | +0.906 | −0.270 | +0.891 | +0.91 | −0.503 | 0.0002 |

ETG (β=0.9, inquiry-only): ground-truth suspect identified at **0.92 when that
slot carries the stereotyped identity vs. 0.00 otherwise** at identical
evidence dose. BUD (belief inflation on target): 0.000 / +0.121 / +0.220 —
monotone in β. The CFR decomposition matters for real models: cross-permutation
flips minus within-permutation (seed-to-seed) flips isolates
identity-attributable decision instability from sampling noise.

## Interpretation

- **Specificity** holds across all nine channels on the identity-blind subject.
- **Sensitivity + dose-response** holds on every channel, including the three
  new ones (TIG, DBI, tone).
- Channels differ in saturation (SCR saturates by β=0.5; TIG/DBI scale
  smoothly) — supporting per-channel bias *profiles* over a single score.

## Known limitations before real-model runs

- Dialect transforms need fluent-speaker validation (flagged in
  `identity/dialect.py`).
- Lexicon tone judge is a CI stand-in; real runs should use `LLMJudge` with a
  human-validation κ split.
- Surface-text variety is procedural; an LLM-backed compiler pass behind the
  same schema (and linter) is the recommended enrichment for frontier subjects.
