# BBQ-Seeded Pilot — Full-Instrument Validation Report

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

## Results (target = per-world BBQ stereotyped group)

| subject | ABS | SCR hostile | tone gap | TIG credence | DBI lead | DBI recheck | EDG rank | CPB | CFR | p |
|---|---|---|---|---|---|---|---|---|---|---|
| unbiased_mock | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.00 | +0.000 | 0.000 | 0.53 |
| biased β=0.5 | +0.413 | +0.800 | +0.498 | −0.150 | −0.106 | +0.497 | +0.46 | −0.380 | 0.689 | 0.0002 |
| biased β=0.9 | +0.517 | +0.800 | +0.909 | −0.270 | −0.185 | +0.904 | +2.20 | −0.507 | 0.741 | 0.0002 |

ETG (β=0.9): ground-truth suspect identified at **0.64 when that slot carries
the stereotyped identity vs. 0.04 otherwise** at identical evidence dose — the
differential-evidence-threshold signature.

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
