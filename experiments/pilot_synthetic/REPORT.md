# Synthetic Pilot — Instrument Validation Report

**Study:** `pilot_synthetic_v1` · 2,880 episodes · 4 world templates × 6 skeletons ×
3 ambiguity/dose conditions × 5 permutations × 4 subjects × 2 seeds ·
category `Race_ethnicity`, target group `African American` (synthetic ground truth).

## Purpose

Validate the full pipeline (world compiler → leakage lint → identity firewall →
episode engines → measures → inference) with *known* bias in and measured bias out:
an identity-blind mock (specificity: everything must null) and identity-conditioned
mocks with planted bias strength β ∈ {0.3, 0.6, 0.9} (sensitivity: every measure
must recover a monotone dose-response).

## Results

| subject | ABS | insufficient rate | SCR hostile gap | EDG rank gap | EDG denial gap | CPB keep gap | CFR | p (ABS null) |
|---|---|---|---|---|---|---|---|---|
| unbiased_mock  | +0.000 | 0.50 | +0.000 | +0.00 | +0.00 | +0.000 | 0.000 | 0.56 |
| biased β=0.3   | +0.154 | 0.34 | +0.800 | +0.49 | +0.36 | −0.025 | 0.544 | 0.0010 |
| biased β=0.6   | +0.326 | 0.19 | +0.800 | +1.25 | +0.60 | −0.300 | 0.768 | 0.0002 |
| biased β=0.9   | +0.354 | 0.07 | +0.800 | +2.29 | +0.88 | −0.500 | 0.832 | 0.0002 |

ETG (β=0.9): picks the ground-truth suspect at 0.67 when that slot carries the
target identity vs. 0.17 otherwise, *invariant to evidence dose* — i.e., the
biased agent "convicts" the target with or without proof, exactly the
differential-evidence-threshold signature the measure was designed to expose.

## Conclusions

1. **Specificity:** the identity-blind subject nulls on all seven measures and the
   permutation test (p = 0.56).
2. **Sensitivity + dose-response:** ABS, CFR, EDG (rank, effort, denial), and CPB
   are monotone in planted β; all biased subjects are flagged at p ≤ 0.001.
3. **Process ≠ outcome:** SCR saturates by β=0.3 while ABS is still small —
   process measures catch weak bias that outcome measures dilute. This is the
   synthetic preview of hypothesis H2 (bias migrates into process).
4. **Leak test:** under the neutral identity package, transcripts are
   byte-identical across permutations for all five templates (CI-enforced) — the
   counterfactual guarantee holds by construction.

## What changes for real runs

Replace mock subjects with `{"kind": "llm", "model": "..."}` in the study config
and set `BIASGAME_API_KEY` / `BIASGAME_BASE_URL`. Nothing else changes: same
worlds, same firewall, same measures, same report.
