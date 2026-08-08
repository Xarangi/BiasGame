"""BiasGame: counterfactual, trajectory-level measurement of social bias in LLM agents.

Core ideas:
- Frozen worlds: scenario skeletons authored entirely in slot space (P1..Pk), with
  no demographic content; facts, evidence, and persona knowledge are fixed.
- Identity firewall: demographics enter only through a render layer that maps slot
  tokens to identity surface forms; all world logic stays identity-blind.
- Counterfactual permutation: each world runs under a Latin square of
  identity-to-slot assignments; behavioral differences across permutations are
  causally attributable to identity alone.
- Bayesian yardstick: evidence items carry log-likelihood ratios, so a normative
  posterior is computable at every turn and bias is deviation from it.
"""

__version__ = "0.1.0"

SLOT_IDS = ("P1", "P2", "P3", "P4", "P5")
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
