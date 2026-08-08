"""Minimal exact/Monte-Carlo inference for pilot reports.

For the synthetic pilot we test a binomial-style null: under identity
irrelevance, each decisive ambiguous decision picks the target-identity slot
with probability 1/k. The Monte-Carlo permutation test simulates that null with
a keyed RNG (deterministic reports).
"""

from __future__ import annotations

from statistics import mean

from biasgame.rng import keyed_rng


def permutation_test(
    picks: list[int], expected_rate: float = 0.2, iterations: int = 5000
) -> dict:
    observed = mean(picks)
    rng = keyed_rng("permtest", len(picks), expected_rate, iterations)
    n = len(picks)
    at_least = 0
    for _ in range(iterations):
        sim = sum(1 for _ in range(n) if rng.random() < expected_rate) / n
        if sim >= observed:
            at_least += 1
    return {
        "n": n,
        "observed_rate": observed,
        "null_rate": expected_rate,
        "p_one_sided": (at_least + 1) / (iterations + 1),
    }
