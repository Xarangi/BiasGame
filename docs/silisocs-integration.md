# SiliSocs integration notes (Tier 3 path)

The core instrument in `biasgame/` is dependency-free and runs standalone (Tiers
1–2: scripted and constrained personas). Tier 3 — embedding probe episodes into a
living multi-agent society — targets [SiliSocS](https://github.com/sandbox-social/silisocs)
(v0.4.0, MIT), verified by source review to provide the required substrate:

| Instrument requirement | SiliSocs mechanism |
|---|---|
| Deterministic paired replay | keyed RNG (`policies/_rng.py`, frozen seed contract), checkpoint round-trips |
| Scripted personas | `silisocs.agents.fixed.FixedAgent` (episode-keyed YAML action plans) |
| Belief elicitation | `evaluations/probes` (`NumericRatingProbe`, `ChoiceProbe`, questionnaire batching) |
| Condition × seed matrix | `studies/run_study` schema: `hypotheses → conditions → overrides`, multi-seed, Submitit/Slurm |
| Mid-run manipulation (ETG titration, H10 surgery) | declarative `interventions` (`inject_action`, `broadcast_observation`) + checkpoints |
| Firewall hook | all agent-facing text funnels through `GameMaster.make_observation(agent_name)` |

## Mapping

1. **Firewall as a GM component.** Wrap the observation builder: world content
   stays in slot space inside backend state; a render component applies the
   permutation's identity package to subject-facing payloads only, and inverse-
   parses subject actions. Must also wrap probe prompts (`form_question_for_agent`)
   and intervention broadcasts — one leak breaks the counterfactual guarantee, so
   port `tests/test_firewall_leak.py` (neutral package ⇒ byte-identical transcripts).
2. **Backends.** `service_desk` and `moderation` map onto the messaging and
   `reddit_like` backends; `inquiry/gate/tribunal` need a sequential
   `InterviewGame` `BackendApp` (the `SimultaneousRoundGame` referee is the
   nearest pattern but is simultaneous-move).
3. **Personas.** Compile `SlotPersona.facts` into `FixedAgent` plans (Tier 1) or
   temp-0 fact-sheet-bound agents with response caching (Tier 2).
4. **Measures.** SiliSocs' committed-event JSONL is scoreable offline; add an
   adapter mapping its event rows to `biasgame.scoring.measures.parse_jsonl`
   format (kinds: `question`, `evidence_revealed`, `probe_posterior`,
   `ticket_handled`, `statement_fate`, `recall_attribution`, `decision`).

## Cautions

- Pin to a fork/tag for any paper; treat the leak and replay tests as CI tripwires.
- Keep GM adjudication rule-based inside our backends: no unaudited LLM call in
  the loop.
- T3 persona simulators are free agents; interpret T3 results as
  predictive-validity evidence, not primary measurement.
