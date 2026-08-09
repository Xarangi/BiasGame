"""Study runner: config -> episode matrix -> JSONL logs -> measures report.

A study config declares worlds x permutations x subjects x seeds; the runner
executes every cell, writes one JSONL per episode plus a manifest, computes all
applicable measures per subject, and runs the permutation test on the headline
gaps. Usage:

    python -m biasgame.studies.run_study --config configs/pilot_synthetic.json --out runs/pilot
"""

from __future__ import annotations

import argparse
import json
import pathlib

from biasgame import __version__
from biasgame.agents.subjects import BiasedMock, LLMSubject, UnbiasedMock
from biasgame.engine.episode import RUNNERS
from biasgame.identity.firewall import Firewall
from biasgame.identity.packages import PACKAGES, package_for_category
from biasgame.identity.permutations import latin_square
from biasgame.scoring.measures import MEASURES, parse_jsonl
from biasgame.worlds.compiler import compile_world
from biasgame.worlds.lint import lint_world
from biasgame.analysis.stats import permutation_test


def build_subject(spec: dict, category: str, seed: int, target_group: str | None = None):
    kind = spec["kind"]
    if kind == "unbiased_mock":
        return UnbiasedMock(seed=seed)
    if kind == "biased_mock":
        return BiasedMock(
            beta=spec["beta"], category=category, seed=seed,
            target_group=target_group, gamma=spec.get("gamma", 0.0),
        )
    if kind == "llm":
        return LLMSubject(model=spec["model"], temperature=spec.get("temperature", 0.7))
    raise ValueError(f"unknown subject kind {kind!r}")


def run_study(config: dict, out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(exist_ok=True)
    category = config["category"]
    target_group = config.get("target_group")  # None => per-world BBQ targets
    package = package_for_category(category)
    manifest = {
        "biasgame_version": __version__,
        "config": config,
        "episodes": [],
    }
    logs_by_subject: dict[str, list[dict]] = {}

    # BBQ seeding: when a bbq_file is configured, each "skeleton" is one
    # documented stereotype probe from the benchmark (deduplicated by
    # question_index x target group), routed to templates matching its harm
    # domain. Without it, worlds compile from raw parameters.
    seeds = None
    if config.get("bbq_file"):
        from biasgame.worlds.bbq import load_seeds, templates_for

        seeds = load_seeds(config["bbq_file"], polarity=config.get("bbq_polarity", "neg"))
        # Keep only seeds whose target exists in the identity package.
        labels = {i.group_label for i in package.identities}
        seeds = [s for s in seeds if s.target_group in labels][: config["skeletons"]]

    def _world_specs():
        if seeds is None:
            for template in config["templates"]:
                for skeleton_index in range(config["skeletons"]):
                    yield template, skeleton_index, None
        else:
            for skeleton_index, seed_item in enumerate(seeds):
                routed = [t for t in templates_for(seed_item) if t in config["templates"]]
                for template in routed or config["templates"][:1]:
                    yield template, skeleton_index, seed_item

    for template, skeleton_index, bbq_seed in _world_specs():
        for ambiguity, dose in config.get("ambiguity_doses", [["ambig", 0]]):
            world = compile_world(
                template, category, skeleton_index, ambiguity, dose, bbq_seed=bbq_seed
            )
            violations = lint_world(world)
            if violations:
                raise SystemExit(
                    f"LINT FAILURE in {world.world_id}: {violations[:3]}"
                )
            perms = latin_square(world.slot_ids(), package)[: config["permutations"]]
            # Horizon manipulation only exists in interview-family runners;
            # other templates would silently duplicate episodes across horizons.
            horizons = (
                config.get("horizons", [0])
                if template in ("inquiry", "gate", "tribunal")
                else [0]
            )
            for perm in perms:
                for subj_spec in config["subjects"]:
                    for seed in range(config["seeds"]):
                        for horizon in horizons:
                            world_target = (
                                world.bbq_meta.get("target_group") or target_group
                            )
                            subject = build_subject(
                                subj_spec, category, seed, target_group=world_target
                            )
                            fw = Firewall(
                                package=package,
                                permutation=perm,
                                world_id=world.world_id,
                                marking=config.get("marking", "explicit"),
                            )
                            log = RUNNERS[template](
                                world, fw, subject, subject.name, seed,
                                phrase_questions=config.get("phrase_questions", False),
                                horizon_filler=horizon,
                                surgery=config.get("surgery", False),
                                explicit_probe=config.get("explicit_probe", False),
                            )
                            fname = (
                                f"{world.world_id}--{perm.permutation_id}--"
                                f"{subject.name}--s{seed}--h{horizon}.jsonl"
                            )
                            (out_dir / "episodes" / fname).write_text(log.to_jsonl())
                            manifest["episodes"].append(fname)
                            logs_by_subject.setdefault(subject.name, []).append(
                                parse_jsonl(log.to_jsonl())
                            )

    report: dict = {"per_subject": {}, "tests": {}}
    for subject_name, logs in logs_by_subject.items():
        results = {}
        for mname, fn in MEASURES.items():
            try:
                results[mname] = fn(logs, target_group)
            except Exception as exc:  # a measure not applicable to these templates
                results[mname] = {"error": str(exc)}
        report["per_subject"][subject_name] = results

    # Bias-horizon curves: the same measures stratified by horizon dose.
    if len(config.get("horizons", [0])) > 1:
        report["per_subject_horizon"] = {}
        for subject_name, logs in logs_by_subject.items():
            by_h: dict[int, list[dict]] = {}
            for log in logs:
                by_h.setdefault(log["meta"].get("horizon_filler", 0), []).append(log)
            for h, lg in sorted(by_h.items()):
                report["per_subject_horizon"][f"{subject_name}|h{h}"] = {
                    "abs": MEASURES["abs"](lg, target_group),
                    "surgery": MEASURES["surgery"](lg, target_group),
                    "eig": MEASURES["eig"](lg, target_group),
                }

    # Permutation test on ABS target-pick indicator: is the target-identity
    # pick rate higher than identity-shuffled chance, per subject?
    for subject_name, logs in logs_by_subject.items():
        picks = []
        for log in logs:
            meta = log["meta"]
            if meta["ambiguity"] != "ambig":
                continue
            dec = [e for e in log["events"] if e["kind"] == "decision"]
            if not dec or dec[0]["slot"] == "INSUFFICIENT_EVIDENCE":
                continue
            tg = meta.get("target_group") or target_group
            picks.append(int(meta["slot_group"].get(dec[0]["slot"]) == tg))
        if picks:
            report["tests"][subject_name] = permutation_test(picks, expected_rate=0.2)

    manifest["report"] = report
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    config = json.loads(pathlib.Path(args.config).read_text())
    report = run_study(config, pathlib.Path(args.out))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
