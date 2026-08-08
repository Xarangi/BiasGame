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


def build_subject(spec: dict, category: str, seed: int):
    kind = spec["kind"]
    if kind == "unbiased_mock":
        return UnbiasedMock(seed=seed)
    if kind == "biased_mock":
        return BiasedMock(beta=spec["beta"], category=category, seed=seed)
    if kind == "llm":
        return LLMSubject(model=spec["model"], temperature=spec.get("temperature", 0.7))
    raise ValueError(f"unknown subject kind {kind!r}")


def run_study(config: dict, out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episodes").mkdir(exist_ok=True)
    category = config["category"]
    target_group = config["target_group"]
    package = package_for_category(category)
    manifest = {
        "biasgame_version": __version__,
        "config": config,
        "episodes": [],
    }
    logs_by_subject: dict[str, list[dict]] = {}

    for template in config["templates"]:
        for skeleton_index in range(config["skeletons"]):
            for ambiguity, dose in config.get("ambiguity_doses", [["ambig", 0]]):
                world = compile_world(template, category, skeleton_index, ambiguity, dose)
                violations = lint_world(world)
                if violations:
                    raise SystemExit(
                        f"LINT FAILURE in {world.world_id}: {violations[:3]}"
                    )
                perms = latin_square(world.slot_ids(), package)[: config["permutations"]]
                for perm in perms:
                    for subj_spec in config["subjects"]:
                        for seed in range(config["seeds"]):
                            subject = build_subject(subj_spec, category, seed)
                            fw = Firewall(
                                package=package,
                                permutation=perm,
                                world_id=world.world_id,
                                marking=config.get("marking", "explicit"),
                            )
                            log = RUNNERS[template](
                                world, fw, subject, subject.name, seed
                            )
                            fname = (
                                f"{world.world_id}--{perm.permutation_id}--"
                                f"{subject.name}--s{seed}.jsonl"
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
            picks.append(int(meta["slot_group"].get(dec[0]["slot"]) == target_group))
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
