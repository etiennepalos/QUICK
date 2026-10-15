#!/usr/bin/env python3
"""Run a predeclared finite ddX discretization study using a retained runtime.

The JSON plan specifies the molecular case, model/grid points and pairwise
comparisons. Jobs run sequentially. A failed refinement gate remains a failed
gate; successful execution and selected-protocol qualification are reported
separately. This script does not change QUICK defaults or source arithmetic.
"""
# (C) Copyright 2026 QUICK contributors. Mozilla Public License, v. 2.0.
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback

import validate_solvation as reference


def write(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def compare(spec: dict, left: dict, right: dict) -> dict:
    energy_error = abs(right["energy"] - left["energy"])
    gradient_error = reference.difference(left["gradient_hartree_per_bohr"],
                                          right["gradient_hartree_per_bohr"])
    return {**spec, "energy_difference_hartree": right["energy"] - left["energy"],
            "absolute_energy_difference_hartree": energy_error,
            "maximum_gradient_difference_hartree_per_bohr": gradient_error,
            "energy_tolerance_hartree": 1.0e-5, "gradient_tolerance_hartree_per_bohr": 5.0e-5,
            "passed": energy_error <= 1.0e-5 and gradient_error <= 5.0e-5,
            "scope": "Finite pairwise refinement at the stated model, geometry and numerical controls"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--launcher", default="")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--electron-count-tolerance", type=float, default=1.0e-3)
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("timeout must be positive")
    args.driver = args.driver.resolve(strict=True)
    args.basis = args.basis.resolve(strict=True)
    args.plan = args.plan.resolve(strict=True)
    args.workdir = args.workdir.resolve()
    args.workdir.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    point_ids = [point["id"] for point in plan["points"]]
    if len(point_ids) != len(set(point_ids)):
        raise ValueError("Duplicate point IDs")
    if any(point["model"] not in ("COSMO", "PCM") for point in plan["points"]):
        raise ValueError("Unsupported continuum model")
    if any(spec[side] not in point_ids for spec in plan["comparisons"] for side in ("left", "right")):
        raise ValueError("Comparison references an unspecified point")
    env = dict(os.environ)
    env.update(QUICK_BASIS=str(args.basis), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    report = {"schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "plan": plan, "plan_sha256": reference.sha256(args.plan),
              "validator_sha256": reference.sha256(Path(__file__)),
              "shared_parser_sha256": reference.sha256(Path(reference.__file__)),
              "runtime_before": reference.runtime(args.driver, env),
              "environment": {key: env[key] for key in ("QUICK_BASIS", "OMP_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
              "runs": [], "checks": [], "comparisons": [], "exceptions": [],
              "passed": False, "operational_checks_passed": False,
              "limitations": ["Pairwise finite discretization evidence does not prove an infinite-order limit.",
                  "No physics conventions, screening, defaults or acceptance gates are changed.",
                  "Historical failed comparisons remain historical failures.",
                  "Timings are execution diagnostics, not a performance benchmark."]}
    write(args.workdir / "protocol-before-science.json", report)
    campaign = reference.Campaign(args, report, env)
    results = {}
    for point in plan["points"]:
        try:
            results[point["id"]] = campaign.run(point["id"], plan["case"], point["model"], "gradient",
                point["grid"], epsilon=plan.get("epsilon", 78.3553), field=plan.get("applied_field", False))
        except Exception:
            report["exceptions"].append({"point": point["id"], "traceback": traceback.format_exc()})
        for spec in plan["comparisons"]:
            if spec["left"] in results and spec["right"] in results:
                if not any(old["left"] == spec["left"] and old["right"] == spec["right"]
                           for old in report["comparisons"]):
                    report["comparisons"].append(compare(spec, results[spec["left"]], results[spec["right"]]))
        write(args.workdir / "partial-report.json", report)
    report["runtime_after"] = reference.runtime(args.driver, env)
    report["runtime_unchanged"] = report["runtime_before"] == report["runtime_after"]
    report["operational_checks_passed"] = (not report["exceptions"] and report["runtime_unchanged"]
        and bool(report["checks"]) and all(check["passed"] for check in report["checks"]))
    report["model_comparison_status"] = {
        model: {"complete": all(point["id"] in results for point in plan["points"] if point["model"] == model),
                "declared_pair_count": sum(spec.get("model") == model for spec in plan["comparisons"]),
                "all_declared_pairs_pass": (any(spec.get("model") == model for spec in plan["comparisons"]) and
                    all(check["passed"] for check in report["comparisons"] if check.get("model") == model))}
        for model in sorted({point["model"] for point in plan["points"]})}
    report["passed"] = (report["operational_checks_passed"] and bool(plan["comparisons"]) and
        len(report["comparisons"]) == len(plan["comparisons"]) and
        all(check["passed"] for check in report["comparisons"]))
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write(args.workdir / "report.json", report)
    print(json.dumps({"passed": report["passed"], "operational_checks_passed": report["operational_checks_passed"],
        "model_comparison_status": report["model_comparison_status"],
        "failed_pairs": [check for check in report["comparisons"] if not check["passed"]],
        "report": str(args.workdir / "report.json")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
