#!/usr/bin/env python3
"""Rank measured CUDA configurations by warm speed and estimated time to result."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(report):
    groups = {}
    for record in report["records"]:
        if record["status"] == "ok":
            groups.setdefault(record["configuration"]["name"], []).append(record)
    rows = []
    for name, records in groups.items():
        samples = [s for r in records for s in r["samples_s"]]
        elapsed = statistics.median(samples)
        preparation = statistics.median(r["cold_preparation_s"] for r in records)
        first_extra = statistics.median(
            max(0, r["first_run_s"] - r["median_run_s"]) for r in records
        )
        rows.append(
            dict(
                name=name,
                configuration=records[0]["configuration"],
                actual_schedule=records[0].get("decision", {}).get("selected"),
                decision=records[0].get("decision", {}),
                rounds=len(records),
                execution_s=elapsed,
                seconds_per_step=elapsed / report["case"]["steps"],
                cold_preparation_s=preparation,
                first_execution_extra_s=first_extra,
                cold_time_to_result_s=preparation + elapsed + first_extra,
                min_execution_s=min(samples),
                max_execution_s=max(samples),
                fresh_process=all(r.get("fresh_process", False) for r in records),
            )
        )
    rows.sort(key=lambda r: r["execution_s"])
    return dict(case=report["case"], device=report["device"], configurations=rows)


def rank(summary, steps, *, cached=False, expected_runs=1):
    """Estimate within tested configurations; preparation changes need validation."""
    values = []
    for row in summary["configurations"]:
        score = expected_runs * steps * row["seconds_per_step"] + (
            0 if cached else row["cold_preparation_s"] + row["first_execution_extra_s"]
        )
        values.append(
            dict(
                seconds=score,
                configuration=row["configuration"],
                fresh_process=row["fresh_process"],
                measured_steps=summary["case"]["steps"],
                extrapolated=steps != summary["case"]["steps"],
            )
        )
    return sorted(values, key=lambda r: r["seconds"])


def recommend(summary, steps, *, cached=False, expected_runs=1):
    values = rank(summary, steps, cached=cached, expected_runs=expected_runs)
    return (
        (values[0]["seconds"], values[0]["configuration"]["name"])
        if values
        else (None, None)
    )


def load_reports(directories, *, fresh_only=False, cases=None):
    """Merge complementary sweeps only when workload and implementation agree."""
    merged = {}
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            report = json.loads(path.read_text())
            if not report.get("records") or (
                cases and report["case"]["name"] not in cases
            ):
                continue
            key = json.dumps(
                [
                    report["case"],
                    report["device"],
                    report.get("jax"),
                    report.get("extension_sha256"),
                    report.get("planner_sha256"),
                    report.get("workload_sha256"),
                    {
                        k: report.get("hardware", {}).get(k)
                        for k in ("uuid", "driver", "power_limit")
                    }
                    if report.get("hardware")
                    else None,
                ],
                sort_keys=True,
            )
            records = [
                r
                for r in report["records"]
                if not fresh_only or r.get("fresh_process", False)
            ]
            if not records:
                continue
            item = merged.setdefault(key, {**report, "records": [], "sources": []})
            item["records"].extend(records)
            item["sources"].append(str(path))
    return list(merged.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--fresh-only", action="store_true")
    parser.add_argument("--steps", type=int, help="Rank for a requested horizon")
    parser.add_argument("--expected-runs", type=int, default=1)
    parser.add_argument("--cached", action="store_true")
    args = parser.parse_args()
    if args.expected_runs < 1 or (args.steps is not None and args.steps < 1):
        parser.error("steps and expected-runs must be positive")
    studies = []
    for report in load_reports(
        args.directories, fresh_only=args.fresh_only, cases=args.cases
    ):
        item = summarize(report)
        item["sources"] = report["sources"]
        item["extension_sha256"] = report.get("extension_sha256")
        item["planner_sha256"] = report.get("planner_sha256")
        item["workload_sha256"] = report.get("workload_sha256")
        item["jax"] = report.get("jax")
        item["predictions"] = [
            dict(
                steps=n,
                cold=dict(
                    zip(
                        ("seconds", "configuration"),
                        recommend(item, n),
                        strict=True,
                    )
                ),
                cached=dict(
                    zip(
                        ("seconds", "configuration"),
                        recommend(item, n, cached=True),
                        strict=True,
                    )
                ),
            )
            for n in (16, 64, 256, 1024, 4096, 16384)
        ]
        item["requested_ranking"] = rank(
            item,
            args.steps or item["case"]["steps"],
            cached=args.cached,
            expected_runs=args.expected_runs,
        )
        if item["requested_ranking"]:
            best = item["requested_ranking"][0]["seconds"]
            item["within_three_percent"] = [
                r["configuration"]["name"]
                for r in item["requested_ranking"]
                if r["seconds"] <= best * 1.03
            ]
        studies.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                method="T = preparation + first-execution overhead + expected_runs * steps * execution_per_step. Cached mode excludes preparation and capture. Common material/state setup and Python import are excluded from ranking. Match the complete workload and hardware; extrapolations require fresh-process validation at the actual horizon. Shared sweeps are screening data, not unbiased cold-start measurements. Differences below 3% should be treated as ties, not proven wins.",
                requested_steps=args.steps,
                expected_runs=args.expected_runs,
                cached=args.cached,
                studies=studies,
            ),
            indent=2,
        )
        + "\n"
    )
    for study in studies:
        best = study["configurations"][0] if study["configurations"] else None
        print(
            study["case"]["name"],
            study["case"]["shape"],
            "warm:",
            best["name"] if best else None,
            "cold256:",
            recommend(study, 256)[1],
            "cold4096:",
            recommend(study, 4096)[1],
        )


if __name__ == "__main__":
    main()
