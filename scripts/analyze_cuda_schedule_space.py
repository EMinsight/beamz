#!/usr/bin/env python3
"""Rank measured CUDA configurations by warm speed and estimated time to result."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path


def summarize(report):
    groups = {}
    for record in report["records"]:
        if record["status"] == "ok" and not record.get(
            "exclude_from_current_recommendations"
        ):
            groups.setdefault(record["configuration"]["name"], []).append(record)
    rows = []
    for name, records in groups.items():
        samples = [s for r in records for s in r["samples_s"]]
        elapsed = statistics.median(samples)
        # A prepared baseline can hit the program cache while alternatives
        # repeat identical source/monitor setup. Exclude that common work.
        preparation = statistics.median(
            r.get(
                "schedule_preparation_s",
                r["cold_preparation_s"] - r.get("program_setup_s", 0),
            )
            for r in records
        )
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


def read_report(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def input_reports(inputs):
    """Read raw directories/files or the portable exported evidence packet."""
    for source in inputs:
        paths = (
            [source]
            if source.is_file()
            else sorted((*source.glob("*.json"), *source.glob("*.json.gz")))
        )
        for path in paths:
            data = read_report(path)
            if "evidence" in data:
                for item in data["evidence"]:
                    if not item["exclusion"]:
                        yield f"{path}:{item['source']}", item["report"]
            else:
                yield str(path), data


def measurement_identity(report, record=None, *, include_steps=True):
    """Common comparison identity for ranking, CSVs, tables and plots."""
    identity = {**report, **(record or {}).get("provenance", {})}
    case = dict(report["case"])
    if not include_steps:
        case.pop("steps", None)
    return dict(
        case=case,
        **{
            key: identity.get(key)
            for key in (
                "device",
                "jax",
                "extension_sha256",
                "planner_sha256",
                "implementation_sha256",
                "workload_sha256",
                "benchmark_sha256",
                "monitor_layout",
            )
        },
        hardware={
            key: identity.get("hardware", {}).get(key)
            for key in ("uuid", "driver", "power_limit")
        }
        if identity.get("hardware")
        else None,
    )


def partition_report(report, *, fresh_only=False):
    """Separate resumed workers before calculating any timing medians."""
    groups = {}
    for record in report["records"]:
        if record.get("exclude_from_current_recommendations") or (
            fresh_only and not record.get("fresh_process", False)
        ):
            continue
        key = json.dumps(measurement_identity(report, record), sort_keys=True)
        group = groups.setdefault(
            key, {**report, **record.get("provenance", {}), "records": []}
        )
        group["records"].append(record)
    return list(groups.values())


def load_reports(directories, *, fresh_only=False, cases=None):
    """Merge complementary sweeps only when workload and implementation agree."""
    merged = {}
    for path, report in input_reports(directories):
        if not report.get("records") or (cases and report["case"]["name"] not in cases):
            continue
        for group in partition_report(report, fresh_only=fresh_only):
            key = json.dumps(measurement_identity(group), sort_keys=True)
            item = merged.setdefault(key, {**group, "records": [], "sources": []})
            item["records"].extend(group["records"])
            if str(path) not in item["sources"]:
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
        item["comparison_identity"] = measurement_identity(report)
        item["sources"] = report["sources"]
        item["extension_sha256"] = report.get("extension_sha256")
        item["planner_sha256"] = report.get("planner_sha256")
        item["implementation_sha256"] = report.get("implementation_sha256")
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
                method="T = schedule preparation + first-execution overhead + expected_runs * steps * execution_per_step. Cached mode excludes preparation and capture. Common material/source/monitor/state setup and Python import are excluded from ranking. Repeated program setup after the prepared baseline is also excluded to avoid cache bias. Match the complete workload and hardware; extrapolations require fresh-process validation at the actual horizon. Shared sweeps are screening data, not unbiased cold-start measurements. Differences below 3% should be treated as ties, not proven wins.",
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
