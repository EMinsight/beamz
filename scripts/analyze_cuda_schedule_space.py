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


def recommend(summary, steps, *, cached=False):
    """Estimate within tested configurations; preparation changes need validation."""
    values = []
    for row in summary["configurations"]:
        score = steps * row["seconds_per_step"] + (
            0 if cached else row["cold_preparation_s"] + row["first_execution_extra_s"]
        )
        values.append((score, row["name"]))
    return min(values) if values else (None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    studies = []
    for directory in args.directories:
        for path in sorted(directory.glob("*.json")):
            report = json.loads(path.read_text())
            if not report.get("records"):
                continue
            item = summarize(report)
            item["source"] = str(path)
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
            studies.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                method="Predictions extrapolate measured per-step execution and startup costs; verify finalists in fresh processes at actual horizons. Shared sweeps are screening data, not unbiased cold-start measurements.",
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
