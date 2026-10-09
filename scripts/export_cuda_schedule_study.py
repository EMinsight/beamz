#!/usr/bin/env python3
"""Preserve CUDA schedule measurements and export tables and a standalone plot.

Use --input-root after measurement, or --archive to regenerate the derived files
from a committed evidence packet without a GPU. No benchmarks run here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from scripts.analyze_cuda_schedule_space import summarize

SCREEN = {"geometry", "survey", "orientation", "refined", "dense-screen"}
FRESH = {
    "large-isolated",
    "vectorized-fresh",
    "finalists256",
    "horizon16",
    "horizon1024",
    "horizon4096",
    "dense-fresh",
    "policy-anchors",
    "layout-anchors",
    "compact-layout-fresh",
    "postlayout-fresh",
}
IDENTITY = (
    "device",
    "jax",
    "hardware",
    "commit",
    "extension_sha256",
    "benchmark_sha256",
    "planner_sha256",
    "implementation_sha256",
    "monitor_layout",
    "workload_sha256",
)


def collect(root):
    evidence = []
    for folder in sorted(SCREEN | FRESH | {"early-fresh", "layout-history"}):
        for path in sorted((root / folder).glob("*.json")):
            report = json.loads(path.read_text())
            exclusion = None
            if report["case"]["name"] == "magnetic":
                exclusion = (
                    "Permeability-only pilot did not produce dense H coefficients."
                )
            if folder == "geometry" and report["case"]["name"] == "large":
                exclusion = "Incomplete shared pilot before material-sharing fix; memory failures."
            if folder == "early-fresh":
                exclusion = "Historical startup before vectorized planning; retained for comparison."
            if folder == "layout-history":
                exclusion = "Historical layout startup before compact monitor-block permutations."
            for record in report["records"]:
                # Recover exact worker identity from independently saved parts,
                # including sweeps made before aggregate provenance was added.
                label = f"{report['case']['name']}--{record['configuration']['name']}--{record['round']}"
                part = (
                    root
                    / folder
                    / "isolated-parts"
                    / f"{label}-{report['case']['steps']}.json"
                )
                if part.exists():
                    worker = json.loads(part.read_text())
                    record["provenance"] = {
                        key: worker[key] for key in IDENTITY if key in worker
                    }
                identity = {**report, **record.get("provenance", {})}
                if (
                    folder in FRESH
                    and tuple(record.get("storage_axes", (0, 1, 2))) != (0, 1, 2)
                    and identity.get("monitor_layout") != "component_blocks"
                ):
                    record["exclude_from_current_recommendations"] = (
                        "Historical rotated startup before compact monitor-block permutations."
                    )
            evidence.append(
                dict(
                    source=str(path.relative_to(root)),
                    phase="fresh" if folder in FRESH else "screening",
                    exclusion=exclusion,
                    report=report,
                )
            )
    return dict(
        scope="Finite hierarchical search on one RTX3090 at the recorded driver/power limit. This does not prove an optimum on every workload or GPU.",
        method="72 available combinations; 38 representative choices, additional rotated refinements, then fresh-process finalists and horizon checks. Shared sweeps establish warm screening, not cold preparation. Main sweep uses borrowed state. Owned-state and public API checks are separate. Differences below 3% are ties, not confidence bounds.",
        evidence=evidence,
        host_plan_comparison=json.loads((root / "plan-vectorization.json").read_text()),
        owned_checks=[
            json.loads(p.read_text()) for p in sorted((root / "owned").glob("*.json"))
        ],
    )


def rows(packet):
    result = []
    for item in packet["evidence"]:
        report = item["report"]
        if item["exclusion"] or item["phase"] != "fresh":
            continue
        summary = summarize(report)
        for row in summary["configurations"]:
            if not row["fresh_process"]:
                continue
            records = [
                r
                for r in report["records"]
                if r["status"] == "ok"
                and r["configuration"]["name"] == row["name"]
                and not r.get("exclude_from_current_recommendations")
            ]
            cells = math.prod(report["case"]["shape"])
            result.append(
                dict(
                    source=item["source"],
                    case=report["case"]["name"],
                    shape="x".join(map(str, report["case"]["shape"])),
                    steps=report["case"]["steps"],
                    configuration=row["name"],
                    actual_schedule=row["actual_schedule"],
                    ownership="borrowed",
                    rounds=row["rounds"],
                    preparation_s=row["cold_preparation_s"],
                    first_execution_s=statistics.median(
                        r["first_run_s"] for r in records
                    ),
                    warm_s=row["execution_s"],
                    warm_gcups=cells
                    * report["case"]["steps"]
                    / row["execution_s"]
                    / 1e9,
                    cold_after_shared_setup_s=statistics.median(
                        r.get(
                            "schedule_time_to_result_s",
                            r["cold_time_to_result_s"] - r.get("program_setup_s", 0),
                        )
                        for r in records
                    ),
                    raw_pipeline_first_result_s=statistics.median(
                        r["total_time_to_result_s"] for r in records
                    ),
                )
            )
    return result


def write_tables(measured, output):
    preferred = {
        "finalists256",
        "dense-fresh",
        "policy-anchors",
        "layout-anchors",
        "compact-layout-fresh",
        "postlayout-fresh",
    }
    bycase = {}
    # The larger anchor sets replace one-round finalists for the same choice.
    for row in sorted(measured, key=lambda r: (r["rounds"], r["source"])):
        if row["steps"] == 256 and row["source"].split("/")[0] in preferred:
            bycase.setdefault(row["case"], {})[row["configuration"]] = row
    lines = [
        "<!-- Generated by scripts/export_cuda_schedule_study.py. -->",
        "| Workload (z×y×x) | Best streamed GCUPS | Best real mstep GCUPS | Best cold choice at 256 steps | Cold seconds after shared setup |",
        "|---|---:|---:|---|---:|",
    ]
    for case, choices in sorted(bycase.items()):
        values = list(choices.values())
        streamed = [r for r in values if r["actual_schedule"] == "streamed"]
        mstep = [r for r in values if r["actual_schedule"] == "mstep"]
        fastest = min(values, key=lambda r: r["cold_after_shared_setup_s"])
        tied = [
            r
            for r in values
            if r["cold_after_shared_setup_s"]
            <= 1.03 * fastest["cold_after_shared_setup_s"]
        ]
        cold = next((r for r in tied if r["configuration"] == "streamed"), fastest)
        best_stream = max((r["warm_gcups"] for r in streamed), default=0)
        best_mstep = (
            f"{max(r['warm_gcups'] for r in mstep):.2f}"
            if mstep
            else (
                "fallback"
                if any(r["configuration"].startswith("mstep-") for r in values)
                else "not tested"
            )
        )
        lines.append(
            f"| {case} ({cold['shape']}) | {best_stream:.2f} | {best_mstep} | {cold['configuration']} | {cold['cold_after_shared_setup_s']:.3f} |"
        )
    lines += [
        "",
        "Cold choice retains ordinary streamed when it is within 3% of the fastest measured finalist. GCUPS uses physical cells. Warm figures compare the tested finalists; the evidence packet also contains the broader 64-step screening. A fallback is streamed execution despite requesting mstep.",
        "",
    ]
    (output / "cuda-schedule-tables.md").write_text("\n".join(lines))


def write_plot(measured, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, case, names in zip(
        axes,
        ("medium", "large", "narrow_x"),
        (
            ("streamed", "mstep-8x16-32"),
            ("streamed", "mstep-16x32-64"),
            ("streamed", "streamed-axes-120"),
        ),
        strict=True,
    ):
        for name in names:
            points = {}
            for row in sorted(measured, key=lambda r: r["rounds"]):
                if (
                    row["case"] == case
                    and row["configuration"] == name
                    and row["steps"] in (16, 256, 1024, 4096)
                    and row["source"].split("/")[0] != "vectorized-fresh"
                ):
                    points[row["steps"]] = row["cold_after_shared_setup_s"]
            ordered = sorted(points)
            if ordered:
                ax.plot(ordered, [points[n] for n in ordered], marker="o", label=name)
        ax.set(
            title=case,
            xlabel="Timesteps",
            ylabel="Seconds after shared setup",
            xscale="log",
            yscale="log",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Measured first-result time on RTX3090 (borrowed state)")
    fig.savefig(output / "cuda-schedule-time-to-result.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-root", type=Path)
    source.add_argument("--archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    packet = (
        collect(args.input_root)
        if args.input_root
        else json.loads(args.archive.read_text())
    )
    if args.validation:
        packet["validation"] = json.loads(args.validation.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    measured = rows(packet)
    packet["counts"] = dict(
        records=sum(len(item["report"]["records"]) for item in packet["evidence"]),
        included_workloads=len(
            {
                item["report"]["case"]["name"]
                for item in packet["evidence"]
                if not item["exclusion"]
            }
        ),
        fresh_summaries=len(measured),
        owned_checks=len(packet["owned_checks"]),
    )
    packet["fresh_summaries"] = measured
    (args.output_dir / "cuda-schedule-space.json").write_text(
        json.dumps(packet, indent=2) + "\n"
    )
    with (args.output_dir / "cuda-schedule-space.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(measured[0]) if measured else []
        )
        writer.writeheader()
        writer.writerows(measured)
    write_tables(measured, args.output_dir)
    write_plot(measured, args.output_dir)
    print(json.dumps(packet["counts"], indent=2))


if __name__ == "__main__":
    main()
