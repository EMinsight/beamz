"""Statistics and artifacts for the RTX 3090 CUDA capacity sweep.

The hardware runner lives in :mod:`scripts.benchmark_rtx3090_capacity`.  Keeping
the aggregation and rendering here makes the calculations unit-testable without
requiring a CUDA device.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from rtx3090_benchmark import TimingStatistics, summarize_timings
except ModuleNotFoundError:
    from scripts.rtx3090_benchmark import TimingStatistics, summarize_timings


SCHEMA_VERSION = "beamz.performance/rtx3090-capacity-v1"
MODAL_WORKLOAD = "modal_waveguide_cpml"
BARE_WORKLOAD = "bare_pec"


@dataclass(frozen=True, slots=True)
class CapacityMeasurement:
    """One warm, already-compiled capacity-sweep measurement."""

    workload: str
    resolution_nm: float
    grid_zyx: tuple[int, int, int]
    timesteps: int
    warmups: int
    warm_runtime_samples_s: tuple[float, ...]
    setup_s: float
    trace_lower_s: float
    executable_compile_s: float
    source_spec_count: int
    peak_bytes_in_use: int
    peak_pool_bytes: int
    live_bytes_in_use: int
    process_memory_bytes: int
    allocator_limit_bytes: int

    def __post_init__(self) -> None:
        if self.workload not in {MODAL_WORKLOAD, BARE_WORKLOAD}:
            raise ValueError(f"unknown capacity workload {self.workload!r}")
        if not math.isfinite(self.resolution_nm) or self.resolution_nm <= 0.0:
            raise ValueError("resolution_nm must be positive and finite")
        if len(self.grid_zyx) != 3 or any(value <= 2 for value in self.grid_zyx):
            raise ValueError("grid_zyx must contain three sizes greater than two")
        if self.timesteps <= 0 or self.warmups <= 0:
            raise ValueError("timesteps and warmups must be positive")
        summarize_timings(self.warm_runtime_samples_s)
        durations = (self.setup_s, self.trace_lower_s, self.executable_compile_s)
        if any(not math.isfinite(value) or value < 0.0 for value in durations):
            raise ValueError("setup and compilation durations must be non-negative")
        byte_values = (
            self.peak_bytes_in_use,
            self.peak_pool_bytes,
            self.live_bytes_in_use,
            self.process_memory_bytes,
            self.allocator_limit_bytes,
        )
        if any(value < 0 for value in byte_values):
            raise ValueError("memory measurements must be non-negative")
        if self.allocator_limit_bytes <= 0:
            raise ValueError("allocator_limit_bytes must be positive")

    @property
    def cells(self) -> int:
        return math.prod(self.grid_zyx)

    @property
    def updated_cells(self) -> int:
        return self.cells * self.timesteps

    @property
    def timing(self) -> TimingStatistics:
        return summarize_timings(self.warm_runtime_samples_s)

    @property
    def median_gcups(self) -> float:
        return self.updated_cells / self.timing.median_s / 1e9

    @property
    def gcups_ci95(self) -> tuple[float, float]:
        timing = self.timing
        return (
            self.updated_cells / timing.median_ci95_high_s / 1e9,
            self.updated_cells / timing.median_ci95_low_s / 1e9,
        )

    @property
    def milliseconds_per_step(self) -> float:
        return self.timing.median_s * 1e3 / self.timesteps

    @property
    def peak_memory_gib(self) -> float:
        return self.peak_bytes_in_use / 2**30

    @property
    def peak_pool_gib(self) -> float:
        return self.peak_pool_bytes / 2**30

    @property
    def process_memory_gib(self) -> float:
        return self.process_memory_bytes / 2**30

    @property
    def allocator_utilization(self) -> float:
        return self.peak_pool_bytes / self.allocator_limit_bytes

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timing"] = asdict(self.timing)
        payload["timing"]["coefficient_of_variation"] = (
            self.timing.coefficient_of_variation
        )
        payload.update(
            {
                "cells": self.cells,
                "updated_cells": self.updated_cells,
                "median_gcups": self.median_gcups,
                "gcups_ci95": self.gcups_ci95,
                "milliseconds_per_step": self.milliseconds_per_step,
                "peak_memory_gib": self.peak_memory_gib,
                "peak_pool_gib": self.peak_pool_gib,
                "process_memory_gib": self.process_memory_gib,
                "allocator_utilization": self.allocator_utilization,
            }
        )
        return payload

    @classmethod
    def from_child_payload(cls, payload: dict[str, Any]) -> CapacityMeasurement:
        return cls(
            workload=str(payload["workload"]),
            resolution_nm=float(payload["resolution_nm"]),
            grid_zyx=tuple(int(value) for value in payload["grid_zyx"]),
            timesteps=int(payload["timesteps"]),
            warmups=int(payload["warmups"]),
            warm_runtime_samples_s=tuple(
                float(value) for value in payload["warm_runtime_samples_s"]
            ),
            setup_s=float(payload["setup_s"]),
            trace_lower_s=float(payload["trace_lower_s"]),
            executable_compile_s=float(payload["executable_compile_s"]),
            source_spec_count=int(payload["source_spec_count"]),
            peak_bytes_in_use=int(payload["peak_bytes_in_use"]),
            peak_pool_bytes=int(payload["peak_pool_bytes"]),
            live_bytes_in_use=int(payload["live_bytes_in_use"]),
            process_memory_bytes=int(payload["process_memory_bytes"]),
            allocator_limit_bytes=int(payload["allocator_limit_bytes"]),
        )


@dataclass(frozen=True, slots=True)
class CapacityFailure:
    """A failed child attempt, normally the upper GPU-memory bracket."""

    workload: str
    resolution_nm: float
    kind: str
    returncode: int
    detail: str


def _measurements_for(
    measurements: Iterable[CapacityMeasurement], workload: str
) -> tuple[CapacityMeasurement, ...]:
    return tuple(
        sorted(
            (item for item in measurements if item.workload == workload),
            key=lambda item: item.cells,
        )
    )


def _linear_fit(
    x_values: Iterable[float], y_values: Iterable[float]
) -> dict[str, float]:
    x = tuple(float(value) for value in x_values)
    y = tuple(float(value) for value in y_values)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("a linear fit needs at least three paired observations")
    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator <= 0.0:
        raise ValueError("linear-fit x values must not all be equal")
    slope = (
        sum(
            (x_value - x_mean) * (y_value - y_mean)
            for x_value, y_value in zip(x, y, strict=True)
        )
        / denominator
    )
    intercept = y_mean - slope * x_mean
    residual = sum(
        (y_value - (intercept + slope * x_value)) ** 2
        for x_value, y_value in zip(x, y, strict=True)
    )
    total = sum((value - y_mean) ** 2 for value in y)
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}


@dataclass(frozen=True, slots=True)
class CapacitySweep:
    """All successful and failed attempts from one controlled GPU sweep."""

    beamz_revision: str
    device: str
    driver_version: str | None
    cuda_version: str | None
    total_gpu_memory_bytes: int
    baseline_gpu_memory_bytes: int
    allocator_fraction: float
    timesteps: int
    samples: int
    warmups: int
    started_at: str
    completed_at: str
    measurements: tuple[CapacityMeasurement, ...]
    failures: tuple[CapacityFailure, ...] = ()

    def __post_init__(self) -> None:
        if not self.beamz_revision.strip() or not self.device.strip():
            raise ValueError("revision and device must be non-empty")
        if self.total_gpu_memory_bytes <= 0 or self.baseline_gpu_memory_bytes < 0:
            raise ValueError("GPU memory metadata is invalid")
        if not 0.0 < self.allocator_fraction <= 1.0:
            raise ValueError("allocator_fraction must lie in (0, 1]")
        if self.timesteps <= 0 or self.samples < 3 or self.warmups <= 0:
            raise ValueError("invalid benchmark repetition counts")
        if not self.measurements:
            raise ValueError("a capacity sweep needs at least one measurement")
        if any(item.timesteps != self.timesteps for item in self.measurements):
            raise ValueError("all points must use the sweep timestep count")
        if any(
            len(item.warm_runtime_samples_s) != self.samples
            for item in self.measurements
        ):
            raise ValueError("all points must use the sweep sample count")

    @property
    def modal(self) -> tuple[CapacityMeasurement, ...]:
        return _measurements_for(self.measurements, MODAL_WORKLOAD)

    @property
    def bare(self) -> tuple[CapacityMeasurement, ...]:
        return _measurements_for(self.measurements, BARE_WORKLOAD)

    @staticmethod
    def _workload_summary(points: tuple[CapacityMeasurement, ...]) -> dict[str, Any]:
        if not points:
            return {}
        best = max(points, key=lambda item: item.median_gcups)
        largest_cells = max(item.cells for item in points)
        saturated = tuple(item for item in points if item.cells >= largest_cells / 2.0)
        saturated_gcups = tuple(item.median_gcups for item in saturated)
        summary: dict[str, Any] = {
            "point_count": len(points),
            "best_median_gcups": best.median_gcups,
            "best_gcups_ci95": best.gcups_ci95,
            "best_resolution_nm": best.resolution_nm,
            "best_cells": best.cells,
            "saturated_gcups_median": statistics.median(saturated_gcups),
            "saturated_gcups_min": min(saturated_gcups),
            "saturated_gcups_max": max(saturated_gcups),
            "saturated_point_count": len(saturated),
            "largest_successful_cells": largest_cells,
        }
        if len(saturated) >= 3:
            fit = _linear_fit(
                (item.updated_cells for item in saturated),
                (item.timing.median_s for item in saturated),
            )
            summary["large_domain_runtime_fit"] = fit
            if fit["slope"] > 0.0:
                summary["fit_asymptotic_gcups"] = 1.0 / fit["slope"] / 1e9
        return summary

    @property
    def summary(self) -> dict[str, Any]:
        modal_summary = self._workload_summary(self.modal)
        bare_summary = self._workload_summary(self.bare)
        result: dict[str, Any] = {
            MODAL_WORKLOAD: modal_summary,
            BARE_WORKLOAD: bare_summary,
        }
        if self.modal:
            largest = max(self.modal, key=lambda item: item.cells)
            result["capacity"] = {
                "largest_successful_resolution_nm": largest.resolution_nm,
                "largest_successful_cells": largest.cells,
                "largest_successful_peak_memory_gib": largest.peak_memory_gib,
                "largest_successful_peak_pool_gib": largest.peak_pool_gib,
                "largest_successful_process_memory_gib": largest.process_memory_gib,
                "largest_successful_allocator_utilization": (
                    largest.allocator_utilization
                ),
                "first_gpu_oom_resolution_nm": next(
                    (
                        failure.resolution_nm
                        for failure in self.failures
                        if failure.kind == "gpu_oom"
                        and failure.workload == MODAL_WORKLOAD
                    ),
                    None,
                ),
                "shared_gpu_safety_stop_resolution_nm": next(
                    (
                        failure.resolution_nm
                        for failure in self.failures
                        if failure.kind == "shared_gpu_safety_stop"
                        and failure.workload == MODAL_WORKLOAD
                    ),
                    None,
                ),
            }
            memory_points = tuple(
                item for item in self.modal if item.cells >= largest.cells / 8.0
            )
            if len(memory_points) >= 3:
                memory_fit = _linear_fit(
                    (item.cells for item in memory_points),
                    (item.peak_bytes_in_use for item in memory_points),
                )
                result["capacity"]["active_memory_fit"] = memory_fit
                if memory_fit["slope"] > 0.0:
                    result["capacity"]["fitted_bytes_per_cell"] = memory_fit["slope"]
                    available = largest.allocator_limit_bytes - memory_fit["intercept"]
                    result["capacity"]["fitted_allocator_capacity_cells"] = max(
                        0, int(available / memory_fit["slope"])
                    )
                    physical_available = (
                        self.total_gpu_memory_bytes
                        - self.baseline_gpu_memory_bytes
                        - memory_fit["intercept"]
                    )
                    fitted_physical_cells = max(
                        0, int(physical_available / memory_fit["slope"])
                    )
                    result["capacity"]["fitted_shared_gpu_capacity_cells"] = (
                        fitted_physical_cells
                    )
                    if fitted_physical_cells > 0:
                        result["capacity"]["fitted_shared_gpu_resolution_nm"] = (
                            largest.resolution_nm
                            * (largest.cells / fitted_physical_cells) ** (1.0 / 3.0)
                        )
        if modal_summary and bare_summary:
            result["bare_to_modal_best_gcups_ratio"] = (
                bare_summary["best_median_gcups"] / modal_summary["best_median_gcups"]
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "beamz_revision": self.beamz_revision,
            "device": self.device,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "total_gpu_memory_bytes": self.total_gpu_memory_bytes,
            "baseline_gpu_memory_bytes": self.baseline_gpu_memory_bytes,
            "allocator_fraction": self.allocator_fraction,
            "timesteps": self.timesteps,
            "samples": self.samples,
            "warmups": self.warmups,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "measurement_definition": {
                "gcups": "prod(grid_zyx) * timesteps / median_warm_runtime_s / 1e9",
                "timing_boundary": (
                    "warm already-compiled full executable; includes all FDTD "
                    "updates, boundaries, sources, and configured monitors"
                ),
                "capacity_boundary": (
                    "fresh child processes with XLA preallocation disabled; first "
                    "observed GPU OOM is the upper bracket"
                ),
            },
            "summary": self.summary,
            "measurements": [item.as_dict() for item in self.measurements],
            "failures": [asdict(item) for item in self.failures],
        }


def _format_cells(cells: int) -> str:
    if cells >= 1_000_000_000:
        return f"{cells / 1e9:.2f}B"
    if cells >= 1_000_000:
        return f"{cells / 1e6:.2f}M"
    return f"{cells / 1e3:.1f}k"


def _csv_rows(sweep: CapacitySweep) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in sorted(
        sweep.measurements, key=lambda point: (point.cells, point.workload)
    ):
        timing = item.timing
        ci_low, ci_high = item.gcups_ci95
        rows.append(
            {
                "workload": item.workload,
                "resolution_nm": item.resolution_nm,
                "nz": item.grid_zyx[0],
                "ny": item.grid_zyx[1],
                "nx": item.grid_zyx[2],
                "domain_cells": item.cells,
                "timesteps": item.timesteps,
                "median_runtime_s": timing.median_s,
                "median_gcups": item.median_gcups,
                "gcups_ci95_low": ci_low,
                "gcups_ci95_high": ci_high,
                "runtime_cv": timing.coefficient_of_variation,
                "milliseconds_per_step": item.milliseconds_per_step,
                "peak_bytes_in_use": item.peak_bytes_in_use,
                "peak_pool_bytes": item.peak_pool_bytes,
                "process_memory_bytes": item.process_memory_bytes,
                "allocator_limit_bytes": item.allocator_limit_bytes,
                "allocator_utilization": item.allocator_utilization,
                "setup_s": item.setup_s,
                "trace_lower_s": item.trace_lower_s,
                "executable_compile_s": item.executable_compile_s,
                "source_spec_count": item.source_spec_count,
            }
        )
    return rows


def _write_csv(sweep: CapacitySweep, path: Path) -> None:
    rows = _csv_rows(sweep)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(sweep: CapacitySweep) -> str:
    summary = sweep.summary
    modal = summary[MODAL_WORKLOAD]
    bare = summary.get(BARE_WORKLOAD, {})
    capacity = summary["capacity"]
    lines = [
        "# RTX 3090 CUDA FDTD capacity sweep",
        "",
        f"BeamZ revision: `{sweep.beamz_revision}`  ",
        f"Device: `{sweep.device}` ({sweep.total_gpu_memory_bytes / 2**30:.2f} GiB)  ",
        (
            f"Protocol: `{sweep.timesteps}` steps, `{sweep.warmups}` warm-ups, "
            f"`{sweep.samples}` timed samples per point; fixed physical waveguide."
        ),
        "",
        "## Headline results",
        "",
        (
            f"- Realistic mode-source + six-face CPML peak: "
            f"**{modal['best_median_gcups']:.3f} GCUPS** at "
            f"{_format_cells(modal['best_cells'])} cells "
            f"(95% median CI {modal['best_gcups_ci95'][0]:.3f}–"
            f"{modal['best_gcups_ci95'][1]:.3f})."
        ),
        (
            f"- Large-domain realistic plateau: "
            f"**{modal['saturated_gcups_median']:.3f} GCUPS** median across "
            f"{modal['saturated_point_count']} points in the upper half of the "
            "successful cell range."
        ),
    ]
    if bare:
        lines.append(
            f"- Matched bare PEC update peak: **{bare['best_median_gcups']:.3f} "
            f"GCUPS**; {summary['bare_to_modal_best_gcups_ratio']:.2f}× the realistic "
            "peak."
        )
    lines.append(
        f"- Largest safely measured realistic domain: "
        f"**{_format_cells(capacity['largest_successful_cells'])} cells** at "
        f"{capacity['largest_successful_resolution_nm']:.3g} nm, using "
        f"{capacity['largest_successful_peak_memory_gib']:.2f} GiB active, "
        f"{capacity['largest_successful_peak_pool_gib']:.2f} GiB pooled, and "
        f"{capacity['largest_successful_process_memory_gib']:.2f} GiB of process "
        "VRAM."
    )
    if capacity["first_gpu_oom_resolution_nm"] is not None:
        lines.append(
            f"- First GPU OOM: **{capacity['first_gpu_oom_resolution_nm']:.3g} nm**, "
            "which supplies the upper capacity bracket."
        )
    elif capacity["shared_gpu_safety_stop_resolution_nm"] is not None:
        lines.append(
            f"- Shared-GPU safety stop: **"
            f"{capacity['shared_gpu_safety_stop_resolution_nm']:.3g} nm** was not "
            "re-run after the allocation transition closed the T3/Chromium GPU "
            "process twice. The active-memory fit projects "
            f"**{_format_cells(capacity['fitted_shared_gpu_capacity_cells'])} "
            "cells** on nominal free VRAM, but that projection is not a measured "
            "capacity point."
        )
    lines.extend(
        [
            "",
            "## Exact measurements",
            "",
            "| Workload | Resolution | Grid (z×y×x) | Cells | Median GCUPS | 95% CI | Peak active memory | CV |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in sorted(
        sweep.measurements, key=lambda point: (point.cells, point.workload)
    ):
        low, high = item.gcups_ci95
        lines.append(
            f"| {item.workload} | {item.resolution_nm:.3g} nm | "
            f"{item.grid_zyx[0]}×{item.grid_zyx[1]}×{item.grid_zyx[2]} | "
            f"{_format_cells(item.cells)} | {item.median_gcups:.3f} | "
            f"{low:.3f}–{high:.3f} | {item.peak_memory_gib:.2f} GiB | "
            f"{item.timing.coefficient_of_variation * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "GCUPS uses the conventional base-cell definition: "
            "`nz × ny × nx × timesteps / warm runtime / 1e9`. Setup, mode solving, "
            "lowering, and first compilation are excluded from warm throughput. The "
            "realistic curve includes heterogeneous materials, a solved 3D mode "
            "source, and PML on all six faces.",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_sweep(sweep: CapacitySweep, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {MODAL_WORKLOAD: "#1f77b4", BARE_WORKLOAD: "#d97706"}
    labels = {
        MODAL_WORKLOAD: "Modal source + CPML",
        BARE_WORKLOAD: "Bare PEC ceiling",
    }
    markers = {MODAL_WORKLOAD: "o", BARE_WORKLOAD: "s"}
    linestyles = {MODAL_WORKLOAD: "-", BARE_WORKLOAD: "--"}
    figure, axes = plt.subplots(2, 2, figsize=(14.2, 10.2), layout="constrained")

    for workload, points in (
        (MODAL_WORKLOAD, sweep.modal),
        (BARE_WORKLOAD, sweep.bare),
    ):
        if not points:
            continue
        cells = [item.cells for item in points]
        gcups = [item.median_gcups for item in points]
        ci = [item.gcups_ci95 for item in points]
        axes[0, 0].errorbar(
            cells,
            gcups,
            yerr=(
                [
                    value - interval[0]
                    for value, interval in zip(gcups, ci, strict=True)
                ],
                [
                    interval[1] - value
                    for value, interval in zip(gcups, ci, strict=True)
                ],
            ),
            color=colors[workload],
            marker=markers[workload],
            linestyle=linestyles[workload],
            capsize=3,
            linewidth=2,
            label=labels[workload],
        )
        axes[0, 1].plot(
            cells,
            [item.milliseconds_per_step for item in points],
            color=colors[workload],
            marker=markers[workload],
            linestyle=linestyles[workload],
            linewidth=2,
            label=labels[workload],
        )
        axes[1, 0].plot(
            cells,
            [item.peak_memory_gib for item in points],
            color=colors[workload],
            marker=markers[workload],
            linestyle=linestyles[workload],
            linewidth=2,
            label=labels[workload],
        )
        axes[1, 1].plot(
            cells,
            [item.timing.coefficient_of_variation * 100.0 for item in points],
            color=colors[workload],
            marker=markers[workload],
            linestyle=linestyles[workload],
            linewidth=2,
            label=labels[workload],
        )

    if sweep.modal:
        modal_cells = [item.cells for item in sweep.modal]
        axes[1, 0].plot(
            modal_cells,
            [item.peak_pool_gib for item in sweep.modal],
            color="#1f77b4",
            marker=".",
            linestyle=":",
            linewidth=1.7,
            label="Modal allocator pool",
        )
        axes[1, 0].plot(
            modal_cells,
            [item.process_memory_gib for item in sweep.modal],
            color="#4b5563",
            marker=".",
            linestyle="-.",
            linewidth=1.5,
            label="Modal process VRAM",
        )

    best = max(sweep.modal, key=lambda item: item.median_gcups)
    axes[0, 0].annotate(
        f"realistic peak {best.median_gcups:.2f} GCUPS",
        xy=(best.cells, best.median_gcups),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=9,
        color="#174a73",
    )
    allocator_limit_gib = (
        max(item.allocator_limit_bytes for item in sweep.measurements) / 2**30
    )
    axes[1, 0].axhline(
        allocator_limit_gib,
        color="#303030",
        linestyle=":",
        linewidth=1.4,
        label=f"JAX allocator limit ({allocator_limit_gib:.1f} GiB)",
    )
    total_memory_gib = sweep.total_gpu_memory_bytes / 2**30
    axes[1, 0].axhline(
        total_memory_gib,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label=f"Physical VRAM ({total_memory_gib:.1f} GiB)",
    )

    axes[0, 0].set_title("FDTD throughput by domain size")
    axes[0, 0].set_ylabel("GCUPS (median, 95% bootstrap CI)")
    axes[0, 1].set_title("Warm time per full timestep")
    axes[0, 1].set_ylabel("milliseconds per timestep")
    axes[1, 0].set_title("Active, pooled, and process GPU memory")
    axes[1, 0].set_ylabel("GiB")
    axes[1, 1].set_title("Warm-run timing variability")
    axes[1, 1].set_ylabel("coefficient of variation (%)")
    for axis in axes.flat:
        axis.set_xscale("log")
        axis.set_xlabel("domain cells (log scale)")
        axis.grid(alpha=0.2, which="both")
        axis.legend(loc="best")
    axes[0, 1].set_yscale("log")
    figure.suptitle(
        f"BeamZ 3D CUDA capacity sweep on {sweep.device}\n"
        f"{sweep.timesteps} steps × {sweep.samples} warm samples; fixed physical waveguide",
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _report_rows(
    sweep: CapacitySweep,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_resolution: dict[float, dict[str, CapacityMeasurement]] = {}
    for item in sweep.measurements:
        by_resolution.setdefault(item.resolution_nm, {})[item.workload] = item
    curve_rows: list[dict[str, Any]] = []
    for resolution_nm, workloads in sorted(by_resolution.items(), reverse=True):
        modal = workloads.get(MODAL_WORKLOAD)
        bare = workloads.get(BARE_WORKLOAD)
        reference = modal or bare
        if reference is None:
            continue
        row: dict[str, Any] = {
            "resolution_nm": resolution_nm,
            "domain_cells": reference.cells,
            "domain_cells_m": reference.cells / 1e6,
            "log10_domain_cells": math.log10(reference.cells),
            "grid": "×".join(str(value) for value in reference.grid_zyx),
        }
        for prefix, item in (("modal", modal), ("bare", bare)):
            if item is None:
                continue
            low, high = item.gcups_ci95
            row.update(
                {
                    f"{prefix}_gcups": item.median_gcups,
                    f"{prefix}_gcups_ci_low": low,
                    f"{prefix}_gcups_ci_high": high,
                    f"{prefix}_ms_per_step": item.milliseconds_per_step,
                    f"{prefix}_peak_memory_gib": item.peak_memory_gib,
                    f"{prefix}_peak_pool_gib": item.peak_pool_gib,
                    f"{prefix}_process_memory_gib": item.process_memory_gib,
                    f"{prefix}_allocator_utilization": item.allocator_utilization,
                    f"{prefix}_cv": item.timing.coefficient_of_variation,
                }
            )
        curve_rows.append(row)
    detail_rows = _csv_rows(sweep)
    return curve_rows, detail_rows


def _report_artifact(sweep: CapacitySweep) -> dict[str, Any]:
    summary = sweep.summary
    modal = summary[MODAL_WORKLOAD]
    bare = summary.get(BARE_WORKLOAD, {})
    capacity = summary["capacity"]
    curve_rows, detail_rows = _report_rows(sweep)
    workload_labels = {
        "modal": "Modal source + CPML",
        "bare": "Bare PEC ceiling",
    }
    chart_rows: list[dict[str, Any]] = []
    for row in curve_rows:
        for prefix, label in workload_labels.items():
            if f"{prefix}_gcups" not in row:
                continue
            chart_rows.append(
                {
                    "workload": label,
                    "resolution_nm": row["resolution_nm"],
                    "domain_cells": row["domain_cells"],
                    "domain_cells_m": row["domain_cells_m"],
                    "log10_domain_cells": row["log10_domain_cells"],
                    "grid": row["grid"],
                    "gcups": row[f"{prefix}_gcups"],
                    "gcups_ci_low": row[f"{prefix}_gcups_ci_low"],
                    "gcups_ci_high": row[f"{prefix}_gcups_ci_high"],
                    "peak_memory_gib": row[f"{prefix}_peak_memory_gib"],
                    "peak_pool_gib": row[f"{prefix}_peak_pool_gib"],
                    "process_memory_gib": row[f"{prefix}_process_memory_gib"],
                    "allocator_utilization": row[f"{prefix}_allocator_utilization"],
                    "runtime_cv": row[f"{prefix}_cv"],
                }
            )
    modal_chart_rows = [
        row for row in chart_rows if row["workload"] == workload_labels["modal"]
    ]
    modal_memory_rows = [
        {
            **{
                key: row[key]
                for key in (
                    "resolution_nm",
                    "domain_cells",
                    "log10_domain_cells",
                    "grid",
                )
            },
            "memory_kind": kind,
            "memory_gib": row[field],
        }
        for row in modal_chart_rows
        for kind, field in (
            ("Active arrays", "peak_memory_gib"),
            ("JAX allocator pool", "peak_pool_gib"),
            ("Process VRAM", "process_memory_gib"),
        )
    ]
    generated_at = sweep.completed_at
    source_id = "rtx3090_capacity_sweep"
    title = "RTX 3090 CUDA FDTD capacity and throughput"
    if capacity["first_gpu_oom_resolution_nm"] is not None:
        capacity_text = (
            f" The first GPU OOM occurred at "
            f"{capacity['first_gpu_oom_resolution_nm']:.3g} nm."
        )
    elif capacity["shared_gpu_safety_stop_resolution_nm"] is not None:
        capacity_text = (
            " The next allocation transition destabilized the shared T3/Chromium "
            "GPU process twice, so the measured capacity is a safe lower bound; "
            f"the active-memory fit projects about "
            f"{_format_cells(capacity['fitted_shared_gpu_capacity_cells'])} cells "
            "against nominal free VRAM."
        )
    else:
        capacity_text = (
            " The sweep ended before observing a GPU OOM, so capacity is "
            "lower-bounded only."
        )
    cards = [
        {
            "id": "modal_peak",
            "description": "Best median warm throughput in the realistic waveguide sweep.",
            "dataset": "summary",
            "sourceId": source_id,
            "metrics": [
                {
                    "label": "Realistic peak GCUPS",
                    "field": "modal_peak_gcups",
                    "format": "number",
                }
            ],
        },
        {
            "id": "modal_plateau",
            "description": "Median throughput across the upper half of successful domain sizes.",
            "dataset": "summary",
            "sourceId": source_id,
            "metrics": [
                {
                    "label": "Large-domain GCUPS",
                    "field": "modal_plateau_gcups",
                    "format": "number",
                }
            ],
        },
        {
            "id": "bare_peak",
            "description": "Best median throughput of the matched source-free PEC update ceiling.",
            "dataset": "summary",
            "sourceId": source_id,
            "metrics": [
                {
                    "label": "Bare update peak GCUPS",
                    "field": "bare_peak_gcups",
                    "format": "number",
                }
            ],
        },
        {
            "id": "capacity",
            "description": "Largest realistic domain measured without destabilizing the shared display GPU.",
            "dataset": "summary",
            "sourceId": source_id,
            "metrics": [
                {
                    "label": "Largest successful cells, M",
                    "field": "largest_cells_m",
                    "format": "number",
                },
                {
                    "label": "Allocator peak",
                    "field": "capacity_utilization",
                    "format": "percent",
                },
            ],
        },
    ]
    charts = [
        {
            "id": "throughput_curve",
            "title": "FDTD throughput by domain cells",
            "subtitle": "Median warm GCUPS; x-axis is logarithmic through log10(cell count).",
            "showDescription": True,
            "intent": "trend",
            "question": "Where does throughput saturate as the regular grid grows?",
            "rationale": "An ordered line exposes launch-limited, saturation, and capacity regimes.",
            "type": "line",
            "dataset": "chart_curves",
            "sourceId": source_id,
            "encodings": {
                "x": {
                    "field": "log10_domain_cells",
                    "type": "quantitative",
                    "label": "log10(domain cells)",
                },
                "y": {
                    "field": "gcups",
                    "type": "quantitative",
                    "label": "GCUPS",
                },
                "color": {
                    "field": "workload",
                    "type": "nominal",
                    "label": "Workload",
                },
                "lineStyle": {
                    "field": "workload",
                    "type": "nominal",
                    "label": "Workload",
                },
                "tooltip": [
                    {
                        "field": "domain_cells",
                        "type": "quantitative",
                        "label": "Domain cells",
                    },
                    {
                        "field": "resolution_nm",
                        "type": "quantitative",
                        "label": "Resolution",
                        "unit": "nm",
                    },
                    {"field": "grid", "type": "text", "label": "Grid (z×y×x)"},
                    {
                        "field": "gcups_ci_low",
                        "type": "quantitative",
                        "label": "95% CI low",
                    },
                    {
                        "field": "gcups_ci_high",
                        "type": "quantitative",
                        "label": "95% CI high",
                    },
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "spec"},
            "layout": "full",
            "surface": {"surface": "card", "viewMode": "both"},
        },
        {
            "id": "memory_curve",
            "title": "Realistic active, pooled, and process GPU memory",
            "subtitle": "The allocator pool grows in bins and can materially exceed live FDTD arrays.",
            "showDescription": True,
            "intent": "trend",
            "question": "How quickly does each workload consume the available GPU memory?",
            "rationale": "The ordered memory curve reveals fixed overhead and bytes-per-cell scaling.",
            "type": "line",
            "dataset": "modal_memory_curves",
            "sourceId": source_id,
            "encodings": {
                "x": {
                    "field": "log10_domain_cells",
                    "type": "quantitative",
                    "label": "log10(domain cells)",
                },
                "y": {
                    "field": "memory_gib",
                    "type": "quantitative",
                    "label": "GPU memory",
                    "unit": "GiB",
                },
                "color": {
                    "field": "memory_kind",
                    "type": "nominal",
                    "label": "Memory counter",
                },
                "lineStyle": {
                    "field": "memory_kind",
                    "type": "nominal",
                    "label": "Memory counter",
                },
                "tooltip": [
                    {
                        "field": "domain_cells",
                        "type": "quantitative",
                        "label": "Domain cells",
                    },
                    {
                        "field": "resolution_nm",
                        "type": "quantitative",
                        "label": "Resolution",
                        "unit": "nm",
                    },
                    {"field": "grid", "type": "text", "label": "Grid (z×y×x)"},
                ],
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "spec"},
            "referenceLines": [
                {
                    "axis": "y",
                    "value": sweep.total_gpu_memory_bytes / 2**30,
                    "label": "Physical VRAM",
                    "color": "neutral",
                    "lineStyle": "dashed",
                },
                {
                    "axis": "y",
                    "value": max(
                        item.allocator_limit_bytes for item in sweep.measurements
                    )
                    / 2**30,
                    "label": "Configured allocator limit",
                    "color": "neutral",
                    "lineStyle": "dotted",
                },
            ],
            "layout": "full",
            "surface": {"surface": "card", "viewMode": "both"},
        },
        {
            "id": "memory_throughput_relationship",
            "title": "Realistic throughput by allocator utilization",
            "subtitle": "One point per spatial resolution; utilization is peak pool bytes divided by the configured allocator limit.",
            "showDescription": True,
            "intent": "relationship",
            "question": "Does the realistic solver retain throughput as memory fills?",
            "rationale": "A scatter separates memory pressure from domain-size ordering.",
            "type": "scatter",
            "dataset": "modal_curves",
            "sourceId": source_id,
            "encodings": {
                "x": {
                    "field": "allocator_utilization",
                    "type": "quantitative",
                    "label": "Allocator utilization",
                    "format": "percent",
                },
                "y": {
                    "field": "gcups",
                    "type": "quantitative",
                    "label": "Median throughput",
                    "unit": "GCUPS",
                },
                "size": {
                    "field": "domain_cells",
                    "type": "quantitative",
                    "label": "Domain cells",
                },
                "tooltip": [
                    {
                        "field": "resolution_nm",
                        "type": "quantitative",
                        "label": "Resolution",
                        "unit": "nm",
                    },
                    {
                        "field": "domain_cells",
                        "type": "quantitative",
                        "label": "Domain cells",
                    },
                    {
                        "field": "runtime_cv",
                        "type": "quantitative",
                        "label": "Runtime CV",
                        "format": "percent",
                    },
                ],
            },
            "layout": "full",
            "palette": {"kind": "sequential"},
            "surface": {"surface": "card", "viewMode": "both"},
        },
    ]
    tables = [
        {
            "id": "measurement_detail",
            "title": "Capacity-sweep measurements",
            "subtitle": "Exact warm medians, uncertainty, memory, and stability for every successful child process.",
            "showDescription": True,
            "dataset": "detail",
            "sourceId": source_id,
            "defaultSort": {"field": "domain_cells", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "workload", "label": "Workload", "type": "text"},
                {
                    "field": "resolution_nm",
                    "label": "Resolution (nm)",
                    "format": "number",
                },
                {"field": "domain_cells", "label": "Domain cells", "format": "compact"},
                {"field": "median_gcups", "label": "Median GCUPS", "format": "number"},
                {"field": "gcups_ci95_low", "label": "95% CI low", "format": "number"},
                {
                    "field": "gcups_ci95_high",
                    "label": "95% CI high",
                    "format": "number",
                },
                {
                    "field": "peak_bytes_in_use",
                    "label": "Peak active bytes",
                    "format": "compact",
                },
                {"field": "runtime_cv", "label": "Runtime CV", "format": "percent"},
            ],
        }
    ]
    source = {
        "id": source_id,
        "label": "RTX 3090 isolated-process CUDA capacity sweep",
        "path": "rtx3090-capacity.sql",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": (
                "SELECT * FROM read_csv_auto('rtx3090-capacity.csv', header = true);"
            ),
            "description": "Builds the fixed-geometry waveguide at increasing regular-grid resolution in fresh GPU processes.",
            "executed_at": generated_at,
            "filters": [
                f"device={sweep.device}",
                f"timesteps={sweep.timesteps}",
                f"samples={sweep.samples}",
                f"allocator_fraction={sweep.allocator_fraction}",
            ],
            "metric_definitions": [
                "GCUPS = nz * ny * nx * full timesteps / median warm executable seconds / 1e9.",
                "Peak active memory is JAX peak_bytes_in_use; allocator utilization uses peak_pool_bytes / bytes_limit.",
            ],
            "tables_used": ["rtx3090-capacity.json", "rtx3090-capacity.csv"],
        },
    }
    summary_row = {
        "modal_peak_gcups": modal["best_median_gcups"],
        "modal_plateau_gcups": modal["saturated_gcups_median"],
        "bare_peak_gcups": bare.get("best_median_gcups", 0.0),
        "largest_cells_m": capacity["largest_successful_cells"] / 1e6,
        "capacity_utilization": capacity["largest_successful_allocator_utilization"],
    }
    technical_summary = (
        "## Technical summary\n\n"
        f"The realistic mode-source waveguide reached **{modal['best_median_gcups']:.3f} "
        f"GCUPS** (95% median CI {modal['best_gcups_ci95'][0]:.3f}–"
        f"{modal['best_gcups_ci95'][1]:.3f}) and sustained a "
        f"**{modal['saturated_gcups_median']:.3f} GCUPS** median across the upper "
        f"half of successful domain sizes. The matched bare PEC path peaked at "
        f"**{bare.get('best_median_gcups', 0.0):.3f} GCUPS**. The largest realistic "
        f"success was **{_format_cells(capacity['largest_successful_cells'])} cells** "
        f"at {capacity['largest_successful_resolution_nm']:.3g} nm.{capacity_text}"
    )
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": technical_summary,
            "sourceId": source_id,
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in cards],
        },
        {
            "id": "throughput_finding",
            "type": "markdown",
            "sourceId": source_id,
            "body": (
                "## Throughput reaches a large-domain plateau\n\n"
                "The realistic curve includes mode injection, heterogeneous material loads, and six-face CPML. Read the solid blue curve as achieved end-to-end update throughput and the dashed orange curve as the matched source-free CUDA ceiling. The gap is the cost of realistic boundary and source work, not host setup."
            ),
        },
        {
            "id": "throughput_chart",
            "type": "chart",
            "chartId": "throughput_curve",
            "layout": "full",
        },
        {
            "id": "capacity_finding",
            "type": "markdown",
            "sourceId": source_id,
            "body": (
                "## Memory growth defines the usable domain limit\n\n"
                f"The largest completed realistic point held {_format_cells(capacity['largest_successful_cells'])} cells with "
                f"{capacity['largest_successful_peak_memory_gib']:.2f} GiB active, "
                f"a {capacity['largest_successful_peak_pool_gib']:.2f} GiB JAX pool, and "
                f"{capacity['largest_successful_process_memory_gib']:.2f} GiB of process VRAM. "
                "The stepwise pool is why a nominally feasible next array footprint can still destabilize another GPU client on a shared display card."
            ),
        },
        {
            "id": "memory_chart",
            "type": "chart",
            "chartId": "memory_curve",
            "layout": "full",
        },
        {
            "id": "pressure_finding",
            "type": "markdown",
            "sourceId": source_id,
            "body": (
                "## Memory pressure does not by itself define peak speed\n\n"
                "Each point below is one realistic resolution. Bubble size represents domain cells, so the plot shows whether GCUPS continues improving, plateaus, or degrades as the allocator fills. The result should be interpreted with the bootstrap intervals and runtime CV in the exact-results table."
            ),
        },
        {
            "id": "pressure_chart",
            "type": "chart",
            "chartId": "memory_throughput_relationship",
            "layout": "full",
        },
        {
            "id": "exact_results_intro",
            "type": "markdown",
            "body": "## Exact points and uncertainty\n\nThe table preserves the measurements behind the curves. Median confidence intervals use deterministic percentile bootstrap resampling of the nine warm launches; CV is the sample standard deviation divided by the mean.",
            "sourceId": source_id,
        },
        {
            "id": "exact_results",
            "type": "table",
            "tableId": "measurement_detail",
            "layout": "full",
        },
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": (
                "## Scope and metric definitions\n\n"
                "**Realistic workload.** A fixed 6.5 × 6.5 × 4.0 µm silicon-on-insulator waveguide, one solved 3D TM mode source, heterogeneous permittivity, and PML on all six faces. Only the uniform regular-grid cell size changes.\n\n"
                "**Bare ceiling.** The identical base grid shape with uniform material, no source, and PEC boundaries. It estimates the best custom-CUDA field-update rate, not application throughput.\n\n"
                "**GCUPS.** `nz × ny × nx × full timesteps ÷ median warm seconds ÷ 1e9`. One conventional cell update means one complete Yee timestep over one base-grid cell."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": source_id,
            "body": (
                "## Isolated-process methodology\n\n"
                f"Each point ran in a fresh process with XLA preallocation disabled and a {sweep.allocator_fraction:.0%} allocator limit. "
                f"After mode solving, lowering, and compilation, {sweep.warmups} launches primed clocks and allocations; the next {sweep.samples} full {sweep.timesteps}-step launches were synchronized and timed. "
                "Rasterization, mode solving, lowering, compilation, and result decoding are reported separately and excluded from GCUPS. The sweep stopped at the shared-desktop safety boundary rather than forcing another allocation transition."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations and robustness checks\n\n"
                "This is a single-card, single-session measurement, so driver, temperature, display load, and clock policy can move absolute values. Bootstrap intervals quantify repeat timing noise, not machine-to-machine uncertainty. The capacity projection comes from a linear active-memory fit and does not model allocator-bin growth or fragmentation; it is not a measured maximum. The allocator counters cover JAX-managed live and pool bytes; process memory from `nvidia-smi` is retained as a cross-check. The bare ceiling deliberately omits source and absorbing-boundary physics and must not be presented as realistic application GCUPS."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended next steps\n\n"
                "1. Re-run the same resolution list under locked application clocks and a quiescent display GPU to establish a controlled reference envelope.\n"
                "2. Profile the largest stable realistic point to attribute the gap between realistic and bare GCUPS to CPML, material loads, and planar source residuals.\n"
                "3. Add this sweep as an opt-in hardware artifact, but gate regressions on two or three representative plateau points instead of deliberately OOMing in CI."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "Would a monitor-bearing modal workload change the plateau materially? How much capacity is recovered by compact CPML face storage? Does a layout-aligned domain outperform the physically exact raster shape enough to justify padding?"
            ),
        },
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "A technical capacity and throughput sweep for BeamZ's 3D CUDA FDTD backend.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [source],
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready"
            if capacity["first_gpu_oom_resolution_nm"] is not None
            else "partial",
            "datasets": {
                "summary": [summary_row],
                "curves": curve_rows,
                "chart_curves": chart_rows,
                "modal_curves": modal_chart_rows,
                "modal_memory_curves": modal_memory_rows,
                "detail": detail_rows,
            },
            **(
                {}
                if capacity["first_gpu_oom_resolution_nm"] is not None
                else {
                    "accessIssues": [
                        {
                            "id": "capacity_not_bounded",
                            "dataset": "curves",
                            "message": "Capacity is a safe shared-desktop lower bound; the next allocator transition closed the T3/Chromium GPU process twice, so no destructive full-VRAM attempt was repeated.",
                        }
                    ]
                }
            ),
        },
        "sources": [source],
    }


def _chart_map() -> dict[str, Any]:
    return {
        "throughput_curve": {
            "section": "Throughput reaches a large-domain plateau",
            "question": "Where does throughput saturate as the grid grows?",
            "family": "trend",
            "type": "line",
            "fields": ["log10_domain_cells", "modal_gcups", "bare_gcups"],
            "claim": "Separates realistic achieved GCUPS from the bare CUDA ceiling.",
            "palette": "blue solid versus orange dashed",
        },
        "memory_curve": {
            "section": "Memory growth defines the usable domain limit",
            "question": "How quickly does each workload consume GPU memory?",
            "family": "trend",
            "type": "line",
            "fields": [
                "log10_domain_cells",
                "modal_peak_memory_gib",
                "bare_peak_memory_gib",
            ],
            "claim": "Shows capacity scaling and workload footprint differences.",
            "palette": "blue solid versus orange dashed",
        },
        "memory_throughput_relationship": {
            "section": "Memory pressure does not by itself define peak speed",
            "question": "Does throughput hold as allocator utilization increases?",
            "family": "relationship",
            "type": "scatter",
            "fields": [
                "modal_allocator_utilization",
                "modal_gcups",
                "domain_cells",
            ],
            "claim": "Tests whether capacity pressure coincides with throughput loss.",
            "palette": "single blue root; bubble size encodes cells",
        },
    }


def write_capacity_artifacts(sweep: CapacitySweep, output_dir: Path) -> dict[str, Path]:
    """Write raw data, exact tables, a static preview, and report input."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "rtx3090-capacity.json",
        "csv": output_dir / "rtx3090-capacity.csv",
        "markdown": output_dir / "rtx3090-capacity.md",
        "graph": output_dir / "rtx3090-capacity.png",
        "artifact": output_dir / "artifact.json",
        "chart_map": output_dir / "chart-map.json",
        "sql": output_dir / "rtx3090-capacity.sql",
    }
    paths["json"].write_text(json.dumps(sweep.as_dict(), indent=2) + "\n")
    _write_csv(sweep, paths["csv"])
    paths["markdown"].write_text(_markdown_report(sweep))
    _plot_sweep(sweep, paths["graph"])
    paths["artifact"].write_text(json.dumps(_report_artifact(sweep), indent=2) + "\n")
    paths["chart_map"].write_text(json.dumps(_chart_map(), indent=2) + "\n")
    paths["sql"].write_text(
        "SELECT *\nFROM read_csv_auto('rtx3090-capacity.csv', header = true);\n"
    )
    return paths
