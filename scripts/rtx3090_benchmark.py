"""Statistical reporting and plotting for the controlled RTX 3090 comparison.

The benchmark runner launches both revisions in fresh Python processes.  Keeping
the statistics and rendering here makes the report testable without a GPU and
prevents presentation code from affecting the timed region.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


def _validate_samples(samples: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in samples)
    if len(values) < 3:
        raise ValueError("at least three warm samples are required")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("warm samples must be positive finite durations")
    return values


def _percentile(sorted_values: tuple[float, ...], fraction: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must lie in [0, 1]")
    position = (len(sorted_values) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (
        position - low
    )


def _bootstrap_median_interval(
    values: tuple[float, ...], *, draws: int = 4_000, seed: int = 3090
) -> tuple[float, float]:
    """Return a deterministic percentile-bootstrap 95% median interval."""
    if draws < 100:
        raise ValueError("bootstrap draws must be at least 100")
    generator = random.Random(seed)
    count = len(values)
    medians = sorted(
        statistics.median(values[generator.randrange(count)] for _ in range(count))
        for _ in range(draws)
    )
    return _percentile(tuple(medians), 0.025), _percentile(tuple(medians), 0.975)


def _bootstrap_median_ratio_interval(
    numerator: tuple[float, ...],
    denominator: tuple[float, ...],
    *,
    draws: int = 4_000,
    seed: int = 3090,
) -> tuple[float, float]:
    """Return an independent-bootstrap 95% interval for a median ratio."""
    if draws < 100:
        raise ValueError("bootstrap draws must be at least 100")
    generator = random.Random(seed)
    numerator_count = len(numerator)
    denominator_count = len(denominator)
    ratios = sorted(
        statistics.median(
            numerator[generator.randrange(numerator_count)]
            for _ in range(numerator_count)
        )
        / statistics.median(
            denominator[generator.randrange(denominator_count)]
            for _ in range(denominator_count)
        )
        for _ in range(draws)
    )
    return _percentile(tuple(ratios), 0.025), _percentile(tuple(ratios), 0.975)


@dataclass(frozen=True, slots=True)
class TimingStatistics:
    """Descriptive statistics for independent, warm executable samples."""

    count: int
    minimum_s: float
    p25_s: float
    median_s: float
    mean_s: float
    p75_s: float
    maximum_s: float
    stdev_s: float
    median_ci95_low_s: float
    median_ci95_high_s: float

    @property
    def coefficient_of_variation(self) -> float:
        return self.stdev_s / self.mean_s if self.mean_s else 0.0


def summarize_timings(samples: Iterable[float]) -> TimingStatistics:
    """Summarize durations and attach a deterministic 95% bootstrap CI."""
    values = _validate_samples(samples)
    ordered = tuple(sorted(values))
    ci_low, ci_high = _bootstrap_median_interval(values)
    return TimingStatistics(
        count=len(values),
        minimum_s=ordered[0],
        p25_s=_percentile(ordered, 0.25),
        median_s=float(statistics.median(ordered)),
        mean_s=float(statistics.fmean(ordered)),
        p75_s=_percentile(ordered, 0.75),
        maximum_s=ordered[-1],
        stdev_s=float(statistics.stdev(ordered)),
        median_ci95_low_s=ci_low,
        median_ci95_high_s=ci_high,
    )


@dataclass(frozen=True, slots=True)
class BackendMeasurement:
    """One isolated revision/backend measurement of the same physical workload."""

    label: str
    revision: str
    backend: str
    device: str
    grid_zyx: tuple[int, int, int]
    timesteps: int
    trace_lower_s: float
    compile_s: float
    warm_runtime_samples_s: tuple[float, ...]
    driver_version: str | None = None
    cuda_version: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.label.strip()
            or not self.revision.strip()
            or not self.backend.strip()
        ):
            raise ValueError("measurement labels must be non-empty")
        if len(self.grid_zyx) != 3 or any(value <= 0 for value in self.grid_zyx):
            raise ValueError("grid_zyx must contain three positive values")
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if not math.isfinite(self.trace_lower_s) or self.trace_lower_s < 0.0:
            raise ValueError("trace_lower_s must be finite and non-negative")
        if not math.isfinite(self.compile_s) or self.compile_s <= 0.0:
            raise ValueError("compile_s must be positive and finite")
        _validate_samples(self.warm_runtime_samples_s)

    @property
    def timing(self) -> TimingStatistics:
        return summarize_timings(self.warm_runtime_samples_s)

    @property
    def updated_cells(self) -> int:
        return math.prod(self.grid_zyx) * self.timesteps

    @property
    def median_gcups(self) -> float:
        return self.updated_cells / self.timing.median_s / 1e9

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timing"] = asdict(self.timing)
        payload["timing"]["coefficient_of_variation"] = (
            self.timing.coefficient_of_variation
        )
        payload["updated_cells"] = self.updated_cells
        payload["median_gcups"] = self.median_gcups
        return payload


@dataclass(frozen=True, slots=True)
class RTX3090Comparison:
    """A comparable main-vs-PR result, with speedup defined as baseline/candidate."""

    baseline: BackendMeasurement
    cuda: BackendMeasurement

    def __post_init__(self) -> None:
        if self.baseline.grid_zyx != self.cuda.grid_zyx:
            raise ValueError("measurements must use the same grid")
        if self.baseline.timesteps != self.cuda.timesteps:
            raise ValueError("measurements must use the same timestep count")
        if self.baseline.device != self.cuda.device:
            raise ValueError("measurements must use the same device")

    @property
    def runtime_speedup(self) -> float:
        return self.baseline.timing.median_s / self.cuda.timing.median_s

    @property
    def gcups_speedup(self) -> float:
        return self.cuda.median_gcups / self.baseline.median_gcups

    @property
    def runtime_speedup_ci95(self) -> tuple[float, float]:
        return _bootstrap_median_ratio_interval(
            self.baseline.warm_runtime_samples_s,
            self.cuda.warm_runtime_samples_s,
        )

    @property
    def compile_speedup(self) -> float:
        return self.baseline.compile_s / self.cuda.compile_s

    @property
    def cuda_is_faster(self) -> bool:
        return self.runtime_speedup > 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "beamz.performance/rtx3090-v1",
            "baseline": self.baseline.as_dict(),
            "cuda": self.cuda.as_dict(),
            "runtime_speedup": self.runtime_speedup,
            "runtime_speedup_ci95": self.runtime_speedup_ci95,
            "gcups_speedup": self.gcups_speedup,
            "compile_speedup": self.compile_speedup,
            "cuda_is_faster": self.cuda_is_faster,
        }


def write_report_artifacts(
    comparison: RTX3090Comparison, output_dir: Path
) -> dict[str, Path]:
    """Write JSON, a compact Markdown table, and an annotated PNG comparison graph."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "rtx3090-cuda-comparison.json",
        "markdown": output_dir / "rtx3090-cuda-comparison.md",
        "graph": output_dir / "rtx3090-cuda-comparison.png",
    }
    paths["json"].write_text(json.dumps(comparison.as_dict(), indent=2) + "\n")
    paths["markdown"].write_text(_markdown_report(comparison))
    _plot_report(comparison, paths["graph"])
    return paths


def _markdown_report(comparison: RTX3090Comparison) -> str:
    baseline = comparison.baseline
    cuda = comparison.cuda
    baseline_timing = baseline.timing
    cuda_timing = cuda.timing
    speedup_low, speedup_high = comparison.runtime_speedup_ci95
    return "\n".join(
        (
            "# RTX 3090 custom CUDA FDTD benchmark",
            "",
            f"Device: `{baseline.device}`  ",
            f"Workload: `{baseline.grid_zyx}` cells × `{baseline.timesteps}` timesteps  ",
            "Timing boundary: already-compiled full FDTD executable; allocator warmups excluded.",
            "",
            "| Implementation | Median runtime | 95% median CI | Median GCUPS | Mean ± stdev | Compile |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {baseline.label} | {baseline_timing.median_s:.6f} s | "
                f"[{baseline_timing.median_ci95_low_s:.6f}, {baseline_timing.median_ci95_high_s:.6f}] s | "
                f"{baseline.median_gcups:.3f} | {baseline_timing.mean_s:.6f} ± {baseline_timing.stdev_s:.6f} s | "
                f"{baseline.compile_s:.3f} s |"
            ),
            (
                f"| {cuda.label} | {cuda_timing.median_s:.6f} s | "
                f"[{cuda_timing.median_ci95_low_s:.6f}, {cuda_timing.median_ci95_high_s:.6f}] s | "
                f"{cuda.median_gcups:.3f} | {cuda_timing.mean_s:.6f} ± {cuda_timing.stdev_s:.6f} s | "
                f"{cuda.compile_s:.3f} s |"
            ),
            "",
            (
                f"Custom CUDA speedup: **{comparison.runtime_speedup:.3f}×** "
                f"(95% bootstrap CI [{speedup_low:.3f}, {speedup_high:.3f}]×; "
                f"{'faster' if comparison.cuda_is_faster else 'slower'})"
            ),
            "",
        )
    )


def _plot_report(comparison: RTX3090Comparison, output_path: Path) -> None:
    # Import lazily so statistics remain usable in minimal test environments.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = comparison.baseline
    cuda = comparison.cuda
    measurements = (baseline, cuda)
    names = (baseline.label, cuda.label)
    colors = ("#7f8c8d", "#1f77b4")
    timing = tuple(measurement.timing for measurement in measurements)
    runtimes = [item.median_s for item in timing]
    runtime_errors = [
        [item.median_s - item.median_ci95_low_s for item in timing],
        [item.median_ci95_high_s - item.median_s for item in timing],
    ]
    gcups = [measurement.median_gcups for measurement in measurements]
    compile_times = [measurement.compile_s for measurement in measurements]

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout="constrained")
    positions = range(len(names))
    bars = axes[0].bar(
        positions,
        runtimes,
        color=colors,
        yerr=runtime_errors,
        capsize=5,
        error_kw={"ecolor": "#1d1d1d", "linewidth": 1.2},
    )
    axes[0].set_title("Warm executable runtime")
    axes[0].set_ylabel("seconds (median, 95% bootstrap CI)")
    throughput_bars = axes[1].bar(positions, gcups, color=colors)
    axes[1].set_title("FDTD throughput")
    axes[1].set_ylabel("GCUPS (median)")
    compile_bars = axes[2].bar(positions, compile_times, color=colors)
    axes[2].set_title("First executable compilation")
    axes[2].set_ylabel("seconds")
    for axis in axes:
        axis.set_xticks(tuple(positions), names, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, runtimes, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, value in zip(throughput_bars, gcups, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, value in zip(compile_bars, compile_times, strict=True):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[0].set_ylim(0.0, max(item.median_ci95_high_s for item in timing) * 1.16)
    axes[1].set_ylim(0.0, max(gcups) * 1.18)
    axes[2].set_ylim(0.0, max(compile_times) * 1.16)
    axes[1].text(
        0.5,
        0.97,
        "CUDA speedup: "
        f"{comparison.runtime_speedup:.2f}× "
        f"[{comparison.runtime_speedup_ci95[0]:.2f}, "
        f"{comparison.runtime_speedup_ci95[1]:.2f}]",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontweight="bold",
        color="#137333" if comparison.cuda_is_faster else "#b3261e",
    )
    figure.suptitle(
        f"BeamZ FDTD on {baseline.device} — {baseline.grid_zyx} × {baseline.timesteps}",
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
