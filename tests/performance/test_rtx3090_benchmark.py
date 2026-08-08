from __future__ import annotations

import json

import pytest

from scripts.rtx3090_benchmark import (
    BackendMeasurement,
    RTX3090Comparison,
    summarize_timings,
    write_report_artifacts,
)


def _measurement(label: str, samples: tuple[float, ...]) -> BackendMeasurement:
    return BackendMeasurement(
        label=label,
        revision="abc123",
        backend="jax" if label == "origin/main JAX/XLA" else "cuda_streamed",
        device="NVIDIA GeForce RTX 3090",
        grid_zyx=(32, 48, 64),
        timesteps=20,
        trace_lower_s=0.1,
        compile_s=1.2,
        warm_runtime_samples_s=samples,
        driver_version="610.43.03",
        cuda_version="12.9",
    )


def test_timing_summary_reports_distribution_and_deterministic_interval():
    summary = summarize_timings((1.0, 1.1, 1.2, 1.3, 1.4))

    assert summary.count == 5
    assert summary.minimum_s == pytest.approx(1.0)
    assert summary.p25_s == pytest.approx(1.1)
    assert summary.median_s == pytest.approx(1.2)
    assert summary.p75_s == pytest.approx(1.3)
    assert summary.maximum_s == pytest.approx(1.4)
    assert summary.median_ci95_low_s <= summary.median_s <= summary.median_ci95_high_s
    assert summary.coefficient_of_variation > 0.0


def test_measurement_rejects_insufficient_or_invalid_samples():
    with pytest.raises(ValueError, match="three warm"):
        _measurement("origin/main JAX/XLA", (1.0, 1.1))
    with pytest.raises(ValueError, match="positive finite"):
        _measurement("origin/main JAX/XLA", (1.0, 0.0, 1.1))


def test_report_writes_machine_readable_statistics_markdown_and_graph(tmp_path):
    report = RTX3090Comparison(
        _measurement("origin/main JAX/XLA", (2.0, 2.1, 2.2, 2.1, 2.0)),
        _measurement("PR CUDA streamed", (1.0, 1.1, 1.0, 1.1, 1.0)),
    )

    paths = write_report_artifacts(report, tmp_path)
    payload = json.loads(paths["json"].read_text())

    assert payload["schema_version"] == "beamz.performance/rtx3090-v1"
    assert payload["runtime_speedup"] == pytest.approx(2.1)
    assert payload["cuda_is_faster"]
    assert "Custom CUDA speedup: **2.100×**" in paths["markdown"].read_text()
    assert paths["graph"].read_bytes().startswith(b"\x89PNG")
