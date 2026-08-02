from __future__ import annotations

from dataclasses import replace

import pytest

from tests.performance.benchmark_schema import (
    SCHEMA_VERSION,
    BenchmarkRecord,
    compare_backend_speedup,
    compare_benchmarks,
)


@pytest.fixture
def baseline_record():
    return BenchmarkRecord(
        beamz_commit="abc123",
        beamz_version="0.4.3",
        python_version="3.11.14",
        jax_version="0.7.2",
        jaxlib_version="0.7.2",
        workload="realistic_3d",
        backend="jax",
        device="NVIDIA H100 80GB",
        device_count=1,
        precision="float32",
        grid_dimensions=(128, 256, 384),
        timesteps=500,
        boundaries=("CPML",),
        sources=("ModeSource",),
        monitors=("ModeMonitor",),
        trace_lower_s=1.2,
        compile_s=8.0,
        warm_runtime_samples_s=(2.0, 1.9, 2.1, 2.0, 2.2),
        warm_end_to_end_samples_s=(2.1, 2.0, 2.2, 2.1, 2.3),
        peak_memory_bytes=4_000_000_000,
    )


def test_benchmark_record_contains_required_reproducibility_metadata(
    baseline_record,
):
    payload = baseline_record.as_dict()

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["material_cells"] == 128 * 256 * 384
    assert payload["updated_cells"] == 128 * 256 * 384 * 500
    assert payload["median_warm_runtime_s"] == pytest.approx(2.0)
    assert payload["median_warm_end_to_end_s"] == pytest.approx(2.1)
    assert payload["kernel_gcups"] == pytest.approx(3.145728)
    assert payload["end_to_end_gcups"] == pytest.approx(2.9959314285714285)
    assert payload["warm_runtime_samples_s"] == (
        2.0,
        1.9,
        2.1,
        2.0,
        2.2,
    )


def test_shared_hardware_comparison_reports_but_does_not_gate(baseline_record):
    current = replace(
        baseline_record,
        beamz_commit="def456",
        warm_runtime_samples_s=(2.2, 2.2, 2.2),
        warm_end_to_end_samples_s=(2.31, 2.31, 2.31),
    )

    comparison = compare_benchmarks(
        baseline_record,
        current,
        controlled_hardware=False,
    )

    assert comparison.warm_runtime_change == pytest.approx(0.10)
    assert comparison.passed is None


def test_controlled_hardware_enforces_runtime_memory_and_compile_limits(
    baseline_record,
):
    passing = replace(
        baseline_record,
        beamz_commit="pass",
        compile_s=8.7,
        warm_runtime_samples_s=(2.08, 2.08, 2.08),
        warm_end_to_end_samples_s=(2.184, 2.184, 2.184),
        peak_memory_bytes=4_160_000_000,
    )
    failing = replace(
        passing,
        beamz_commit="fail",
        warm_runtime_samples_s=(2.2, 2.2, 2.2),
        warm_end_to_end_samples_s=(2.31, 2.31, 2.31),
    )

    assert compare_benchmarks(baseline_record, passing, controlled_hardware=True).passed
    assert not compare_benchmarks(
        baseline_record, failing, controlled_hardware=True
    ).passed


def test_benchmark_record_requires_multiple_warm_samples(baseline_record):
    with pytest.raises(ValueError, match="three warm"):
        replace(baseline_record, warm_runtime_samples_s=(1.0, 1.1))


def test_benchmark_comparison_rejects_feature_or_backend_mismatch(baseline_record):
    with pytest.raises(ValueError, match="same workload"):
        compare_benchmarks(
            baseline_record,
            replace(baseline_record, backend="cuda_streamed"),
            controlled_hardware=True,
        )

    with pytest.raises(ValueError, match="same workload"):
        compare_benchmarks(
            baseline_record,
            replace(baseline_record, monitors=()),
            controlled_hardware=True,
        )


def test_backend_speedup_allows_implementation_change_on_same_hardware(
    baseline_record,
):
    candidate = replace(
        baseline_record,
        backend="cuda_streamed",
        warm_runtime_samples_s=(1.0, 1.0, 1.0),
        warm_end_to_end_samples_s=(1.4, 1.4, 1.4),
        compile_s=4.0,
        peak_memory_bytes=3_000_000_000,
    )

    speedup = compare_backend_speedup(baseline_record, candidate)

    assert speedup.kernel_speedup == pytest.approx(2.0)
    assert speedup.end_to_end_speedup == pytest.approx(1.5)
    assert speedup.compile_speedup == pytest.approx(2.0)
    assert speedup.memory_ratio == pytest.approx(0.75)
