"""A measured schedule choice must preserve capability and memory contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from beamz.lattice import component_shapes
from beamz.simulation import _cuda_abi as abi
from beamz.simulation.backend import CUDA_DEFAULT_FLAGS
from beamz.simulation.compile import CompiledProgramKey, _preparation_key
from beamz.simulation.cuda import schedule


@pytest.mark.parametrize(
    "backend,choice,expected",
    [
        ("cuda", None, "auto"),
        ("auto", None, "auto"),
        ("cuda_streamed", None, "streamed"),
        ("cuda_mstep", None, "mstep"),
        ("cuda", "mstep", "mstep"),
        ("cuda", "streamed", "streamed"),
    ],
)
def test_schedule_request_and_compatibility_aliases(backend, choice, expected):
    assert schedule.normalize_schedule_request(backend, choice, 3) == (expected, 3)


@pytest.mark.parametrize(
    "backend,choice,runs",
    [
        ("cuda", "unknown", 1),
        ("cuda_streamed", "mstep", 1),
        ("cuda_mstep", "auto", 1),
        ("jax", "mstep", 1),
        ("cuda", "auto", 0),
        ("cuda", "auto", True),
        ("cuda", "auto", 1.5),
    ],
)
def test_invalid_schedule_requests_fail(backend, choice, runs):
    with pytest.raises(ValueError):
        schedule.normalize_schedule_request(backend, choice, runs)


def _program(shape=(256, 512, 512), steps=256, runs=1):
    cfg = SimpleNamespace(
        backend="cuda_streamed",
        cuda_schedule="auto",
        cuda_expected_runs=runs,
        num_steps=steps,
        cuda_flags=CUDA_DEFAULT_FLAGS,
        is_3d=True,
        sharding=SimpleNamespace(enabled=False),
        cuda_storage_axes=(0, 1, 2),
        metric_kind="isotropic_uniform",
        cuda_memory_policy="auto",
    )
    coeffs = SimpleNamespace(e_inverse_offdiagonal=np.empty(0))
    for axis in "xyz":
        setattr(coeffs, f"e_decay_{axis}", np.ones(2, np.float32))
        setattr(coeffs, f"e_source_{axis}", np.zeros(1, np.int32))
        for kind in ("decay", "source"):
            setattr(coeffs, f"h_{kind}_{axis}", np.array(1, np.float32))
    terms = tuple(
        SimpleNamespace(slab=SimpleNamespace(low=12, high=12, shape=(1, 1, 1)))
        for _ in range(6)
    )
    cpml = SimpleNamespace(
        enabled=True, h_terms=terms, e_terms=terms, metallic_edges=frozenset()
    )
    supports = component_shapes(shape)
    return SimpleNamespace(
        config=cfg,
        coefficients=coeffs,
        boundary=SimpleNamespace(cpml=cpml, logical_component_shapes=supports),
        monitors=(SimpleNamespace(freq_count=3, dft_point_count=100),) * 2,
        sharding=SimpleNamespace(
            mesh=None,
            layout=SimpleNamespace(
                enabled=False, num_devices=1, padded_shapes=supports, axis=None
            ),
        ),
    )


def _geometry(shape=(257, 513, 513), active=480, depth=229):
    return SimpleNamespace(
        shape=shape,
        core_cells=52_000_000,
        summary=lambda: {
            "core_fraction": 0.78,
            "active_tiles": active,
            "column_depth": depth,
            "owned_quadrants": active * 4,
        },
    )


def test_selector_requires_compilation_to_be_amortized():
    program = _program()
    candidate, reason, details = schedule.automatic_candidate(
        program, _geometry(), device_kind="NVIDIA GeForce RTX 3090"
    )
    assert not candidate
    assert "compilation" in reason
    assert details["predicted_saved_execution_s"] < details["extra_compile_s"]
    program.config.cuda_expected_runs = 16
    assert schedule.automatic_candidate(
        program, _geometry(), device_kind="NVIDIA GeForce RTX 3090"
    )[0]


@pytest.mark.parametrize("change", ["gpu", "columns", "dense", "monitors", "precision"])
def test_selector_stays_streamed_outside_measured_envelope(change):
    program = _program(runs=32)
    geometry = _geometry()
    gpu = "NVIDIA GeForce RTX 3090"
    if change == "gpu":
        gpu = "NVIDIA H100"
    if change == "columns":
        geometry = _geometry(active=28)
    if change == "dense":
        program.coefficients.h_source_x = np.ones((3, 3, 3), np.float32)
    if change == "monitors":
        program.monitors *= 2
    if change == "precision":
        program.config.cuda_flags |= abi.CUDA_BF16_PSI
    assert not schedule.automatic_candidate(program, geometry, device_kind=gpu)[0]


@pytest.mark.parametrize(
    "available,chosen", [(1, "streamed"), (None, "streamed"), (20 << 30, "mstep")]
)
def test_auto_selection_checks_actual_workspace(monkeypatch, available, chosen):
    from beamz.simulation import backend, memory
    from beamz.simulation.cuda import mstep

    program = _program(runs=32)
    monkeypatch.setattr(schedule, "cuda_device_kind", lambda: "NVIDIA GeForce RTX 3090")
    monkeypatch.setattr(
        backend,
        "cuda_backend_status",
        lambda **kw: SimpleNamespace(targets=(abi.CUDA_MSTEP_TARGET,)),
    )
    monkeypatch.setattr(mstep, "build_mstep_geometry", lambda *args: _geometry())
    plan = SimpleNamespace(summary=lambda: {"scratch_bytes": 8192, "plan_bytes": 1024})
    monkeypatch.setattr(mstep, "build_mstep_plan", lambda *args: plan)
    monkeypatch.setattr(memory, "cuda_available_bytes", lambda program: available)
    result, decision = schedule.choose_execution_schedule(
        program, (None,) * 9, None, native_supported=True, donate_state=False
    )
    assert decision["selected"] == chosen
    assert (result is plan) == (chosen == "mstep")


def test_schedule_and_reuse_are_execution_cache_inputs():
    import beamz as bz

    sim = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
    )
    request = sim.to_request(
        num_steps=2, backend="cuda", schedule="auto", expected_runs=4
    )
    keys = [
        CompiledProgramKey.from_request(
            replace(
                request,
                run=replace(request.run, cuda_schedule=choice, cuda_expected_runs=runs),
            )
        )
        for choice, runs in (("auto", 4), ("streamed", 4), ("auto", 1))
    ]
    assert len(set(keys)) == 3
    assert len({_preparation_key(key) for key in keys}) == 1


def test_auto_backend_cannot_silently_accept_mstep_on_2d():
    import beamz as bz
    from beamz.simulation.backend import CudaBackendUnavailable

    sim = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
    )
    with pytest.raises(CudaBackendUnavailable, match="3D"):
        sim.compile(backend="auto", schedule="mstep")


@pytest.mark.parametrize(
    "tile,segment",
    [("8x32", "32"), ("8x16", "128"), ("16x16", "256"), ("16x32", "whole")],
)
def test_mstep_variant_flags_are_validated_and_cacheable(monkeypatch, tile, segment):
    from beamz.simulation.cuda.mstep_config import (
        variant_flags_from_env,
        variant_summary,
    )

    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_TILE", tile)
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_SEGMENT", segment)
    flags = variant_flags_from_env()
    assert variant_summary(flags) == {
        "tile": tuple(map(int, tile.split("x"))),
        "column_segment": 0 if segment == "whole" else int(segment),
    }
    monkeypatch.setenv("BEAMZ_CUDA_MSTEP_TILE", "bad")
    with pytest.raises(ValueError, match="TILE"):
        variant_flags_from_env()
