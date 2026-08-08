from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import beamz as bz
from beamz.design import MaterialGrid, RectilinearGrid
from beamz.design.raster import Grid, Material, RasterOptions, Scene, rasterize
from beamz.simulation import backend as backend_runtime


class _FakeDevice:
    platform = "gpu"
    device_kind = "NVIDIA H100 80GB HBM3"
    compute_capability = (9, 0)


def _extension(*targets):
    return SimpleNamespace(
        __version__="0.1.0",
        registrations=lambda: {target: object() for target in targets},
    )


def test_auto_preserves_jax_when_cuda_is_not_visible(monkeypatch):
    monkeypatch.setattr(backend_runtime, "_gpu_devices", lambda: ())

    assert backend_runtime.resolve_backend("auto") == "jax"
    with pytest.raises(backend_runtime.CudaBackendUnavailable, match="no visible"):
        backend_runtime.resolve_backend("cuda")


def test_auto_preserves_jax_for_source_free_2d_simulations(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "auto" else backend,
    )
    simulation = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
    )

    program = simulation.compile(backend="auto")

    assert program.config.backend == "jax"


def test_explicit_streamed_cuda_accepts_source_free_2d_simulations(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "cuda_streamed" else backend,
    )
    simulation = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
    )

    program = simulation.compile(backend="cuda_streamed")

    assert program.config.backend == "cuda_streamed"


def test_auto_preserves_jax_for_2d_cpml_simulations(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "auto" else backend,
    )
    simulation = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
        boundaries=[bz.PML(thickness=0.1 * bz.um, formulation="cpml")],
    )

    program = simulation.compile(backend="auto")

    assert program.config.backend == "jax"


def test_explicit_cuda_rejects_2d_cpml_simulations():
    simulation = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
        boundaries=[bz.PML(thickness=0.1 * bz.um, formulation="cpml")],
    )

    with pytest.raises(
        backend_runtime.CudaBackendUnavailable, match="without CPML"
    ):
        simulation.compile(backend="cuda_streamed")


def _rectilinear_3d_simulation():
    grid = RectilinearGrid(
        np.asarray((0.0, 0.1, 0.3)),
        np.asarray((0.0, 0.15, 0.3)),
        np.asarray((0.0, 0.1, 0.3)),
    )
    shape = (2, 2, 2)
    materials = MaterialGrid(
        permittivity=np.ones(shape, dtype=np.float32),
        conductivity=np.float32(0.0),
        permeability=np.float32(1.0),
        resolution=0.1,
        shape=shape,
        grid=grid,
    )
    return bz.Simulation(material_grid=materials, time=np.arange(3) * 1e-17)


def test_auto_selects_streamed_cuda_for_rectilinear_3d_simulations(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "auto" else backend,
    )

    program = _rectilinear_3d_simulation().compile(backend="auto")

    assert program.config.backend == "cuda_streamed"


def test_explicit_streamed_cuda_accepts_rectilinear_3d_simulations(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "cuda_streamed" else backend,
    )

    program = _rectilinear_3d_simulation().compile(backend="cuda_streamed")

    assert program.config.backend == "cuda_streamed"
    assert program.config.metric_kind == "rectilinear"


def test_hopper_cuda_rejects_rectilinear_3d_simulations():
    with pytest.raises(
        backend_runtime.CudaBackendUnavailable, match="Hopper-specific kernel"
    ):
        _rectilinear_3d_simulation().compile(backend="cuda_hopper")


def _full_tensor_3d_simulation():
    material_grid = MaterialGrid.from_raster_result(
        rasterize(
            Scene(
                (
                    Material(
                        epsilon_r=(
                            (3.0, 0.2, 0.0),
                            (0.2, 2.0, 0.0),
                            (0.0, 0.0, 1.0),
                        )
                    ),
                )
            ),
            Grid.uniform((0.0, 0.0, 0.0), (0.3, 0.3, 0.3), (3, 3, 3)),
            options=RasterOptions(smoothing="farjadpour_full"),
        )
    )
    return bz.Simulation(material_grid=material_grid, time=np.arange(3) * 1e-17)


def test_auto_preserves_jax_for_full_tensor_permittivity(monkeypatch):
    monkeypatch.setattr(
        backend_runtime,
        "resolve_backend",
        lambda backend: "cuda_streamed" if backend == "auto" else backend,
    )

    program = _full_tensor_3d_simulation().compile(backend="auto")

    assert program.config.backend == "jax"


def test_explicit_cuda_rejects_full_tensor_permittivity():
    with pytest.raises(
        backend_runtime.CudaBackendUnavailable, match="full-tensor permittivity"
    ):
        _full_tensor_3d_simulation().compile(backend="cuda_streamed")


def test_cuda_status_exposes_complete_diagnostics():
    status = backend_runtime.CudaBackendStatus(
        available=True,
        extension_version="0.1.0",
        targets=("beamz_cuda_streamed",),
        gpu_devices=("NVIDIA H100 80GB HBM3",),
        compute_capabilities=(90,),
    )

    assert status.as_dict() == {
        "available": True,
        "extension_version": "0.1.0",
        "targets": ("beamz_cuda_streamed",),
        "gpu_devices": ("NVIDIA H100 80GB HBM3",),
        "compute_capabilities": (90,),
        "reason": None,
    }


def test_cuda_defaults_to_validated_streamed_target_on_sm90(monkeypatch):
    extension = _extension("beamz_cuda_streamed", "beamz_cuda_hopper")
    monkeypatch.setattr(backend_runtime, "_gpu_devices", lambda: (_FakeDevice(),))
    monkeypatch.setattr(backend_runtime, "_load_extension", lambda: extension)
    monkeypatch.setattr(
        backend_runtime,
        "register_cuda_ffi_targets",
        lambda module=None: tuple(sorted(extension.registrations())),
    )

    assert backend_runtime.resolve_backend("cuda") == "cuda_streamed"
    assert backend_runtime.resolve_backend("cuda_streamed") == "cuda_streamed"
    assert backend_runtime.resolve_backend("cuda_hopper") == "cuda_hopper"
    assert backend_runtime.resolve_backend("auto") == "cuda_streamed"


def test_explicit_hopper_rejects_pre_sm90(monkeypatch):
    device = _FakeDevice()
    device.compute_capability = (8, 0)
    extension = _extension("beamz_cuda_streamed", "beamz_cuda_hopper")
    monkeypatch.setattr(backend_runtime, "_gpu_devices", lambda: (device,))
    monkeypatch.setattr(backend_runtime, "_load_extension", lambda: extension)
    monkeypatch.setattr(
        backend_runtime,
        "register_cuda_ffi_targets",
        lambda module=None: tuple(sorted(extension.registrations())),
    )

    with pytest.raises(backend_runtime.CudaBackendUnavailable, match="SM90"):
        backend_runtime.resolve_backend("cuda_hopper")
    assert backend_runtime.resolve_backend("cuda") == "cuda_streamed"


def test_hopper_only_extension_requires_explicit_opt_in(monkeypatch):
    extension = _extension("beamz_cuda_hopper")
    monkeypatch.setattr(backend_runtime, "_gpu_devices", lambda: (_FakeDevice(),))
    monkeypatch.setattr(backend_runtime, "_load_extension", lambda: extension)
    monkeypatch.setattr(
        backend_runtime,
        "register_cuda_ffi_targets",
        lambda module=None: tuple(sorted(extension.registrations())),
    )

    assert backend_runtime.resolve_backend("auto") == "jax"
    assert backend_runtime.resolve_backend("cuda_hopper") == "cuda_hopper"
    with pytest.raises(backend_runtime.CudaBackendUnavailable, match="compatible"):
        backend_runtime.resolve_backend("cuda")


def test_typed_ffi_registrations_use_cuda_api_v1(monkeypatch):
    extension = _extension(
        "beamz_cuda_streamed",
        "beamz_cuda_streamed_2d_steps",
        "beamz_cuda_streamed_steps",
        "beamz_cuda_streamed_cpml_steps",
        "beamz_cuda_streamed_source_cpml_steps",
        "beamz_cuda_streamed_source_monitor_cpml_steps",
        "beamz_cuda_hopper",
    )
    registrations = []
    monkeypatch.setattr(backend_runtime, "_REGISTERED_MODULE", None)
    monkeypatch.setattr(
        backend_runtime.jax.ffi,
        "register_ffi_target",
        lambda name, capsule, **kwargs: registrations.append((name, capsule, kwargs)),
    )

    targets = backend_runtime.register_cuda_ffi_targets(extension)

    assert targets == (
        "beamz_cuda_hopper",
        "beamz_cuda_streamed",
        "beamz_cuda_streamed_2d_steps",
        "beamz_cuda_streamed_cpml_steps",
        "beamz_cuda_streamed_source_cpml_steps",
        "beamz_cuda_streamed_source_monitor_cpml_steps",
        "beamz_cuda_streamed_steps",
    )
    assert {name for name, _, _ in registrations} == set(targets)
    assert all(
        kwargs == {"platform": "CUDA", "api_version": 1}
        for _, _, kwargs in registrations
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("xla", "jax"),
        ("cuda-streamed", "cuda_streamed"),
        ("hopper", "cuda_hopper"),
    ],
)
def test_backend_aliases(value, expected):
    assert backend_runtime.normalize_backend(value) == expected


def test_backend_participates_in_compiled_program_identity():
    from beamz.simulation.compile import CompiledProgramKey

    simulation = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
    )
    jax_key = CompiledProgramKey.from_request(
        simulation.to_request(num_steps=2, backend="jax")
    )
    cuda_key = CompiledProgramKey.from_request(
        simulation.to_request(num_steps=2, backend="cuda_streamed")
    )

    assert jax_key != cuda_key
    assert jax_key.backend == "jax"
    assert cuda_key.backend == "cuda_streamed"
