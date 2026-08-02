from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import beamz as bz
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
    extension = _extension("beamz_cuda_streamed", "beamz_cuda_hopper")
    registrations = []
    monkeypatch.setattr(backend_runtime, "_REGISTERED_MODULE", None)
    monkeypatch.setattr(
        backend_runtime.jax.ffi,
        "register_ffi_target",
        lambda name, capsule, **kwargs: registrations.append(
            (name, capsule, kwargs)
        ),
    )

    targets = backend_runtime.register_cuda_ffi_targets(extension)

    assert targets == ("beamz_cuda_hopper", "beamz_cuda_streamed")
    assert {name for name, _, _ in registrations} == set(targets)
    assert all(kwargs == {"platform": "CUDA", "api_version": 1} for _, _, kwargs in registrations)


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
