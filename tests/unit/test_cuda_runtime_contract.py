from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from beamz.simulation import kernels
from beamz.simulation.compile import _elide_uniform_grid
from beamz.simulation.cuda import runtime as cuda_runtime
from beamz.simulation.execute import initial_program_state
from tests.performance.h100_workloads import H100Workload


def _program_and_state(*, cpml: bool):
    workload = H100Workload(
        name="cuda_contract",
        shape_zyx=(8, 10, 12),
        timesteps=3,
        resolution=100e-9,
        pml_cells=2,
        heterogeneous=True,
        cpml=cpml,
    )
    simulation = workload.build()
    program = simulation.compile(backend="jax")
    state = initial_program_state(
        program,
        t=float(simulation.time[0]),
        current_step=0,
        monitor_steps=3,
    )
    config = replace(program.config, backend="cuda_streamed")
    context = kernels.CompiledStepContext(
        config=config,
        boundary=program.boundary,
        source_batches={},
        metrics=program.metrics,
        resolution=config.resolution,
        dt=config.dt,
        dt_scalar=jnp.asarray(config.dt, dtype=jnp.float32),
        is_3d=True,
    )
    return program, state, context


def test_cuda_ffi_phase_packs_cpml_and_aliases_state(monkeypatch):
    program, state, context = _program_and_state(cpml=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            nterms = int(attributes["nterms"])
            psi_start = 13 + 3 * nterms
            return (*arguments[:3], *arguments[psi_start : psi_start + nterms])

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.update_h(state, context, program.coefficients)

    target, results, options, arguments, attributes = captured[0]
    assert target == "beamz_cuda_streamed"
    assert len(results) == 9
    assert len(arguments) == 37
    assert attributes == {
        "abi_version": np.int32(1),
        "phase": np.int32(0),
        "nterms": np.int32(6),
        "dt": np.float32(context.dt),
        "resolution": np.float32(context.resolution),
        "metallic_edges": np.int32(0),
    }
    assert options["input_output_aliases"] == {
        0: 0,
        1: 1,
        2: 2,
        31: 3,
        32: 4,
        33: 5,
        34: 6,
        35: 7,
        36: 8,
    }
    assert next_state.cpml_psi_h_terms == state.cpml_psi_h_terms


def test_cuda_ffi_phase_supports_non_cpml_yee_update(monkeypatch):
    program, state, context = _program_and_state(cpml=False)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return arguments[:3]

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.update_e(state, context, program.coefficients)

    _, results, options, arguments, attributes = captured[0]
    assert len(results) == 3
    assert len(arguments) == 13
    assert attributes["phase"] == 1
    assert attributes["nterms"] == 0
    assert options["input_output_aliases"] == {0: 0, 1: 1, 2: 2}
    assert next_state.cpml_psi_e_terms == ()


def test_cuda_multi_step_ffi_aliases_all_fields(monkeypatch):
    program, state, context = _program_and_state(cpml=False)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return arguments[:6]

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_steps(state, context, program.coefficients, 7)

    target, results, options, arguments, attributes = captured[0]
    assert target == "beamz_cuda_streamed_steps"
    assert len(results) == 6
    assert len(arguments) == 18
    assert options["input_output_aliases"] == {index: index for index in range(6)}
    assert attributes == {
        "abi_version": np.int32(1),
        "nsteps": np.int32(7),
        "dt": np.float32(context.dt),
        "resolution": np.float32(context.resolution),
        "metallic_edges": np.int32(63),
    }
    assert next_state.hx is state.hx
    assert next_state.ez is state.ez


def test_cuda_backend_selects_hybrid_jax_orchestration_kernel():
    _program, _state, context = _program_and_state(cpml=True)

    selected = kernels.select_update_kernel(context)

    assert selected.kind == "cuda_streamed"
    assert selected.update_h is cuda_runtime.update_h
    assert selected.update_e is cuda_runtime.update_e


def test_hopper_backend_uses_sm90_tiled_target(monkeypatch):
    program, state, context = _program_and_state(cpml=False)
    context = replace(context, config=replace(context.config, backend="cuda_hopper"))
    targets = []

    def fake_ffi_call(target, result_metadata, **options):
        del result_metadata, options
        targets.append(target)
        return lambda *arguments, **attributes: arguments[:3]

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    cuda_runtime.update_h(state, context, program.coefficients)

    assert targets == ["beamz_cuda_hopper"]


def test_uniform_cuda_coefficients_are_compacted_without_rounding():
    uniform = jnp.full((3, 4, 5), np.float32(1.25))
    varied = uniform.at[1, 2, 3].set(np.nextafter(np.float32(1.25), np.float32(2.0)))

    compact = _elide_uniform_grid(uniform)

    assert compact.shape == ()
    assert float(compact) == 1.25
    assert _elide_uniform_grid(varied).shape == varied.shape
