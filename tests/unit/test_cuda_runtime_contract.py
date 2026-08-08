from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from beamz.simulation import kernels
from beamz.simulation.compile import _elide_uniform_grid
from beamz.simulation.cuda import runtime as cuda_runtime
from beamz.simulation.execute import initial_program_state
from tests.performance.h100_workloads import H100Workload


def _program_and_state(*, cpml: bool, source: bool = False, monitor: bool = False):
    workload = H100Workload(
        name="cuda_contract",
        shape_zyx=(8, 10, 12),
        timesteps=3,
        resolution=100e-9,
        pml_cells=2,
        heterogeneous=True,
        cpml=cpml,
        source=source,
        monitor=monitor,
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
    assert len(arguments) == 40
    assert attributes == {
        "abi_version": np.int32(2),
        "phase": np.int32(0),
        "nterms": np.int32(6),
        "dt": np.float32(context.dt),
        "resolution": np.float32(context.resolution),
        "metallic_edges": np.int32(0),
        "metric_kind": np.int32(0),
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
    assert len(arguments) == 16
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
    assert len(arguments) == 24
    assert options["input_output_aliases"] == {index: index for index in range(6)}
    assert attributes == {
        "abi_version": np.int32(2),
        "nsteps": np.int32(7),
        "dt": np.float32(context.dt),
        "resolution": np.float32(context.resolution),
        "metallic_edges": np.int32(63),
        "metric_kind": np.int32(0),
    }
    assert next_state.hx is state.hx
    assert next_state.ez is state.ez


def test_cuda_cpml_multi_step_ffi_aliases_fields_and_psi(monkeypatch):
    program, state, context = _program_and_state(cpml=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (*arguments[:6], *arguments[31:37], *arguments[62:68])

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_steps(state, context, program.coefficients, 7)

    target, results, options, arguments, attributes = captured[0]
    assert target == "beamz_cuda_streamed_cpml_steps"
    assert len(results) == 18
    assert len(arguments) == 74
    assert options["input_output_aliases"] == {
        **{index: index for index in range(6)},
        **{31 + index: 6 + index for index in range(6)},
        **{62 + index: 12 + index for index in range(6)},
    }
    assert attributes["nsteps"] == np.int32(7)
    assert next_state.cpml_psi_h_terms == state.cpml_psi_h_terms
    assert next_state.cpml_psi_e_terms == state.cpml_psi_e_terms


def test_cuda_source_cpml_graph_packs_source_and_aliases_state(monkeypatch):
    program, state, context = _program_and_state(cpml=True, source=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (*arguments[:6], *arguments[31:37], *arguments[62:68])

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_source_steps(
        state, context, program.coefficients, program.sources[0], 3
    )

    target, results, options, arguments, attributes = captured[0]
    assert target == "beamz_cuda_streamed_source_cpml_steps"
    assert len(results) == 18
    assert len(arguments) == 77
    assert options["input_output_aliases"] == {
        **{index: index for index in range(6)},
        **{31 + index: 6 + index for index in range(6)},
        **{62 + index: 12 + index for index in range(6)},
    }
    assert attributes["source_component"] == np.int32(2)
    assert attributes["cpml_enabled"] == np.int32(1)
    assert next_state.cpml_psi_h_terms == state.cpml_psi_h_terms


def test_cuda_source_group_graph_packs_all_phases_and_aliases_state(monkeypatch):
    program, state, context = _program_and_state(cpml=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (*arguments[:6], *arguments[31:37], *arguments[62:68])

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)
    source_group = SimpleNamespace(
        coeffs=jnp.ones((2, 3, 4, 5), dtype=jnp.float32),
        waveforms=jnp.ones((2, 8), dtype=jnp.float32),
        starts=jnp.zeros((2, 3), dtype=jnp.int32),
    )
    groups = (source_group, None, None, None, None, None, None, None, None)

    next_state = cuda_runtime.run_source_group_steps(
        state, context, program.coefficients, groups, 3
    )

    target, results, options, arguments, attributes = captured[0]
    assert target == "beamz_cuda_streamed_source_groups_cpml_steps"
    assert len(results) == 18
    assert len(arguments) == 102
    assert arguments[74] is source_group.coeffs
    assert arguments[75] is source_group.waveforms
    assert arguments[76] is source_group.starts
    assert arguments[101] is state.current_step
    assert options["input_output_aliases"] == {
        **{index: index for index in range(6)},
        **{31 + index: 6 + index for index in range(6)},
        **{62 + index: 12 + index for index in range(6)},
    }
    assert attributes["nsteps"] == np.int32(3)
    assert attributes["cpml_enabled"] == np.int32(1)
    assert next_state.cpml_psi_h_terms == state.cpml_psi_h_terms


def test_cuda_source_group_graph_requires_all_phase_component_slots():
    program, state, context = _program_and_state(cpml=False)

    with np.testing.assert_raises_regex(ValueError, "nine phase/component groups"):
        cuda_runtime.run_source_group_steps(
            state, context, program.coefficients, (None,) * 8, 3
        )


def test_cuda_source_graph_supports_pec_without_cpml(monkeypatch):
    program, state, context = _program_and_state(cpml=False, source=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return arguments[:6]

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_source_steps(
        state, context, program.coefficients, program.sources[0], 3
    )

    target, results, options, arguments, attributes = captured[0]
    assert target == "beamz_cuda_streamed_source_cpml_steps"
    assert len(results) == 6
    assert len(arguments) == 27
    assert options["input_output_aliases"] == {index: index for index in range(6)}
    assert attributes["cpml_enabled"] == np.int32(0)
    assert next_state.ez is state.ez


def test_cuda_source_monitor_graph_aliases_dft_accumulators(monkeypatch):
    program, state, context = _program_and_state(cpml=True, source=True, monitor=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (
                *arguments[:6],
                *arguments[31:37],
                *arguments[62:68],
                *arguments[91:94],
            )

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_source_monitor_steps(
        state,
        context,
        program.coefficients,
        program.sources[0],
        program.monitors[0],
        3,
    )

    target, results, options, arguments, attributes = captured[0]
    assert target == "beamz_cuda_streamed_source_monitor_cpml_steps"
    assert len(results) == 21
    assert len(arguments) == 95
    assert options["input_output_aliases"][91] == 18
    assert options["input_output_aliases"][92] == 19
    assert options["input_output_aliases"][93] == 20
    assert attributes["frequency_count"] == np.int32(3)
    assert attributes["cpml_enabled"] == np.int32(1)
    assert next_state.dft_vec_re is state.dft_vec_re


def test_cuda_source_monitor_graph_supports_pec_without_cpml(monkeypatch):
    program, state, context = _program_and_state(cpml=False, source=True, monitor=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (*arguments[:6], *arguments[41:44])

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_source_monitor_steps(
        state,
        context,
        program.coefficients,
        program.sources[0],
        program.monitors[0],
        3,
    )

    _target, results, options, arguments, attributes = captured[0]
    assert len(results) == 9
    assert len(arguments) == 45
    assert options["input_output_aliases"][41] == 6
    assert options["input_output_aliases"][42] == 7
    assert options["input_output_aliases"][43] == 8
    assert attributes["cpml_enabled"] == np.int32(0)
    assert next_state.dft_vec_re is state.dft_vec_re


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
