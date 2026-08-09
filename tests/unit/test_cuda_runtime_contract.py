from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from beamz.simulation import kernels
from beamz.simulation.compile import _elide_uniform_grid
from beamz.simulation.cuda import runtime as cuda_runtime
from beamz.simulation.execute import (
    CUDA_GRAPH_MAX_STEPS,
    build_scan,
    initial_program_state,
)
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


def test_cuda_boundary_code_packs_only_uniform_two_sided_cpml():
    uniform = tuple(
        SimpleNamespace(slab=SimpleNamespace(low=4, high=4)) for _ in range(6)
    )
    asymmetric = (
        *uniform[:-1],
        SimpleNamespace(slab=SimpleNamespace(low=4, high=0)),
    )

    assert cuda_runtime._boundary_code(frozenset({"front"}), uniform) == (4 << 8) | 1
    assert cuda_runtime._boundary_code(frozenset({"front"}), asymmetric) == 1
    assert cuda_runtime._boundary_code(frozenset({"right"})) == 1 << 5


def test_cuda_cpml_bf16_state_is_explicit_and_preserves_continuation(monkeypatch):
    program, state, _context = _program_and_state(cpml=True)
    cuda_program = replace(
        program, config=replace(program.config, backend="cuda_streamed")
    )
    seeded = state._replace(
        cpml_psi_h_terms=tuple(value + 0.125 for value in state.cpml_psi_h_terms),
        cpml_psi_e_terms=tuple(value - 0.25 for value in state.cpml_psi_e_terms),
    )
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", "bf16")

    converted = initial_program_state(
        cuda_program,
        t=float(cuda_program.config.dt),
        current_step=1,
        continuation=seeded,
        monitor_steps=3,
    )

    assert all(value.dtype == jnp.bfloat16 for value in converted.cpml_psi_h_terms)
    assert all(value.dtype == jnp.bfloat16 for value in converted.cpml_psi_e_terms)
    np.testing.assert_allclose(
        np.asarray(converted.cpml_psi_h_terms[0], dtype=np.float32), 0.125
    )
    np.testing.assert_allclose(
        np.asarray(converted.cpml_psi_e_terms[0], dtype=np.float32), -0.25
    )


def test_cuda_cpml_rejects_unknown_psi_precision(monkeypatch):
    program, _state, _context = _program_and_state(cpml=True)
    cuda_program = replace(
        program, config=replace(program.config, backend="cuda_streamed")
    )
    monkeypatch.setenv("BEAMZ_CUDA_CPML_PSI_PRECISION", "fp8")

    with np.testing.assert_raises_regex(ValueError, "must be 'fp32' or 'bf16'"):
        initial_program_state(cuda_program, t=0.0, current_step=0, monitor_steps=3)


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
        "abi_version": np.int32(cuda_runtime.CUDA_ABI_VERSION),
        "phase": np.int32(0),
        "nterms": np.int32(6),
        "dt": np.float32(context.dt),
        "resolution": np.float32(context.resolution),
        # Low six bits encode PEC faces; high bits carry the uniform CPML width.
        "metallic_edges": np.int32(2 << 8),
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
    coefficients = program.coefficients._replace(
        **{
            name: jnp.asarray(1.0, dtype=jnp.float32)
            for name in (
                "h_decay_x",
                "h_decay_y",
                "h_decay_z",
                "h_source_x",
                "h_source_y",
                "h_source_z",
                "e_decay_x",
                "e_decay_y",
                "e_decay_z",
                "e_source_x",
                "e_source_y",
                "e_source_z",
            )
        }
    )
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return arguments[:12]

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_steps(state, context, coefficients, 7)

    target, results, options, arguments, attributes = captured[0]
    assert target == "beamz_cuda_temporal_steps"
    assert len(results) == 12
    assert len(arguments) == 30
    assert options["input_output_aliases"] == {index: index for index in range(12)}
    assert attributes == {
        "abi_version": np.int32(cuda_runtime.CUDA_ABI_VERSION),
        "nsteps": np.int32(7),
        "dt": np.float32(context.dt),
        "resolution": np.float32(context.resolution),
        "metallic_edges": np.int32(63),
        "metric_kind": np.int32(0),
    }
    assert next_state.hx is state.hx
    assert next_state.ez is state.ez


def test_cuda_scan_replays_one_bounded_graph_and_advances_chunk_clocks(monkeypatch):
    program, state, _context = _program_and_state(cpml=False)
    requested_steps = 2 * CUDA_GRAPH_MAX_STEPS + 7
    program = replace(
        program,
        config=replace(
            program.config,
            backend="cuda_streamed",
            num_steps=requested_steps,
        ),
    )
    traced_step_counts = []

    def fake_run_steps(chunk_state, _context, _coefficients, nsteps):
        traced_step_counts.append(nsteps)
        return chunk_state._replace(ex=chunk_state.ex + np.float32(nsteps))

    monkeypatch.setattr(
        "beamz.simulation.cuda.run_steps",
        fake_run_steps,
    )

    next_state = build_scan(program)(state, program.coefficients)

    assert traced_step_counts == [CUDA_GRAPH_MAX_STEPS, 7]
    assert int(next_state.current_step) == requested_steps
    np.testing.assert_allclose(
        next_state.t,
        state.t + np.float32(program.config.dt * requested_steps),
    )
    np.testing.assert_allclose(next_state.ex, state.ex + np.float32(requested_steps))


def test_cuda_scan_routes_one_slab_source_to_narrow_graph(monkeypatch):
    program, state, _context = _program_and_state(cpml=True, source=True)
    program = replace(
        program,
        config=replace(program.config, backend="cuda_streamed"),
    )
    calls = []

    def fake_run_source_steps(chunk_state, _context, _coefficients, source, nsteps):
        calls.append((source, nsteps))
        return chunk_state

    monkeypatch.setattr("beamz.simulation.cuda.run_source_steps", fake_run_source_steps)
    monkeypatch.setattr(
        "beamz.simulation.cuda.run_source_group_steps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("general source graph should not be selected")
        ),
    )

    build_scan(program)(state, program.coefficients)

    assert calls == [(program.sources[0], program.config.num_steps)]


def test_cuda_scan_routes_simple_source_monitor_to_narrow_graph(monkeypatch):
    program, state, _context = _program_and_state(cpml=True, source=True, monitor=True)
    program = replace(
        program,
        config=replace(program.config, backend="cuda_streamed"),
    )
    calls = []

    def fake_run_source_monitor_steps(
        chunk_state, _context, _coefficients, source, monitor, nsteps
    ):
        calls.append((source, monitor, nsteps))
        return chunk_state

    monkeypatch.setattr(
        "beamz.simulation.cuda.run_source_monitor_steps",
        fake_run_source_monitor_steps,
    )
    monkeypatch.setattr(
        "beamz.simulation.cuda.run_program_steps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("general program graph should not be selected")
        ),
    )

    build_scan(program)(state, program.coefficients)

    assert calls == [
        (program.sources[0], program.monitors[0], program.config.num_steps)
    ]


def test_cuda_scan_keeps_scheduled_monitor_on_general_graph(monkeypatch):
    program, state, _context = _program_and_state(cpml=True, source=True, monitor=True)
    program = replace(
        program,
        config=replace(program.config, backend="cuda_streamed"),
        monitors=(replace(program.monitors[0], dft_record_interval=2),),
    )
    calls = []

    def fake_run_program_steps(
        chunk_state, _context, _coefficients, groups, monitors, nsteps
    ):
        calls.append((groups, monitors, nsteps))
        return chunk_state

    monkeypatch.setattr(
        "beamz.simulation.cuda.run_program_steps", fake_run_program_steps
    )
    monkeypatch.setattr(
        "beamz.simulation.cuda.run_source_monitor_steps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("narrow monitor graph should not be selected")
        ),
    )

    build_scan(program)(state, program.coefficients)

    assert len(calls) == 1
    assert calls[0][1] is not None
    assert calls[0][2] == program.config.num_steps


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
        starts_tuple=((0, 0, 0), (0, 0, 0)),
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
    assert attributes["coincident_source_group_mask"] == np.int32(1)
    assert next_state.cpml_psi_h_terms == state.cpml_psi_h_terms


def test_cuda_source_group_graph_uses_temporal_cpml_field_banks(monkeypatch):
    program, state, context = _program_and_state(cpml=True)
    coefficients = program.coefficients._replace(
        h_decay_x=jnp.asarray(1.0, dtype=jnp.float32),
        h_decay_y=jnp.asarray(1.0, dtype=jnp.float32),
        h_decay_z=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_x=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_y=jnp.asarray(1.0, dtype=jnp.float32),
        h_source_z=jnp.asarray(1.0, dtype=jnp.float32),
        e_source_x=jnp.zeros((1,), dtype=jnp.int32),
        e_source_y=jnp.zeros((1,), dtype=jnp.int32),
        e_source_z=jnp.zeros((1,), dtype=jnp.int32),
    )
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (
                *arguments[:6],
                *arguments[74:80],
                *arguments[31:37],
                *arguments[62:68],
            )

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)

    next_state = cuda_runtime.run_source_group_steps(
        state, context, coefficients, (None,) * 9, 3
    )

    target, results, options, arguments, attributes = captured[0]
    assert target == "beamz_cuda_temporal_source_groups_cpml_steps"
    assert len(results) == 24
    assert len(arguments) == 108
    assert arguments[107] is state.current_step
    assert options["input_output_aliases"] == {
        **{index: index for index in range(6)},
        **{74 + index: 6 + index for index in range(6)},
        **{31 + index: 12 + index for index in range(6)},
        **{62 + index: 18 + index for index in range(6)},
    }
    assert "cpml_enabled" not in attributes
    assert attributes["nsteps"] == np.int32(3)
    assert next_state.hx is arguments[74]
    assert next_state.cpml_psi_h_terms == state.cpml_psi_h_terms


def test_cuda_source_group_graph_requires_all_phase_component_slots():
    program, state, context = _program_and_state(cpml=False)

    with np.testing.assert_raises_regex(ValueError, "nine phase/component groups"):
        cuda_runtime.run_source_group_steps(
            state, context, program.coefficients, (None,) * 8, 3
        )


def test_cuda_program_graph_packs_monitor_batch_and_aliases_accumulators(monkeypatch):
    program, state, context = _program_and_state(cpml=True, source=True, monitor=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (
                *arguments[:6],
                *arguments[31:37],
                *arguments[62:68],
                *arguments[108:111],
            )

        return call

    monkeypatch.setattr(cuda_runtime.jax.ffi, "ffi_call", fake_ffi_call)
    source_group = SimpleNamespace(
        coeffs=jnp.ones((1, 2, 3, 4), dtype=jnp.float32),
        waveforms=jnp.ones((1, 8), dtype=jnp.float32),
        starts=jnp.zeros((1, 3), dtype=jnp.int32),
        starts_tuple=((0, 0, 0),),
    )
    groups = (source_group, None, None, None, None, None, None, None, None)
    packed = cuda_runtime.pack_dft_monitors(program.monitors)

    next_state = cuda_runtime.run_program_steps(
        state, context, program.coefficients, groups, packed, 3
    )

    target, results, options, arguments, attributes = captured[0]
    assert target == "beamz_cuda_streamed_program_cpml_steps"
    assert len(results) == 21
    assert len(arguments) == 113
    assert packed[0].shape[:2] == (1, 6)
    assert options["input_output_aliases"][108] == 18
    assert options["input_output_aliases"][109] == 19
    assert options["input_output_aliases"][110] == 20
    assert attributes["monitor_count"] == np.int32(1)
    assert attributes["coincident_source_group_mask"] == np.int32(1)
    np.testing.assert_array_equal(next_state.dft_vec_re, state.dft_vec_re)


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
                *arguments[95:97],
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
    assert len(results) == 23
    assert len(arguments) == 97
    assert options["input_output_aliases"][91] == 18
    assert options["input_output_aliases"][92] == 19
    assert options["input_output_aliases"][93] == 20
    assert options["input_output_aliases"][95] == 21
    assert options["input_output_aliases"][96] == 22
    assert attributes["frequency_count"] == np.int32(3)
    assert attributes["cpml_enabled"] == np.int32(1)
    np.testing.assert_array_equal(next_state.dft_vec_re, state.dft_vec_re)


def test_cuda_source_monitor_graph_supports_pec_without_cpml(monkeypatch):
    program, state, context = _program_and_state(cpml=False, source=True, monitor=True)
    captured = []

    def fake_ffi_call(target, result_metadata, **options):
        def call(*arguments, **attributes):
            captured.append((target, result_metadata, options, arguments, attributes))
            return (*arguments[:6], *arguments[41:44], *arguments[45:47])

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
    assert len(results) == 11
    assert len(arguments) == 47
    assert options["input_output_aliases"][41] == 6
    assert options["input_output_aliases"][42] == 7
    assert options["input_output_aliases"][43] == 8
    assert options["input_output_aliases"][45] == 9
    assert options["input_output_aliases"][46] == 10
    assert attributes["cpml_enabled"] == np.int32(0)
    np.testing.assert_array_equal(next_state.dft_vec_re, state.dft_vec_re)


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
