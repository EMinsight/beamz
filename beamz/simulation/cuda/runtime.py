"""Lower fused 3D Yee phases to the optional BeamZ CUDA typed FFI targets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation.model import SimulationState

CUDA_ABI_VERSION = 2
_PHASE_H = 0
_PHASE_E = 1
_COMPONENT_CODE = {name: index for index, name in enumerate(("Hx", "Hy", "Hz"))}
_COMPONENT_CODE.update({name: index for index, name in enumerate(("Ex", "Ey", "Ez"))})
_EMPTY = jnp.empty((0,), dtype=jnp.float32)
_METRIC_KIND_CODE = {
    "isotropic_uniform": 0,
    "axis_uniform": 1,
    "rectilinear": 2,
}


def _metallic_edge_mask(edges: frozenset[str]) -> int:
    order = ("front", "back", "bottom", "top", "left", "right")
    return sum(1 << index for index, name in enumerate(order) if name in edges)


def _term_metadata(terms) -> jnp.ndarray:
    return jnp.asarray(
        [
            (
                _COMPONENT_CODE[term.component],
                int(term.axis),
                int(term.slab.low),
                int(term.slab.high),
                1 if float(term.sign) > 0.0 else -1,
            )
            for term in terms
        ],
        dtype=jnp.int32,
    ).reshape((len(terms), 5))


def _shape(value):
    return jax.ShapeDtypeStruct(value.shape, value.dtype)


def _phase_metrics(ctx, phase: int):
    """Return CUDA-axis-ordered derivative metrics for one Yee phase."""
    metrics = ctx.metrics
    if phase == _PHASE_H:
        return (metrics.e_to_h_z, metrics.e_to_h_y, metrics.e_to_h_x)
    return (metrics.h_to_e_z, metrics.h_to_e_y, metrics.h_to_e_x)


def _metric_kind_code(ctx) -> np.int32:
    try:
        return np.int32(_METRIC_KIND_CODE[ctx.config.metric_kind])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported CUDA derivative metric kind: {ctx.config.metric_kind!r}"
        ) from exc


def _ffi_phase(
    target: str,
    phase: int,
    targets,
    sources,
    materials,
    terms,
    psi_terms,
    metrics,
    *,
    metric_kind,
    dt,
    resolution,
    metallic_edges,
):
    if len(terms) != len(psi_terms):
        raise ValueError(
            "CUDA CPML terms and recurrence buffers must have equal length"
        )
    if terms and len(terms) != 6:
        raise ValueError("3D CUDA CPML requires exactly six derivative terms per phase")
    metadata = _term_metadata(terms)
    term_arrays = tuple(
        value for term in terms for value in (term.a, term.b, term.inv_kappa)
    )
    arguments = (
        *targets,
        *sources,
        *materials,
        metadata,
        *term_arrays,
        *psi_terms,
        *metrics,
    )
    result_metadata = tuple(_shape(value) for value in (*targets, *psi_terms))
    psi_start = 13 + 3 * len(terms)
    aliases = {0: 0, 1: 1, 2: 2}
    aliases.update({psi_start + index: 3 + index for index in range(len(psi_terms))})
    call = jax.ffi.ffi_call(
        target,
        result_metadata,
        input_output_aliases=aliases,
        vmap_method="sequential",
    )
    outputs = call(
        *arguments,
        abi_version=np.int32(CUDA_ABI_VERSION),
        phase=np.int32(phase),
        nterms=np.int32(len(terms)),
        dt=np.float32(dt),
        resolution=np.float32(resolution),
        metallic_edges=np.int32(_metallic_edge_mask(metallic_edges)),
        metric_kind=np.int32(metric_kind),
    )
    return tuple(outputs)


def update_h(state, ctx, coeffs) -> SimulationState:
    """Advance the three magnetic fields and optional CPML memory on CUDA."""
    target = (
        "beamz_cuda_hopper"
        if ctx.config.backend == "cuda_hopper"
        else "beamz_cuda_streamed"
    )
    terms = ctx.boundary.cpml.h_terms
    materials = (
        (
            coeffs.h_sigma_m_x,
            coeffs.h_sigma_m_y,
            coeffs.h_sigma_m_z,
            _EMPTY,
            _EMPTY,
            _EMPTY,
        )
        if ctx.config.backend == "cuda_hopper"
        else (
            coeffs.h_decay_x,
            coeffs.h_decay_y,
            coeffs.h_decay_z,
            coeffs.h_source_x,
            coeffs.h_source_y,
            coeffs.h_source_z,
        )
    )
    outputs = _ffi_phase(
        target,
        _PHASE_H,
        (state.hx, state.hy, state.hz),
        (state.ex, state.ey, state.ez),
        materials,
        terms,
        state.cpml_psi_h_terms,
        _phase_metrics(ctx, _PHASE_H),
        metric_kind=_metric_kind_code(ctx),
        dt=ctx.dt,
        resolution=ctx.resolution,
        metallic_edges=ctx.boundary.cpml.metallic_edges,
    )
    return state._replace(
        hx=outputs[0],
        hy=outputs[1],
        hz=outputs[2],
        cpml_psi_h_terms=outputs[3:],
    )


def update_e(state, ctx, coeffs) -> SimulationState:
    """Advance the three electric fields and optional CPML memory on CUDA."""
    target = (
        "beamz_cuda_hopper"
        if ctx.config.backend == "cuda_hopper"
        else "beamz_cuda_streamed"
    )
    terms = ctx.boundary.cpml.e_terms
    materials = (
        (
            coeffs.e_conductivity_x,
            coeffs.e_conductivity_y,
            coeffs.e_conductivity_z,
            coeffs.e_permittivity_x,
            coeffs.e_permittivity_y,
            coeffs.e_permittivity_z,
        )
        if ctx.config.backend == "cuda_hopper"
        else (
            coeffs.e_decay_x,
            coeffs.e_decay_y,
            coeffs.e_decay_z,
            coeffs.e_source_x,
            coeffs.e_source_y,
            coeffs.e_source_z,
        )
    )
    outputs = _ffi_phase(
        target,
        _PHASE_E,
        (state.ex, state.ey, state.ez),
        (state.hx, state.hy, state.hz),
        materials,
        terms,
        state.cpml_psi_e_terms,
        _phase_metrics(ctx, _PHASE_E),
        metric_kind=_metric_kind_code(ctx),
        dt=ctx.dt,
        resolution=ctx.resolution,
        metallic_edges=ctx.boundary.cpml.metallic_edges,
    )
    return state._replace(
        ex=outputs[0],
        ey=outputs[1],
        ez=outputs[2],
        cpml_psi_e_terms=outputs[3:],
    )


def _cpml_graph_io(state, ctx, coeffs):
    fields = (state.hx, state.hy, state.hz, state.ex, state.ey, state.ez)
    h_terms = ctx.boundary.cpml.h_terms
    e_terms = ctx.boundary.cpml.e_terms
    h_materials = (
        coeffs.h_decay_x,
        coeffs.h_decay_y,
        coeffs.h_decay_z,
        coeffs.h_source_x,
        coeffs.h_source_y,
        coeffs.h_source_z,
    )
    e_materials = (
        coeffs.e_decay_x,
        coeffs.e_decay_y,
        coeffs.e_decay_z,
        coeffs.e_source_x,
        coeffs.e_source_y,
        coeffs.e_source_z,
    )
    h_payload = (
        *h_materials,
        _term_metadata(h_terms),
        *(value for term in h_terms for value in (term.a, term.b, term.inv_kappa)),
        *state.cpml_psi_h_terms,
    )
    e_payload = (
        *e_materials,
        _term_metadata(e_terms),
        *(value for term in e_terms for value in (term.a, term.b, term.inv_kappa)),
        *state.cpml_psi_e_terms,
    )
    arguments = (
        *fields,
        *h_payload,
        *e_payload,
        *_phase_metrics(ctx, _PHASE_H),
        *_phase_metrics(ctx, _PHASE_E),
    )
    result_values = (*fields, *state.cpml_psi_h_terms, *state.cpml_psi_e_terms)
    aliases = {index: index for index in range(6)}
    aliases.update({31 + index: 6 + index for index in range(6)})
    aliases.update({62 + index: 12 + index for index in range(6)})
    return arguments, result_values, aliases


def _yee_graph_io(state, ctx, coeffs):
    fields = (state.hx, state.hy, state.hz, state.ex, state.ey, state.ez)
    materials = (
        coeffs.h_decay_x,
        coeffs.h_decay_y,
        coeffs.h_decay_z,
        coeffs.h_source_x,
        coeffs.h_source_y,
        coeffs.h_source_z,
        coeffs.e_decay_x,
        coeffs.e_decay_y,
        coeffs.e_decay_z,
        coeffs.e_source_x,
        coeffs.e_source_y,
        coeffs.e_source_z,
    )
    arguments = (
        *fields,
        *materials,
        *_phase_metrics(ctx, _PHASE_H),
        *_phase_metrics(ctx, _PHASE_E),
    )
    return arguments, fields, {index: index for index in range(6)}


def _graph_io(state, ctx, coeffs):
    return (
        _cpml_graph_io(state, ctx, coeffs)
        if ctx.boundary.cpml.enabled
        else _yee_graph_io(state, ctx, coeffs)
    )


def _replace_graph_outputs(state, outputs) -> SimulationState:
    return state._replace(
        hx=outputs[0],
        hy=outputs[1],
        hz=outputs[2],
        ex=outputs[3],
        ey=outputs[4],
        ez=outputs[5],
        cpml_psi_h_terms=outputs[6:12],
        cpml_psi_e_terms=outputs[12:18],
    )


def run_source_steps(state, ctx, coeffs, source, nsteps: int) -> SimulationState:
    """Advance one packed pre-E source through one CUDA graph call."""
    if nsteps < 1:
        raise ValueError("CUDA step count must be positive")
    if (
        source.timing != "pre_e"
        or source.component not in {"Ex", "Ey", "Ez"}
        or not source.is_slab
        or source.slab_starts is None
        or source.slab_sizes is None
        or tuple(source.coeff.shape) != tuple(source.slab_sizes)
    ):
        raise ValueError("CUDA source graph requires one packed pre-E slab source")
    arguments, result_values, aliases = _graph_io(state, ctx, coeffs)
    call = jax.ffi.ffi_call(
        "beamz_cuda_streamed_source_cpml_steps",
        tuple(_shape(value) for value in result_values),
        input_output_aliases=aliases,
        vmap_method="sequential",
    )
    outputs = call(
        *arguments,
        source.coeff,
        source.waveform,
        state.current_step,
        abi_version=np.int32(CUDA_ABI_VERSION),
        nsteps=np.int32(nsteps),
        dt=np.float32(ctx.dt),
        resolution=np.float32(ctx.resolution),
        metallic_edges=np.int32(_metallic_edge_mask(ctx.boundary.cpml.metallic_edges)),
        metric_kind=_metric_kind_code(ctx),
        cpml_enabled=np.int32(ctx.boundary.cpml.enabled),
        source_component=np.int32(_COMPONENT_CODE[source.component]),
        source_start_z=np.int32(source.slab_starts[0]),
        source_start_y=np.int32(source.slab_starts[1]),
        source_start_x=np.int32(source.slab_starts[2]),
    )
    if ctx.boundary.cpml.enabled:
        return _replace_graph_outputs(state, outputs)
    return state._replace(
        hx=outputs[0],
        hy=outputs[1],
        hz=outputs[2],
        ex=outputs[3],
        ey=outputs[4],
        ez=outputs[5],
    )


def run_source_monitor_steps(
    state, ctx, coeffs, source, monitor, nsteps: int
) -> SimulationState:
    """Advance one packed source and one rectangular plane DFT in a CUDA graph."""
    if (
        not monitor.dft_enabled
        or monitor.dft_record_interval != 1
        or monitor.dft_t_start != 0.0
        or np.isfinite(monitor.dft_t_end)
        or monitor.dft_window_code != 0
        or monitor.dft_normalization_code != 0
        or monitor.freq_count < 1
        or monitor.dft_point_count < 1
        or state.dft_vec_re.dtype != jnp.float32
    ):
        raise ValueError("CUDA monitor graph requires a full-time float32 plane DFT")
    arguments, result_values, aliases = _graph_io(state, ctx, coeffs)
    state_output_count = len(result_values)
    result_values = (
        *result_values,
        state.dft_vec_re,
        state.dft_vec_im,
        state.dft_weight_sum,
    )
    aliases = {
        **aliases,
        len(arguments) + 17: state_output_count,
        len(arguments) + 18: state_output_count + 1,
        len(arguments) + 19: state_output_count + 2,
    }
    call = jax.ffi.ffi_call(
        "beamz_cuda_streamed_source_monitor_cpml_steps",
        tuple(_shape(value) for value in result_values),
        input_output_aliases=aliases,
        vmap_method="sequential",
    )
    outputs = call(
        *arguments,
        source.coeff,
        source.waveform,
        state.current_step,
        *monitor.dft_flat_idx,
        *monitor.dft_weights,
        monitor.freq_hz,
        monitor.dft_component_mask,
        state.dft_vec_re,
        state.dft_vec_im,
        state.dft_weight_sum,
        state.t,
        abi_version=np.int32(CUDA_ABI_VERSION),
        nsteps=np.int32(nsteps),
        dt=np.float32(ctx.dt),
        resolution=np.float32(ctx.resolution),
        metallic_edges=np.int32(_metallic_edge_mask(ctx.boundary.cpml.metallic_edges)),
        metric_kind=_metric_kind_code(ctx),
        cpml_enabled=np.int32(ctx.boundary.cpml.enabled),
        source_component=np.int32(_COMPONENT_CODE[source.component]),
        source_start_z=np.int32(source.slab_starts[0]),
        source_start_y=np.int32(source.slab_starts[1]),
        source_start_x=np.int32(source.slab_starts[2]),
        frequency_count=np.int32(monitor.freq_count),
        point_count=np.int32(monitor.dft_point_count),
    )
    next_state = (
        _replace_graph_outputs(state, outputs)
        if ctx.boundary.cpml.enabled
        else state._replace(
            hx=outputs[0],
            hy=outputs[1],
            hz=outputs[2],
            ex=outputs[3],
            ey=outputs[4],
            ez=outputs[5],
        )
    )
    return next_state._replace(
        dft_vec_re=outputs[state_output_count],
        dft_vec_im=outputs[state_output_count + 1],
        dft_weight_sum=outputs[state_output_count + 2],
    )


def run_steps(state, ctx, coeffs, nsteps: int) -> SimulationState:
    """Advance a source-free, monitor-free Yee run through one CUDA FFI call."""
    if nsteps < 1:
        raise ValueError("CUDA step count must be positive")
    fields = (state.hx, state.hy, state.hz, state.ex, state.ey, state.ez)
    if ctx.boundary.cpml.enabled:
        if not ctx.is_3d:
            raise ValueError("CUDA 2D multi-step execution does not yet support CPML")
        arguments, result_values, aliases = _cpml_graph_io(state, ctx, coeffs)
        call = jax.ffi.ffi_call(
            "beamz_cuda_streamed_cpml_steps",
            tuple(_shape(value) for value in result_values),
            input_output_aliases=aliases,
            vmap_method="sequential",
        )
        outputs = call(
            *arguments,
            abi_version=np.int32(CUDA_ABI_VERSION),
            nsteps=np.int32(nsteps),
            dt=np.float32(ctx.dt),
            resolution=np.float32(ctx.resolution),
            metallic_edges=np.int32(
                _metallic_edge_mask(ctx.boundary.cpml.metallic_edges)
            ),
            metric_kind=_metric_kind_code(ctx),
        )
        return _replace_graph_outputs(state, outputs)
    materials = (
        coeffs.h_decay_x,
        coeffs.h_decay_y,
        coeffs.h_decay_z,
        coeffs.h_source_x,
        coeffs.h_source_y,
        coeffs.h_source_z,
        coeffs.e_decay_x,
        coeffs.e_decay_y,
        coeffs.e_decay_z,
        coeffs.e_source_x,
        coeffs.e_source_y,
        coeffs.e_source_z,
    )
    target = (
        "beamz_cuda_streamed_steps"
        if ctx.is_3d
        else "beamz_cuda_streamed_2d_steps"
    )
    call = jax.ffi.ffi_call(
        target,
        tuple(_shape(value) for value in fields),
        input_output_aliases={index: index for index in range(6)},
        vmap_method="sequential",
    )
    attributes = {
        "abi_version": np.int32(CUDA_ABI_VERSION),
        "nsteps": np.int32(nsteps),
        "dt": np.float32(ctx.dt),
        "resolution": np.float32(ctx.resolution),
        "metallic_edges": np.int32(
            _metallic_edge_mask(ctx.boundary.cpml.metallic_edges)
        ),
        "metric_kind": _metric_kind_code(ctx),
    }
    if not ctx.is_3d:
        attributes["polarization"] = np.int32(
            0 if ctx.config.polarization_2d == "tm" else 1
        )
    outputs = call(
        *fields,
        *materials,
        *_phase_metrics(ctx, _PHASE_H),
        *_phase_metrics(ctx, _PHASE_E),
        **attributes,
    )
    return state._replace(
        hx=outputs[0],
        hy=outputs[1],
        hz=outputs[2],
        ex=outputs[3],
        ey=outputs[4],
        ez=outputs[5],
    )
