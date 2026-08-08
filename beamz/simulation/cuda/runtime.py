"""Lower fused 3D Yee phases to the optional BeamZ CUDA typed FFI targets."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from beamz.simulation.model import SimulationState

CUDA_ABI_VERSION = 1
_PHASE_H = 0
_PHASE_E = 1
_COMPONENT_CODE = {name: index for index, name in enumerate(("Hx", "Hy", "Hz"))}
_COMPONENT_CODE.update({name: index for index, name in enumerate(("Ex", "Ey", "Ez"))})
_EMPTY = jnp.empty((0,), dtype=jnp.float32)


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


def _ffi_phase(
    target: str,
    phase: int,
    targets,
    sources,
    materials,
    terms,
    psi_terms,
    *,
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
    arguments = (*fields, *h_payload, *e_payload)
    result_values = (*fields, *state.cpml_psi_h_terms, *state.cpml_psi_e_terms)
    aliases = {index: index for index in range(6)}
    aliases.update({31 + index: 6 + index for index in range(6)})
    aliases.update({62 + index: 12 + index for index in range(6)})
    return arguments, result_values, aliases


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
    """Advance one packed pre-E source and CPML through one CUDA graph call."""
    if nsteps < 1:
        raise ValueError("CUDA step count must be positive")
    if not ctx.boundary.cpml.enabled:
        raise ValueError("CUDA source graph requires CPML")
    if (
        source.timing != "pre_e"
        or source.component not in {"Ex", "Ey", "Ez"}
        or not source.is_slab
        or source.slab_starts is None
        or source.slab_sizes is None
        or tuple(source.coeff.shape) != tuple(source.slab_sizes)
    ):
        raise ValueError("CUDA source graph requires one packed pre-E slab source")
    arguments, result_values, aliases = _cpml_graph_io(state, ctx, coeffs)
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
        source_component=np.int32(_COMPONENT_CODE[source.component]),
        source_start_z=np.int32(source.slab_starts[0]),
        source_start_y=np.int32(source.slab_starts[1]),
        source_start_x=np.int32(source.slab_starts[2]),
    )
    return _replace_graph_outputs(state, outputs)


def run_steps(state, ctx, coeffs, nsteps: int) -> SimulationState:
    """Advance a source-free, monitor-free Yee run through one CUDA FFI call."""
    if nsteps < 1:
        raise ValueError("CUDA step count must be positive")
    fields = (state.hx, state.hy, state.hz, state.ex, state.ey, state.ez)
    if ctx.boundary.cpml.enabled:
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
    call = jax.ffi.ffi_call(
        "beamz_cuda_streamed_steps",
        tuple(_shape(value) for value in fields),
        input_output_aliases={index: index for index in range(6)},
        vmap_method="sequential",
    )
    outputs = call(
        *fields,
        *materials,
        abi_version=np.int32(CUDA_ABI_VERSION),
        nsteps=np.int32(nsteps),
        dt=np.float32(ctx.dt),
        resolution=np.float32(ctx.resolution),
        metallic_edges=np.int32(_metallic_edge_mask(ctx.boundary.cpml.metallic_edges)),
    )
    return state._replace(
        hx=outputs[0],
        hy=outputs[1],
        hz=outputs[2],
        ex=outputs[3],
        ey=outputs[4],
        ez=outputs[5],
    )
