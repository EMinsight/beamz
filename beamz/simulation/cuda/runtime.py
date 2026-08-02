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
_COMPONENT_CODE.update(
    {name: index for index, name in enumerate(("Ex", "Ey", "Ez"))}
)
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
        raise ValueError("CUDA CPML terms and recurrence buffers must have equal length")
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
    aliases.update(
        {psi_start + index: 3 + index for index in range(len(psi_terms))}
    )
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
    target = "beamz_cuda_hopper" if ctx.config.backend == "cuda_hopper" else "beamz_cuda_streamed"
    terms = ctx.boundary.cpml.h_terms
    outputs = _ffi_phase(
        target,
        _PHASE_H,
        (state.hx, state.hy, state.hz),
        (state.ex, state.ey, state.ez),
        (
            coeffs.h_sigma_m_x,
            coeffs.h_sigma_m_y,
            coeffs.h_sigma_m_z,
            _EMPTY,
            _EMPTY,
            _EMPTY,
        ),
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
    target = "beamz_cuda_hopper" if ctx.config.backend == "cuda_hopper" else "beamz_cuda_streamed"
    terms = ctx.boundary.cpml.e_terms
    outputs = _ffi_phase(
        target,
        _PHASE_E,
        (state.ex, state.ey, state.ez),
        (state.hx, state.hy, state.hz),
        (
            coeffs.e_conductivity_x,
            coeffs.e_conductivity_y,
            coeffs.e_conductivity_z,
            coeffs.e_permittivity_x,
            coeffs.e_permittivity_y,
            coeffs.e_permittivity_z,
        ),
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

