"""Whole-column two-step scheduling, derived from PR #308's temporal.fut.

The numerical operations remain in the CUDA Yee primitives. This module only
proves which cells can advance twice and which intermediate cells must exist.
Tiles are 16x32; eligibility and scratch allocation use 8x16 quadrants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamz.simulation.backend import CudaBackendUnavailable

TILE_Y, TILE_X = 16, 32
QUAD_Y, QUAD_X = TILE_Y // 2, TILE_X // 2


@dataclass(frozen=True)
class MstepPlan:
    shape: tuple[int, int, int]
    bounds: np.ndarray
    core_masks: np.ndarray
    slots: np.ndarray
    first_cells: np.ndarray
    edge_cells: np.ndarray
    scratch_cells: int
    core_cells: int

    def summary(self):
        return {
            "schedule": "cuda_mstep",
            "tile": (TILE_Y, TILE_X),
            "temporal_steps": 2,
            "core_fraction": self.core_cells / int(np.prod(self.shape)),
            "first_step_cells": self.first_cells.size,
            "edge_cells": self.edge_cells.size,
            "scratch_bytes": 6 * self.scratch_cells * 4,
            "plan_bytes": sum(
                a.nbytes
                for a in (
                    self.bounds,
                    self.core_masks,
                    self.slots,
                    self.first_cells,
                    self.edge_cells,
                )
            ),
        }


def build_mstep_plan(field_shapes, cpml, groups, monitor_indices=None):
    """Build static dependency queues without allocating a volume-sized mask.

    Every second-step edge cell needs first-step E at its positive neighbours;
    those E values need first-step H at their negative neighbours. A one-cell
    Chebyshev margin covers both, including curl cross-axis dependencies.
    Monitors additionally request their sample cells and negative neighbours.
    Source intersections exclude an entire column quadrant and its two-cell
    halo, so the tiled kernel never needs to inject a source.
    """
    shape = tuple(max(s[a] for s in field_shapes) for a in range(3))
    nz, ny, nx = shape
    if int(np.prod(shape)) >= np.iinfo(np.int32).max:
        raise CudaBackendUnavailable("cuda_mstep cell indices exceed int32")
    nq_y = (ny + QUAD_Y - 1) // QUAD_Y
    nq_x = (nx + QUAD_X - 1) // QUAD_X
    # Include padding quadrants so each tile always has four mask bits.
    nq_y += nq_y % 2
    nq_x += nq_x % 2
    low = np.ones(3, dtype=int)
    high = np.asarray(shape) - 3
    for term in (*cpml.h_terms, *cpml.e_terms):
        axis = int(term.axis)
        c = "xyz".index(term.component[1]) + (0 if term.component[0] == "H" else 3)
        low[axis] = max(low[axis], int(term.slab.low))
        if term.slab.high:
            high[axis] = min(
                high[axis], field_shapes[c][axis] - int(term.slab.high) - 1
            )
    zb, zt = int(low[0] + 2), int(high[0])
    qy = np.arange(nq_y)[:, None] * QUAD_Y
    qx = np.arange(nq_x)[None, :] * QUAD_X
    core = (qy - 2 >= low[1]) & (qy + QUAD_Y + 1 <= high[1])
    core = core & (qx - 2 >= low[2]) & (qx + QUAD_X + 1 <= high[2])
    if zt - zb < 4:
        core[:] = False
    for group in groups:
        if group is None:
            continue
        extent = np.asarray(group.coeffs.shape[1:])
        for start in np.asarray(group.starts):
            end = start + extent
            if np.any(end <= 0) or np.any(start >= shape):
                continue
            intersects = (qy - 2 < end[1]) & (qy + QUAD_Y + 2 > start[1])
            intersects = intersects & (qx - 2 < end[2]) & (qx + QUAD_X + 2 > start[2])
            core[intersects] = False
    masks = sum(
        core[dy::2, dx::2].astype(np.int32) << (2 * dy + dx)
        for dy in range(2)
        for dx in range(2)
    )
    edge_xy = ~np.repeat(np.repeat(core, QUAD_Y, axis=0), QUAD_X, axis=1)[:ny, :nx]
    padded = np.pad(edge_xy, 1, constant_values=True)
    first_xy = np.logical_or.reduce(
        [padded[dy : dy + ny, dx : dx + nx] for dy in range(3) for dx in range(3)]
    )
    watched = {}
    if monitor_indices is not None:
        # DFT plans use E,H order; native field banks use H,E order.
        for c, dims in enumerate((*field_shapes[3:], *field_shapes[:3])):
            indices = np.asarray(monitor_indices)[:, c].reshape(-1)
            indices = np.unique(indices[(indices >= 0) & (indices < np.prod(dims))])
            zz, rest = np.divmod(indices, dims[1] * dims[2])
            yy, xx = np.divmod(rest, dims[2])
            for dz, dy, dx in ((0, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, -1)):
                for z, y, x in zip(zz + dz, yy + dy, xx + dx, strict=True):
                    if 0 <= z < nz and 0 <= y < ny and 0 <= x < nx:
                        watched.setdefault(int(z), []).append(int(y * nx + x))
    slots = np.full((nz, nq_y, nq_x), -1, dtype=np.int32)
    first, edge = [], []
    all_xy = np.arange(ny * nx, dtype=np.int32)
    edge_rows = np.flatnonzero(edge_xy).astype(np.int32)
    slot_count = 0
    for z in range(nz):
        is_core_z = zb <= z < zt
        edge.append((edge_rows if is_core_z else all_xy) + z * ny * nx)
        mask = first_xy.copy() if zb + 1 <= z < zt - 1 else np.ones((ny, nx), bool)
        if z in watched:
            mask.reshape(-1)[watched[z]] = True
        rows = np.flatnonzero(mask).astype(np.int32)
        first.append(rows + z * ny * nx)
        yy, xx = np.divmod(rows, nx)
        occupied = np.unique((yy // QUAD_Y) * nq_x + xx // QUAD_X)
        slots[z].reshape(-1)[occupied] = np.arange(
            slot_count, slot_count + occupied.size
        )
        slot_count += occupied.size
    # Missing intermediate cells map to one initialized spare quadrant. Values
    # from it may feed discarded edge calculations, never an owned result.
    slots[slots < 0] = slot_count
    if (slot_count + 1) * QUAD_Y * QUAD_X >= np.iinfo(np.int32).max:
        raise CudaBackendUnavailable("cuda_mstep scratch offsets exceed int32")
    return MstepPlan(
        shape,
        np.asarray((zb, zt), np.int32),
        masks,
        slots,
        np.concatenate(first),
        np.concatenate(edge),
        (slot_count + 1) * QUAD_Y * QUAD_X,
        int(core.sum()) * QUAD_Y * QUAD_X * max(0, zt - zb),
    )


def run_mstep(
    state,
    ctx,
    coeffs,
    groups,
    packed_monitors,
    plan,
    nsteps,
    *,
    observation_origin,
    observation_step_offset,
):
    """Lower the shared CUDA program with compact intermediate field storage."""
    import jax.numpy as jnp

    from beamz.simulation import _cuda_abi as abi
    from beamz.simulation.cuda import runtime as rt

    fields = rt._fields(state)
    if any(f.dtype != jnp.float32 for f in fields):
        raise CudaBackendUnavailable("cuda_mstep requires float32 fields")
    arguments, values, aliases = rt._graph_io(state, ctx, coeffs)
    state_count = len(values)
    final = tuple(jnp.empty_like(f) for f in fields)
    scratch = tuple(jnp.zeros((plan.scratch_cells,), jnp.float32) for _ in fields)
    for value in (*final, *scratch):
        aliases[len(arguments)] = len(values)
        arguments = (*arguments, value)
        values = (*values, value)
    arguments = (*arguments, *rt._source_group_arguments(groups))
    if packed_monitors is None:
        packed_monitors = (
            jnp.zeros((0, 6, 1, 1), jnp.int32),
            jnp.zeros((0, 6, 1, 1), jnp.float32),
            jnp.zeros((0, 1), jnp.float32),
            jnp.zeros((0, 6), jnp.float32),
            jnp.zeros((0, 5), jnp.int32),
            jnp.zeros((0, 2), jnp.int32),
            jnp.zeros((0, 3), jnp.float32),
        )
        dft = (jnp.zeros((0,), jnp.float32),) * 3
    else:
        dft = (state.dft_vec_re, state.dft_vec_im, state.dft_weight_sum)
    arguments = (*arguments, *packed_monitors)
    monitor_start = len(values)
    count, frequencies = packed_monitors[2].shape
    phase = tuple(
        jnp.empty((max(1, count), frequencies), jnp.float32) for _ in range(3)
    )
    for value in (*dft, *phase):
        aliases[len(arguments)] = len(values)
        arguments = (*arguments, value)
        values = (*values, value)
    arguments = (
        *arguments,
        observation_origin,
        state.current_step,
        jnp.asarray(observation_step_offset, jnp.int32),
        *(
            jnp.asarray(a)
            for a in (
                plan.bounds,
                plan.core_masks,
                plan.slots,
                plan.first_cells,
                plan.edge_cells,
            )
        ),
    )
    flags = abi.NATIVE_SCHEDULE_SOURCES
    if ctx.boundary.cpml.enabled:
        flags |= abi.NATIVE_SCHEDULE_CPML
    if rt._uniform_cpml_thickness(ctx) > 0:
        flags |= abi.NATIVE_SCHEDULE_UNIFORM_CPML
    if rt._packed_e_material(coeffs):
        flags |= abi.NATIVE_SCHEDULE_PACKED_MATERIAL
    if count:
        flags |= abi.NATIVE_SCHEDULE_MONITORS
    if ctx.config.cuda_flags & abi.CUDA_GRAPH_CACHE:
        flags |= abi.NATIVE_SCHEDULE_GRAPH_CACHE
    schedule = rt.NativeSchedulePlan(
        layout=abi.PROGRAM_LAYOUT_YEE_IN_PLACE,
        flags=flags,
        monitor_count=count,
        coincident_source_group_mask=rt._coincident_source_group_mask(groups),
        disjoint_source_group_mask=rt._disjoint_source_group_mask(groups),
        temporal_steps=2,
    )
    outputs = rt._ffi_call(abi.CUDA_MSTEP_TARGET, values, aliases)(
        *arguments, **rt._program_attributes(ctx, nsteps, schedule)
    )
    bank = state_count if (nsteps // 2) % 2 else 0
    result = rt._replace_fields(state, outputs, bank)
    if ctx.boundary.cpml.enabled:
        result = result._replace(
            cpml_psi_h_terms=outputs[6:12], cpml_psi_e_terms=outputs[12:18]
        )
    if count:
        result = result._replace(
            dft_vec_re=outputs[monitor_start],
            dft_vec_im=outputs[monitor_start + 1],
            dft_weight_sum=outputs[monitor_start + 2],
        )
    return result
