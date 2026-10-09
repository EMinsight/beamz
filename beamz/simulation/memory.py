"""Capacity-aware CUDA scheduling, independent of input-buffer ownership.

This is a conservative workspace estimate, not an allocator reservation. A
choice is frozen when a scan is built; it does not change inside a timestep.
"""

from __future__ import annotations

import math
import os

import jax

from beamz.simulation import _cuda_abi as abi


def cuda_memory_policy_from_env() -> str:
    policy = os.environ.get("BEAMZ_CUDA_MEMORY_POLICY", "auto").strip().lower()
    if policy not in {"auto", "speed", "capacity"}:
        raise ValueError("BEAMZ_CUDA_MEMORY_POLICY must be auto, speed, or capacity")
    return policy


def cuda_workspace_estimate(
    program, *, donate_state: bool, mstep_plan=None, low_memory=False
) -> int:
    """Estimate optional fast-schedule workspace on the busiest device.

    Include both field banks for pair kernels, rotated materials for layout
    transforms, replicated normal CPML slabs, and retained input ownership.
    This intentionally overestimates rather than promising an exact peak.
    """
    layout = program.sharding.layout
    count = layout.num_devices if layout.enabled else 1
    fields = (
        sum(math.prod(shape) * 4 for shape in layout.padded_shapes.values()) // count
    )
    psi = sum(
        math.prod(term.slab.shape)
        * 4
        // (count if layout.enabled and term.axis != layout.axis else 1)
        for term in (*program.boundary.cpml.h_terms, *program.boundary.cpml.e_terms)
    )
    state = fields + psi
    # Frozen-input streamed graphs may use an alternate bank even without the
    # temporal-pair flag. Pair graphs can use a third bank. Count those possible
    # banks conservatively; capacity mode uses only the in-place output state.
    bank_count = (
        1
        if low_memory
        else 3
        if program.config.cuda_flags & abi.CUDA_TEMPORAL_PAIR
        else 2
    )
    if mstep_plan is not None:
        # Output state, alternate bank, compact intermediate fields and static
        # dependency arrays. Retained caller state is accounted for below.
        summary = mstep_plan.summary()
        workspace = state + fields + summary["scratch_bytes"] + summary["plan_bytes"]
        workspace += sum(
            int(m.freq_count) * int(m.dft_point_count) * 6 * 2 * 4
            for m in program.monitors
        )
    else:
        workspace = bank_count * state
    if (layout.enabled and layout.axis == 2) or program.config.cuda_storage_axes != (
        0,
        1,
        2,
    ):
        materials = (
            sum(value.size * value.dtype.itemsize for value in program.coefficients)
            // count
        )
        workspace += 2 * state + materials
    if not donate_state:
        workspace += state
    return workspace


def cuda_capacity_schedule(program, *, donate_state: bool) -> bool:
    """Keep fast kernels when they fit; use existing in-place kernels otherwise."""
    if program.config.backend != "cuda_streamed":
        return False
    policy = program.config.cuda_memory_policy
    if policy != "auto":
        return policy == "capacity"
    available = cuda_available_bytes(program)
    # CPU contract tests and older backends may not expose allocator statistics.
    # Explicit capacity mode is available without querying or initializing a GPU.
    if available is None:
        return False
    return (
        cuda_workspace_estimate(program, donate_state=donate_state) + (64 << 20)
        > available
    )


def cuda_available_bytes(program) -> int | None:
    """Return the smallest remaining CUDA allocator budget, when available."""
    mesh = program.sharding.mesh
    devices = tuple(mesh.devices.flat) if mesh is not None else tuple(jax.devices())
    budgets = []
    for device in devices:
        if device.platform != "gpu":
            continue
        stats = device.memory_stats()
        if stats and "bytes_limit" in stats and "bytes_in_use" in stats:
            budgets.append(int(stats["bytes_limit"]) - int(stats["bytes_in_use"]))
    return min(budgets) if budgets else None


def check_mstep_workspace(program, plan, *, donate_state):
    """Check a forced temporal workspace before lowering its allocations."""
    from beamz.simulation.backend import CudaBackendUnavailable

    required = cuda_workspace_estimate(
        program, donate_state=donate_state, mstep_plan=plan
    )
    available = cuda_available_bytes(program)
    if program.config.cuda_memory_policy == "capacity":
        raise CudaBackendUnavailable(
            "schedule='mstep' requires additional field workspace; "
            "memory policy 'capacity' requires schedule='streamed' or 'auto'"
        )
    if available is not None and required + (64 << 20) > available:
        raise CudaBackendUnavailable(
            f"schedule='mstep' estimates {required / 2**20:.0f} MiB of workspace "
            f"plus 64 MiB headroom, but only {available / 2**20:.0f} MiB remains; "
            "use schedule='streamed' or 'auto', or reduce the simulation size"
        )
    return required, available
