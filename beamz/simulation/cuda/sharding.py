"""Explicit slab communication around local CUDA Yee phases.

The first supported domain is a uniform, all-PEC 3D grid without CPML.
Global masks remain in the JAX timestep; the native call sees a one-cell
halo and treats the partition axis as an interior axis. No native ABI change
is needed. Graph programs cannot be used here: every phase needs new halos.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from beamz.simulation import _cuda_abi as abi
from beamz.simulation.backend import CudaBackendUnavailable

_FACES = (("front", "back"), ("bottom", "top"), ("left", "right"))
_ALL_FACES = frozenset(face for pair in _FACES for face in pair)
_MESH_AXIS = "fdtd"


def _validate_devices(mesh):
    if any(device.platform != "gpu" for device in mesh.devices.flat):
        raise CudaBackendUnavailable("CUDA sharding requires a GPU device mesh")


def validate_sharded_config(config, boundary, plan):
    """Reject combinations whose local native boundary semantics are unsupported."""
    if (
        config.backend != "cuda_streamed"
        or not config.is_3d
        or config.metric_kind != "isotropic_uniform"
        or boundary.cpml.enabled
        or boundary.cpml.metallic_edges != _ALL_FACES
    ):
        raise CudaBackendUnavailable(
            "CUDA sharding currently requires cuda_streamed, a 3D isotropic "
            "uniform grid, and PEC on all faces without CPML. Use backend='jax' "
            "for other sharded configurations."
        )
    if plan is None or not plan.layout.enabled or plan.mesh is None:
        raise ValueError("CUDA sharding requires a compiled device layout")
    axis = plan.layout.axis
    if len({shape[axis] for shape in plan.layout.padded_shapes.values()}) != 1:
        raise ValueError("CUDA component partitions must share global interfaces")
    _validate_devices(plan.mesh)


def exchange_halos(value, *, axis, num_devices):
    """Attach adjacent one-cell faces inside a manual ``fdtd`` mesh.

    The outer faces are zero, never periodic. Input contains only owned cells;
    callers must discard halo outputs before assembling the global result.
    """
    low = jax.lax.slice_in_dim(value, 0, 1, axis=axis)
    high = jax.lax.slice_in_dim(
        value, value.shape[axis] - 1, value.shape[axis], axis=axis
    )
    from_low = jax.lax.ppermute(
        high, _MESH_AXIS, [(i, i + 1) for i in range(num_devices - 1)]
    )
    from_high = jax.lax.ppermute(
        low, _MESH_AXIS, [(i + 1, i) for i in range(num_devices - 1)]
    )
    return jnp.concatenate((from_low, value, from_high), axis=axis)


def _pad_local(value, axis):
    if value.ndim == 0:
        return value
    padding = [(0, 0)] * value.ndim
    padding[axis] = (1, 1)
    return jnp.pad(value, padding)


def _phase(state, ctx, coeffs, *, phase):
    from . import runtime

    plan = ctx.sharding_plan
    axis, count = plan.layout.axis, plan.layout.num_devices
    spec = P(*(_MESH_AXIS if i == axis else None for i in range(3)))
    prefix = "h" if phase == 0 else "e"
    targets = tuple(getattr(state, prefix + c) for c in "xyz")
    sources = tuple(getattr(state, ("e" if phase == 0 else "h") + c) for c in "xyz")
    materials = tuple(
        getattr(coeffs, f"{prefix}_{kind}_{c}")
        for kind in ("decay", "source")
        for c in "xyz"
    )
    material_specs = tuple(P() if value.ndim == 0 else spec for value in materials)
    edges = ctx.boundary.cpml.metallic_edges - frozenset(_FACES[axis])

    # Keep the compatibility import local: older supported JAX installations
    # expose shard_map through the experimental module.
    try:
        shard_map = jax.shard_map
    except AttributeError:
        from jax.experimental.shard_map import shard_map

    @partial(
        shard_map,
        mesh=plan.mesh,
        in_specs=((spec,) * 3, (spec,) * 3, material_specs),
        out_specs=(spec,) * 3,
    )
    def local_update(local_targets, local_sources, local_materials):
        outputs = runtime._ffi_phase(
            abi.CUDA_STREAMED_TARGET,
            phase,
            tuple(_pad_local(value, axis) for value in local_targets),
            tuple(
                exchange_halos(value, axis=axis, num_devices=count)
                for value in local_sources
            ),
            tuple(_pad_local(value, axis) for value in local_materials),
            (),
            (),
            runtime._phase_metrics(ctx, phase),
            metric_kind=0,
            dt=ctx.dt,
            resolution=ctx.resolution,
            cuda_flags=ctx.config.cuda_flags,
            metallic_edges=edges,
        )
        return tuple(
            jax.lax.slice_in_dim(value, 1, value.shape[axis] - 1, axis=axis)
            for value in outputs
        )

    outputs = local_update(targets, sources, materials)
    return state._replace(
        **dict(zip((prefix + c for c in "xyz"), outputs, strict=True))
    )


def select_sharded_kernel(ctx):
    from beamz.simulation.kernels import StepUpdateKernel

    validate_sharded_config(ctx.config, ctx.boundary, ctx.sharding_plan)
    return StepUpdateKernel(
        "cuda_streamed_sharded",
        partial(_phase, phase=0),
        partial(_phase, phase=1),
    )
