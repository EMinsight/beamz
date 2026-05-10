"""Utilities for Beamz Yee-lattice coordinates."""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp
import numpy as np


def _component_axis_offsets_3d(component: str) -> dict[str, float]:
    """Return physical Yee offsets, in grid-cell units, for a 3D component."""
    if component == "Ex":
        return {"z": 0.0, "y": 0.0, "x": 0.5}
    if component == "Ey":
        return {"z": 0.0, "y": 0.5, "x": 0.0}
    if component == "Ez":
        return {"z": 0.5, "y": 0.0, "x": 0.0}
    if component == "Hx":
        return {"z": 0.5, "y": 0.5, "x": 0.0}
    if component == "Hy":
        return {"z": 0.5, "y": 0.0, "x": 0.5}
    if component == "Hz":
        return {"z": 0.0, "y": 0.5, "x": 0.5}
    raise ValueError(f"Unsupported component {component!r}")


def component_axis_offsets_3d(component: str) -> dict[str, float]:
    """Return the physical Yee offsets, in grid-cell units, for a 3D component."""
    return dict(_component_axis_offsets_3d(component))


def component_shape_3d(
    component: str, grid_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Return the stored 3D Beamz field shape for a Yee component."""
    nz, ny, nx = (int(v) for v in grid_shape)
    if component == "Ex":
        return (nz, ny, nx - 1)
    if component == "Ey":
        return (nz, ny - 1, nx)
    if component == "Ez":
        return (nz - 1, ny, nx)
    if component == "Hx":
        return (nz - 1, ny - 1, nx)
    if component == "Hy":
        return (nz - 1, ny, nx - 1)
    if component == "Hz":
        return (nz, ny - 1, nx - 1)
    raise ValueError(f"Unsupported component {component!r}")


def component_coordinates_3d_um(
    component: str,
    grid_shape: tuple[int, int, int],
    dx_um: float,
) -> dict[str, np.ndarray]:
    """Return Beamz raw Yee sample coordinates for a stored 3D component.

    Beamz stores only the owned interior samples for the shortened Yee axis.
    Physical coordinates follow the standard Yee lattice: each component is
    offset by half a cell only along its own axis (for E) or the two transverse
    axes (for H).
    """

    shape = component_shape_3d(component, grid_shape)
    offsets = _component_axis_offsets_3d(component)
    return {
        "z": (np.arange(shape[0], dtype=np.float64) + offsets["z"]) * dx_um,
        "y": (np.arange(shape[1], dtype=np.float64) + offsets["y"]) * dx_um,
        "x": (np.arange(shape[2], dtype=np.float64) + offsets["x"]) * dx_um,
    }


def component_coordinates_3d_um_serializable(
    component: str,
    grid_shape: tuple[int, int, int],
    dx_um: float,
) -> dict[str, list[float]]:
    coords = component_coordinates_3d_um(component, grid_shape, dx_um)
    return {axis: values.tolist() for axis, values in coords.items()}


def component_index_arrays_3d(
    component: str,
    grid_shape: tuple[int, int, int],
    *,
    stored_shape: tuple[int, int, int] | None = None,
    region: tuple[slice, slice, slice] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return voxel indices sampled by a 3D Yee component on a cell-centered raster.

    Beamz's rasterizer stores material properties at cell centers. To evaluate a
    piecewise-constant voxel grid at a Yee location, we sample the voxel that owns
    that physical point, i.e. ``floor(index + offset)`` in grid-cell units.
    """

    shape = (
        tuple(int(v) for v in stored_shape)
        if stored_shape is not None
        else component_shape_3d(component, grid_shape)
    )
    offsets = _component_axis_offsets_3d(component)
    axes = ("z", "y", "x")
    region = region or (slice(None), slice(None), slice(None))

    indices: list[np.ndarray] = []
    for axis, dim, grid_dim, axis_region in zip(
        axes, shape, grid_shape, region, strict=False
    ):
        coord = np.arange(dim, dtype=np.float64) + offsets[axis]
        idx = np.floor(coord).astype(np.int32)
        idx = np.clip(idx, 0, int(grid_dim) - 1)
        indices.append(idx[axis_region])

    return tuple(indices)  # type: ignore[return-value]


def sample_voxel_grid_at_component_3d(
    grid,
    component: str,
    *,
    stored_shape: tuple[int, int, int] | None = None,
    region: tuple[slice, slice, slice] | None = None,
):
    """Sample a cell-centered 3D raster on a Yee component lattice.

    Staggered E components use symmetric centered sampling. H components retain
    owner-cell sampling because their material terms are already collocated with
    the magnetic lattice convention used by the update.
    """

    if component in {"Ex", "Ey", "Ez"}:
        return sample_voxel_grid_at_e_component_3d_centered(
            grid,
            component,
            stored_shape=stored_shape,
            region=region,
        )

    z_idx, y_idx, x_idx = component_index_arrays_3d(
        component,
        tuple(int(v) for v in np.asarray(grid).shape),
        stored_shape=stored_shape,
        region=region,
    )
    sampled = jnp.asarray(grid)
    sampled = jnp.take(sampled, jnp.asarray(z_idx), axis=0)
    sampled = jnp.take(sampled, jnp.asarray(y_idx), axis=1)
    sampled = jnp.take(sampled, jnp.asarray(x_idx), axis=2)
    return sampled


def sample_voxel_grid_at_e_component_3d_centered(
    grid,
    component: str,
    *,
    stored_shape: tuple[int, int, int] | None = None,
    region: tuple[slice, slice, slice] | None = None,
):
    """Sample a cell-centered 3D raster onto a staggered E site by symmetric averaging.

    The underlying raster stores cell-centered material values. For staggered E-field
    sites, owner-cell sampling introduces a directional bias because the physical Yee
    location sits halfway between neighboring cell centers along the component axis.
    This helper preserves the existing compact scalar material model while removing
    that low-side bias by averaging the two adjacent cell-centered samples only along
    the staggered E axis.
    """

    if component not in {"Ex", "Ey", "Ez"}:
        raise ValueError(
            "sample_voxel_grid_at_e_component_3d_centered only supports Ex/Ey/Ez"
        )

    grid_shape = tuple(int(v) for v in np.asarray(grid).shape)
    shape = (
        tuple(int(v) for v in stored_shape)
        if stored_shape is not None
        else component_shape_3d(component, grid_shape)
    )
    offsets = _component_axis_offsets_3d(component)
    axes = ("z", "y", "x")
    stagger_axis = component[-1].lower()
    region = region or (slice(None), slice(None), slice(None))

    sampled_lo = jnp.asarray(grid)
    sampled_hi = jnp.asarray(grid)

    for axis_index, (axis, dim, grid_dim, axis_region) in enumerate(
        zip(axes, shape, grid_shape, region, strict=False)
    ):
        coord = np.arange(dim, dtype=np.float64) + offsets[axis]
        lo = np.floor(coord).astype(np.int32)
        lo = np.clip(lo, 0, int(grid_dim) - 1)
        if axis == stagger_axis:
            hi = np.clip(lo + 1, 0, int(grid_dim) - 1)
        else:
            hi = lo

        lo = lo[axis_region]
        hi = hi[axis_region]
        sampled_lo = jnp.take(sampled_lo, jnp.asarray(lo), axis=axis_index)
        sampled_hi = jnp.take(sampled_hi, jnp.asarray(hi), axis=axis_index)

    return 0.5 * (sampled_lo + sampled_hi)


def component_shape_2d(
    component: str,
    grid_shape: tuple[int, int],
    plane: str,
) -> tuple[int, int]:
    """Return the canonical 2D Beamz field shape for a component."""

    dim0, dim1 = (int(v) for v in grid_shape)
    if plane == "xy":
        mapping = {
            "Ex": (dim0, dim1 - 1),
            "Ey": (dim0 - 1, dim1),
            "Ez": (dim0 + 1, dim1 + 1),
            "Hx": (dim0, dim1 + 1),
            "Hy": (dim0 + 1, dim1),
            "Hz": (dim0 - 1, dim1 - 1),
        }
    elif plane == "yz":
        mapping = {
            "Ex": (dim0, dim1),
            "Ey": (dim0, dim1 - 1),
            "Ez": (dim0 - 1, dim1),
            "Hx": (dim0 - 1, dim1 - 1),
            "Hy": (dim0, dim1 - 1),
            "Hz": (dim0 - 1, dim1),
        }
    elif plane == "xz":
        mapping = {
            "Ex": (dim0, dim1 - 1),
            "Ey": (dim0, dim1),
            "Ez": (dim0 - 1, dim1),
            "Hx": (dim0, dim1 - 1),
            "Hy": (dim0 - 1, dim1 - 1),
            "Hz": (dim0 - 1, dim1),
        }
    else:
        raise ValueError(f"Unsupported plane {plane!r}")

    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(f"Unsupported component {component!r}") from exc


def _component_axis_offsets_2d(
    component: str, plane: str
) -> tuple[tuple[str, str], dict[str, float]]:
    """Return canonical 2D Yee offsets for stored component samples."""

    if plane == "xy":
        offsets = {
            "Ex": {"y": 0.5, "x": 0.0},
            "Ey": {"y": 0.0, "x": 0.5},
            "Ez": {"y": 0.0, "x": 0.0},
            "Hx": {"y": 0.5, "x": 0.0},
            "Hy": {"y": 0.0, "x": 0.5},
            "Hz": {"y": 0.0, "x": 0.0},
        }
        axes = ("y", "x")
    elif plane == "yz":
        offsets = {
            "Ex": {"z": 0.5, "y": 0.5},
            "Ey": {"z": 0.5, "y": 0.0},
            "Ez": {"z": 0.0, "y": 0.5},
            "Hx": {"z": 0.0, "y": 0.0},
            "Hy": {"z": 0.5, "y": 0.0},
            "Hz": {"z": 0.0, "y": 0.5},
        }
        axes = ("z", "y")
    elif plane == "xz":
        offsets = {
            "Ex": {"z": 0.5, "x": 0.0},
            "Ey": {"z": 0.5, "x": 0.5},
            "Ez": {"z": 0.0, "x": 0.5},
            "Hx": {"z": 0.5, "x": 0.0},
            "Hy": {"z": 0.0, "x": 0.0},
            "Hz": {"z": 0.0, "x": 0.5},
        }
        axes = ("z", "x")
    else:
        raise ValueError(f"Unsupported plane {plane!r}")

    try:
        return axes, offsets[component]
    except KeyError as exc:
        raise ValueError(f"Unsupported component {component!r}") from exc


def component_coordinates_2d_um(
    component: str,
    grid_shape: tuple[int, int],
    dx_um: float,
    plane: str,
) -> dict[str, np.ndarray]:
    """Return canonical 2D component coordinates."""

    if plane == "xy" and component in {"Ez", "Hx", "Hy"}:
        return tm_xy_full_component_coordinates_2d_um(component, grid_shape, dx_um)

    shape = component_shape_2d(component, grid_shape, plane)
    axes, offsets = _component_axis_offsets_2d(component, plane)
    return {
        axis: (np.arange(length, dtype=np.float64) + offsets[axis]) * dx_um
        for axis, length in zip(axes, shape, strict=True)
    }


def sample_voxel_grid_at_component_2d(
    grid,
    component: str,
    plane: str,
    *,
    stored_shape: tuple[int, int] | None = None,
    region: tuple[slice, slice] | None = None,
):
    """Sample a cell-centered 2D raster at canonical 2D component locations."""

    if plane == "xy" and component in {"Ez", "Hx", "Hy"}:
        sampled = sample_voxel_grid_at_tm_xy_full_component_2d(grid, component)
        canonical_shape = tm_xy_full_component_shape_2d(
            component, tuple(int(v) for v in np.asarray(grid).shape)
        )
        if stored_shape is not None and tuple(int(v) for v in stored_shape) != canonical_shape:
            raise ValueError(
                f"stored_shape={stored_shape!r} does not match canonical xy TM "
                f"shape {canonical_shape!r} for component {component!r}"
            )
        region = region or (slice(None), slice(None))
        return sampled[region]

    grid_np = np.asarray(grid)
    shape = (
        tuple(int(v) for v in stored_shape)
        if stored_shape is not None
        else component_shape_2d(component, tuple(int(v) for v in grid_np.shape), plane)
    )
    region = region or (slice(None), slice(None))

    axes, offsets = _component_axis_offsets_2d(component, plane)
    index_axes = []
    for axis, dim, grid_dim, axis_region in zip(
        axes,
        shape,
        grid_np.shape,
        region,
        strict=False,
    ):
        coord = np.arange(dim, dtype=np.float64) + offsets[axis]
        idx = np.floor(coord).astype(np.int32)
        idx = np.clip(idx, 0, int(grid_dim) - 1)
        index_axes.append(idx[axis_region])

    dim0_idx, dim1_idx = index_axes
    sampled = jnp.asarray(grid)
    sampled = jnp.take(sampled, jnp.asarray(dim0_idx), axis=0)
    sampled = jnp.take(sampled, jnp.asarray(dim1_idx), axis=1)
    return sampled


def tm_xy_full_component_shape_2d(
    component: str,
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    """Return the native 2D TMz Yee shape for an xy-plane component.

    The input ``grid_shape`` is the cell-centered material raster shape
    ``(ny, nx)``. Meep's native TMz lattice stores:

    - ``Ez`` on the integer grid nodes: ``(ny + 1, nx + 1)``
    - ``Hx`` on the x-normal faces / y-staggered edges: ``(ny, nx + 1)``
    - ``Hy`` on the y-normal faces / x-staggered edges: ``(ny + 1, nx)``
    """

    ny, nx = (int(v) for v in grid_shape)
    mapping = {
        "Ez": (ny + 1, nx + 1),
        "Hx": (ny, nx + 1),
        "Hy": (ny + 1, nx),
    }
    try:
        return mapping[component]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported TMxy full-state component {component!r}"
        ) from exc


def tm_xy_full_component_coordinates_2d_um(
    component: str,
    grid_shape: tuple[int, int],
    dx_um: float,
) -> dict[str, np.ndarray]:
    """Return native 2D TMz Yee coordinates for xy-plane Ez/Hx/Hy."""

    ny, nx = (int(v) for v in grid_shape)
    if component == "Ez":
        return {
            "y": np.arange(ny + 1, dtype=np.float64) * dx_um,
            "x": np.arange(nx + 1, dtype=np.float64) * dx_um,
        }
    if component == "Hx":
        return {
            "y": (np.arange(ny, dtype=np.float64) + 0.5) * dx_um,
            "x": np.arange(nx + 1, dtype=np.float64) * dx_um,
        }
    if component == "Hy":
        return {
            "y": np.arange(ny + 1, dtype=np.float64) * dx_um,
            "x": (np.arange(nx, dtype=np.float64) + 0.5) * dx_um,
        }
    raise ValueError(f"Unsupported TMxy full-state component {component!r}")


def sample_voxel_grid_at_tm_xy_full_component_2d(
    grid,
    component: str,
):
    """Sample a cell-centered 2D raster at native 2D TMz Yee locations.

    The material raster owns cell centers. Native TMz field samples that lie on
    integer-aligned nodes/edges are therefore gathered from the low-side voxel,
    with the high boundary clipped to the last cell, matching the benchmark's
    Meep ``epsilon_func`` ownership convention.
    """

    grid_np = np.asarray(grid)
    ny, nx = (int(v) for v in grid_np.shape)

    if component == "Ez":
        y_idx = np.clip(
            np.floor(np.arange(ny + 1, dtype=np.float64)).astype(np.int32),
            0,
            ny - 1,
        )
        x_idx = np.clip(
            np.floor(np.arange(nx + 1, dtype=np.float64)).astype(np.int32),
            0,
            nx - 1,
        )
    elif component == "Hx":
        y_idx = np.clip(
            np.floor(np.arange(ny, dtype=np.float64) + 0.5).astype(np.int32),
            0,
            ny - 1,
        )
        x_idx = np.clip(
            np.floor(np.arange(nx + 1, dtype=np.float64)).astype(np.int32),
            0,
            nx - 1,
        )
    elif component == "Hy":
        y_idx = np.clip(
            np.floor(np.arange(ny + 1, dtype=np.float64)).astype(np.int32),
            0,
            ny - 1,
        )
        x_idx = np.clip(
            np.floor(np.arange(nx, dtype=np.float64) + 0.5).astype(np.int32),
            0,
            nx - 1,
        )
    else:
        raise ValueError(f"Unsupported TMxy full-state component {component!r}")

    sampled = jnp.asarray(grid)
    sampled = jnp.take(sampled, jnp.asarray(y_idx), axis=0)
    sampled = jnp.take(sampled, jnp.asarray(x_idx), axis=1)
    return sampled


def nearest_support_indices_3d(
    coords_zyx_um: Mapping[str, np.ndarray],
    center_um: tuple[float, float, float],
    width_um: float,
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """Gaussian support weights on an existing 3D Yee component lattice."""
    x0, y0, z0 = (float(v) for v in center_um)
    sigma = float(width_um)
    dx_um = float(np.median(np.diff(np.asarray(coords_zyx_um["x"], dtype=np.float64))))
    radius_cells = int(np.ceil(4.0 * sigma / dx_um))

    x = np.asarray(coords_zyx_um["x"], dtype=np.float64)
    y = np.asarray(coords_zyx_um["y"], dtype=np.float64)
    z = np.asarray(coords_zyx_um["z"], dtype=np.float64)

    cx = int(np.argmin(np.abs(x - x0)))
    cy = int(np.argmin(np.abs(y - y0)))
    cz = int(np.argmin(np.abs(z - z0)))

    x_start, x_end = max(0, cx - radius_cells), min(x.size, cx + radius_cells + 1)
    y_start, y_end = max(0, cy - radius_cells), min(y.size, cy + radius_cells + 1)
    z_start, z_end = max(0, cz - radius_cells), min(z.size, cz + radius_cells + 1)

    zz, yy, xx = np.meshgrid(
        z[z_start:z_end], y[y_start:y_end], x[x_start:x_end], indexing="ij"
    )
    dist_sq = (xx - x0) ** 2 + (yy - y0) ** 2 + (zz - z0) ** 2
    weights = np.exp(-dist_sq / (2.0 * sigma**2))

    voxels: list[tuple[int, int, int]] = []
    for iz in range(z_start, z_end):
        for iy in range(y_start, y_end):
            for ix in range(x_start, x_end):
                voxels.append((iz, iy, ix))
    return voxels, np.asarray(weights, dtype=np.float64).reshape(-1)
