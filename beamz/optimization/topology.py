"""Topology optimization helper placeholders for BEAMZ.

This module contains the high-level hooks that the application layer calls
while keeping the actual implementation details encapsulated.  Each helper
currently raises ``NotImplementedError`` because the adjoint workflow is not
implemented yet.  Replace the bodies with working code when the optimization
pipeline is ready.
"""

from __future__ import annotations

import numpy as _np

from .optimizers import Optimizer as _Optimizer

DensityArray = _np.ndarray


def initialize_density_from_region(design_region, resolution: float) -> DensityArray:
    """Return a normalized density grid covering ``design_region``."""

    width = getattr(design_region, "width", None)
    height = getattr(design_region, "height", None)
    if width is None or height is None:
        raise ValueError("Design region must expose width and height attributes.")

    nx = max(1, int(round(width / resolution)))
    ny = max(1, int(round(height / resolution)))

    density = _np.full((ny, nx), 0.5, dtype=float)
    material = getattr(design_region, "material", None)
    if material is not None:
        grid = getattr(material, "permittivity_grid", None)
        if grid is not None:
            density = _resample_to_shape(grid, (ny, nx))
    return density


def blur_density(density: DensityArray, radius: float, *, backend: str = "numpy") -> DensityArray:
    """Apply a separable box blur with a configurable radius."""

    radius_cells = max(0, int(round(radius)))
    if radius_cells <= 0:
        return density

    kernel = _np.ones((2 * radius_cells + 1,), dtype=float)
    kernel /= kernel.size

    blurred = _np.apply_along_axis(lambda row: _np.convolve(row, kernel, mode="same"), axis=1, arr=density)
    blurred = _np.apply_along_axis(lambda col: _np.convolve(col, kernel, mode="same"), axis=0, arr=blurred)
    return blurred


def project_density(density: DensityArray, beta: float, eta: float) -> DensityArray:
    """Sigmoid projection toward binary values."""

    if beta <= 0:
        return density

    exp_term = _np.exp(-beta * (density - eta))
    projected = 1.0 / (1.0 + exp_term)
    return projected


def compute_overlap_gradient(forward_fields, adjoint_fields, *, monitor=None) -> DensityArray:
    """Compute overlap gradient using Ez components as a placeholder."""

    forward_ez = _extract_field(forward_fields, "Ez")
    adjoint_ez = _extract_field(adjoint_fields, "Ez")

    if forward_ez is None or adjoint_ez is None:
        raise ValueError("Forward/adjoint data must contain an 'Ez' entry.")

    gradient = _np.real(_np.conj(adjoint_ez) * forward_ez)

    min_shape = tuple(min(f, a) for f, a in zip(forward_ez.shape, adjoint_ez.shape))
    slices = tuple(slice(0, n) for n in min_shape)
    gradient = gradient[slices]
    return gradient


def run_forward(sim, source, monitor, *, live: bool = False):
    """Run a forward simulation with optional live visualization."""

    sim.sources = [source]
    sim.initialize_simulation(save=True, live=live)
    while sim.step():
        pass
    return sim.finalize_simulation()


def run_adjoint(
    density,
    sim,
    source,
    forward_results,
    monitor=None,
    *,
    live: bool = False,
) -> DensityArray:
    """Placeholder adjoint run returning a zero gradient."""

    sim.sources = [source]
    sim.initialize_simulation(save=True, live=live)
    while sim.step():
        pass
    sim.finalize_simulation()
    return _np.zeros_like(density, dtype=float)


def compute_objective(results, monitor=None) -> float:
    """Simple objective: total Ez energy of the final field snapshot."""

    ez = _extract_field(results, "Ez")
    if ez is None:
        return 0.0
    return float(_np.sum(_np.abs(ez) ** 2))


def apply_optimizer_step(
    density: DensityArray,
    gradient: DensityArray,
    optimizer_state: dict | None,
    *,
    method: str = "adam",
    learning_rate: float = 1e-2,
    **kwargs,
) -> tuple[DensityArray, dict | None]:
    """Update density using the generic :class:`Optimizer` helper."""

    if gradient.shape != density.shape:
        raise ValueError("Gradient and density must share shape.")

    if optimizer_state is None:
        optimizer = _Optimizer(method=method, learning_rate=learning_rate, options=kwargs)
    else:
        optimizer = optimizer_state["optimizer"]

    update = optimizer.step(gradient)
    density_updated = density + update
    density_clamped = _np.clip(density_updated, 0.0, 1.0)

    return density_clamped, {"optimizer": optimizer}


def update_design_region_material(design_region, density: DensityArray) -> None:
    """Push the projected density to the region's CustomMaterial grid."""

    material = getattr(design_region, "material", None)
    if material is None or not hasattr(material, "update_grid"):
        return

    permittivity = _np.interp(density, [0.0, 1.0], [_background_eps(), _target_eps()])
    material.update_grid("permittivity", permittivity)


def _normalize_to_unit_interval(values: _np.ndarray) -> _np.ndarray:
    v_min = values.min()
    v_max = values.max()
    if _np.isclose(v_min, v_max):
        return _np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


def _resample_to_shape(grid: _np.ndarray, shape: tuple[int, int]) -> _np.ndarray:
    ny, nx = shape
    y_src = _np.linspace(0.0, 1.0, grid.shape[0])
    y_dst = _np.linspace(0.0, 1.0, ny)
    x_src = _np.linspace(0.0, 1.0, grid.shape[1])
    x_dst = _np.linspace(0.0, 1.0, nx)

    temp = _np.zeros((ny, grid.shape[1]), dtype=float)
    for col in range(grid.shape[1]):
        temp[:, col] = _np.interp(y_dst, y_src, grid[:, col])

    resampled = _np.zeros((ny, nx), dtype=float)
    for row in range(ny):
        resampled[row, :] = _np.interp(x_dst, x_src, temp[row, :])

    return _normalize_to_unit_interval(resampled)


def _background_eps() -> float:
    # Placeholder background permittivity (cladding)
    return 1.444 ** 2


def _target_eps() -> float:
    # Placeholder target permittivity (core)
    return 2.25 ** 2


def _extract_field(container, key: str):
    """Helper to pull Ez-like arrays irrespective of container format."""

    if container is None:
        return None

    if isinstance(container, dict):
        value = container.get(key)
        if isinstance(value, list) and value:
            return value[-1]
        return value

    return getattr(container, key, None)


