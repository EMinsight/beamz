"""Topology optimization helper placeholders for BEAMZ.

This module contains the high-level hooks that the application layer calls
while keeping the actual implementation details encapsulated.  Each helper
currently raises ``NotImplementedError`` because the adjoint workflow is not
implemented yet.  Replace the bodies with working code when the optimization
pipeline is ready.
"""

from __future__ import annotations

import typing as _t

import numpy as _np

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
            density[:] = _normalize_to_unit_interval(grid)
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


def apply_optimizer_step(
    density: DensityArray,
    gradient: DensityArray,
    optimizer_state: dict | None,
    *,
    method: str = "adam",
    learning_rate: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> tuple[DensityArray, dict | None]:
    """Update the density using a simple Adam optimizer."""

    if gradient.shape != density.shape:
        raise ValueError("Gradient and density must share shape.")

    if optimizer_state is None:
        optimizer_state = {"m": _np.zeros_like(density), "v": _np.zeros_like(density), "t": 0}

    m = optimizer_state["m"]
    v = optimizer_state["v"]
    optimizer_state["t"] += 1
    t = optimizer_state["t"]

    m[:] = beta1 * m + (1 - beta1) * gradient
    v[:] = beta2 * v + (1 - beta2) * (gradient ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    density_updated = density + learning_rate * m_hat / (_np.sqrt(v_hat) + epsilon)
    density_clamped = _np.clip(density_updated, 0.0, 1.0)

    return density_clamped, optimizer_state


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


