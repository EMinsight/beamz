from __future__ import annotations

from dataclasses import dataclass
from typing import TextIO

import numpy as np

from beamz.const import LIGHT_SPEED

_STATUS_MARKER = "● "


def get_si_scale_and_label(value):
    """Convert a value to appropriate SI unit and return scale factor and label."""
    if value >= 1e-3:
        return 1e3, "mm"
    elif value >= 1e-6:
        return 1e6, "µm"
    elif value >= 1e-9:
        return 1e9, "nm"
    else:
        return 1e12, "pm"


def check_fdtd_stability(dt, dx, dy=None, dz=None, n_max=1.0, safety_factor=1.0):
    """
    Check FDTD stability with the Courant-Friedrichs-Lewy (CFL) condition.

    Args:
        dt: Time step
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction (None for 1D)
        dz: Grid spacing in z direction (None for 1D/2D)
        n_max: Maximum refractive index in the simulation
        safety_factor: Factor to apply to the theoretical Courant limit (0-1).
                       Use 1.0 to evaluate against the theoretical limit 1/sqrt(dims).

    Returns:
        tuple: (is_stable, courant_number, max_allowed)
    """
    # Determine dimensionality
    dims = 1
    min_spacing = dx
    if dy is not None:
        dims = 2
        min_spacing = min(dx, dy)
    if dz is not None:
        dims = 3
        min_spacing = min(dx, dy, dz)
    # Courant number defined with vacuum speed (conservative and standard for Yee grid)
    c0 = LIGHT_SPEED
    courant = c0 * dt / min_spacing
    # Theoretical stability limit
    max_allowed = 1.0 / np.sqrt(dims)
    # Apply safety factor
    safe_limit = safety_factor * max_allowed
    return courant <= safe_limit, courant, safe_limit


def calc_optimal_fdtd_params(
    wavelength,
    n_max,
    dims=2,
    safety_factor=0.999,
    points_per_wavelength=10,
    width=None,
    height=None,
    depth=None,
):
    """
    Calculate optimal FDTD grid resolution and time step based on wavelength and material properties.

    Args:
        wavelength: Light wavelength in vacuum
        n_max: Maximum refractive index in the simulation
        dims: Dimensionality of simulation (1, 2, or 3)
        safety_factor: Fraction of the theoretical Courant limit to target (0-1).
                       0.95 operates close to the limit; reduce for additional margin.
        points_per_wavelength: Number of grid points per wavelength in the highest index material
        width, height, depth: Optional physical dimensions to estimate total grid size and performance

    Returns:
        tuple: (resolution, dt) - optimal spatial resolution and time step
    """
    # Calculate wavelength in the highest index material
    lambda_material = wavelength / n_max
    # Calculate optimal grid resolution based on desired points per wavelength
    resolution = lambda_material / points_per_wavelength
    # Calculate theoretical Courant limit (dt_max = dx / (c * sqrt(dims)))
    dt_max = resolution / (LIGHT_SPEED * np.sqrt(dims))
    # Apply safety factor (vacuum-based Courant condition)
    dt = safety_factor * dt_max

    # Grid size warning
    if width and height:
        nx = int(width / resolution)
        ny = int(height / resolution)
        nz = int(depth / resolution) if (dims == 3 and depth) else 1
        total_cells = nx * ny * nz

        if total_cells > 5e6:
            display_status(
                f"Warning: Large simulation grid detected ({total_cells / 1e6:.1f}M cells). "
                f"3D simulations can be slow. Consider reducing points_per_wavelength (current: {points_per_wavelength}) "
                f"if performance is an issue.",
                "warning",
            )

    # Verify stability
    try:
        _, courant, limit = check_fdtd_stability(
            dt,
            resolution,
            dy=resolution if dims >= 2 else None,
            dz=resolution if dims >= 3 else None,
            n_max=n_max,
            safety_factor=1.0,
        )
        assert courant <= limit + 1e-15, (
            "Internal error: calculated time step exceeds stability limit"
        )
    except Exception:
        pass

    return resolution, dt


def dxdt(
    wavelength,
    n_max=1.0,
    dims=2,
    safety_factor=0.999,
    points_per_wavelength=10,
    **kwargs,
):
    """Convenience alias returning (dx, dt) for FDTD setup."""
    return calc_optimal_fdtd_params(
        wavelength=wavelength,
        n_max=n_max,
        dims=dims,
        safety_factor=safety_factor,
        points_per_wavelength=points_per_wavelength,
        **kwargs,
    )


def display_status(status: str, status_type: str = "info") -> None:
    """Display a plain status message."""
    prefix_map = {
        "info": "Info",
        "success": "Done",
        "warning": "Warning",
        "error": "Error",
    }
    prefix = prefix_map.get(status_type, "Info")
    print(f"{_STATUS_MARKER}{prefix}: {status}")


@dataclass
class _PlainTask:
    description: str
    total: int | None
    completed: int = 0


class PlainProgress:
    """Minimal progress helper with no colors, spinners, bars, or styled markup."""

    def __init__(self, *, file: TextIO | None = None, inline: bool = True):
        self._file = file
        self._inline = inline
        self._tasks: dict[int, _PlainTask] = {}
        self._next_task_id = 0

    @property
    def _output(self) -> TextIO:
        import sys

        return self._file if self._file is not None else sys.stdout

    def __enter__(self) -> "PlainProgress":
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        status = "failed" if exc_type is not None else "done"
        for task in self._tasks.values():
            total = task.total
            if total is None:
                self._write_line(f"{task.description} {status}")
            else:
                completed = min(task.completed, total)
                self._write_line(f"{task.description} {status} ({completed}/{total})")

    def add_task(self, description: str, total: int | None = None) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        normalized_total = None if total is None else max(int(total), 0)
        self._tasks[task_id] = _PlainTask(
            description=description,
            total=normalized_total,
        )
        if normalized_total is None:
            self._write_line(description)
        else:
            self._write_progress(description, completed=0, total=normalized_total)
        return task_id

    def update(
        self, task_id: int, *, advance: int = 0, completed: int | None = None
    ) -> None:
        task = self._tasks[task_id]
        if completed is not None:
            task.completed = max(int(completed), 0)
        else:
            task.completed = max(task.completed + int(advance), 0)
        if task.total is not None:
            self._write_progress(
                task.description,
                completed=min(task.completed, task.total),
                total=task.total,
            )

    def _can_inline(self) -> bool:
        isatty = getattr(self._output, "isatty", None)
        return bool(self._inline and callable(isatty) and isatty())

    def _write_line(self, message: str) -> None:
        if self._can_inline():
            print(f"\r{_STATUS_MARKER}{message}", file=self._output, flush=True)
        else:
            print(f"{_STATUS_MARKER}{message}", file=self._output)

    def _write_progress(self, description: str, *, completed: int, total: int) -> None:
        message = _format_progress_message(
            completed,
            total,
            label=description.rstrip("."),
            unit="items",
        )
        if self._can_inline():
            print(f"\r{_STATUS_MARKER}{message}", end="", file=self._output, flush=True)
        elif completed == 0:
            print(f"{_STATUS_MARKER}{description}", file=self._output)


def _format_progress_message(
    completed: int,
    total: int,
    *,
    label: str = "Progress",
    unit: str = "steps",
) -> str:
    """Format a simple progress message."""
    safe_total = max(int(total), 1)
    safe_completed = min(max(int(completed), 0), safe_total)
    pct = 100.0 * safe_completed / safe_total
    return f"{label}: {pct:.0f}% ({safe_completed}/{safe_total} {unit})"


def _print_inline_progress(
    completed: int,
    total: int,
    *,
    label: str = "Progress",
    unit: str = "steps",
    file: TextIO | None = None,
) -> None:
    """Print a plain single-line progress update."""
    import sys

    output = file if file is not None else sys.stdout
    message = _format_progress_message(completed, total, label=label, unit=unit)
    print(f"\r{_STATUS_MARKER}{message}", end="", file=output, flush=True)


def _finish_inline_progress(*, file: TextIO | None = None) -> None:
    """Finish a plain inline progress line."""
    import sys

    output = file if file is not None else sys.stdout
    print(file=output, flush=True)


def create_plain_progress() -> PlainProgress:
    """Create a plain progress reporter for tracking processes."""
    return PlainProgress()
