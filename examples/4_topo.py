import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from beamz import *
from beamz.optimization import Optimizer


# General parameters for the simulation
W, H = 15 * µm, 15 * µm
WG_W = 0.5 * µm
N_CORE, N_CLAD = 2.25, 1.444
EPS_CORE, EPS_CLAD = N_CORE ** 2, N_CLAD ** 2
WL = 1.55 * µm
OPT_STEPS, LR = 4, 5e-3
TIME = 15 * WL / LIGHT_SPEED
DX, DT = calc_optimal_fdtd_params(
    WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.95, points_per_wavelength=10
)


def make_density_mask(structure, grid):
    """Create a boolean mask covering the cells that belong to ``structure``."""

    mask = np.zeros_like(grid.permittivity, dtype=bool)
    cell_dx = getattr(grid, "dx", grid.resolution)
    cell_dy = getattr(grid, "dy", grid.resolution)

    x_coords = (np.arange(mask.shape[1]) + 0.5) * cell_dx
    y_coords = (np.arange(mask.shape[0]) + 0.5) * cell_dy

    bbox = structure.get_bounding_box()
    if len(bbox) == 6:
        min_x, min_y, _, max_x, max_y, _ = bbox
    else:
        min_x, min_y, max_x, max_y = bbox

    x_indices = np.where((x_coords >= min_x) & (x_coords <= max_x))[0]
    y_indices = np.where((y_coords >= min_y) & (y_coords <= max_y))[0]

    for iy in y_indices:
        y = y_coords[iy]
        for ix in x_indices:
            x = x_coords[ix]
            if structure.point_in_polygon(x, y):
                mask[iy, ix] = True
    return mask


def box_blur(density, radius_cells):
    """Apply a simple box blur to ``density`` using an integer ``radius_cells``."""

    radius = int(max(0, round(radius_cells)))
    if radius < 1:
        return density.copy()

    size = 2 * radius + 1
    padded = np.pad(density, radius, mode="edge")
    window = sliding_window_view(padded, (size, size))
    return window.mean(axis=(-2, -1))


def smooth_projection(density, beta=2.0, eta=0.5):
    """Project the density through a smoothed Heaviside."""

    tanh = np.tanh
    return (
        tanh(beta * eta)
        + tanh(beta * (density - eta))
    ) / (
        tanh(beta * eta)
        + tanh(beta * (1.0 - eta))
    )


def filtered_density(density, mask, blur_radius, beta, eta):
    """Apply blur and projection filters inside the masked region."""

    filtered = density.copy()
    filtered_mask = box_blur(filtered, blur_radius)
    filtered_mask = smooth_projection(filtered_mask, beta=beta, eta=eta)
    filtered = np.where(mask, filtered_mask, filtered)
    return np.clip(filtered, 0.0, 1.0)


def density_to_permittivity(base_permittivity, density, mask, eps_min, eps_max):
    """Blend the base permittivity with density-driven updates inside ``mask``."""

    permittivity = base_permittivity.copy()
    permittivity[mask] = eps_min + density[mask] * (eps_max - eps_min)
    return permittivity


def preview_permittivity(permittivity_grid, dx, dy, title, filename=None):
    """Visualize the permittivity grid for debugging/inspection."""

    extent = (0, permittivity_grid.shape[1] * dx, 0, permittivity_grid.shape[0] * dy)
    plt.figure(figsize=(6, 5))
    plt.imshow(permittivity_grid, origin="lower", extent=extent, cmap="magma")
    plt.colorbar(label=r"$\epsilon_r$")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()


def extract_objective(results):
    """Return the first objective value stored in the simulation results."""

    objectives = results.get("objectives", {})
    if not objectives:
        return 0.0
    first_key = next(iter(objectives))
    return float(objectives[first_key])


def monitor_objective(monitor: Monitor, normalize: bool = True) -> float:
    """Objective that maximizes transmitted power at the monitor plane."""

    total_power = 0.0
    if monitor.power_history:
        history = np.asarray(monitor.power_history)
        total_power = float(np.sum(np.abs(history)))
        if normalize:
            total_power /= len(history)
    elif monitor.power_accumulated is not None and monitor.power_accumulation_count > 0:
        accumulated = np.asarray(monitor.power_accumulated)
        total_power = float(np.sum(np.abs(accumulated)))
        if normalize:
            total_power /= monitor.power_accumulation_count
    return -total_power


# Design with a custom inhomogeneous material which we will update during optimization
design = Design(width=W, height=H, pml_size=2 * µm)
design += Rectangle(
    position=(0 * µm, H / 2 - WG_W / 2),
    width=3.5 * µm,
    height=WG_W,
    material=Material(permittivity=EPS_CORE),
)
design += Rectangle(
    position=(W / 2 - WG_W / 2, H),
    width=WG_W,
    height=-3.5 * µm,
    material=Material(permittivity=EPS_CORE),
)

# Inverse design region
design_region = Rectangle(
    position=(W / 2 - 4 * µm, H / 2 - 4 * µm),
    width=8 * µm,
    height=8 * µm,
)
design += design_region

# Define the signal
t = np.arange(0, TIME, DT)
signal = ramped_cosine(
    t=t,
    amplitude=1.0,
    frequency=LIGHT_SPEED / WL,
    t_max=TIME,
    ramp_duration=5 * WL / LIGHT_SPEED,
    phase=0,
)

# Rasterize the initial design
grid = RegularGrid(design=design, resolution=DX)
base_permittivity = grid.permittivity.copy()

# Initialize design material with a grid definition matching the FDTD mesh
initial_material_grid = base_permittivity.copy()
design_material = CustomMaterial(
    permittivity_grid=initial_material_grid,
    bounds=((0, W), (0, H)),
)
design_region.material = design_material

# Mask and density initialization for topology updates
mask = make_density_mask(design_region, grid)
density = np.zeros_like(base_permittivity)
density[mask] = np.random.uniform(0.0, 1.0, size=np.count_nonzero(mask))


def update_design_from_density(density_values, label_prefix="iteration", preview=False, step=0):
    filtered = filtered_density(density_values, mask, blur_radius=WL / DX, beta=3.0, eta=0.5)
    permittivity_grid = density_to_permittivity(base_permittivity, filtered, mask, EPS_CLAD, EPS_CORE)
    design_material.update_grid("permittivity", permittivity_grid)
    grid.permittivity = permittivity_grid
    if preview:
        preview_permittivity(
            permittivity_grid,
            getattr(grid, "dx", DX),
            getattr(grid, "dy", DX),
            f"{label_prefix} {step}: permittivity",
        )
    return filtered, permittivity_grid


# Define the optimizer
optimizer = Optimizer(method="adam", learning_rate=LR)
objective_history = []


for opt_step in range(1, OPT_STEPS + 1):
    print(f"\n--- Optimization step {opt_step}/{OPT_STEPS} ---")

    density, permittivity_grid = update_design_from_density(
        density, label_prefix="Iteration", preview=True, step=opt_step
    )

    forward = FDTD(
        design=grid,
        devices=[
            ModeSource(
                design=design,
                position=(2.5 * µm, H / 2),
                width=WG_W * 4,
                wavelength=WL,
                signal=signal,
                direction="+x",
            ),
            Monitor(
                design=design,
                start=(W / 2 - WG_W * 2, H - 2.5 * µm),
                end=(W / 2 + WG_W * 2, H - 2.5 * µm),
                objective_function=monitor_objective,
                accumulate_power=True,
                name="output_monitor",
            ),
        ],
        time=t,
    )

    forward_results = forward.run(
        live=False,
        axis_scale=[-1, 1],
        save_memory_mode=True,
        accumulate_power=True,
        save_fields=["Ez"],
        fields_to_cache=["Ez"],
    )

    objective_value = extract_objective(forward_results)
    objective_history.append(objective_value)
    print(f"Objective value: {objective_value:.6e}")

    forward_fields = forward_results.get("Ez", [])
    num_forward_steps = len(forward_fields)

    adjoint = FDTD(
        design=grid,
        devices=[
            ModeSource(
                design=design,
                position=(W / 2, H - 2.5 * µm),
                width=WG_W * 4,
                wavelength=WL,
                signal=signal,
                direction="-y",
            )
        ],
        time=t,
    )

    adjoint.initialize_simulation(
        save=False,
        live=False,
        accumulate_power=False,
        save_memory_mode=True,
        fields_to_cache=None,
    )

    overlap_gradient = np.zeros_like(density, dtype=np.float64)
    dt = adjoint.dt

    for step_index in range(adjoint.num_steps):
        if not forward_fields:
            break
        if not adjoint.step():
            break
        adjoint_field = adjoint.backend.to_numpy(adjoint.Ez)
        forward_field = np.asarray(forward_fields.pop())
        overlap_gradient += np.real(adjoint_field * np.conj(forward_field)) * dt

    adjoint.finalize_simulation()
    forward_fields.clear()

    if num_forward_steps > 0:
        overlap_gradient /= num_forward_steps

    overlap_gradient *= mask
    grad_norm = np.max(np.abs(overlap_gradient)) or 1.0
    overlap_gradient /= grad_norm

    update = optimizer.step(overlap_gradient * objective_value)
    density = np.where(mask, np.clip(density + update, 0.0, 1.0), density)
    density[~mask] = 0.0


# Final design evaluation
density, permittivity_grid = update_design_from_density(
    density, label_prefix="Final", preview=True, step="final"
)

final_sim = FDTD(
    design=grid,
    devices=[
        ModeSource(
            design=design,
            position=(2.5 * µm, H / 2),
            width=WG_W * 4,
            wavelength=WL,
            signal=signal,
            direction="+x",
        ),
        Monitor(
            design=design,
            start=(W / 2 - WG_W * 2, H - 2.5 * µm),
            end=(W / 2 + WG_W * 2, H - 2.5 * µm),
            objective_function=monitor_objective,
            accumulate_power=True,
            name="output_monitor",
        ),
    ],
    time=t,
)

final_results = final_sim.run(
    live=False,
    axis_scale=[-1, 1],
    save_memory_mode=True,
    accumulate_power=True,
    save_fields=["Ez"],
)

final_objective = extract_objective(final_results)
print(f"Final objective value: {final_objective:.6e}")


# Save diagnostic images
permittivity_filename = "final_design_permittivity.png"
preview_permittivity(
    permittivity_grid,
    getattr(grid, "dx", DX),
    getattr(grid, "dy", DX),
    "Final permittivity distribution",
    filename=permittivity_filename,
)

plt.figure(figsize=(6, 4))
plt.plot(objective_history, marker="o")
plt.xlabel("Optimization step")
plt.ylabel("Objective value")
plt.title("Objective history")
plt.grid(True)
plt.tight_layout()
objective_history_filename = "objective_history.png"
plt.savefig(objective_history_filename, dpi=200)
plt.show()

print(f"Saved final permittivity map to {permittivity_filename}")
print(f"Saved objective history to {objective_history_filename}")
