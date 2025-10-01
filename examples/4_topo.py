import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
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
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.95, points_per_wavelength=10)

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

def box_blur(density, radius_cells, mask=None):
    """Apply a simple box blur to ``density`` and respect ``mask`` if provided."""


    radius = int(max(0, round(radius_cells)))
    if radius < 1:
        return density.copy()

    size = 2 * radius + 1
    padded = np.pad(density, radius, mode="edge")
    window = sliding_window_view(padded, (size, size))

    if mask is None:
        return window.mean(axis=(-2, -1))

    padded_mask = np.pad(mask.astype(float), radius, mode="constant", constant_values=0.0)
    mask_window = sliding_window_view(padded_mask, (size, size))
    weighted_sum = (window * mask_window).sum(axis=(-2, -1))
    weights = mask_window.sum(axis=(-2, -1))
    weights = np.where(weights == 0.0, 1.0, weights)
    return weighted_sum / weights

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

    blurred = box_blur(density, blur_radius, mask=mask)
    projected = smooth_projection(blurred, beta=beta, eta=eta)
    filtered = np.where(mask, projected, density)
    return np.clip(filtered, 0.0, 1.0)

def density_to_permittivity(base_permittivity, density, mask, eps_min, eps_max):
    """Blend the base permittivity with density-driven updates inside ``mask``."""

    permittivity = base_permittivity.copy()
    permittivity[mask] = eps_min + density[mask] * (eps_max - eps_min)
    return permittivity

def preview_permittivity(permittivity_grid,dx,dy,title,filename=None,*,show=True,cmap="magma",
    vmin=None, vmax=None, highlight_mask=None):
    """Visualize the permittivity grid for debugging/inspection."""

    extent = (0, permittivity_grid.shape[1] * dx, 0, permittivity_grid.shape[0] * dy)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(permittivity_grid, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=r"$\epsilon_r$")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    if highlight_mask is not None:
        plt.contour(np.where(highlight_mask, 1.0, 0.0), levels=[0.5], colors=["cyan"], linewidths=0.8, 
            extent=extent, origin="lower")
    if filename: plt.savefig(filename, dpi=200, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()

def plot_density_map(density, dx,dy,title,filename=None,*,show=True,cmap="viridis",vmin=0.0,vmax=1.0,highlight_mask=None):
    """Plot an optimization density grid (0-1) with optional highlighting."""

    extent = (0, density.shape[1] * dx, 0, density.shape[0] * dy)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(density,origin="lower",extent=extent,cmap=cmap,vmin=vmin,vmax=vmax)
    plt.colorbar(im, label="Density")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    if highlight_mask is not None:
        plt.contour(highlight_mask.astype(float),levels=[0.5],colors=["cyan"],linewidths=0.8,extent=extent,origin="lower")
    if filename: plt.savefig(filename, dpi=200, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()

def plot_permittivity_delta(permittivity_grid,reference_grid,dx,dy,title,filename=None,*,show=True,highlight_mask=None):
    """Plot the difference between current and reference permittivity."""
    delta = permittivity_grid - reference_grid
    vmax = np.max(np.abs(delta)) or 1.0
    extent = (0, permittivity_grid.shape[1] * dx, 0, permittivity_grid.shape[0] * dy)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(delta,origin="lower",extent=extent,cmap="RdBu_r",vmin=-vmax,vmax=vmax)
    plt.colorbar(im, label=r"$\Delta \epsilon_r$")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    if highlight_mask is not None:
        plt.contour(np.where(highlight_mask, 1.0, 0.0),levels=[0.5],colors=["cyan"],linewidths=0.8,extent=extent,origin="lower")
    if filename: plt.savefig(filename, dpi=200, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()

def preview_permittivity_zoom(permittivity_grid, mask, dx, dy, title, filename=None, *, show=True, cmap="magma", 
    vmin=None, vmax=None):
    """Plot the permittivity within the masked design region."""

    indices = np.argwhere(mask)
    if indices.size == 0: return

    margin = 2
    min_y, min_x = indices.min(axis=0)
    max_y, max_x = indices.max(axis=0)
    min_y = max(min_y - margin, 0)
    max_y = min(max_y + margin, permittivity_grid.shape[0] - 1)
    min_x = max(min_x - margin, 0)
    max_x = min(max_x + margin, permittivity_grid.shape[1] - 1)

    subset = permittivity_grid[min_y : max_y + 1, min_x : max_x + 1]
    extent = (min_x * dx, (max_x + 1) * dx, min_y * dy, (max_y + 1) * dy)

    plt.figure(figsize=(5, 4))
    im = plt.imshow(subset, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=r"$\epsilon_r$")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    if filename: plt.savefig(filename, dpi=200, bbox_inches="tight")
    if show: plt.show()
    else: plt.close()

def plot_field_map(field, dx, dy, title, filename=None, cmap="RdBu", use_abs=True,
    colorbar_label="Amplitude", show=True):
    """Plot a 2D field map with the given ``title`` and optional ``filename``."""
    data = np.abs(field) if use_abs else field
    extent = (0, data.shape[1] * dx, 0, data.shape[0] * dy)
    plt.figure(figsize=(6, 5))
    im = plt.imshow(data, origin="lower", extent=extent, cmap=cmap)
    plt.colorbar(im, label=colorbar_label)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    if filename:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

def extract_objective(results):
    """Return the first objective value stored in the simulation results."""
    objectives = results.get("objectives", {})
    if not objectives: return 0.0
    first_key = next(iter(objectives))
    return float(objectives[first_key])

def build_forward_devices(accumulate_power: bool = True):
    """Create source and monitor devices for forward simulations."""
    return [ModeSource(design=design, position=(2.5 * µm, H / 2), width=WG_W * 4, wavelength=WL, signal=signal, direction="+x"),
            Monitor(design=design, start=(W / 2 - WG_W * 2, H - 2.5 * µm), end=(W / 2 + WG_W * 2, H - 2.5 * µm),
            objective_function=monitor_objective, accumulate_power=accumulate_power, name="output_monitor")]

def run_forward_simulation(cache_fields: bool = True):
    """Run a forward simulation with optional field caching for adjoint reuse."""
    devices = build_forward_devices(accumulate_power=True)
    forward = FDTD(design=grid, devices=devices, time=t)
    run_kwargs = dict(live=False, axis_scale=[-1, 1], save_memory_mode=True, accumulate_power=True)
    if cache_fields: run_kwargs.update(save_fields=["Ez"], fields_to_cache=["Ez"])
    else: run_kwargs.update(save_fields=None, fields_to_cache=None)
    return forward.run(**run_kwargs)

def monitor_objective(monitor: Monitor) -> float:
    """Objective that minimizes negative transmitted power using time averaging."""

    if not monitor.power_history or not monitor.power_timestamps: return 0.0

    times = np.asarray(monitor.power_timestamps, dtype=float)
    powers = np.asarray(monitor.power_history)
    if np.iscomplexobj(powers): powers = np.abs(powers)
    powers = powers.astype(float, copy=False)
    if times.size < 2: mean_power = float(np.mean(powers))
    else:
        order = np.argsort(times)
        times = times[order]
        powers = powers[order]
        duration = times[-1] - times[0]
        if duration <= 0.0: mean_power = float(np.mean(powers))
        else:
            if hasattr(np, "trapezoid"): integrated = np.trapezoid(powers, times)
            else: integrated = np.trapz(powers, times)
            mean_power = float(integrated / duration)
    return -mean_power


# Design with an inhomogeneous region that will be updated via the rasterized grid
design = Design(width=W, height=H, pml_size=2 * µm)
design += Rectangle(position=(0 * µm, H / 2 - WG_W / 2), width=3.5 * µm, height=WG_W, material=Material(permittivity=EPS_CORE))
design += Rectangle(position=(W / 2 - WG_W / 2, H), width=WG_W, height=-3.5 * µm, material=Material(permittivity=EPS_CORE))
# Inverse design region
design_region = Rectangle(position=(W / 2 - 4 * µm, H / 2 - 4 * µm), width=8 * µm, height=8 * µm,
    material=Material(permittivity=EPS_CLAD))
design += design_region

# Define the signal
t = np.arange(0, TIME, DT)
signal = ramped_cosine(t=t, amplitude=1.0, frequency=LIGHT_SPEED / WL, t_max=TIME, ramp_duration=5 * WL / LIGHT_SPEED, phase=0)

# Rasterize the initial design
grid = RegularGrid(design=design, resolution=DX)
base_permittivity = grid.permittivity.copy()

# Mask and density initialization for topology updates
mask = make_density_mask(design_region, grid)

rng = np.random.default_rng(seed=42)
design_density = np.zeros_like(base_permittivity)
initial_permittivity = rng.uniform(EPS_CLAD, EPS_CORE, size=np.count_nonzero(mask))
design_density[mask] = (initial_permittivity - EPS_CLAD) / (EPS_CORE - EPS_CLAD)

def update_design_from_density(density_values, label_prefix="iteration", preview=False, step=0,
    blur_radius_cells=2, beta=4.0, eta=0.5):
    filtered = filtered_density(density_values, mask, blur_radius=blur_radius_cells, beta=beta, eta=eta)

    permittivity_grid = density_to_permittivity(base_permittivity, filtered, mask, EPS_CLAD, EPS_CORE)

    if hasattr(grid, "permittivity"):
        if grid.permittivity.shape == permittivity_grid.shape:
            np.copyto(grid.permittivity, permittivity_grid)
        else:
            grid.permittivity = permittivity_grid.copy()
    else:
        grid.permittivity = permittivity_grid.copy()

    if preview:
        preview_permittivity(permittivity_grid, getattr(grid, "dx", DX), getattr(grid, "dy", DX),
            f"{label_prefix} {step}: permittivity", highlight_mask=mask, vmin=EPS_CLAD, vmax=EPS_CORE, show=False)
    return filtered, permittivity_grid

# Define the optimizer
optimizer = Optimizer(method="adam", learning_rate=LR)
objective_history = []
for opt_step in range(1, OPT_STEPS + 1):
    print(f"\n--- Optimization step {opt_step}/{OPT_STEPS} ---")

    filtered_density_values, permittivity_grid = update_design_from_density(
        design_density, label_prefix="Iteration", preview=True, step=opt_step)
    baseline_permittivity = permittivity_grid.copy()

    forward_results = run_forward_simulation(cache_fields=True)
    current_objective = extract_objective(forward_results)
    print(f"Objective value (negative transmission): {current_objective:.6e}")
    print(f"Transmission estimate: {-current_objective:.6e}")

    forward_fields = list(forward_results.get("Ez", []))
    num_forward_steps = len(forward_fields)
    forward_snapshot = forward_fields[-1].copy() if forward_fields else None

    adjoint = FDTD(design=grid, devices=[ModeSource(design=design, position=(W / 2, H - 2.5 * µm), width=WG_W * 4,
        wavelength=WL, signal=signal, direction="-y")], time=t)

    adjoint.initialize_simulation(save=False, live=False, accumulate_power=False, save_memory_mode=True, fields_to_cache=None)
    overlap_gradient = np.zeros_like(design_density, dtype=np.float64)
    adjoint_snapshot = None

    for step_index in range(adjoint.num_steps):
        if not forward_fields: break
        if not adjoint.step(): break
        adjoint_field = adjoint.backend.to_numpy(adjoint.Ez)
        forward_field = np.asarray(forward_fields.pop())
        if adjoint_snapshot is None: adjoint_snapshot = adjoint_field.copy()
        overlap_gradient += np.real(adjoint_field * np.conj(forward_field))

    adjoint.finalize_simulation()
    forward_fields.clear()

    overlap_gradient *= mask
    grad_norm = np.max(np.abs(overlap_gradient)) or 1.0
    normalized_gradient = overlap_gradient / grad_norm

    density_gradient = np.where(mask, -normalized_gradient, 0.0)
    density_gradient = box_blur(density_gradient, 1, mask=mask)
    density_gradient = np.where(mask, density_gradient, 0.0)

    update_step = optimizer.step(density_gradient)
    update_step[~mask] = 0.0

    print(
        f"Gradient max: {np.max(np.abs(density_gradient)):.3e}, "
        f"update max: {np.max(np.abs(update_step)):.3e}"
    )

    design_density = np.clip(design_density + update_step, 0.0, 1.0)
    design_density[~mask] = 0.0

    masked_density = design_density[mask]
    if masked_density.size:
        print(
            "Design density stats after update: "
            f"min {masked_density.min():.3e}, max {masked_density.max():.3e}, "
            f"mean {masked_density.mean():.3e}"
        )

    filtered_density_post, updated_permittivity_grid = update_design_from_density(design_density,
        label_prefix="Post-update", preview=False, step=opt_step)

    delta_masked = (updated_permittivity_grid - baseline_permittivity)[mask]
    change_norm = np.linalg.norm(delta_masked)
    print(f"Permittivity grid update norm inside mask: {change_norm:.6e}")
    if change_norm == 0.0:
        print("Warning: No change detected in design region permittivity after update.")

    updated_filename = f"updated_permittivity_step{opt_step}.png"
    preview_permittivity(updated_permittivity_grid, getattr(grid, "dx", DX), getattr(grid, "dy", DX),
        f"Post-update permittivity, step {opt_step}", filename=updated_filename, show=False, highlight_mask=mask,
        vmin=EPS_CLAD, vmax=EPS_CORE)
    print(f"Saved updated design permittivity to {updated_filename}")

    density_snapshot_filename = f"design_density_step{opt_step}.png"
    plot_density_map(design_density, getattr(grid, "dx", DX), getattr(grid, "dy", DX), f"Design density, step {opt_step}",
        filename=density_snapshot_filename, show=False, highlight_mask=mask)
    print(f"Saved design density map to {density_snapshot_filename}")

    filtered_snapshot_filename = f"filtered_density_step{opt_step}.png"
    plot_density_map(filtered_density_post, getattr(grid, "dx", DX), getattr(grid, "dy", DX),
        f"Filtered density, step {opt_step}", filename=filtered_snapshot_filename, show=False, highlight_mask=mask)
    print(f"Saved filtered density map to {filtered_snapshot_filename}")

    zoom_filename = f"permittivity_zoom_step{opt_step}.png"
    preview_permittivity_zoom(updated_permittivity_grid, mask, getattr(grid, "dx", DX), getattr(grid, "dy", DX),
        f"Design region permittivity, step {opt_step}", filename=zoom_filename, show=False, vmin=EPS_CLAD, vmax=EPS_CORE)
    print(f"Saved zoomed permittivity to {zoom_filename}")

    delta_filename = f"permittivity_delta_step{opt_step}.png"
    plot_permittivity_delta(updated_permittivity_grid, base_permittivity, getattr(grid, "dx", DX), getattr(grid, "dy", DX),
        f"Permittivity delta, step {opt_step}", filename=delta_filename, show=False, highlight_mask=mask)
    print(f"Saved permittivity delta map to {delta_filename}")

    np.save(f"design_density_step{opt_step}.npy", design_density)
    np.save(f"filtered_density_step{opt_step}.npy", filtered_density_post)
    np.save(f"permittivity_step{opt_step}.npy", updated_permittivity_grid)

    objective_history.append(current_objective)
    print(
        "Updated design density using Adam optimizer. "
        f"Stored objective {current_objective:.6e} (transmission {-current_objective:.6e})."
    )

    if forward_snapshot is not None:
        plot_field_map(forward_snapshot, getattr(grid, "dx", DX), getattr(grid, "dy", DX),
            f"Forward |Ez|, step {opt_step}", filename=f"forward_field_step{opt_step}.png", show=False)
    if adjoint_snapshot is not None:
        plot_field_map(adjoint_snapshot, getattr(grid, "dx", DX), getattr(grid, "dy", DX),
            f"Adjoint |Ez|, step {opt_step}", filename=f"adjoint_field_step{opt_step}.png", show=False)
    plot_field_map(normalized_gradient, getattr(grid, "dx", DX), getattr(grid, "dy", DX),
        f"Overlap gradient, step {opt_step}", filename=f"overlap_gradient_step{opt_step}.png",
        cmap="magma", use_abs=False, colorbar_label="Gradient (a.u.)", show=False)


# Final design evaluation
final_density, permittivity_grid = update_design_from_density(design_density, label_prefix="Final", preview=True, step="final")
final_results = run_forward_simulation(cache_fields=True)
final_objective = extract_objective(final_results)
print(f"Final objective value: {final_objective:.6e}")
print(f"Final transmission estimate: {-final_objective:.6e}")

# Save diagnostic images
permittivity_filename = "final_design_permittivity.png"
preview_permittivity(permittivity_grid, getattr(grid, "dx", DX), getattr(grid, "dy", DX),
    "Final permittivity distribution", filename=permittivity_filename, show=False)

plt.figure(figsize=(6, 4))

transmission_history = -np.asarray(objective_history)
plt.plot(transmission_history, marker="o")
plt.xlabel("Optimization step")
plt.ylabel("Transmission (a.u.)")
plt.title("Transmission history")

plt.grid(True)
plt.tight_layout()
objective_history_filename = "objective_history.png"
plt.savefig(objective_history_filename, dpi=200)
plt.close()

print(f"Saved final permittivity map to {permittivity_filename}")
print(f"Saved objective history to {objective_history_filename}")
