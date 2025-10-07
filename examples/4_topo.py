import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

plt.switch_backend("Agg")
from beamz import *
from beamz.optimization.topology import project_density
from beamz.optimization.optimizers import Optimizer
from beamz.devices.mode import slab_mode_source

# Parameters
W = H = 15*µm
WG_W = 0.5*µm
N_CORE, N_CLAD = 2.25, 1.444
EPS_CORE, EPS_CLAD = N_CORE**2, N_CLAD**2
WL = 1.55*µm
STEPS, LR = 30, 0.07
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE,N_CLAD), dims=2, safety_factor=0.99, points_per_wavelength=10)
TIME = 25*WL/LIGHT_SPEED
t = np.arange(0, TIME, DT)

FILTER_RADIUS = 0.15*µm
PROJECTION_BETA = 8.0
PROJECTION_ETA = 0.5
TRANSMISSION_WEIGHT = 0.5
MODE_WEIGHT = 0.5

# Create the design
signal = ramped_cosine(t=t,amplitude=1.0, frequency=LIGHT_SPEED/WL, t_max=TIME, ramp_duration=5*WL/LIGHT_SPEED, phase=0)
design = Design(width=W, height=H, pml_size=2*µm)
design += Rectangle(position=(0*µm,H/2-WG_W/2), width=3.5*µm, height=WG_W, material=Material(permittivity=EPS_CORE))
design += Rectangle(position=(W/2-WG_W/2,H), width=WG_W, height=-3.5*µm, material=Material(permittivity=EPS_CORE))
region = Rectangle(position=(W/2-4*µm,H/2-4*µm), width=8*µm, height=8*µm, material=Material(permittivity=EPS_CLAD))
design += region
design.show()

# Rasterize the design
grid = RegularGrid(design=design, resolution=DX)

# Mask and density initialization for topology updates
base = grid.permittivity.copy()
mask = np.zeros_like(base,bool)
dx, dy = getattr(grid,"dx",DX), getattr(grid,"dy",DX)
min_cell = float(min(dx, dy))
filter_radius_cells = max(1, int(round(FILTER_RADIUS / min_cell)))
xs, ys = (np.arange(mask.shape[1])+0.5)*dx, (np.arange(mask.shape[0])+0.5)*dy
minx, miny, _, maxx, maxy, _ = region.get_bounding_box()
mask[(ys[:,None]>=miny)&(ys[:,None]<=maxy)&(xs[None,:]>=minx)&(xs[None,:]<=maxx)] = True

def masked_box_blur(values, mask, radius):
    """Return masked box blur and the corresponding weight map."""

    radius = int(max(0, radius))
    masked_values = np.where(mask, values, 0.0)

    if radius <= 0:
        weights = np.where(mask, 1.0, 1.0)
        return masked_values, weights

    padded_values = np.pad(masked_values, radius, mode="edge")
    padded_mask = np.pad(mask.astype(float), radius, mode="constant", constant_values=0.0)
    window_shape = (2 * radius + 1, 2 * radius + 1)

    values_view = sliding_window_view(padded_values, window_shape)
    mask_view = sliding_window_view(padded_mask, window_shape)

    weighted_sum = (values_view * mask_view).sum(axis=(-2, -1))
    weights = mask_view.sum(axis=(-2, -1))
    weights = np.where(weights == 0.0, 1.0, weights)

    blurred = weighted_sum / weights
    blurred = np.where(mask, blurred, 0.0)
    weights = np.where(mask, weights, 1.0)
    return blurred, weights


def masked_box_blur_backprop(grad_output, mask, weights, radius):
    """Propagate gradients through a masked box blur."""

    radius = int(max(0, radius))
    grad_output = np.where(mask, grad_output, 0.0)
    weights = np.where(weights == 0.0, 1.0, weights)
    contributions = grad_output / weights

    if radius <= 0:
        return np.where(mask, contributions, 0.0)

    padded = np.pad(contributions, radius, mode="constant", constant_values=0.0)
    window_shape = (2 * radius + 1, 2 * radius + 1)
    contrib_view = sliding_window_view(padded, window_shape)
    grad_input = contrib_view.sum(axis=(-2, -1))
    grad_input = np.where(mask, grad_input, 0.0)
    return grad_input


def apply_density_filters(values):
    """Apply masked blur followed by projection to obtain physical densities."""

    blurred, weights = masked_box_blur(values, mask, filter_radius_cells)
    projected = project_density(blurred, beta=PROJECTION_BETA, eta=PROJECTION_ETA)
    projected = np.where(mask, projected, 0.0)
    return blurred, projected, weights


def build_output_monitor(design):
    """Create an output monitor with objective favoring transmission and mode shape."""

    monitor_start = (W/2 - WG_W*2, H - 2.5*µm)
    monitor_end = (W/2 + WG_W*2, H - 2.5*µm)
    monitor = Monitor(
        design=design,
        start=monitor_start,
        end=monitor_end,
        record_fields=True,
        accumulate_power=True,
        record_interval=1,
        max_history_steps=None,
        name="out",
    )

    grid_points = monitor.get_grid_points_2d(dx, dy)
    if not grid_points:
        raise RuntimeError("Monitor line produced no grid points; check resolution or monitor bounds.")
    x_positions = np.array([pt[0] * dx for pt in grid_points], dtype=float)
    center_x = 0.5 * (monitor_start[0] + monitor_end[0])
    relative_x = x_positions - center_x
    target_profile, _ = slab_mode_source(
        x=relative_x,
        w=WG_W,
        n_WG=N_CORE,
        n0=N_CLAD,
        wavelength=WL,
        ind_m=0,
        x0=0.0,
    )
    target_samples = np.asarray(target_profile, dtype=complex)
    target_norm = np.linalg.norm(target_samples)
    if target_norm == 0.0:
        target_norm = 1.0
    target_samples /= target_norm

    def objective_fn(m) -> float:
        if not m.fields.get("Ez"):
            return 0.0
        field_history = [np.asarray(sample, dtype=complex) for sample in m.fields["Ez"]]
        overlaps = []
        ts_norm = np.linalg.norm(target_samples)
        for field_vec in field_history:
            field_norm = np.linalg.norm(field_vec)
            if field_norm == 0.0:
                overlaps.append(0.0)
                continue
            overlap = np.abs(np.vdot(target_samples, field_vec)) / (ts_norm * field_norm)
            overlaps.append(float(overlap))
        mode_score = max(overlaps) if overlaps else 0.0
        power_score = float(np.trapz(np.abs(m.power_history), dx=DT)) if m.power_history else 0.0
        m.mode_score = mode_score
        m.power_score = power_score
        return -(power_score * mode_score)

    monitor.objective_function = objective_fn
    monitor.target_samples = target_samples
    monitor.target_positions = x_positions
    monitor.center_x = center_x
    return monitor


def build_adjoint_source(design, signal, target_positions, target_samples):
    """Construct adjoint source shaped to the desired mode profile."""

    adjoint = ModeSource(
        design=design,
        position=(W/2, H - 2.5*µm),
        width=4 * WG_W,
        wavelength=WL,
        signal=signal,
        direction="-y",
        num_modes=1,
    )
    adjoint.enforce_direction = False
    if adjoint.mode_profiles:
        profile = adjoint.mode_profiles[0]
        profile_positions = np.array([pt[1] for pt in profile], dtype=float)
        real_interp = np.interp(profile_positions, target_positions, target_samples.real, left=0.0, right=0.0)
        imag_interp = np.interp(profile_positions, target_positions, target_samples.imag, left=0.0, right=0.0)
        for idx, point in enumerate(profile):
            target_component = MODE_WEIGHT * (real_interp[idx] + 1j * imag_interp[idx])
            transmission_component = TRANSMISSION_WEIGHT
            profile[idx][0] = transmission_component + target_component
    return adjoint


def build_forward_source(design, signal):
    """Return a forward mode source that injects the input waveguide mode without mirrors."""

    source = ModeSource(
        design=design,
        position=(2.5*µm, H/2),
        width=WG_W * 4,
        wavelength=WL,
        signal=signal,
        direction="+x",
        num_modes=1,
    )
    source.enforce_direction = False
    return source
rng = np.random.default_rng(0)
design_density = np.zeros_like(base)
design_density[mask] = rng.random(np.count_nonzero(mask))
objective_history = []
optimizer = Optimizer(method="adam", learning_rate=LR)

# Optimization loop
for step in range(1,STEPS+1):
    
    blurred_density, physical_density, blur_weights = apply_density_filters(design_density)

    # Update the permittivity
    eps = base.copy()
    eps[mask] = EPS_CLAD + physical_density[mask]*(EPS_CORE-EPS_CLAD)
    np.copyto(grid.permittivity, eps)
    
    # Forward simulation using non-reflective source and mode-aware monitor
    source = build_forward_source(design, signal)
    monitor = build_output_monitor(design)
    forward = FDTD(design=grid, devices=[source, monitor], time=t)
    fres = forward.run(live=True, save_memory_mode=True, accumulate_power=True, save_fields=["Ez"], fields_to_cache=["Ez"])
    forward.plot_power(db_colorbar=False)
    forward_power_path = f"forward_power_step{step:03d}.png"
    forward.fig.savefig(forward_power_path, dpi=200, bbox_inches="tight")
    plt.close(forward.fig)
    ffields = list(fres.get("Ez",[]))
    
    # Adjoint simulation, computing the overlap gradient
    adj_source = build_adjoint_source(design, signal, monitor.target_positions, monitor.target_samples)
    adj = FDTD(design=grid, devices=[adj_source], time=t)
    adj.initialize_simulation(save=False, live=True, accumulate_power=True, save_memory_mode=True, fields_to_cache=None)
    grad = np.zeros_like(base)
    for _ in range(adj.num_steps):
        if not ffields or not adj.step(): break
        grad += np.real(adj.backend.to_numpy(adj.Ez)*np.conj(ffields.pop()))
    adj.finalize_simulation()
    adj.plot_power(db_colorbar=False)
    adj_power_path = f"adjoint_power_step{step:03d}.png"
    adj.fig.savefig(adj_power_path, dpi=200, bbox_inches="tight")
    plt.close(adj.fig)

    grad_plot = np.zeros_like(grad)
    grad_plot[mask] = grad[mask]
    grad_scale = np.max(np.abs(grad_plot)) or 1.0
    grad_fig, grad_ax = plt.subplots(figsize=(6, 6))
    grad_im = grad_ax.imshow(grad_plot, origin="lower", extent=(0, design.width, 0, design.height), cmap="RdBu", aspect="equal", vmin=-grad_scale, vmax=grad_scale)
    plt.colorbar(grad_im, ax=grad_ax, orientation="vertical", label="Adjoint Gradient")
    grad_ax.set_title(f"Overlap Gradient Step {step}")
    grad_ax.set_xlabel("x (m)")
    grad_ax.set_ylabel("y (m)")
    grad_path = f"overlap_gradient_step{step:03d}.png"
    grad_fig.savefig(grad_path, dpi=200, bbox_inches="tight")
    plt.close(grad_fig)

    # Apply blur and projection filtered update with Adam optimizer
    grad_density = grad * (EPS_CORE - EPS_CLAD)
    proj_derivative = PROJECTION_BETA * physical_density * (1.0 - physical_density)
    grad_after_projection = grad_density * proj_derivative
    grad_design = masked_box_blur_backprop(
        grad_after_projection,
        mask,
        blur_weights,
        filter_radius_cells,
    )
    grad_design = np.where(mask, grad_design, 0.0)
    grad_norm = grad_design / ((np.abs(grad_design).max()) or 1.0)
    adam_update = optimizer.step(-grad_norm)
    adam_update = np.where(mask, adam_update, 0.0)
    new_design_density = design_density + adam_update
    new_design_density = np.clip(new_design_density, 0.0, 1.0)
    density_delta = new_design_density - design_density
    design_density = np.where(mask, new_design_density, design_density)
    design_density[~mask] = 0.0  # Reset density outside the design region
    update_norm = float(np.linalg.norm(density_delta[mask])) if np.any(mask) else 0.0
    max_update = float(np.max(np.abs(density_delta[mask]))) if np.any(mask) else 0.0
    blurred_density, physical_density, blur_weights = apply_density_filters(design_density)
    density_fig, density_ax = plt.subplots(figsize=(6, 6))
    density_im = density_ax.imshow(physical_density, origin="lower", extent=(0, design.width, 0, design.height), cmap="viridis", aspect="equal", vmin=0.0, vmax=1.0)
    plt.colorbar(density_im, ax=density_ax, orientation="vertical", label="Density")
    density_ax.set_title(f"Projected Density Step {step}")
    density_ax.set_xlabel("x (m)")
    density_ax.set_ylabel("y (m)")
    density_path = f"density_step{step:03d}.png"
    density_fig.savefig(density_path, dpi=200, bbox_inches="tight")
    plt.close(density_fig)
    obj = float(next(iter(fres.get("objectives",{"out":0}).values())))
    combined_objective = -obj
    transmission = getattr(monitor, "power_score", 0.0)
    mode_score = getattr(monitor, "mode_score", 0.0)
    objective_history.append(combined_objective)
    history_fig, history_ax = plt.subplots(figsize=(6, 4))
    history_ax.plot(range(1, len(objective_history)+1), objective_history, marker="o", linewidth=2)
    history_ax.set_xlabel("Optimization Step")
    history_ax.set_ylabel("Combined Objective")
    history_ax.set_title("Objective History (higher is better)")
    history_ax.grid(True, alpha=0.3)
    history_fig.savefig("objective_history.png", dpi=200, bbox_inches="tight")
    plt.close(history_fig)
    print(
        f"step {step}: obj {combined_objective:.4e} | power {transmission:.4e} | "
        f"mode {mode_score:.3f} | update norm {update_norm:.3e} | max density update {max_update:.3e}"
    )

# Final transmission
_, physical_density, _ = apply_density_filters(design_density)
eps = base.copy()
eps[mask] = EPS_CLAD+physical_density[mask]*(EPS_CORE-EPS_CLAD)
np.copyto(grid.permittivity, eps)
final_source = build_forward_source(design, signal)
final_monitor = build_output_monitor(design)
final = FDTD(design=grid, devices=[final_source, final_monitor], time=t).run(
    live=True,
    save_memory_mode=True,
    accumulate_power=True,
    save_fields=["Ez"],
    fields_to_cache=None,
)
final_obj = -float(next(iter(final.get("objectives", {"out": 0}).values())))
print(
    "final objective",
    final_obj,
    "power",
    getattr(final_monitor, "power_score", 0.0),
    "mode",
    getattr(final_monitor, "mode_score", 0.0),
)
