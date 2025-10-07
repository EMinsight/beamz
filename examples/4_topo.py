import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Agg")
from beamz import *
from beamz.optimization.topology import apply_density_update

# Parameters
W = H = 15*µm
WG_W = 0.5*µm
N_CORE, N_CLAD = 2.25, 1.444
EPS_CORE, EPS_CLAD = N_CORE**2, N_CLAD**2
WL = 1.55*µm
STEPS, LR = 30, 0.5
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE,N_CLAD), dims=2, safety_factor=0.99, points_per_wavelength=10)
TIME = 25*WL/LIGHT_SPEED
t = np.arange(0, TIME, DT)

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
xs, ys = (np.arange(mask.shape[1])+0.5)*dx, (np.arange(mask.shape[0])+0.5)*dy
minx, miny, _, maxx, maxy, _ = region.get_bounding_box()
mask[(ys[:,None]>=miny)&(ys[:,None]<=maxy)&(xs[None,:]>=minx)&(xs[None,:]<=maxx)] = True
rng = np.random.default_rng(0)
density = np.zeros_like(base)
density[mask] = rng.random(np.count_nonzero(mask))

# Optimization loop
for step in range(1,STEPS+1):
    
    # Update the permittivity
    eps = base.copy()
    eps[mask] = EPS_CLAD + density[mask]*(EPS_CORE-EPS_CLAD)
    np.copyto(grid.permittivity, eps)
    
    # Forward simulation, saving the Ez fields and objective value for all time steps
    source = ModeSource(design=design, position=(2.5*µm,H/2), width=WG_W*4, wavelength=WL, signal=signal, direction="+x")
    monitor = Monitor(design=design, start=(W/2-WG_W*2,H-2.5*µm), end=(W/2+WG_W*2,H-2.5*µm),
        objective_function = lambda m: -float(np.sum(np.abs(m.power_history))) if m.power_history else 0.0, name="out")
    forward = FDTD(design=grid, devices=[source, monitor], time=t)
    fres = forward.run(live=True, save_memory_mode=True, accumulate_power=True, save_fields=["Ez"], fields_to_cache=["Ez"])
    forward.plot_power(db_colorbar=True)
    forward_power_path = f"forward_power_step{step:03d}.png"
    forward.fig.savefig(forward_power_path, dpi=200, bbox_inches="tight")
    plt.close(forward.fig)
    ffields = list(fres.get("Ez",[]))
    
    # Adjoint simulation, computing the overlap gradient
    adj = FDTD(design=grid, devices=[ModeSource(design=design, position=(W/2,H-2.5*µm), width=WG_W*4,
        wavelength=WL, signal=signal, direction="-y")], time=t)
    adj.initialize_simulation(save=False, live=True, accumulate_power=True, save_memory_mode=True, fields_to_cache=None)
    grad = np.zeros_like(base)
    for _ in range(adj.num_steps):
        if not ffields or not adj.step(): break
        grad += np.real(adj.backend.to_numpy(adj.Ez)*np.conj(ffields.pop()))
    adj.finalize_simulation()
    adj.plot_power(db_colorbar=True)
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

    # Apply the density update
    density, _, _, _, _ = apply_density_update(density, grad/((np.abs(grad).max()) or 1.0), mask, learning_rate=LR, blur_radius=1)
    density[~mask] = 0.0 # Reset density outside the design region
    density_fig, density_ax = plt.subplots(figsize=(6, 6))
    density_im = density_ax.imshow(density, origin="lower", extent=(0, design.width, 0, design.height), cmap="viridis", aspect="equal", vmin=0.0, vmax=1.0)
    plt.colorbar(density_im, ax=density_ax, orientation="vertical", label="Density")
    density_ax.set_title(f"Density Step {step}")
    density_ax.set_xlabel("x (m)")
    density_ax.set_ylabel("y (m)")
    density_path = f"density_step{step:03d}.png"
    density_fig.savefig(density_path, dpi=200, bbox_inches="tight")
    plt.close(density_fig)
    obj = float(next(iter(fres.get("objectives",{"out":0}).values())))
    print(f"step {step}: transmission {-obj:.4e}")

# Final transmission
eps = base.copy()
eps[mask] = EPS_CLAD+density[mask]*(EPS_CORE-EPS_CLAD)
np.copyto(grid.permittivity, eps)
source = ModeSource(design=design, position=(2.5*µm,H/2), width=WG_W*4, wavelength=WL, signal=signal, direction="+x")
monitor = Monitor(design=design, start=(W/2-WG_W*2,H-2.5*µm), end=(W/2+WG_W*2,H-2.5*µm),
    objective_function = lambda m:-float(np.sum(np.abs(m.power_history))) if m.power_history else 0.0, name="out")
final = FDTD(design=grid, devices=[source, monitor], time=t).run(live=True, save_memory_mode=True,
    accumulate_power=True, save_fields=["Ez"], fields_to_cache=None)
print("final transmission", -float(next(iter(final.get("objectives",{"out":0}).values()))))