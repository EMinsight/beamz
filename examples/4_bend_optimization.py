from beamz import *
import numpy as np

# 90-degree bend topology optimization (2D TE)
# - Input straight waveguide from left
# - Output straight waveguide upwards
# - Square design region at the corner, optimized via pixel permittivity updates

# Parameters
WL = 1.55*µm
N_CORE = 2.04   # Si3N4
N_CLAD = 1.444  # SiO2
WG_W = 0.6*µm
PML = 1.0*µm

# Domain
X = 12*µm
Y = 12*µm

# Discretization
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.55)
TIME = 120*WL/LIGHT_SPEED

# Design region (square at corner)
DR_SIZE = 4.0*µm
DR_X0 = X/2 - DR_SIZE/2
DR_Y0 = Y/2 - DR_SIZE/2

# Optimization
PIXELS = 40  # DR_SIZE will be discretized into PIXELS x PIXELS
BETA = 0.5   # step size for gradient descent
ITERS = 3
SMOOTH_MIN = N_CLAD**2
SMOOTH_MAX = N_CORE**2

# Utility to build design from pixel map

def build_design_from_rho(rho):
    design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=PML)
    # Input and output access waveguides
    design += Rectangle(position=(0, Y/2 - WG_W/2), width=X/2 - DR_SIZE/2, height=WG_W, material=Material(N_CORE**2))
    design += Rectangle(position=(X/2 - WG_W/2, Y/2), width=WG_W, height=Y/2 - DR_SIZE/2, material=Material(N_CORE**2))
    # Fill design region from rho
    px_w = DR_SIZE / PIXELS
    px_h = DR_SIZE / PIXELS
    for i in range(PIXELS):
        for j in range(PIXELS):
            eps_val = SMOOTH_MIN + (SMOOTH_MAX - SMOOTH_MIN) * rho[j, i]
            if eps_val <= SMOOTH_MIN + 1e-12:  # skip pure cladding to keep geometry sparse
                continue
            x = DR_X0 + i * px_w
            y = DR_Y0 + j * px_h
            design += Rectangle(position=(x, y), width=px_w, height=px_h, material=Material(eps_val))
    return design

# Sources and monitors builder

def add_io(design, signal):
    # Forward source: left to right (+x)
    src = ModeSource(
        design=design,
        start=(2*µm, Y/2-1.2*µm), end=(2*µm, Y/2+1.2*µm),
        wavelength=WL, signal=signal, direction="+x"
    )
    design += src
    # Output monitor line at top waveguide (measures transmission into vertical arm)
    mon = Monitor(design=design,
                  start=(X/2 - 1.2*µm, Y - 2*µm), end=(X/2 + 1.2*µm, Y - 2*µm),
                  record_fields=True, accumulate_power=True)
    design += mon
    return src, mon

# Objective: power through top monitor at late time (proxy for transmission)

def objective_from_monitor(mon):
    stats = mon.get_power_statistics()
    return stats.get('mean_power', 0.0)

# Simple adjoint using reciprocity: excite from output back towards design region
# We approximate gradient: dJ/deps ~ Re( E_fwd * E_adj ) accumulated in DR


def run_fdtd(design, time_steps, resolution):
    sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=resolution)
    sim.run(live=False, save_memory_mode=True, accumulate_power=False)
    # Pull Ez snapshot at final time for simplicity
    Ez = sim.backend.to_numpy(sim.Ez)
    return Ez, sim


def sample_E_in_DR(Ez, design):
    # Extract Ez in the DR bounds using grid spacing from sim
    # We map continuous DR to grid indices via design extents and mesh in FDTD
    # Here we will approximate by slicing based on physical coordinates
    return Ez

if __name__ == "__main__":
    # Time signal
    t = np.arange(0, TIME, DT)
    signal = ramped_cosine(t, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0,
                           ramp_duration=WL*20/LIGHT_SPEED, t_max=TIME/2)

    # Initialize density field in DR
    rho = 0.5 * np.ones((PIXELS, PIXELS))

    best_obj = -1
    for it in range(ITERS):
        # Build current design
        design = build_design_from_rho(rho)
        design.show()
        # IO
        fwd_src, out_mon = add_io(design, signal)
        # Forward run
        Ez_fwd, sim_fwd = run_fdtd(design, t, DX)
        J = objective_from_monitor(out_mon)
        print(f"Iteration {it+1}/{ITERS} - objective ~ {J:.6g}")
        best_obj = max(best_obj, J)

        # Build adjoint design: use same structure, but source at output monitor line pointing -y into the bend
        design_adj = build_design_from_rho(rho)
        adj_src = ModeSource(
            design=design_adj,
            start=(X/2 - 1.2*µm, Y - 2*µm), end=(X/2 + 1.2*µm, Y - 2*µm),
            wavelength=WL, signal=signal, direction="-y"
        )
        design_adj += adj_src
        Ez_adj, sim_adj = run_fdtd(design_adj, t, DX)

        # Very simple gradient proxy (not rigorous):
        # Map fields to DR pixel grid by averaging Ez magnitudes inside each pixel
        grad = np.zeros_like(rho)
        px_w = DR_SIZE / PIXELS
        px_h = DR_SIZE / PIXELS
        # Convert FDTD grid to physical extents
        dy, dx = sim_fwd.dy, sim_fwd.dx
        ny, nx = sim_fwd.ny, sim_fwd.nx
        xs = np.arange(nx) * dx
        ys = np.arange(ny) * dy
        for i in range(PIXELS):
            for j in range(PIXELS):
                x0 = DR_X0 + i*px_w
                x1 = x0 + px_w
                y0 = DR_Y0 + j*px_h
                y1 = y0 + px_h
                xi0 = np.searchsorted(xs, x0)
                xi1 = np.searchsorted(xs, x1)
                yi0 = np.searchsorted(ys, y0)
                yi1 = np.searchsorted(ys, y1)
                if xi1 <= xi0 or yi1 <= yi0:
                    continue
                Ef = Ez_fwd[yi0:yi1, xi0:xi1]
                Ea = Ez_adj[yi0:yi1, xi0:xi1]
                # Use overlap as sensitivity proxy
                grad[j, i] = np.real(np.mean(Ef * np.conj(Ea)))

        # Normalize and take ascent for increasing transmission
        if np.max(np.abs(grad)) > 0:
            grad /= (np.max(np.abs(grad)) + 1e-12)
        rho = np.clip(rho + BETA * grad, 0.0, 1.0)

    # Final design visualization
    final_design = build_design_from_rho(rho)
    final_design.show()

    # Final forward simulation with accumulation for power map
    sim = FDTD(design=final_design, time=t, mesh="regular", resolution=DX)
    sim.run(live=True, save_memory_mode=True, accumulate_power=True)
    sim.plot_power(db_colorbar=True)
