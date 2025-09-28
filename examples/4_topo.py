import numpy as np
from beamz import *
from beamz.helpers import calc_optimal_fdtd_params
from beamz.design.materials import CustomMaterial

# General parameters
W, H = 20*µm, 20*µm
WG_W = 0.5*µm
N_CORE, N_CLAD = 2.25, 1.444
WL = 1.55*µm
# Optimization parameters
OPT_STEPS = 2
LEARNING_RATE = 1e-2
# Calculate the optimal FDTD parameters
TIME = 10*WL/LIGHT_SPEED
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2)

# Design
design = Design(width=W, height=H, pml_size=2.5*µm)
design += Rectangle(position=(2*µm, 9*µm), width=3.1*µm, height=WG_W, material=Material(permittivity=N_CORE**2))
design += Rectangle(position=(10*µm, 15*µm), width=WG_W, height=3.1*µm, material=Material(permittivity=N_CORE**2))

# Create design region for topology optimization
design_region_size = 8.2*µm
design_region_resolution = 50  # 50x50 grid for optimization

# Create initial permittivity grid (random values between N_CLAD^2 and N_CORE^2)
initial_eps_grid = np.random.random((design_region_resolution, design_region_resolution)) * (N_CORE**2 - N_CLAD**2) + N_CLAD**2

# Create CustomMaterial with grid-based permittivity
design_material = CustomMaterial(permittivity_grid=initial_eps_grid, bounds=((5.1*µm, 5.1*µm + design_region_size), 
                                    (7*µm, 7*µm + design_region_size)), interpolation='nearest')


design += Rectangle(position=(5.1*µm, 7*µm), width=design_region_size, height=design_region_size,
    material=design_material)

# Source & Monitor
t = np.linspace(0, TIME, int(TIME/DT))
signal = ramped_cosine(time_steps=t, amplitude=1.0, frequency=LIGHT_SPEED/WL, t_max=TIME, phase=0)
forward_source = ModeSource(design=design, position=(3*µm, 9.25*µm), width=WG_W*4, wavelength=WL, signal=signal, direction="+x")
adjoint_source = ModeSource(design=design, position=(9.5*µm, 15.0*µm), width=WG_W*4, wavelength=WL, signal=signal, direction="-x")
design += forward_source
design += adjoint_source
# Output monitor placed across the vertical waveguide core (improves adjoint coupling)
design += Monitor(design=design, start=(9.5*µm, 15.0*µm), end=(10.5*µm, 15.0*µm))

design.show()

# Rasterize the design
grid = RegularGrid(design=design, resolution=DX)
grid.show(field="permittivity")


# Define the optimization run manually
fdtd = FDTD(design, time=t, resolution=DX) # Rasterize the design before simulating it
objective_history = []

# Simple gradient descent optimizer
class SimpleOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def step(self, params, gradients):
        for param, grad in zip(params, gradients):
            param -= self.learning_rate * grad

optimizer = SimpleOptimizer(learning_rate=LEARNING_RATE)
def objective_function(ModeSource, Monitor):
    # Defines what and how we measure the performance of the design we are optimizing for
    # Use the latest recorded Ez field along the monitor line
    if not Monitor.fields or not Monitor.fields.get('Ez') or len(Monitor.fields['Ez']) == 0:
        # Fallback to total power if available
        if getattr(Monitor, 'power_history', None):
            try: return float(np.real(Monitor.power_history[-1]))
            except Exception: return 0.0
        return 0.0

    Ez_line = np.array(Monitor.fields['Ez'][-1])
    if Ez_line is None or len(Ez_line) == 0: return 0.0

    # Build a local mode definition along the monitor line to target the desired waveguide mode
    try:
        target_src = type(ModeSource)(
            design=Monitor.design,
            start=(Monitor.start[0], Monitor.start[1]),
            end=(Monitor.end[0], Monitor.end[1]),
            wavelength=getattr(ModeSource, 'wavelength', WL),
            signal=0,
            num_modes=1
        )
        # Fundamental mode profile along the line
        prof = target_src.mode_vectors[:, 0] if getattr(target_src, 'mode_vectors', None) is not None else None
    except Exception: prof = None

    # If mode profile couldn't be computed, use a flat reference as a very weak fallback
    if prof is None: prof = np.ones_like(Ez_line, dtype=complex)
    else:
        # Resample profile to match monitor sampling length if needed
        if prof.shape[0] != len(Ez_line):
            idx = np.rint(np.linspace(0, prof.shape[0] - 1, len(Ez_line))).astype(int)
            prof = prof[idx]

    # Compute normalized overlap efficiency between measured field and target mode shape
    e = np.asarray(Ez_line, dtype=complex)
    m = np.asarray(prof, dtype=complex)
    # Avoid trivial zeros
    if np.allclose(m, 0): m = np.ones_like(e)
    num = np.vdot(m, e)  # conj(m) * e summed
    den = (np.vdot(m, m) * np.vdot(e, e))
    if np.real(den) <= 0: return 0.0
    eta = (np.abs(num) ** 2) / np.real(den)
    # Return a real scalar in [0, 1]
    return float(np.real(eta))

# Run the optimization/simulation loop
def run():
    for step in range(OPT_STEPS):
        # Run forward simulation, save the fields at every step t and compute the objective
        forward_history, objective = fdtd.forward_sim(objective_function=objective_function)
        objective_history.append(objective)
        # Run adjoint simulation and compute the overlap at every step t and compute the average overlap while doing dp
        overlap = fdtd.adjoint_sim(forward_history, compute_overlap=True)
        # Update the design region using the overlap scaled by the optimizer
        # Find the design region (Rectangle with CustomMaterial)
        region = None
        for s in fdtd.design.structures:
            if isinstance(s, Rectangle) and isinstance(s.material, CustomMaterial):
                region = s
                break
        if region is not None:
            # Get the current permittivity grid
            eps_grid = region.material.permittivity_grid
            g = np.array(overlap, dtype=float)
            
            # Normalize gradient for stable steps
            gnorm = np.max(np.abs(g)) + 1e-12
            g = g / gnorm
            
            # Ascend objective -> subtract negative gradient
            optimizer.step([eps_grid], [-g])
            
            # Clamp to physical bounds
            eps_min = N_CLAD**2
            eps_max = N_CORE**2
            np.clip(eps_grid, eps_min, eps_max, out=eps_grid)
            
            # Update the material grid
            region.material.update_grid('permittivity', eps_grid)