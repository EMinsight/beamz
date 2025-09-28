import numpy as np
from beamz import *
from beamz.helpers import calc_optimal_fdtd_params
from beamz.design.materials import CustomMaterial
from beamz.optimization import topology as topo

# General parameters
W, H = 15*µm, 15*µm
WG_W = 0.5*µm
N_CORE, N_CLAD = 2.25, 1.444
WL = 1.55*µm
# Optimization parameters
OPT_STEPS = 2
LEARNING_RATE = 1e-2
# Calculate the optimal FDTD parameters
TIME = 40*WL/LIGHT_SPEED
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.95, points_per_wavelength=10)

# Design
design = Design(width=W, height=H, pml_size=2*µm)
design += Rectangle(position=(0*µm, H/2-WG_W/2), width=3.5*µm, height=WG_W, material=Material(permittivity=N_CORE**2))
design += Rectangle(position=(W/2-WG_W/2, H), width=WG_W, height=-3.5*µm, material=Material(permittivity=N_CORE**2))
design_material = CustomMaterial(permittivity_func=lambda x, y, z=None: np.random.uniform(N_CLAD**2, N_CORE**2))
design_region = Rectangle(position=(W/2-4*µm, H/2-4*µm), width=8*µm, height=8*µm, material=design_material)
design += design_region

# Source & Monitor
t = np.linspace(0, TIME, int(TIME/DT))
signal = ramped_cosine(t=t, amplitude=1.0, frequency=LIGHT_SPEED/WL, t_max=TIME, ramp_duration=WL*10/LIGHT_SPEED, phase=0)
forward_source = ModeSource(design=design, position=(2.5*µm, H/2), width=WG_W*4, wavelength=WL, signal=signal, direction="+x")
adjoint_source = ModeSource(design=design, position=(W/2, H-2.5*µm), width=WG_W*4, wavelength=WL,
                               signal=signal, direction="-y")
design += forward_source
design += adjoint_source
# Output monitor placed across the vertical waveguide core (improves adjoint coupling)
monitor = Monitor(design=design, start=(9.5*µm, 15.0*µm), end=(10.5*µm, 15.0*µm))
design += monitor
design.show()

# Rasterize the design
grid = RegularGrid(design=design, resolution=DX)
grid.show(field="permittivity")

# Setup the FDTD simulation
sim = FDTD(design=design, time=t, resolution=DX)


# Initialize density field tied to the design region (actual implementation lives in topo).
density = topo.initialize_density_from_region(design_region, resolution=DX)
optimizer_state = None


for step in range(OPT_STEPS):
    # 1. Apply blur filter to enforce minimum feature size.
    # 2. Project densities toward binary structures for fabrication.
    # 3. Update design region materials with the projected density.
    blurred = topo.blur_density(density, radius=WL/2)
    projected = topo.project_density(blurred, beta=2.0, eta=0.5)
    topo.update_design_region_material(design_region, projected)

    # 4. Run forward FDTD simulation to evaluate the objective and store fields.
    forward_fields, objective = sim.forward(sources=forward_source, monitors=monitor,
        objective=lambda fields: np.sum(np.abs(fields['Ez'][-1])**2),live=True)
    # 5. Run adjoint simulation to obtain sensitivity information 
    # 6. And compute the field-overlap gradient using stored forward/adjoint fields.
    adjoint_fields, overlap_gradient = sim.adjoint(sources=adjoint_source, forward_fields=forward_fields, live=True, 
        save_adjoint_fields=False)

    # 7. Update the density field using the optimizer step.
    density, optimizer_state = topo.apply_optimizer_step(density, overlap_gradient,
        optimizer_state, method="adam", learning_rate=LEARNING_RATE)