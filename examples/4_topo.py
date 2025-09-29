import numpy as np
from beamz import *

# General parameters for the simulation
W, H = 15*µm, 15*µm
WG_W = 0.5*µm
N_CORE, N_CLAD = 2.25, 1.444
WL = 1.55*µm
OPT_STEPS, LR = 2, 1e-2
TIME = 15*WL/LIGHT_SPEED
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.95, points_per_wavelength=10)

# Design with a custom inhomogeneous material which we will update during optimization
design = Design(width=W, height=H, pml_size=2*µm)
design += Rectangle(position=(0*µm, H/2-WG_W/2), width=3.5*µm, height=WG_W, material=Material(permittivity=N_CORE**2))
design += Rectangle(position=(W/2-WG_W/2, H), width=WG_W, height=-3.5*µm, material=Material(permittivity=N_CORE**2))
design_material = CustomMaterial(permittivity_func= lambda x, y, z=None: np.random.uniform(N_CLAD**2, N_CORE**2))
design += Rectangle(position=(W/2-4*µm, H/2-4*µm), width=8*µm, height=8*µm, material=design_material)

# Define the signal
t = np.linspace(0, TIME, int(TIME/DT))
signal = ramped_cosine(t=t, amplitude=1.0, frequency=LIGHT_SPEED/WL, t_max=TIME, ramp_duration=WL*5/LIGHT_SPEED, phase=0)

# Rasterize the initial design
grid = RegularGrid(design=design, resolution=DX)

# Define the objective function
def objective_function():
    pass

# Define the optimizer
optimizer = Optimizer(method="adam", learning_rate=LR)
for opt_step in range(OPT_STEPS):
    # TODO: Where do we apply the blur and the filter???
    # Run the forward FDTD simulation
    forward = FDTD(design=grid, devices=[ModeSource(design=design, position=(2.5*µm, H/2), width=WG_W*4, wavelength=WL,
        signal=signal, direction="+x"), Monitor(design=design, start=(W/2-WG_W*2, H-2.5*µm), end=(W/2+WG_W*2, H-2.5*µm))], time=t)
    forward_fields = forward.run(live=True, axis_scale=[-1, 1], save_memory_mode=True) # TODO: only save the Ez field!!!
    # Run the adjoint FDTD simulation step-by-step and accumulate the overlap field
    adjoint = FDTD(design=grid, devices=[ModeSource(design=design, position=(W/2, H-2.5*µm), width=WG_W*4, wavelength=WL,
        signal=signal, direction="-y")], time=t)
    overlap_gradient = np.zeros_like(forward_fields["Ez"]) # TODO: Initialize with the correct shape!!!
    for step in t:
        adjoint_field = adjoint.step() # Simulate one step of the adjoint FDTD simulation
        overlap_gradient += compute_overlap_gradient(forward_fields, adjoint_field) / len(t) # Accumulate overlap gradient
        forward_fields.pop() # Delete the forward field that was just used to free up memory
    # Update the grid permittivity with the overlap gradient
    grid.permittivity = grid.permittivity + LR * overlap_gradient # TODO: Update with the optimizer



