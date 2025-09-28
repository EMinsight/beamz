import numpy as np
from beamz import *
from beamz.helpers import calc_optimal_fdtd_params
from beamz.design.materials import CustomMaterial

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
design_material = CustomMaterial(permittivity_func= lambda x, y, z=None: np.random.uniform(N_CLAD**2, N_CORE**2))
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




sim = FDTD(design=design, time=t, resolution=DX)


# Objective function
def obj_func(fields): return np.sum(np.abs(fields['Ez'][-1])**2)
objective_history = []

def Adagrad_optimizer(design_region, overlap_field, learning_rate):
    design_region = design_region + learning_rate * overlap_field
    return design_region




for step in range(OPT_STEPS):
    # Project the design region to a blurred binary mask

    # Run forward simulation, measure the objective value & save the fields for every time-step in a list
    forward_fields, objective = sim.forward(sources=forward_source, monitors=monitor, objective=obj_func, live=True)
    objective_history.append(objective)

    # Run adjoint simulation while calculating the average overlap of the adjoint field with the reversed forward fields
    overlap = sim.adjoint(sources=adjoint_source, forward_fields=forward_fields, live=True)

    # Update the design region without the binary mask using the overlap field x scaler factor from the optimizer
