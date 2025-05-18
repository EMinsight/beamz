from beamz import *
import numpy as np
from beamz.helpers import calc_optimal_fdtd_params

# Parameters
X = 18*µm
Y = 7*µm
WL = 1.55*µm
TIME = 40*WL/LIGHT_SPEED
N_CORE = 2.04 # Si3N4
N_CLAD = 1.444 # SiO2
WG_W = 0.565*µm
OFFSET = 0.5*µm
H = 3*µm
W = 5*µm
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2)

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
design += Rectangle(position=(0,3.5*µm-WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, 3.5*µm + OFFSET - WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, 3.5*µm - OFFSET - WG_W/2), width=X/2, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2-W/2,Y/2-H/2), width=W, height=H, material=Material(N_CORE**2))

grid = RegularGrid(design=design, resolution=DX)
grid.show(field="permittivity")
grid.show(field="conductivity")
grid.show(field="permeability")

# Define the source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*5/LIGHT_SPEED, t_max=TIME/2)
source = ModeSource(design=design, start=(2*µm, 3.5*µm-1.2*µm), end=(2*µm, 3.5*µm+1.2*µm), wavelength=WL, signal=signal)
design += source
design.show()

sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=True, save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)