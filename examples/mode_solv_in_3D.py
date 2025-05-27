from beamz import *
import numpy as np
from beamz.helpers import calc_optimal_fdtd_params

# Parameters
X, Y, Z = 7*µm, 7*µm, 3*µm # domain width, height, depth
WL = 1.55*µm # wavelength
TIME = 40*WL/LIGHT_SPEED # total simulation duration
N_CORE, N_CLAD = 2.04, 1.444 # Si3N4, SiO2
WG_W = 0.565*µm # width of the waveguide
H, W, OFFSET = 3.5*µm, 9*µm, 1.05*µm # height, length, offset of the MMI
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=3, safety_factor=0.40) 

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, depth=Z, material=Material(permittivity=1.0, permeability=1.0, conductivity=0.0), pml_size=WL)
design += Rectangle(position=(0, 0, 0), width=X, height=Y, depth=Z/2, material=Material(N_CLAD**2))
design += Rectangle(position=(0, Y/2-WG_W/2, Z/2), width=X, height=WG_W, depth=Z/8, material=Material(N_CORE**2))
design.show()

# Create the signal
#time_steps = np.arange(0, TIME, DT)
#signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*30/LIGHT_SPEED, t_max=TIME/2)
#design += ModeSource(design=design, start=(2*µm, 3.5*µm-1.2*µm), end=(2*µm, 3.5*µm+1.2*µm), wavelength=WL, signal=signal)
#design.show()

# Run the simulation
#sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
#sim.run(live=True, save_memory_mode=True, accumulate_power=True)
#sim.plot_power(db_colorbar=True)