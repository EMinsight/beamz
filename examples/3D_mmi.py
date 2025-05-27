from beamz import *
import numpy as np

# Parameters
X, Y, Z = 20*µm, 10*µm, 5*µm # domain width, height, depth
WL = 1.55*µm # wavelength
TIME = 40*WL/LIGHT_SPEED # total simulation duration
N_CORE, N_CLAD = 2.04, 1.444 # Si3N4, SiO2
WG_W = 0.565*µm # width of the waveguide
H, W, OFFSET = 3.5*µm, 9*µm, 1.05*µm # height, length, offset of the MMI
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=3, safety_factor=0.40) 

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, depth=Z, material=Material(permittivity=1.0, permeability=1.0, conductivity=0.0), pml_size=WL)
design += Rectangle(position=(0, 0, 0), width=X, height=Y, depth=Z/2, material=Material(N_CLAD**2))
design += Rectangle(position=(0, Y/2-WG_W/2, Z/2), width=X/2, height=WG_W, depth=Z/8, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 + OFFSET - WG_W/2, Z/2), width=X/2, height=WG_W, depth=Z/8, material=Material(N_CORE**2))
design += Rectangle(position=(X/2, Y/2 - OFFSET - WG_W/2, Z/2), width=X/2, height=WG_W, depth=Z/8, material=Material(N_CORE**2))
design += Rectangle(position=(X/2-W/2, Y/2-H/2, Z/2), width=W, height=H, depth=Z/8, material=Material(N_CORE**2))

# Add monitors
design += Monitor(start=(0, 0, Z/2+Z/16), end=(X, Y, Z/2+Z/16), record_fields=False, accumulate_power=True)

# Define the source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*5/LIGHT_SPEED, t_max=TIME/2)
source = ModeSource(design=design, start=(2*µm, Y/2-1.2*µm, Z/2+Z/16), end=(2*µm, Y/2+1.2*µm, Z/2+Z/16), wavelength=WL, signal=signal)
design += source
design.show()

