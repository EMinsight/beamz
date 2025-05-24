"""
paper inspiration: https://photonics.intec.ugent.be/download/pub_3105.pdf
"""
from beamz import *
import numpy as np

# Parameters
WL = 1.55*µm
TIME = 120*WL/LIGHT_SPEED
X = 20*µm
Y = 19*µm
N_CORE = 2.04 # Si3N4
N_CLAD = 1.444 # SiO2
WG_WIDTH = 0.565*µm
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD))
RING_RADIUS = 6*µm

# Create the design
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)
design += Rectangle(position=(0,WL*2), width=X, height=WG_WIDTH, material=Material(N_CORE**2))
design += Ring(position=(X/2, WL*2+WG_WIDTH+RING_RADIUS+WG_WIDTH/2+0.2*WG_WIDTH), 
               inner_radius=RING_RADIUS-WG_WIDTH/2, outer_radius=RING_RADIUS+WG_WIDTH/2, 
               material=Material(N_CORE**2))

# Define the signal & source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, 
                       ramp_duration=WL*20/LIGHT_SPEED, t_max=TIME/3)
design += ModeSource(design=design, start=(WL*2, WL*2+WG_WIDTH/2-1.5*µm), end=(WL*2, WL*2+WG_WIDTH/2+1.5*µm), 
                     wavelength=WL, signal=signal)
design.show()

# Run the simulation
#sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX, backend="numpy")
#sim.run(live=True, save_memory_mode=True, accumulate_power=True)
#sim.plot_power(db_colorbar=True)