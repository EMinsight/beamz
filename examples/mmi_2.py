from beamz import *
import numpy as np

# Parameters
X = 150*µm # domain width (increased to accommodate larger MMI)
Y = 10*µm # domain height
WL = 1.55*µm # wavelength
TIME = 40*WL/LIGHT_SPEED # total simulation duration
N_CORE = 2.04 # Si3N4
N_CLAD = 1.444 # SiO2
WG_W = 0.565*µm # width of the waveguide
H = 3.5*µm # height of the MMI
W = 125*µm # length of the MMI (corrected for proper self-imaging)
OFFSET = 1.05*µm # offset of the output waveguides from center of the MMI
TAPER_LEN = 10*µm # length of input/output tapers
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.60) 

# Design the MMI with input and output waveguides
design = Design(width=X, height=Y, material=Material(N_CLAD**2), pml_size=WL)

# Input waveguide
design += Rectangle(position=(0, Y/2-WG_W/2), width=X/2-W/2-TAPER_LEN, height=WG_W, material=Material(N_CORE**2))

# Input taper
design += Polygon(vertices=[
    (X/2-W/2-TAPER_LEN, Y/2-WG_W/2),
    (X/2-W/2, Y/2-H/2),
    (X/2-W/2, Y/2+H/2),
    (X/2-W/2-TAPER_LEN, Y/2+WG_W/2)
], material=Material(N_CORE**2))

# MMI section
design += Rectangle(position=(X/2-W/2, Y/2-H/2), width=W, height=H, material=Material(N_CORE**2))

# Output tapers
design += Polygon(vertices=[
    (X/2+W/2, Y/2+OFFSET-H/2),
    (X/2+W/2+TAPER_LEN, Y/2+OFFSET-WG_W/2),
    (X/2+W/2+TAPER_LEN, Y/2+OFFSET+WG_W/2),
    (X/2+W/2, Y/2+OFFSET+H/2)
], material=Material(N_CORE**2))

design += Polygon(vertices=[
    (X/2+W/2, Y/2-OFFSET-H/2),
    (X/2+W/2+TAPER_LEN, Y/2-OFFSET-WG_W/2),
    (X/2+W/2+TAPER_LEN, Y/2-OFFSET+WG_W/2),
    (X/2+W/2, Y/2-OFFSET+H/2)
], material=Material(N_CORE**2))

# Output waveguides
design += Rectangle(position=(X/2+W/2+TAPER_LEN, Y/2+OFFSET-WG_W/2), width=X/2-W/2-TAPER_LEN, height=WG_W, material=Material(N_CORE**2))
design += Rectangle(position=(X/2+W/2+TAPER_LEN, Y/2-OFFSET-WG_W/2), width=X/2-W/2-TAPER_LEN, height=WG_W, material=Material(N_CORE**2))

# Define the source (matching exactly to waveguide width)
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*5/LIGHT_SPEED, t_max=TIME/2)
source = ModeSource(design=design, start=(2*µm, Y/2-WG_W/2), end=(2*µm, Y/2+WG_W/2), wavelength=WL, signal=signal)
design += source
design.show()

# Run the simulation and show results
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=True, save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)