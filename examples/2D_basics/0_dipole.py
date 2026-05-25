import beamz as bz
import numpy as np

µm = bz.µm
WL = 0.6 * µm # wavelength of the source
TIME = 25 * WL / bz.LIGHT_SPEED # total simulation duration
N_CLAD, N_CORE = 1, 2 # refractive indices of the core and cladding
DX, DT = bz.dxdt(WL, max(N_CORE, N_CLAD), dims=2, points_per_wavelength=8)

# Create the design
design = bz.Design(8*µm, 8*µm, material=bz.Material(N_CLAD**2))
design += bz.Rectangle(width=4*µm, height=4*µm, material=bz.Material(N_CORE**2))

# Define the signal and source
t = np.arange(0, TIME, DT)
signal = bz.ramped_cosine(
    t,
    amplitude=1,
    frequency=bz.LIGHT_SPEED/WL, 
    ramp_duration=3*WL/bz.LIGHT_SPEED, 
    t_max=TIME/2
)
source = bz.GaussianSource(position=(4*µm, 5*µm), width=WL/6, signal=signal)

# Define the simulation (with added PML boundaries)
sim = bz.Simulation(
    design=design, 
    sources=[source], 
    boundaries=[bz.PML(edges='all', thickness=2*WL)], 
    time=t, 
    resolution=DX)

# Run the simulation, interactively showing the $E_z$ field
sim.run(animate_live="Ez", animation_interval=1, clean_visualization=True)