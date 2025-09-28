from beamz import *
import numpy as np

WL = 0.6*µm # wavelength of the source
TIME = 15*WL/LIGHT_SPEED # total simulation duration
N_CLAD = 1; N_CORE = 2 # refractive indices of the core and cladding

DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2, safety_factor=0.999, points_per_wavelength=10)

# Check the actual Courant number achieved
from beamz.helpers import check_fdtd_stability
is_stable, courant, limit = check_fdtd_stability(DT, DX, dy=DX, n_max=max(N_CORE, N_CLAD), safety_factor=1.0)
print(f"Courant number: {courant:.4f} (theoretical limit: {limit:.4f})")
print(f"Achievement: {courant/limit*100:.1f}% of theoretical limit")
print(f"Stable: {is_stable}")

# Create the design
design = Design(8*µm, 8*µm, material=Material(N_CLAD**2), pml_size=WL*1.5)
design += Rectangle(width=4*µm, height=4*µm, material=Material(N_CORE**2))
design.show()

# Define the signal
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
design += GaussianSource(position=(4*µm, 5*µm), width=WL/6, signal=signal)

# Run the simulation
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=True, axis_scale=[-1, 1], save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)

# Note: With safety_factor=0.95 (vacuum-based Courant), DT targets ~95% of the theoretical limit (1/sqrt(dims)).

# courant, time (S), timesteps
# 0.999 -> 16.9s -> 850 
# 0.50 -> 35.1s -> 1698