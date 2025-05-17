from beamz import *
import numpy as np
from beamz.helpers import calc_optimal_fdtd_params

WL = 1.55*µm
TIME = 40*WL/LIGHT_SPEED
n_core = 2.3
n_clad = 1

# Calculate optimal resolution and time step
resolution, dt = calc_optimal_fdtd_params(WL, max(n_core, n_clad), safety_factor=0.10, points_per_wavelength=20)

T = np.arange(0, TIME, dt)
print(T)

design = Design(width=6*µm, height=6*µm, material=Material(n_clad**2), pml_size=WL/3)
design.add(Rectangle(position=(0,1*µm), width=6*µm, height=0.4*µm, material=Material(n_core**2)))
design.add(Ring(position=(3*µm, 3.35*µm), inner_radius=1.5*µm, outer_radius=1.9*µm, material=Material(n_core**2)))
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=TIME/5, t_max=TIME/3)
design.add(ModeSource(design=design, start=(1*µm, 0.5*µm), end=(1*µm, 1.9*µm), wavelength=WL, signal=signal))
design.show()

sim = FDTD(design=design, time=T, mesh="regular", resolution=resolution, backend="numpy")
sim.run(live=True)
sim.plot_power(db_colorbar=True)