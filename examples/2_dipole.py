from beamz import *
import numpy as np
from beamz.design.sources import GaussianSource

WL = 0.6*µm
TIME = 40*WL/LIGHT_SPEED
N_CLAD = 1; N_CORE = 2
DX, DT = calc_optimal_fdtd_params(WL, max(N_CORE, N_CLAD), dims=2)

design = Design(8*µm, 8*µm, material=Material(N_CLAD**2), pml_size=WL*1.5)
design += Rectangle(width=4*µm, height=4*µm, material=Material(N_CORE**2))
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
design += GaussianSource(position=(4*µm, 5*µm), width=WL/6, signal=signal)
sim = FDTD(design=design, time=time_steps, mesh="regular", resolution=DX)
sim.run(live=True, axis_scale=[-1, 1], save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)