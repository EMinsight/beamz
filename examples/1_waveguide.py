from beamz import *
import numpy as np
from beamz.helpers import calc_optimal_fdtd_params

WL = 1.55*µm
TIME = 120*WL/LIGHT_SPEED
n_core = 2
n_clad = 1
resolution, dt = calc_optimal_fdtd_params(WL, max(n_core, n_clad), dims=2, safety_factor=0.45)
T = np.arange(0, TIME, dt)


design = Design(width=15*µm, height=4.5*µm, material=Material(n_clad**2), pml_size=WL/2)
design.add(Rectangle(position=(0,2.25*µm-0.55*µm), width=15*µm, height=1.1*µm, material=Material(n_core**2)))

signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*30/LIGHT_SPEED, t_max=TIME/2)
import beamz
beamz.design.signals.plot_signal(signal, T)
source = ModeSource(design=design, start=(1.1*µm, 2.25*µm-1.43*µm), end=(1.1*µm, 2.25*µm+1.43*µm), wavelength=WL, signal=signal)
source.show()
design.add(source)
design.show()

sim = FDTD(design=design, time=T, mesh="regular", resolution=resolution)
sim.run(live=True, save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)