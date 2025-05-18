from beamz import *
import numpy as np
from beamz.helpers import calc_optimal_fdtd_params

WL = 1.55*µm
TIME = 120*WL/LIGHT_SPEED
n_core = 2.04 # Si3N4
n_clad = 1.444 # SiO2
wg_width = 0.565*µm
resolution, dt = calc_optimal_fdtd_params(WL, max(n_core, n_clad), dims=2)
T = np.arange(0, TIME, dt)


design = Design(width=18*µm, height=7*µm, material=Material(n_clad**2), pml_size=WL)
design.add(Rectangle(position=(0,3.5*µm-wg_width/2), width=18*µm, height=wg_width, material=Material(n_core**2)))
design.show()

#grid = RegularGrid(design=design, resolution=resolution)
#grid.show(field="permittivity")
#grid.show(field="conductivity")
#grid.show(field="permeability")

signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*30/LIGHT_SPEED, t_max=TIME/2)
#import beamz
#beamz.design.signals.plot_signal(signal, T)

source = ModeSource(design=design, start=(2*µm, 3.5*µm-1.2*µm), end=(2*µm, 3.5*µm+1.2*µm), wavelength=WL, signal=signal)
#source.show()
design.add(source)
design.show()

sim = FDTD(design=design, time=T, mesh="regular", resolution=resolution)
sim.run(live=True, save_memory_mode=True, accumulate_power=True)
sim.plot_power(db_colorbar=True)