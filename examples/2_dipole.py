from beamz import *
import numpy as np
from beamz.design.sources import GaussianSource

WL = 1.55*µm
TIME = 40*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.015*WL/LIGHT_SPEED)))
design = Design(8*µm, 8*µm, depth=None, material=Material(1), pml_size=WL/1.8)
design.add(Rectangle(width=4*µm, height=4*µm, material=Material(4)))
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
design.add(GaussianSource(position=(4*µm, 5*µm), width=WL/6, signal=signal))
sim = FDTD(design=design, time=T, mesh="regular", resolution=WL/25)
sim.run(live=True, axis_scale=[-2, 2])
sim.plot_power(db_colorbar=True)