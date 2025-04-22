from beamz import *
import numpy as np
from beamz.design.sources import GaussianSource

WL = 1.55*µm
TIME = 40*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.015*WL/LIGHT_SPEED)))
design = Design(8*µm, 8*µm, depth=None, material=Material(1), pml_size=WL/2)
design.add(Rectangle(width=3*µm, height=3*µm, material=Material(4)))
signal_1 = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
signal_2 = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=np.pi/2, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
design.add(GaussianSource(position=(3*µm, 4.5*µm), width=WL/6, signal=signal_1))
design.add(GaussianSource(position=(4.5*µm, 2*µm), width=WL/6, signal=signal_2))
design.show()
sim = FDTD(design=design, time=T, mesh="regular", resolution=WL/25)
sim.run(live=True, axis_scale=[-2, 2])
sim.plot_power(db_colorbar=True)