from beamz import *
import numpy as np

WL = 1.55*µm
TIME = 25*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.015*WL/LIGHT_SPEED)))
design = Design(width=20*µm, height=4.5*µm, material=Material(2.1), pml_size=WL/3)
design.add(Rectangle(position=(0,2.25*µm-0.55*µm), width=20*µm, height=1.1*µm, material=Material(4)))
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*5/LIGHT_SPEED, t_max=TIME/5)

source = ModeSource(design=design, start=(1.1*µm, 2.25*µm-1.2*µm), end=(1.1*µm, 2.25*µm+1.2*µm), wavelength=WL, signal=signal)
source.show()
design.add(source)
design.show()

sim = FDTD(design=design, time=T, mesh="regular", resolution=WL/25)
sim.run(live=True, axis_scale=[-0.5,0.5])
sim.plot_power(log_scale=False, db_colorbar=True)