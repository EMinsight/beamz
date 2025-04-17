from beamz import *
import numpy as np
from beamz.simulation.signals import plot_signal, ramped_cosine
from beamz.design.sources import LineSource

WL = 1.55*µm
TIME = 40*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.015*WL/LIGHT_SPEED)))

design = Design(width=6*µm, height=6*µm, material=Material(1), pml_size=0.5*µm)
i = 0
while (i-1)*µm*0.4 < 6*µm: 
    design.add(Rectangle(position=(1*i*µm,2.5*µm), width=0.5*µm, height=1*µm, material=Material(6.25)))
    i += 1
design.add(Rectangle(position=(0,0), width=6*µm, height=2.5*µm, material=Material(6.25)))


signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=TIME/4, t_max=TIME)
plot_signal(signal, T)
design.add(LineSource(start=(0.3*µm,1*µm), end=(5.7*µm,0.2*µm), distribution=None, direction="+x", signal=signal))


design.show()
FDTD(design=design, time=T, mesh="regular", resolution=WL/40).run(live=True, axis_scale=[-1,1])