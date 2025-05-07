from beamz import *
import numpy as np

WL = 1.55*µm
TIME = 20*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.015*WL/LIGHT_SPEED)))

WG_WIDTH = 3.2*µm
W = 8*WG_WIDTH
H = W



design = Design(width=W, height=H, material=Material(1.42**2), pml_size=WL/2)
#design.add(Rectangle(position=(0,WL*2), width=W, height=WG_WIDTH, material=Material(1.42**2)))


tapper = Rectangle(position=(WL*4,WL*2), width=W*1.5, height=WG_WIDTH, material=Material(1.42**2)).rotate(45)
design.add(tapper)

signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*5/LIGHT_SPEED, t_max=TIME/5)
design.add(ModeSource(design=design, start=(1.1*µm, 2.25*µm-1.2*µm), end=(1.1*µm, 2.25*µm+1.2*µm), wavelength=WL, signal=signal))
design.show()

#sim = FDTD(design=design, time=T, mesh="regular", resolution=WL/25)
#sim.run(live=True, axis_scale=[-0.5,0.5])
#sim.plot_power(log_scale=False, db_colorbar=True)