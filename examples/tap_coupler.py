from beamz import *
import numpy as np

ANGLE = 5
W = 100*µm; H = 22*µm
WL = 1.55*µm
TIME = 115*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.035*WL/LIGHT_SPEED)))
WG_WIDTH = 2.8*µm; CLAD = 1.4**2; CORE = 1.5**2
design = Design(width=W, height=H, material=Material(CLAD), pml_size=2*WL)
design.add(Rectangle(position=(0,WL*3.5), width=W, height=WG_WIDTH, material=Material(CORE)))
design.add(Rectangle(position=(WL*8,WL*3.5), width=W, height=WG_WIDTH, material=Material(CORE)).rotate(ANGLE, point=(WL*8, WL*3.5)))
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*10/LIGHT_SPEED, t_max=TIME/10)
design.add(ModeSource(design=design, start=(2.7*WL, 2.2*WL), end=(2.7*WL, WG_WIDTH+5*WL), wavelength=WL, signal=signal))
design.show()
sim = FDTD(design=design, time=T, mesh="regular", resolution=WL/15)
sim.run(live=True, axis_scale=[-0.5,0.5], accumulate_power=True, save_memory_mode=True)
sim.plot_power(log_scale=False, db_colorbar=True)