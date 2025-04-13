from beamz import *
import numpy as np

WL = 1.55*µm
TIME = 20*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.05*WL/LIGHT_SPEED)))
design = Design(20*µm, 20*µm, depth=None, material=Material(1.0, 1.0, 0.0))
design.add(Rectangle(width=10*µm, height=10*µm, material=Material(4.0, 1.0)))
signal_1 = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=TIME/2)
signal_2 = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=np.pi/2, ramp_duration=TIME/2)
design.add(PointSource(position=(10*µm, 10*µm), signal=signal_1))
design.add(PointSource(position=(10*µm, 15*µm), signal=signal_2))
FDTD(design=design, time=T, mesh="regular", resolution=WL/20).run()