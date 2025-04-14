from beamz import *
import numpy as np

WL = 1.55*µm
TIME = 40*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.015*WL/LIGHT_SPEED)))
design = Design(width=6*µm, height=6*µm, material=Material(2.1))
design.add(Rectangle(position=(0,1*µm), width=6*µm, height=0.4*µm, material=Material(6.25)))
design.add(Ring(position=(3*µm, 3.35*µm), inner_radius=1.5*µm, outer_radius=1.9*µm, material=Material(6.25)))
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=TIME/2, t_max=TIME)
design.add(ModeSource(design=design, start=(1.5*µm, 0.8*µm), end=(1.5*µm, 1.6*µm), wavelength=WL, signal=signal))
FDTD(design=design, time=T, mesh="regular", resolution=WL/40).run(live=True, axis_scale=[-1,1])