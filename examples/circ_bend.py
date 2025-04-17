from beamz import *
import numpy as np

WL = 1.55*µm
TIME = 20*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.015*WL/LIGHT_SPEED)))
design = Design(width=6*µm, height=6*µm, material=Material(2.1), pml_size=WL/4)
design.add(Rectangle(position=(0,3*µm), width=1.2*µm, height=0.9*µm, material=Material(4)))
design.add(CircularBend(position=(1.2*µm, 1.2*µm), inner_radius=1.8*µm, outer_radius=2.7*µm, angle=90, rotation=0, material=Material(4)))
design.add(Rectangle(position=(3*µm, 0*µm), width=0.9*µm, height=1.2*µm, material=Material(4)))

signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=TIME/3, t_max=TIME)
design.add(ModeSource(design=design, start=(0.5*µm, 2.7*µm), end=(0.5*µm, 4.1*µm), wavelength=WL, signal=signal))
design.show()

sim = FDTD(design=design, time=T, mesh="regular", resolution=WL/40).run()


#sim.run(live=True, axis_scale=[-1,1], save_animation=True, clean_visualization=True, animation_filename="resring2.mp4")