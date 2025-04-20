from beamz import *
import numpy as np

WL = 1.55*µm
TIME = 15*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.015*WL/LIGHT_SPEED)))
design = Design(width=WL*6, height=WL*3.5, material=Material(2.1), pml_size=WL/2)
design.add(Rectangle(position=(0,WL*1.75-0.55*µm), width=WL*6, height=1.1*µm, material=Material(4)))
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=TIME/5, t_max=TIME)
design.add(ModeSource(design=design, start=(1.1*µm, WL*1.75-0.55*µm-0.5*µm), end=(1.1*µm, WL*1.75+0.55*µm+0.5*µm), wavelength=WL, signal=signal))
design.show()

sim = FDTD(design=design, time=T, mesh="regular", resolution=WL/30)
sim.run(live=False)

sim.plot_power(log_scale=True)
#sim.run(live=True, axis_scale=[-1,1], save_animation=True, clean_visualization=True, animation_filename="resring2.mp4")