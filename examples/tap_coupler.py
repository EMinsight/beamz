from beamz import *
import numpy as np

WL = 1.55*µm
TIME = 115*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/(0.04*WL/LIGHT_SPEED)))

ANGLE = 5
WG_WIDTH = 3.2*µm
W = 100*µm
H = 16*µm


clad = 1.42**2
core = 1.45**2
print(f"clad: {clad}, core: {core}")


design = Design(width=W, height=H, material=Material(clad), pml_size=WL)
design.add(Rectangle(position=(0,WL*2), width=W, height=WG_WIDTH, material=Material(core)))
design.add(Rectangle(position=(WL*4,WL*2), width=W, 
                   height=WG_WIDTH, material=Material(core)).rotate(ANGLE, point=(WL*4, WL*2)))


signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=WL*5/LIGHT_SPEED, t_max=TIME/5)
design.add(ModeSource(design=design, start=(1.5*WL, WL), end=(1.5*WL, WG_WIDTH+3*WL), wavelength=WL, signal=signal))
design.show()

sim = FDTD(design=design, time=T, mesh="regular", resolution=WL/10)
print(sim.estimate_memory_usage())
sim.run(live=False, axis_scale=[-0.5,0.5], accumulate_power=True, decimate_save=10)
sim.plot_power(log_scale=False, db_colorbar=True)