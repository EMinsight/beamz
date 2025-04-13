from beamz import *
import numpy as np

WL = 1.55*µm # wavelength of the light
TIME = 20*WL/LIGHT_SPEED # duration of the simulation
T = np.linspace(0, TIME, int(TIME/(0.01*WL/LIGHT_SPEED))) # time with 200 steps
# Define the materials used in the design
SiN = Material(permittivity=3.5**2, permeability=1.0, conductivity=0.0)  # n ≈ 3.5
SiO2 = Material(permittivity=1.45**2, permeability=1.0, conductivity=0.0)  # n ≈ 1.45
design = Design(width=8*µm, height=4*µm, material=SiO2)
design.add(Rectangle(position=(0*µm, 1.5*µm), width=8*µm, height=1*µm, material=SiN))
design.show()


signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL, phase=0, ramp_duration=TIME/2, t_max=TIME)
source = ModeSource(design=design, start=(1.5*µm, 1.3*µm), end=(1.5*µm, 2.6*µm), wavelength=WL, signal=signal)
source.show()
design.add(source)

# Run FDTD with finer resolution
FDTD(design=design, time=T, mesh="regular", resolution=0.03*µm).run()




