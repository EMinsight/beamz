from beamz import *
import numpy as np

WL = 1.55*µm # wavelength of the light
TIME = 20*WL/LIGHT_SPEED # duration of the simulation
T = np.linspace(0, TIME, int(TIME/200)) # time with 200 steps
# Define the materials used in the design
SiN = Material(permittivity=2.5, permeability=1.0, conductivity=0.0)
design = Design(width=6*µm, height=2.5*µm, material=Material(permittivity=1.45, permeability=1.0, conductivity=0.0))
design.add(Rectangle(position=(0*µm, 1.05*µm), width=6*µm, height=0.4*µm, material=SiN))
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL*1.5, phase=0, ramp_duration=TIME/2, t_max=TIME)
design.add(ModeSource(design=design, start=(1*µm, 1.6*µm), end=(1*µm, 0.9*µm), wavelength=WL, signal=signal))
# Run the FDTD simulation with a regular mesh of 20 nm resolution
FDTD(design=design, time=T, mesh="regular", resolution=0.02*µm).run()