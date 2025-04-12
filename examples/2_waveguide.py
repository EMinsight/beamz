from beamz.design.materials import Material
from beamz.const import µm, LIGHT_SPEED
from beamz.design.structures import *
from beamz.design.sources import ModeSource
from beamz.simulation.signals import ramped_cosine
from beamz.simulation.fdtd import FDTD

WL = 1.55*µm # wavelength of the light
TIME = 20*WL/LIGHT_SPEED
DT = 0.05*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/DT))
# Define the materials used in the design
SiN = Material(permittivity=2.5, permeability=1.0, conductivity=0.0)
design = Design(width=6*µm, height=2.5*µm, material=Material(permittivity=1.45, permeability=1.0, conductivity=0.0))
design.add(Rectangle(position=(0*µm, 1.05*µm), width=6*µm, height=0.4*µm, material=SiN))
signal = ramped_cosine(T, amplitude=1.0, frequency=LIGHT_SPEED/WL*1.5, phase=0, ramp_duration=TIME/2, t_max=TIME)
design.add(ModeSource(design=design, start=(1*µm, 1.6*µm), end=(1*µm, 0.9*µm), wavelength=WL, signal=signal))
design.show()
# Run the FDTD simulation
FDTD(design=design, time=T, mesh="regular", resolution=0.02*µm).run()