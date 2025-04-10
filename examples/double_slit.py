from beamz.design.materials import Material
from beamz.const import µm, LIGHT_SPEED
from beamz.design.structures import *
from beamz.simulation.signals import plot_signal, ramped_cosine
from beamz.simulation.fdtd import FDTD

WL = 1.55*µm # wavelength of the light
TIME = 20*WL/LIGHT_SPEED
DT = 0.05*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/DT))

Air = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
design = Design(width=4*µm, height=4*µm, material=Air, border_color="black")
design.add(Rectangle(position=(2*µm, 0*µm), width=1*µm, height=4*µm, material=Air))
design.add(Rectangle(position=(0*µm, 2*µm), width=4*µm, height=1*µm, material=Air))
signal = ramped_cosine(T, amplitude=1.0, frequency=(LIGHT_SPEED/WL)*1.5, phase=0, ramp_duration=TIME/2, t_max=TIME)
design.show()

FDTD(design=design, mesh="regular", resolution=0.02*µm).run()