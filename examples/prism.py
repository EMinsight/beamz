from beamz.design.materials import Material
from beamz.const import µm
from beamz.design.structures import *

# Define the materials used in the design
Air = Material(permittivity=1.0, permeability=1.0, conductivity=0.0)
Crystal = Material(permittivity=2.5, permeability=1.0, conductivity=0.0)

# Define the design and show it
design = Design(width=10*µm, height=10*µm, material=Air, color=rgb_to_hex(0, 0, 0))
T = Polygon(vertices=[(0*µm, 0*µm), (4*µm, 0*µm), (2*µm, 3.1*µm)], material=Crystal, color=rgb_to_hex(50, 50, 50)).shift(3*µm, 3.5*µm).scale(1.5)
design.add(T)
design.show()

#wavelengths = [0.5, 0.6, 0.7] # µm (red, green, blue)


# Add sources and monitors
#design.add(Source(position=(0*µm, 1.25*µm), angle=90, wavelength=1.55*µm))
#design.add(Monitor(position=(3.1*µm, 1.25*µm), angle=90, wavelength=1.55*µm))

# Run the simulation
#design.run()
