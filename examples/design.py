from beamz.design.materials import Material
from beamz.design.structures import *

# Define the materials used in the design
SiO2 = Material(permittivity=1.45, permeability=1.0, conductivity=0.0)
SiN = Material(permittivity=2.5, permeability=1.0, conductivity=0.0)

# Define the design and show it
design = Design(width=200e-6, height=200e-6, depth=None)
design.add(Rectangle(position=(0,0), width=200e-6, height=200e-6, material=SiO2))
design.add(Rectangle(position=(0,15e-6), width=200e-6, height=20e-6, material=SiN))
#design.add(Circle(position=(100, 100), radius=50, material=SiN))
design.add(Ring(position=(100e-6, 120e-6), inner_radius=50e-6, outer_radius=70e-6, material=SiN))
design.show()