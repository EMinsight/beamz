from beamz.design.materials import Material
from beamz.design.structures import *
µm = 1e-6

# Define the materials used in the design
SiO2 = Material(permittivity=1.45, permeability=1.0, conductivity=0.0)
SiN = Material(permittivity=2.5, permeability=1.0, conductivity=0.0)
# Define the design and show it
design = Design(width=6*µm, height=6*µm, depth=None)
design.add(Rectangle(position=(0,0), width=6*µm, height=6*µm, material=SiO2))
design.add(Rectangle(position=(0,1*µm), width=6*µm, height=0.4*µm, material=SiN))
design.add(Ring(position=(3*µm, 3.5*µm), inner_radius=1.5*µm, outer_radius=1.9*µm, material=SiN))
design.show()




