from beamz.design.materials import Material
from beamz.const import µm
from beamz.design.structures import *

# Define the materials used in the design
SiO2 = Material(permittivity=1.45, permeability=1.0, conductivity=0.0)
SiN = Material(permittivity=2.5, permeability=1.0, conductivity=0.0)
# Define the design and show it
design = Design(width=6*µm, height=6*µm, material=SiO2)
design.add(Rectangle(position=(0,1*µm), width=6*µm, height=0.4*µm, material=SiN))
design.add(Ring(position=(3*µm, 3.5*µm), inner_radius=1.5*µm, outer_radius=1.9*µm, material=SiN))
design.add(CircularBend(position=(3*µm, 3.5*µm), inner_radius=0.9*µm, outer_radius=1.3*µm, angle=90, rotation=-180, material=SiN))
design.add(Polygon(vertices=[(3*µm, 3.5*µm), (5.5*µm, 3.5*µm), (6*µm, 6*µm), (4*µm, 6*µm)], material=SiN).shift(0, 0.5*µm))
design.show()