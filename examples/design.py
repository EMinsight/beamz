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
design.add(CircularBend(position=(3*µm, 3.5*µm), inner_radius=0.9*µm, outer_radius=1.3*µm, angle=90, rotation=-200, material=SiN))

design.scatter()

p = Polygon(vertices=[(3*µm, 3.5*µm), (5.5*µm, 3.5*µm), (6*µm, 6*µm), (4*µm, 6*µm)], material=SiN)
p.shift(-1.5*µm, -1.5*µm)
p.scale(0.04)

for i in range(1000):
    new_p = p.copy()
    new_p.shift(random.uniform(-5*µm, 5*µm), random.uniform(-5*µm, 5*µm))
    new_p.scale(random.uniform(0.05, 1))
    new_p.rotate(random.uniform(0, 360))
    design.add(new_p)

design.show()