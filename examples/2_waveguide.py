from beamz.design.materials import Material
from beamz.design.structures import *
from beamz.const import *

WAVELENGTH = 1.55*µm

# Define SiN and SiO2 materials
SiN = Material(name="SiN", permittivity=2.5, permeability=1.0)
SiO2 = Material(name="SiO2", permittivity=1.45, permeability=1.0)

wg = Rectangle(position=(0,0), width=100*µm, height=100*µm, material=SiN)


# Define waveguide dimensions
width = 100*µm
height = 100*µm
# Create waveguide structure

# Setup the simulation
sim = bz.Simulation(
    name="waveguide_sim",
    type="2D",
    size=(width, height),
    grid=bz.StandardGrid(cell_size=wavelength/20),
    materials=[SiN, SiO2],
    structures=[],
)

# Run the simulation
results = sim.run(save=True, animate_live=True)