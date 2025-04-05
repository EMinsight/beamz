from beamz import *

µm = 1e-6
wavelength = 1.55*µm

# Define SiN and SiO2 materials
SiN = Material(name="SiN", permittivity=2.5, permeability=1.0)
SiO2 = Material(name="SiO2", permittivity=1.45, permeability=1.0)


wg = Rectangle(position=(0,0), width=100, height=100, material=SiN)


# Define waveguide dimensions
width = 100
height = 100
wavelength = 1.55

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