from beamz.sim import Simulation, StandardGrid
from beamz.matlib import Material
from beamz.sources import Dipole, Wave
from beamz.const import LIGHT_SPEED

# Materials
Air = Material(
    name="Air",
    permittivity=1.0,
    permeability=1.0,
    conductivity=0.0,
    color="white"
)

# Structures
# no structures in this example



# Sources
wavelength = 1550e-9 # m
frequency = LIGHT_SPEED / wavelength # (m/s)/m = Hz
ramp = 10 / frequency # s

dipole = PointSource(
    position=(0, 0),
    signal=Wave(
        direction=(0, 1),
        amplitude=1.0,
        frequency=LIGHT_SPEED/wavelength,
        ramp_up_time=10*wavelength, # 10 wavelengths
        ramp_down_time=10*wavelength
    )
)

# Detectors
# no detectors in this example

# Combine all the information into a single simulation object
sim = Simulation(
    name="dipole_sim",
    type="2D",
    size=(100, 100),
    grid=StandardGrid(cell_size=1),
    structures=,
    sources=[dipole],
    monitors=[]
)

# Rund the simulation
sim.run()

