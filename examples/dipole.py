from beamz.sim import Simulation, StandardGrid, PML
from beamz.materials import Material
from beamz.sources import PointSource, Wave
from beamz.const import LIGHT_SPEED

# Parameters
wavelength = 1.55 # all units in µm
frequency = LIGHT_SPEED / wavelength # (m/s)/m = Hz
sim_time, ramp = 50 / frequency, 5 / frequency # s

# Materials
Air = Material(permittivity=1.00058986, permeability=1.00000037)

# Sources
dipole = PointSource(
    position=(150, 150),  # Center of the grid
    signal=Wave(
        amplitude=10.0,
        frequency=LIGHT_SPEED/wavelength,
        ramp_up_time=10*wavelength/LIGHT_SPEED,  # Convert to seconds
        ramp_down_time=10*wavelength/LIGHT_SPEED
    )
)

# Combine all the information into a single simulation object
sim = Simulation(
    size=(300, 300),
    grid=StandardGrid(cell_size=wavelength/20),  # 20 cells per wavelength
    sources=[dipole],
    structures=[PML(thickness=50, sigma_max=1.0, m=3.5)],
    time=sim_time,
    dt=wavelength/LIGHT_SPEED/20
)

# Run the simulation
sim.run(save=True, animate_live=True)