from beamz.sim import Simulation, StandardGrid
from beamz.materials import Material
from beamz.sources import PointSource, Wave
from beamz.const import LIGHT_SPEED

# Parameters
wavelength = 1.55 # all units in µm
frequency = LIGHT_SPEED / wavelength # (m/s)/m = Hz
ramp = 10 / frequency # s
sim_time = 100 / frequency

# Materials
Air = Material(
    permittivity=1.00058986,  # Relative permittivity at STP (0°C, 1 atm)
    permeability=1.00000037,  # Relative permeability at STP (0°C, 1 atm)
    conductivity=0.0,         # Perfect insulator
    color="white"
)

# Sources
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

# Combine all the information into a single simulation object
sim = Simulation(
    name="dipole_sim",
    type="2D",
    size=(100, 100),
    grid=StandardGrid(cell_size=1),
    structures=None,
    sources=[dipole],
    monitors=None,
    device="cpu"
)

# Run the simulation
results = sim.run(save=True, animate_live=True)
# Visualize the results
results.plot_field(field="Ez", t=sim_time/2)