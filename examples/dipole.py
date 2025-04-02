from beamz.sim import Simulation, StandardGrid
from beamz.materials import Material
from beamz.sources import PointSource, Wave
from beamz.const import LIGHT_SPEED

# Parameters
wavelength = 1.55 # all units in µm
frequency = LIGHT_SPEED / wavelength # (m/s)/m = Hz
ramp = 10 / frequency # s
sim_time = 100 / frequency # Total simulation time

# Materials
Air = Material(
    name="air",
    permittivity=1.00058986,  # Relative permittivity at STP (0°C, 1 atm)
    permeability=1.00000037,  # Relative permeability at STP (0°C, 1 atm)
    conductivity=0.0,         # Perfect insulator
    color="white"
)

# Sources
dipole = PointSource(
    position=(50, 50),  # Center of the grid
    signal=Wave(
        direction=(0, 1),
        amplitude=1.0,
        frequency=LIGHT_SPEED/wavelength,
        ramp_up_time=10*wavelength/LIGHT_SPEED,  # Convert to seconds
        ramp_down_time=10*wavelength/LIGHT_SPEED
    )
)

# Combine all the information into a single simulation object
sim = Simulation(
    name="dipole_sim",
    type="2D",
    size=(100, 100),
    grid=StandardGrid(cell_size=wavelength/20),  # 20 cells per wavelength
    structures=None,
    sources=[dipole],
    monitors=None,
    device="cpu"
)

# Set simulation time
sim.time = sim_time
sim.num_steps = int(sim_time / sim.dt)

# Run the simulation
results = sim.run(save=True, animate_live=True)

# Visualize the results
sim.plot_field(field="Ez", t=sim_time)