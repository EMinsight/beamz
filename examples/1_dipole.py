from beamz.const import *
from beamz.design.materials import Material
from beamz.design.structures import *
from beamz.design.sources import PointSource
from beamz.simulation.signals import cosine, sigmoid, plot_signal
from beamz.simulation.meshing import RegularGrid
from beamz.simulation.fdtd import Simulation

# Define wavelength and simulation time
WL = 1.55*µm
FREQ = LIGHT_SPEED/WL
# Create a 2D slice of air
design = Design(20*µm, 20*µm, depth=None, material=Material(1.0, 1.0, 0.0))
design.add(Rectangle(width=10*µm, height=10*µm, material=Material(4.0, 1.0)))
design.show()
# Set time for the simulation
TIME = 20*WL/LIGHT_SPEED
DT = 0.05*WL/LIGHT_SPEED
T = np.linspace(0, TIME, int(TIME/DT))
print(f"TIME: {TIME}, DT: {DT}, STEPS: {int(TIME/DT)}")
# Define the signal for the first dipole
signal_1 = sigmoid(T, min=0, max=1, duration=TIME/4, t0=TIME/16)
signal_1 *= cosine(T, amplitude=1.0, frequency=FREQ, phase=0)
signal_1 *= sigmoid(T, min=1, max=0, duration=TIME/4, t0=TIME - TIME/4 - TIME/16)
plot_signal(signal_1, T)
# Define the signal for the second dipole
signal_2 = sigmoid(T, min=0, max=1, duration=TIME/4, t0=TIME/16)
signal_2 *= cosine(T, amplitude=1.0, frequency=FREQ, phase=np.pi/2)
signal_2 *= sigmoid(T, min=1, max=0, duration=TIME/4, t0=TIME - TIME/4 - TIME/16)
plot_signal([signal_1, signal_2], T)
# Setup the FDTD simulation, converting the design into a grid, 
sim = Simulation(design=RegularGrid(design, resolution=WL/20), sources=PointSource(position=(100*µm, 100*µm), signal=None))
# Show the rasterized design and sources
sim.show()
# Run the simulation and live preview the results
sim.run(save=False, animate_live=True)