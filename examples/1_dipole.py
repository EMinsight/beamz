from beamz.const import *
from beamz.design.materials import Material
from beamz.design.structures import *
from beamz.simulation.sources import PointSource
from beamz.simulation.signals import cosine, sigmoid, plot_signal

# Define wavelength and simulation time
WL = 1.55*µm
TIME = 50*WL/LIGHT_SPEED
FREQ = LIGHT_SPEED/WL

# Create a 2D slice of air
design = Design(20*µm, 20*µm, depth=None, material=Material(permittivity=1.0, permeability=1.0, conductivity=0.0))
design.add(Rectangle(width=10*µm, height=10*µm, material=Material(4.0, 1.0)))
design.show()

# Define the signal function
t = np.linspace(0, TIME, 1000)
signal = sigmoid(t, min=0, max=1, duration=TIME/4, t0=TIME/16)
signal *= cosine(t, amplitude=1.0, frequency=FREQ, phase=0)
signal *= sigmoid(t, min=1, max=0, duration=TIME/4, t0=TIME - TIME/4 - TIME/16)
plot_signal(signal, t)

# Define the source as a point in the design modulated by the signal
source = PointSource(position=(100*µm, 100*µm), signal=signal)

# Setup the FDTD simulation, converting the design into a grid, 
sim = FDTD(design=RegularGrid(design, resolution=WL/20), sources=source)

# Run the simulation and live preview the results
sim.run(save=False, animate_live=True)