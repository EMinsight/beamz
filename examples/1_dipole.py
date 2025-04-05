from beamz.const import *
from beamz.design.materials import Material
from beamz.design.structures import *
from beamz.simulation.sources import PointSource
from beamz.simulation.signals import Wave, SigmoidRamp

# Define wavelength and simulation time
WL = 1.55*µm
TIME = 50*WL / LIGHT_SPEED
FREQ = LIGHT_SPEED / (5*WL)
RAMP = FREQ*3
# Create a 2D slice of air
design = Design(20*µm, 20*µm, depth=None, material=Material(permittivity=1.0, permeability=1.0, conductivity=0.0))
design.add(Rectangle(width=10*µm, height=10*µm, material=Material(4.0, 1.0)))
#design.show()
# Add a dipole source
#signal = SigmoidRamp(carrier=Wave(amplitude=1.0, frequency=WL), duration=RAMP, padding=RAMP/2)
# signal = Signal(duration=TIME, step=TIME/1000)
#signal = SigmoidRamp(carrier=Wave(amplitude=1.0, frequency=FREQ), duration=RAMP, padding=0)

signal = Wave(amplitude=1.0, frequency=FREQ)



# Show the signal
t = np.linspace(0, TIME, 1000)
a = signal.get_amplitude(t)
plt.plot(t, a, color='black')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal')
plt.show()



#source = PointSource(position=(100*µm, 100*µm), signal=Wave(amplitude=1.0, frequency=LIGHT_SPEED/WL))

# Setup the FDTD simulation
#sim = FDTD(design=RegularGrid(design, resolution=WL/20), sources=source, time=TIME, ramp=RAMP)
# Run the simulation and live preview the results
#sim.run(save=False, animate_live=True)