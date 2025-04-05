from beamz.design.materials import Material
from beamz.const import µm, LIGHT_SPEED
from beamz.simulation import FDTD, Wave, PointSource, RegularGrid
from beamz.design.structures import *
# Create a 2D slice of air
design = Design(200*µm, 200*µm, depth=None, material=Material(permittivity=1.0, permeability=1.0))
design.add(PML(thickness=45*µm, sigma_max=1.0, m=3.5))
# Add a dipole source
WAVELENGTH = 1.55*µm
SIM_TIME, RAMP = 50*WAVELENGTH / LIGHT_SPEED, 5*WAVELENGTH / LIGHT_SPEED # s
dipole = PointSource(position=(100*µm, 100*µm),
                     signal=Wave(amplitude=10.0,
                                 frequency=LIGHT_SPEED/WAVELENGTH,
                                 ramp_up_time=RAMP,
                                 ramp_down_time=RAMP))
# Setup the FDTD simulation
sim = FDTD(size=(200*µm, 200*µm),
    design=RegularGrid(design, resolution=WAVELENGTH/20),
    sources=dipole,
    time=SIM_TIME,
    dt=WAVELENGTH/LIGHT_SPEED/20
)
# Run the simulation and live preview the results
sim.run(save=False, animate_live=True)