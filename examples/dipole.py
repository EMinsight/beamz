from beamz import *

wavelength = 1.55 # all units in µm
sim_time, ramp = 50*wavelength / LIGHT_SPEED, 5*wavelength / LIGHT_SPEED # s

# Air = Material(permittivity=1.00058986, permeability=1.00000037)

dipole = PointSource(
    position=(150, 150),  # Center of the grid
    signal=Wave(
        amplitude=10.0,
        frequency=LIGHT_SPEED/wavelength,
        ramp_up_time=10*wavelength/LIGHT_SPEED,  # Convert to seconds
        ramp_down_time=10*wavelength/LIGHT_SPEED
    )
)

sim = Simulation(
    size=(300, 300),
    grid=StandardGrid(cell_size=wavelength/20),  # 20 cells per wavelength
    sources=[dipole],
    boundaries=Boundaries(
        top=PML(thickness=50, sigma_max=1.0, m=3.5),
        bottom=PML(thickness=50, sigma_max=1.0, m=3.5),
        left=PML(thickness=50, sigma_max=1.0, m=3.5),
        right=PML(thickness=50, sigma_max=1.0, m=3.5)
    ),
    time=sim_time,
    dt=wavelength/LIGHT_SPEED/20
)

sim.run(save=False, animate_live=False)