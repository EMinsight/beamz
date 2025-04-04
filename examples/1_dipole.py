from beamz import *

µm = 1e-6
wavelength = 1.55*µm
sim_time, ramp = 50*wavelength / LIGHT_SPEED, 5*wavelength / LIGHT_SPEED # s

dipole = PointSource(
    position=(100*µm, 100*µm),  # Center of the grid
    signal=Wave(
        amplitude=10.0,
        frequency=LIGHT_SPEED/wavelength,
        ramp_up_time=10*wavelength/LIGHT_SPEED,  # Convert to seconds
        ramp_down_time=10*wavelength/LIGHT_SPEED
    )
)

sim = FDTD(
    size=(200*µm, 200*µm),
    grid=StandardGrid(cell_size=wavelength/20), # 1.55/20 µm
    sources=[dipole],
    boundaries=Boundaries(all=PML(thickness=40, sigma_max=1.0, m=3.5)),
    time=sim_time,
    dt=wavelength/LIGHT_SPEED/20
)

sim.run(save=False, animate_live=True)