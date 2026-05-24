import numpy as np

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    Material,
    ModeSource,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    µm,
)

WL = 1.55 * µm
TIME = 45 * WL / LIGHT_SPEED
N_CORE, N_CLAD = 2.04, 1.444  # Si3N4, SiO2
WG_WIDTH = 0.565 * µm
PML_THICKNESS = 1.2 * WL
SOURCE_X = PML_THICKNESS + 1.0 * WL
SOURCE_AMPLITUDE = 1e-4
DX, DT = calc_optimal_fdtd_params(
    WL, max(N_CORE, N_CLAD), safety_factor=0.999, points_per_wavelength=20
)

# Create the design
design = Design(width=18 * µm, height=7 * µm, material=Material(N_CLAD**2))
design += Rectangle(
    position=(0, 3.5 * µm - WG_WIDTH / 2),
    width=18 * µm,
    height=WG_WIDTH,
    material=Material(N_CORE**2),
)
design.show()

# Rasterize the design
grid = design.rasterize(resolution=DX)
grid.show(field="permittivity")

# Create the signal & source
time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(
    time_steps,
    amplitude=SOURCE_AMPLITUDE,
    frequency=LIGHT_SPEED / WL,
    phase=0,
    ramp_duration=WL * 8 / LIGHT_SPEED,
    t_max=TIME,
)
source = ModeSource(
    grid=grid,
    center=(SOURCE_X, design.height / 2),
    width=WG_WIDTH
    * 3.5,  # Slightly wider than waveguide to capture mode tails, but not so wide to hit PML/boundaries
    wavelength=WL,
    pol="tm",
    signal=signal,
    direction="+x",
)
source.show_signal(t=time_steps)

# Run the simulation
sim = Simulation(
    design=design,
    sources=[source],
    boundaries=[PML(edges="all", thickness=PML_THICKNESS)],
    time=time_steps,
    resolution=DX,
)
results = sim.run(save_fields=["Ez"], field_subsample=20, progress=False)
results.show(field="Ez", cmap="RdBu")
