import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = Path(__file__).resolve().parents[1]

for path in (REPO_ROOT, EXAMPLES_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from beamz import (
    LIGHT_SPEED,
    Design,
    GaussianSource,
    Material,
    Monitor,
    PML,
    Rectangle,
    Simulation,
    calc_optimal_fdtd_params,
    ramped_cosine,
    µm,
)

WL = 1.0 * µm
N_CORE, N_CLAD = 2.0, 1.0
DX, DT = calc_optimal_fdtd_params(
    WL,
    max(N_CORE, N_CLAD),
    dims=2,
    safety_factor=0.95,
    points_per_wavelength=10,
)
time = np.arange(0, 18 * WL / LIGHT_SPEED, DT)

design = Design(width=6 * µm, height=4 * µm, material=Material(N_CLAD**2))
design += Rectangle(
    position=(0, 2 * µm - 0.25 * µm),
    width=6 * µm,
    height=0.5 * µm,
    material=Material(N_CORE**2),
)

signal = ramped_cosine(
    time,
    amplitude=1.0,
    frequency=LIGHT_SPEED / WL,
    ramp_duration=3 * WL / LIGHT_SPEED,
    t_max=time[-1] / 2,
)
source = GaussianSource(position=(1.2 * µm, 2 * µm), width=0.2 * µm, signal=signal)
monitor = Monitor(
    start=(4.8 * µm, 1.4 * µm),
    end=(4.8 * µm, 2.6 * µm),
    name="output_line",
)

sim = Simulation(
    design=design,
    sources=[source],
    monitors=[monitor],
    boundaries=[PML(edges="all", thickness=0.8 * µm)],
    time=time,
    resolution=DX,
)

source_ds = source.to_xarray(t=time)
source_ds["signal"].plot()

results = sim.run(save_fields=["Ez"], field_subsample=5, progress=False)
field_ds = results.to_xarray()

last_ez = field_ds["Ez"].isel(t=-1)
last_ez.plot(x="x", y="y", cmap="RdBu")

center_ez = field_ds["Ez"].sel(y=2.0 * µm, method="nearest")
center_ez.plot(x="x", y="t", cmap="RdBu")

monitor_ds = results.monitor_results["output_line"].to_xarray()
monitor_ds["power"].plot()
