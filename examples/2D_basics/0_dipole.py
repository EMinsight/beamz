import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = Path(__file__).resolve().parents[1]

for path in (REPO_ROOT, EXAMPLES_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from beamz import *
import numpy as np

from _mpl_helpers import save_snapshot_video, show_snapshots

WL = 0.6*µm # wavelength of the source
TIME = 25*WL/LIGHT_SPEED # total simulation duration
N_MEDIUM = 1 # refractive index of the homogeneous background medium
DX, DT = calc_optimal_fdtd_params(WL, N_MEDIUM, dims=2, safety_factor=0.999, points_per_wavelength=8)

# Create a homogeneous design so the example isolates dipole radiation and PML absorption.
design = Design(8*µm, 8*µm, material=Material(N_MEDIUM**2))

time_steps = np.arange(0, TIME, DT)
signal = ramped_cosine(time_steps, amplitude=1.0, frequency=LIGHT_SPEED/WL, ramp_duration=3*WL/LIGHT_SPEED, t_max=TIME/2)
source = GaussianSource(position=(4*µm, 4*µm), width=WL/6, signal=signal)

# Add source and PML boundaries to the simulation.
sim = Simulation(design=design, sources=[source], boundaries=[PML(edges='all', thickness=2*WL)], time=time_steps, resolution=DX)
results = sim.run(snapshot_field="Ez", snapshot_interval=1, store_snapshots=True)

# Save or show the snapshots.
headless = not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")
if headless:
    output = Path.cwd() / "dipole_ez.mp4"
    save_snapshot_video(results.snapshots, filename=output, clean_visualization=True)
    print(f"Saved snapshot video to {output}")
else:
    show_snapshots(results.snapshots, clean_visualization=True)
