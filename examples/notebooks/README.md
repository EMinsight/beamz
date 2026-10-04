# Metasurface notebooks

These two independently implemented Beamz notebooks follow the linked Tidy3D
examples' physical setups, experiment order, and plot layouts. Saved notebook
outputs contain actual Beamz FDTD results. The adjacent `_metasurface_utils.py`
provides geometry plotting and monitor-acquisition helpers; retain it when
copying a notebook.

- [Huygens' dielectric surfaces](huygens_surface.ipynb): silicon disks with
  242 nm radius, 220 nm height, and 666 nm period; 1.1–1.6 µm transmission and
  phase spectra in the source's two separate backgrounds; empty/device
  calibration; and all seven lateral mesh resolutions from P/2 to P/128.
  [Tidy3D community source](https://home.flexcompute.com/tidy3d/community/notebooks/Huygens/).
- [Dielectric metasurface absorber](dielectric_metasurface_absorber.ipynb):
  Drude silicon resonators on lossy PDMS; periodic-cell R/T/A and resonant field
  maps over 0.4–0.8 THz; a 15 × 15 finite array under Gaussian illumination;
  and a comparison against a periodic cell at the same coarse mesh target.
  [Tidy3D example source](https://www.flexcompute.com/tidy3d/examples/notebooks/DielectricMetasurfaceAbsorber/).

## Run

From the repository root, install the development environment and register its
kernel. CPU JAX works; for the finite array a GPU with a compatible JAX CUDA
installation is recommended.

```bash
uv sync --extra dev
# Optional NVIDIA GPU support, using the CUDA version supported by your machine:
uv pip install 'jax[cuda12]==0.9.0'  # Match this repository's locked JAX version.
uv run python -m ipykernel install --user --name beamz-notebooks --display-name Beamz
uv run python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=beamz-notebooks \
  --ExecutePreprocessor.timeout=-1 examples/notebooks/huygens_surface.ipynb
uv run python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=beamz-notebooks \
  --ExecutePreprocessor.timeout=-1 examples/notebooks/dielectric_metasurface_absorber.ipynb
```

All runs explicitly select the single-device JAX backend, including on a GPU.
The final cells write portable numerical spectra to `examples/results/*.npz`.
Those files are outputs, not an execution cache; rerunning recomputes the
simulations. Saved plots can be inspected without any execution.

Both saved notebooks executed on an NVIDIA RTX 3090 (24 GB) with Python 3.11.15
and JAX 0.10.2. The 93 focused solver regression tests passed separately in the
repository's CPU JAX 0.9.0 environment. Execution and validation details are in
`examples/results/metasurface_notebooks_validation.json`.

## Differences from the source notebooks

The main periodic cases use explicit graded rectilinear meshes and Beamz's native
cylinder rasterization/subpixel averaging. Their interfaces are snapped to grid edges;
the reference and device runs use identical grids. Grid locations and subpixel
algorithms differ from Tidy3D, so identical mesh targets do not imply identical
spectra. We do not apply Tidy3D's mirror-symmetry reductions.

The Huygens notebook corrects the source's extra squaring of a power-flux ratio.
It extracts phase from area-averaged complex field amplitudes and keeps source
and monitor positions fixed during mesh refinement. Its mesh study changes
only lateral spacing; vertical and runtime convergence require further checks.

The absorber retains Tidy3D's Drude coefficients with an explicit cyclic-to-angular
frequency conversion, and represents PDMS with the same constant-permittivity /
constant-conductivity model as `Medium.from_nk`. Its periodic mesh has the
source's 40-cell wavelength target. The saved 225-disk finite-array example uses
a **coarse 8-cell target**, instead of the source's 30, to fit a 24 GB GPU and
32 GB system RAM. Change
`finite_steps_per_wavelength` to 30 for a larger-machine reproduction. The
notebook prints the cell budget for both choices. The Gaussian beam source currently
requires a cubic uniform mesh, so the finite array and its coarse periodic
comparison use uniform cells. Their domain extents expand by less than one cell
to accommodate integer cell counts. The 8 µm PDMS sheet is thinner than a coarse
cell and is represented by native subpixel material fractions. Sources and power monitors
exclude transverse CPML, and the finite-array comparison includes an equally
coarse periodic run. Finite-array `1 - R - T` includes uncollected lateral
scattering and is labeled apparent absorptance.

The original notebook authors and research papers are credited in each notebook.
The notebooks reproduce the physical experiments with Beamz code rather than
including source notebook code or plots.
