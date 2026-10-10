# Bloch boundaries and meta-atoms

PR #233 supplies uniform normal-incidence excitation and zero-phase periodic
boundaries. Bloch boundaries extend unit-cell simulations to oblique incidence.

```python
import numpy as np
import beamz as bz

frequency = 5e14
angle = np.deg2rad(25)
kx = 2 * np.pi * frequency / bz.LIGHT_SPEED * np.sin(angle)
vector = (kx, 0.0, 0.0)  # rad/m in physical Cartesian coordinates
period = 320 * bz.nm

simulation = bz.Simulation(
    size=(period, period, 3.2 * bz.um),
    sources=[bz.PlaneWaveSource(
        center=(0, 0, -0.9 * bz.um),
        size=(period, period, 0),
        direction="+z",
        transverse_wavevector=vector,
        source_time=bz.GaussianPulse(frequency, 0.05 * frequency),
    )],
    boundaries=[
        bz.Bloch(axes=("x", "y"), wavevector=vector),
        bz.PML(edges=("front", "back"), thickness=0.4 * bz.um,
               formulation="cpml"),
    ],
    grid_spec=bz.GridSpec.uniform(20 * bz.nm, courant=0.7),
    run_time=240e-15,
)
results = simulation.run(backend="jax", progress=False)
```

## Convention and supported execution

The field convention is `exp(-i omega t)` and the forward seam relation is
`F(r + L_i e_i) = exp(+i k_i L_i) F(r)`. Lower-face ghosts therefore use the
inverse phase. Axes are physical Cartesian axes, including in `xy`, `xz`, and
`yz` 2D simulations. Opposite faces belong to one boundary specification;
conflicting periodic/Bloch wavevectors or overlapping PEC/PML faces are rejected.

Nonzero wavevectors use complex64 fields on single-device JAX. Both 2D
polarizations and 3D support uniform and rectilinear derivative metrics, CPML,
and scalar dispersive media. Complex dispersion retains independent states for
both conjugate poles, including normal-interface constitutive updates. Zero
wavevectors preserve the real periodic path. Native CUDA, multi-device Bloch
execution, and Bloch mode sources/monitors are not supported.

The initial oblique plane-wave source supports 3D, a full transverse aperture,
and analytic quadrature pulses with `fwidth <= 0.1 * freq0`. Its transverse
wavevector must match the boundary. The sheet must be inside a homogeneous,
lossless, nondispersive background. `pol_angle=0` selects s polarization and
`pol_angle=pi/2` selects p at oblique incidence. At normal incidence the existing
Cartesian polarization convention remains in effect.

Normal-incidence plane-wave sheets also require a homogeneous, lossless,
nondispersive injection background on uniform and rectilinear grids. Its
refractive index must match `background_index`; materials away from the active
injection support may be lossy or dispersive.

## Fixed angle and frequency

A fixed transverse wavevector gives different incidence angles at different
frequencies. The source's surface-current impedance and longitudinal delay are
computed at its center frequency. Its oblique pulse is consequently a narrowband
approximation, not a broadband fixed-angle source. For a fixed-angle spectrum,
run one narrowband simulation per frequency and recompute the wavevector.
This distinction and the spatial source phase are also described in the
[Meep plane-wave documentation](https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#angular-reflectance-spectrum-of-a-planar-interface).

## Calibrated transmission and phase

Run the empty cell with identical grid, boundaries, source, acquisition window,
and monitor locations. Divide device transmitted flux by reference transmitted
flux. For reflection, subtract reference complex E/H from the incident-side
device acquisition **before** calculating flux. Subtracting powers would retain
incident/reflected interference.

Project the transmitted complex field onto the reference field to obtain the
co-polarized zeroth-order transmission amplitude. Its argument is the device's
phase shift relative to empty-cell propagation. Total transmission may contain
cross polarization and additional diffraction orders.

The runnable `examples/meta_atom_bloch.py` demonstrates a dielectric pillar at
normal and oblique incidence, saves complex amplitude and power, and compares
successive meshes. Its Fourier analysis reports each propagating diffraction
order separately and checks the difference between their sum and total flux.
The monitor must span the full cell and lie in homogeneous vacuum away from the
device's near field. Evanescent orders and exactly grazing orders are excluded
from far-field power. Mesh refinement should reduce the order-sum discrepancy.

```bash
uv run python examples/meta_atom_bloch.py \
  --frequencies 5e14 --angles 0 25 --spacings-nm 40 20 \
  --output /tmp/meta-atom-bloch.json
```

A CPU JAX run at 500 THz produced the following values for the example pillar:

| Angle | Mesh | Transmission | Reflection | Phase relative to empty cell |
| --- | --- | --- | --- | --- |
| 0° | 40 nm | 1.00000 | 0.00000 | 0.6943 rad |
| 0° | 20 nm | 0.99992 | 0.00008 | 0.6550 rad |
| 25° | 40 nm | 0.99496 | 0.00504 | 0.7524 rad |
| 25° | 20 nm | 0.99455 | 0.00545 | 0.7118 rad |

The phase changes by about 0.04 rad under this refinement; two meshes do not
establish convergence. The lossless energy residual was below 2e-6, while the
order-sum discrepancy fell from about 0.0036 to 0.0009. Refine further for phase
accuracy, and repeat with a longer acquisition and thicker CPML to check runtime
and absorber convergence. The full measured output is retained in
`examples/results/meta_atom_bloch.json`.

Independent analytical tests compare oblique dielectric and Lorentz slabs with
Fresnel reflection and complex transmission for s and p polarization. Additional
tests cover extended-domain seam updates, zero-phase equivalence, constitutive
transfer functions, continuation, complex DFT/recording, graded meshes, and
separation of known diffraction orders.

## Metasurface notebooks

`examples/notebooks/huygens_surface.ipynb` follows the
[Tidy3D Huygens’ surfaces community example](https://home.flexcompute.com/tidy3d/community/notebooks/Huygens/):
reference-calibrated transmission and complex phase in its two original
background configurations, followed by a seven-resolution lateral mesh study.
The power-flux ratio is used directly rather than squared a second time.

`examples/notebooks/dielectric_metasurface_absorber.ipynb` follows the
[Tidy3D dielectric absorber example](https://www.flexcompute.com/tidy3d/examples/notebooks/DielectricMetasurfaceAbsorber/):
Drude silicon on conductive PDMS, periodic R/T/A and resonant fields, and a
15 × 15 Gaussian-illuminated array. The unit-cell mesh preserves the published
40-cell wavelength target; the saved finite array uses a clearly labeled coarse
8-cell target, with the source’s 30-cell target available as a parameter.
The Gaussian-beam array and its equally coarse periodic comparison use cubic
uniform cells; the main periodic case uses a graded grid. The uniform mesh
represents the thin PDMS sheet with subpixel material fractions. The comparison
makes mesh shifts visible. Finite-array
`1 - R - T` is apparent absorptance and includes uncollected lateral scattering.

The periodic sections use normal incidence and zero-phase periodic boundaries. They
complement the oblique Bloch script above and retain computed plots and portable
NPZ spectra. See `examples/notebooks/README.md` for execution commands and the
explicit differences in mesh generation, normalization, and collection aperture.
