# Physical CPML alpha: issue #316

The previous automatic alpha was `2 * EPS_0 * a / dt`, where `a` was 0.1 in
2D and 0.05 in 3D. This changed the physical absorber whenever the timestep
changed. The replacement is `0.1 * EPS_0 * LIGHT_SPEED / L`, where `L` is the
resolved physical thickness, in both 2D and 3D. In the CPML recurrence,
`alpha / EPS_0` has units of inverse seconds: the new maximum shift corresponds
to an e-folding time of ten vacuum transits across the layer. This provides a
small, nonzero physical scale without requiring source metadata. The coefficient
is a validated heuristic for the cases below, not an optimality claim.

Explicit alpha, sigma, kappa and the sponge default are preserved. Omitted
thickness still means 12 cells; use a fixed thickness for mesh convergence.
The zero-thickness layer has no active samples and uses one grid spacing solely
to keep automatic alpha metadata finite. `sim.pml_data["resolved_parameters"]`
retains each absorbing boundary's resolved parameters and edges, including when
several boundaries are merged. These are parameter maxima, not sampled maxima.

## Annulus controls

Base: `ab84951b0ed7c6dbf41b9c531e070606bedd1e35`, with this patch. JAX float32 on
RTX 3090, raster extension rebuilt from the base source; independent Meep 1.34.0
on CPU. The driver is `scripts/benchmark_cpml_convergence.py`. The full-tensor
2D TE annulus has radii 0.4/0.25 µm, relative permittivity 12.1/2.1 in vacuum,
1024 polygon vertices in BeamZ and exact cylinders in Meep. Domain 2.4 µm square,
PML 0.4 µm, angular order 6, Courant 0.5 unless stated. A finite Hz drive ends
at time 12 µm/c; two independent damped-sinusoid fit windows cover 20–60 and
30–70 µm/c. Runs end at 75 µm/c. Frequencies below are in c/µm.

[Machine-readable settings and both fits](cpml-physical-alpha-2026-10-07.json).

| Case | Cells/µm | Frequency | Q |
| --- | ---: | ---: | ---: |
| Default, cos | 80 | 1.685295255 | 73.402070 |
| Default, cos | 160 | 1.687312055 | 72.180705 |
| Half timestep, cos | 160 | 1.687254190 | 72.185677 |
| Explicit alpha=0, cos | 160 | 1.687312056 | 72.180716 |
| PML 0.6 µm, domain 2.8 µm, same onset | 160 | 1.687312067 | 72.180653 |
| PML 0.4 µm, domain 2.8 µm, more clearance | 160 | 1.687312066 | 72.180600 |
| Meep, cos | 160 | 1.687307354 | 72.180317 |
| Default, cos | 320 | 1.687889556 | 71.795161 |
| Default, sin | 320 | 1.687904860 | 71.777676 |
| Legacy alpha=339765.59718310 S/m, sin | 320 | 1.688005464 | 72.488817 |
| Meep, sin | 320 | 1.687899636 | 71.777503 |

For L=0.4 µm, the new alpha is 663.604682 S/m at all these mesh/timestep settings;
sigma is 183360.749809 S/m and kappa is 2. Thickness and clearance changes shift
Q by less than 2 ppm at the tested mesh. Halving the timestep changes Q by about
69 ppm while preserving the physical absorber; timestep dispersion remains.
The 320-cell sine case agrees with Meep within 3.1 ppm in frequency and 2.5 ppm
in Q, whereas the legacy alpha reproduces the issue's Q anomaly.

The outgoing-wave analytic annulus pole has frequency 1.688233595 and Q
71.574044. Mesh refinement reduces the remaining error, but same-grid agreement
with Meep does not prove continuum accuracy. This patch does not eliminate
material discretization error.

Example reproduction:

```bash
PYTHONPATH=. python scripts/benchmark_cpml_convergence.py \
  --resolution 320 --angular-phase 1.5707963267948966 --output /tmp/cpml-beamz
python scripts/benchmark_cpml_convergence.py --engine meep \
  --resolution 320 --angular-phase 1.5707963267948966 --output /tmp/cpml-meep
```

Use `--courant 0.25`, `--alpha-max 0`, or `--pml 0.6 --padding 0.2` for the
other controls. The Meep command requires a Python environment containing Meep.

## Regression coverage and limits

- Parameter invariance under mesh/timestep refinement in 2D/3D; unchanged explicit
  alpha; sampled profiles in TE/TM/3D; immutable per-boundary diagnostics.
- Time-separated normal-incidence reflected packet power below −40 dB, both
  polarizations, two timesteps, multiple resolutions and refractive indices.
- Free-space normal/75-degree angular packets and broadband Gaussian-derivative
  pulses: residual field energy below −50 dB at 20 µm/c and −60 dB at 40 µm/c,
  decreasing between those times, for TE and TM.
- Existing guided slab-mode turnoff and PML physics checks pass.
- A 160-cell/µm annulus test compares default and explicit zero alpha for both
  angular partners, with independent source-free fit windows. Both cases fail
  when the old default formula is restored and pass with the physical default.
- Nine existing native CUDA parity cases pass. Two additional cases compare
  fields, CPML memories and accumulated DFTs through four continuation intervals,
  using three separate boundary objects and two timesteps.

The oblique pulse is a finite angular packet, not a near-90-degree plane-wave
reflection oracle. Native CUDA parity is backend evidence, not independent 3D
physical validation. Exhaustive 3D grazing/evanescent sweeps and the downstream
GDS_FDTD reruns requested in the broader issue have not been performed here.
