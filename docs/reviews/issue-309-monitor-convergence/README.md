# Issue #309: sharp S-bend monitor dependence

The reported plane dependence has a material-consistency contribution. The
3D modal projector discarded the component material coefficients used by
propagation and reconstructed coefficients from scalar permittivity. For the
default Farjadpour diagonal raster smoothing, these are different constitutive
models. Identical effective indices at different output planes do not detect
this mismatch: all those planes solve the same incorrect material model.

The correction retains the cell material tensors and direct Yee coefficients
in detached monitor material regions and passes them to the shared mode-plane
solver. It also handles full-domain material snapshots. The retained Yee
regions include an upper support sample, since node-aligned components extend
beyond the corresponding voxel support. Sources and propagation are unchanged.

## Experiment

Measured locally on an RTX 3090 with JAX 0.9.0/CUDA 12, NumPy 2.4.1, SciPy
1.17.0, and BeamZ 0.5.2 checkout `ee092cb6` plus the material-retention fix.
The unmodified mesh-10 reproduction agrees with the issue's JAX 0.10.2 values
to within 1e-6 dB across all 15 samples.

The fixture is the standalone geometry supplied in
[issue #309](https://github.com/beamzorg/beamz/issues/309), originating from
GDS_FDTD `sbend_dontfabme` at commit
`66b28f1c26649d4362aec2375d891ad0cd60c67e` (`devices.gds` SHA256
`efcb1f267ccc1b0296c6531814127b29c080e6c7b9b03c126f75f1e9849218ee`).
The committed JSON fixture avoids a GDS or PDK dependency.

All output probes within each sweep sample one FDTD run. Source and input
monitor stay fixed at 1.4 and 0.5 µm before the bend. Output distances are
0.25, 0.50, 0.75, 1.00, 1.25, 1.50, and 1.75 µm beyond the bend. Aperture widths
are 1.35, 2.7, and 4.0 µm, with heights scaled from the original 2.7 × 2.196 µm
aspect ratio. Frequencies correspond directly to 1.6, 1.55, and 1.5 µm.

The straight control replaces the device and leads with a uniform 0.5 × 0.22 µm
waveguide and aligns the output center with the input. It keeps the source,
input plane, domain, pulse, and sampling settings unchanged. No commercial
solver was rerun and no probe was selected for agreement with a reference.

The original absorber-normal material-variation warning remains visible for
the bend. No boundary modification is mixed into the extraction comparison.
The outermost output plane is only 0.25 µm before the nominal absorber; its
inclusion is a diagnostic, not a recommendation for absorber clearance.
Likewise, the width-4 aperture has a height of 3.2533 µm and reaches about
0.017 µm into the nominal 1 µm vertical absorber margin (actual absorber edges
are grid-rounded). Treat that largest aperture as a boundary-adjacent diagnostic.
A clean aperture-convergence claim also needs a larger vertical domain or a
smaller clear aperture; boundary/domain convergence was not established here.

## Measured results

At **1.55 µm**, using the original four straight-lead planes (0.25–1.00 µm):

| Mesh | Aperture (µm) | Original spread (dB) | Corrected spread (dB) |
|---:|---|---:|---:|
| 10 | 2.7 × 2.196 | 0.314585 | 0.096661 |
| 10 | 4.0 × 3.2533 | 0.267890 | 0.052641 |
| 20 | 2.7 × 2.196 | 0.130122 | 0.026648 |
| 20 | 4.0 × 3.2533 | 0.125453 | 0.021645 |

Corrected transmission at all sampled straight-lead planes:

| Distance (µm) | Mesh 10, width 2.7 (dB) | Mesh 10, width 4 (dB) | Mesh 20, width 2.7 (dB) | Mesh 20, width 4 (dB) |
|---:|---:|---:|---:|---:|
| 0.25 | -5.794066 | -5.794292 | -5.661466 | -5.674706 |
| 0.50 | -5.760451 | -5.772871 | -5.660597 | -5.677538 |
| 0.75 | -5.791937 | -5.781329 | -5.668655 | -5.681986 |
| 1.00 | -5.857111 | -5.825511 | -5.687245 | -5.696352 |
| 1.25 | -5.850577 | -5.825513 | -5.670702 | -5.680157 |
| 1.50 | -5.810594 | -5.805403 | -5.671689 | -5.684456 |
| 1.75 | -5.771195 | -5.775312 | -5.665571 | -5.679926 |

Across all seven planes, the corrected width-4 spread is **0.052643 dB** at
mesh 10 and **0.021645 dB** at mesh 20. The maximum matched-plane change between
widths 2.7 and 4 is **0.031600 dB** at mesh 10 and **0.016941 dB** at mesh 20.
The width-1.35 aperture still gives a noticeably different transmission level;
a smaller plane spread there is insufficient evidence of aperture convergence.

The corrected straight-guide control at mesh 10 has a seven-plane spread of
**0.042917 dB** at width 2.7 and **0.043008 dB** at width 4. The comparable
sampling floor in a nonradiating guide rules out attributing all remaining
variation to bend radiation. The narrow width-1.35 aperture gives an apparent
positive transmission of roughly 0.14–0.18 dB in this lossless control, another
reason to check aperture dependence rather than merely plane spread.

For this sampled setup, mesh 20 and widths 2.7–4 show plane/aperture changes
within 0.05 dB over 0.25–1.75 µm. The boundary-adjacent widest aperture prevents
interpreting that as a clean aperture-convergence certificate. These results
also do **not** establish absolute
mesh convergence: matched-plane mesh-10 to mesh-20 changes reach **0.179876 dB**
at width 2.7 and **0.145357 dB** at width 4. A 0.01 dB specification is not met.
No minimum distance independent of mesh, aperture, and requested tolerance is
supported by these measurements.

All bend runs terminated as `converged`; mesh 10 used 8,448 steps and mesh 20
used 16,384 steps. Incident
validity masks passed. The committed records retain the termination residuals,
incident powers, effective indices, and raw modal amplitudes.

## Interpretation and limits

The material correction accounts for much of the original same-run variation.
Increasing the aperture alone did not remove the old variation. Remaining
plane spread and aperture offsets are measurable: this is not evidence that
mesh 10 is fully converged or that a universal minimum distance exists.

The remaining error has not been fully separated into normal interpolation,
transverse colocation, finite-aperture truncation, and radiation contributions.
The straight-guide control establishes that radiation alone is insufficient to
explain it. A correction to monitor sampling itself would require separate
validation; this change only restores the material contract shared with sources.

A large projection residual in a radiating field is not automatically an
extraction failure. Conversely, a well-conditioned modal basis and a converged
temporal termination report cannot certify spatial convergence. The regression
bounds below intentionally do not assert agreement with the historical
commercial-engine insertion loss.

For downstream compact models, use the
[plane/aperture/mesh protocol](../../modal-monitor-convergence.md). Report the
spread over eligible planes, aperture changes, and mesh changes separately.
If the desired error is below the remaining spread, refine further and extend
the lead/domain as necessary. Do not label one monitor's value spatially
converged merely because time stepping has stopped.

## Reproduction and regression coverage

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
  python -m tests.characterization.sbend_monitor_case --mesh 10 --output corrected10.json
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
  python -m tests.characterization.sbend_monitor_case --mesh 20 --output corrected20.json
# Add --straight to replace the bend with a uniform guide.
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
  python -m pytest tests/characterization/test_sbend_monitor_planes.py
```

The GPU characterization checks the four original straight-lead planes at mesh
10: spread below 0.12 dB for width 2.7 µm, below 0.08 dB for width 4.0 µm, and
matched-plane aperture differences below 0.04 dB. The scalar-only baseline fails
the plane-spread bounds. These bounds guard the defect; they are not an accuracy
specification. This test is marked `hardware`, `slow`, and `characterization`,
and is excluded from the compact CPU gate.

The fast contracts in `tests/contracts/test_modal_material_snapshot.py` cover
x-, y-, and z-normal planes with both thin and full material snapshots. They
check component coefficient values, crop origins, immutability, independence
from the original arrays, and delivery to the mode solver. Existing scalar
material snapshots continue to use the scalar reconstruction fallback.

## Raw records

- [Original mesh-10 sweep](baseline10.json): three wavelengths, original
  `out_0`–`out_4` probes plus the aperture/distance matrix. `out_0` is inside the
  bend and is excluded from every straight-lead spread.
- [Original mesh-20 sweep](baseline20.json): the same aperture/distance matrix
  at all three wavelengths.
- [Corrected mesh-10 sweep](corrected10.json): a fresh end-to-end run through
  production result detachment and analysis, at all three wavelengths.
- [Corrected mesh-20 extraction](corrected20-center.json): direct 1.55 µm
  extraction from saved baseline fields after restoring the original raster's
  coefficients to the detached material snapshots. This changes only analysis;
  it is not a second FDTD run. The fresh mesh-10 run independently confirms this
  re-extraction procedure.
- [Original straight control](straight-baseline10.json) and
  [corrected straight extraction](straight-corrected10-center.json): the latter
  uses the same saved fields with the actual straight-guide raster coefficients.

The two-resolution results are characterization, not a Richardson convergence
order estimate. The standalone script produces fresh corrected results without
using any of these recorded files.

## Verification

- `make audit`: 1,974 passed, 482 deselected, one pre-existing expected failure;
  lint, type checking, dead-code checks, and every coverage policy passed.
- GPU S-bend characterization: passed on the RTX 3090.
- Material-snapshot contracts: all six axis/storage combinations passed.
- `make build`: wheel, source distribution, and distribution checks passed.
- Strict MkDocs build: passed.

The [full notebook regression checks](notebook-regressions.md) additionally run
both the modal sources/monitors and cosine crossing notebooks on the baseline
and corrected code. Recorded fields and flux remain identical; expected changes
in modal decomposition are quantified separately.
