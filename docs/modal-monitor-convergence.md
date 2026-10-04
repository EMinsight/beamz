# Modal monitor convergence

An automatically terminated simulation can still give position-dependent modal
transmission. `termination.converged` describes temporal decay and monitor
stability. It does not test spatial resolution, monitor aperture, or whether the
solved port mode adequately separates guided fields from radiation.

For a bend, junction, or other radiating device:

1. Keep the source and incident monitor fixed. Place several output monitors in
   the same uniform straight lead, outside the device and absorber. Use the same
   transverse center, aperture, polarization, and mode index.
2. Sample each wavelength of interest directly. At each frequency compute the
   spread `max(S21_dB) - min(S21_dB)` across all eligible planes. Do not select the
   plane that happens to match another simulator.
3. Repeat with wider apertures at those same planes. The aperture must contain
   the guided-mode tails and remain clear of the absorber. Compare both the
   plane spread and the change at each plane. A narrow aperture can have a small
   plane spread while giving a biased amplitude.
4. Refine the mesh and repeat. Report the plane spread, aperture change, and
   mesh change separately against a tolerance appropriate to the application.
   Extend the straight lead and domain if the usable planes or wider apertures
   would approach an absorber.

There is no universal required straight-lead distance or aperture. A distance
that suffices for a straight-guide control may not suffice after a sharp bend.
Moving farther downstream is not guaranteed to improve the estimate
monotonically. Check an interval of planes rather than just two endpoints.

Inspect `diagnostics["valid_mask"]`, incident power, effective indices,
`projection_residual`, and `condition_number` alongside those comparisons.
A high residual means the selected guided modes do not reconstruct much of the
measured field; radiation can produce this legitimately. A condition number
near one only establishes that the selected modal basis is well conditioned.
Neither diagnostic certifies spatial convergence.

Issue [#309](https://github.com/beamzorg/beamz/issues/309) exposed a separate
material-consistency defect: detached 3D modal analysis reconstructed coefficients
from scalar permittivity even when propagation and the source used the raster's
smoothed, component-specific coefficients. Analysis now retains and uses those
coefficients, including the Yee support halo, and the cell material tensors.
This reduces the observed plane dependence but does not remove all sampling
error. Existing scalar-only material snapshots cannot recover omitted
coefficients without the original raster; rerun the simulation for corrected
detached results.

The [S-bend investigation](reviews/issue-309-monitor-convergence/README.md)
records the measured limits and the straight-guide control. Reproduce the sweep
from a checkout with:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m tests.characterization.sbend_monitor_case --mesh 10 --output probe10.json
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m tests.characterization.sbend_monitor_case --mesh 20 --output probe20.json
```

Use `--straight` for the straight-guide control, or `--widths` and `--distances`
to select subsets. Widths are in micrometres and preserve the original aperture
aspect ratio. The default sweep covers widths 1.35, 2.7, and 4.0 µm and distances
0.25–1.75 µm. The width-4 aperture reaches the nominal vertical absorber margin;
it is retained as a diagnostic, not as a clear-aperture convergence certificate.
Use a larger domain to validate that aperture independently of the absorber.
These are experimental settings, not universal placement rules.
