A Monitor is a functional mapping

𝑀
:
(
𝐸
,
𝐻
,
𝑡
)
→
observable
(
𝑡
)
M:(E,H,t)→observable(t)

Implementation:

class Monitor:
    def __init__(self, region, op, temporal_reduction=None, spatial_reduction=None):
        self.region = region          # bounding box or surface
        self.op = op                  # function on E,H per cell
        self.temporal_reduction = temporal_reduction  # "instant", "DFT", "accumulate"
        self.spatial_reduction = spatial_reduction    # "none", "integrate", "average"


op could be a lambda or GPU shader snippet like dot(E, H) or abs(E)**2.

temporal_reduction defines whether to store time-series, accumulate FFTs, or just instantaneous values.

spatial_reduction defines whether to integrate or store field maps.

Then:

“flux monitor” = op = cross(E,H).n, temporal=DFT, spatial=integrate

“field recorder” = selected field components, temporal=interval, spatial=none

“mode monitor” = op = dot(E, E_mode*), temporal=DFT, spatial=integrate

“energy monitor” = op = |E|^2 + |H|^2, temporal=accumulate, spatial=integrate

All the same code path, just different parameters.

# 2D modal phase convention

Modal extraction fits forward and backward waves on the monitor's staggered
electric and magnetic Yee supports. Each basis wave carries its own spatial
phase sign, so a mixture of both directions does not receive a forward-only
phase correction. Amplitudes retain the magnetic-plane reference; temporal
leapfrog alignment is applied separately. The 3D collocated projection is
unchanged.

The two amplitudes are extracted with Lorentz power overlaps: measured E is
paired with modal H, and measured H with modal E. This preserves rejection of
power-orthogonal fields, including on weighted transverse grids; a fit dominated
by the larger numerical E values can contaminate TE transmission with radiation.

# 3D modal sampling and power

Three-dimensional mode monitors interpolate the native Yee fields onto their
physical analysis plane. Modal extraction applies the same normal interpolation
to each directional basis wave, using its own propagation phase. This matters
at both node-aligned and off-grid planes: interpolating staggered E and H can
attenuate them differently and otherwise create a spurious backward amplitude.

Each directional mode is normalized **after** applying this measurement, using
its discrete Poynting self-overlap on the monitor aperture. Thus squared modal
amplitudes and flux monitors use the same watts. This is a real scalar applied
to the complete E/H basis; the direction-dependent interpolation phases remain
intact, preserving rejection of false backward waves.

`ModeSource.power`, `SimulationResults.launched_power()` for a mode source,
`ModeMonitorData.power` (`abs(amps)**2`), and `ModePower` objectives use this
discrete flux convention. Source compilation calibrates against the production
source-plane Poynting quadrature. This is shared setup and analysis code for
JAX and CUDA, uniform and rectilinear grids, and diagonal and full-tensor
materials. It does not change the FDTD update kernels.

For measured-field accounting, compare `modes.modal_flux` with `modes.flux`.
`modes.unresolved_flux` is their difference. The modal reconstruction includes
sampled-basis overlap cross terms; summing squared amplitudes and subtracting
that from sampled flux can omit interference terms in a nonorthogonal basis.

Matched-reference transmission is also shared API:

```python
incident = reference_results.mode("input")
output = device_results.mode("output")
transmission = output.normalized_to(incident, mode_index=0, direction="+")
fundamental_transmission = transmission.power.sel(direction="+", mode_index=0)
```

Both results must use the same raw/source-normalization convention, and their
frequency coordinates must match exactly. The default reference is 1 W, so the
numerical modal powers equal power-transmission ratios. To report watts for a
different incident power, pass `power=...`. All amplitudes and flux diagnostics
receive the same reference scale, preserving measured-field power accounting.
No clipping or per-output peak calibration is applied. A zero/nonfinite incident
mode or a flux array supplied in place of modal reference data is rejected.

At finite mesh spacing, moving a plane to a different fraction of a Yee cell
can change the interpolated flux. Modal watts track the flux at that same
plane, rather than claiming position-independent pre-interpolation power.
Use equivalent sampling positions or matched references for transmission and
converge the mesh, aperture, runtime and physical CPML thickness independently.
A finite mode set can leave unresolved radiation; coherent nonorthogonal modes
can have cross terms. Neither residual is removed by renormalization or clipping.
See the [joint flux/modal validation](../../../benchmarks/reports/issue361/README.md#source-flux-and-modal-power).

Issue #356 validation (JAX CPU): a 0.6 × 0.4 µm, index-2 guide in index-1
cladding at wavelength 1.5 µm, run for 160 fs. Straight-guide measurements use
1.2 µm axial CPML; boundary sensitivity compares against 1.8 µm CPML by
subtracting complex fields before projection. Values below are reflected power
ratios in dB, normalized to incident modal power.

| Grid spacing (nm) | Original projection | Corrected projection | Boundary difference |
| ---: | ---: | ---: | ---: |
| 100 | -34.0 | -70.9 | -95.9 |
| 66.7 | -41.0 | -68.7 | -111.6 |
| 50 | -45.4 | -65.0 | -122.1 |
| 40 | -48.8 | -64.9 | -130.9 |

At 100 nm, increasing axial CPML from 0.3 to 1.2 µm reduces boundary sensitivity
from -19.4 to -95.9 dB. The sampling fix removes the false reflection floor,
but this is not proof of full mesh convergence: a 0.4 µm-long index-2.02 segment
with matched-reference subtraction remains nonmonotonic. Its 50→40 nm change
in complex reflection reaches 2.1e-3 across wavelengths 1.43–1.58 µm, exceeding
the 1e-4 target; doubling the coarse-grid runtime changes it by only 5.2e-9.
