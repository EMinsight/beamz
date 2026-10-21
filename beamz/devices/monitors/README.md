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

Each directional mode is normalized after monitor interpolation, so squared
modal amplitudes, `ModeSource.power`, and flux monitors share the same watts.
Source compilation calibrates against source-plane Poynting quadrature.
`modes.modal_flux` includes sampled-basis overlap cross terms;
`modes.unresolved_flux` is `modes.flux - modes.modal_flux`.

For matched-reference transmission, use
`output.normalized_to(incident, mode_index=0, direction="+")`, where both values
are `ModeMonitorData` with matching frequencies and source normalization.
The default reference is 1 W; pass `power=...` to report another incident power.
Amplitudes and flux diagnostics receive the same scale without clipping.

Finite mesh spacing and aperture can leave interpolation error and unresolved
radiation. Use equivalent sampling positions or matched references and converge
the mesh, aperture, runtime, and physical CPML thickness independently.

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
