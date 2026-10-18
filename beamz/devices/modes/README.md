# Native mode solver

`beamz.devices.modes` owns BeamZ's finite-difference frequency-domain mode
solver. It is shared by mode sources, mode monitors, ports, and direct
rasterized mode analysis; it does not participate in FDTD time stepping.

The high-level BeamZ workflow uses `ModeSpec` on a `ModeSource`, `ModeMonitor`,
or `Port`. For direct solver access, provide an already-rasterized transverse
permittivity grid:

```python
import numpy as np

from beamz import LIGHT_SPEED
from beamz.devices.modes import solve_grid

eps = np.ones((40, 30))
eps[15:25, 12:18] = 3.48**2
result = solve_grid(
    eps_xx=eps,
    x_edges=np.linspace(-2.0, 2.0, eps.shape[0] + 1),
    y_edges=np.linspace(-1.5, 1.5, eps.shape[1] + 1),
    freqs=[LIGHT_SPEED / 1.55e-6],
    num_modes=2,
    target_neff=3.0,
)
```

Coordinates passed to the direct raster API are measured in micrometres and
frequencies are measured in hertz. The result contains labeled effective
indices, field components, and solver diagnostics; BeamZ analysis owns plotting
and persistence.

For a uniformly bent cross-section, add `bend_radius=10.0` (micrometres) and
optionally `bend_axis="x"` or `"y"` (local radial coordinate). Radius is measured
from coordinate zero; positive radius puts the curvature center at `-R`.
`bend_radius=None` preserves straight solves. See
[circular-bend modes](../../../docs/bent-waveguide-modes.md) for the material
assumptions, field conventions, validation results, and limitations, and
[the runnable example](../../../examples/bent_waveguide_modes.py).

The discrete launch path converts solved modes onto BeamZ's component-specific
Yee supports, normalizes their signed power, and applies guarded refinement
only when field overlap, impedance, energy, power, and discrete-Maxwell
validation all accept the candidate.

On uniform grids, y-directed ports are reflected into the x-normal frame for
the same discrete solve and refinement, then transformed back with the axial
magnetic-field sign. Sources and modal analysis share these physical profiles;
this prevents port orientation from changing the mode contract.

`plane.py` owns the shared finite-plane extraction used by source launch and
modal projection, so both consumers solve identical material crops and Yee
supports.

See `UPSTREAM.md` for the imported MicroMode revision and source mapping.

## Optional mode symmetry

`ModeSpec(force_symmetry=False)` is the default for previews, sources, mode
monitors, and port projections. No mirror-parity projection is performed: a
solved off-center field is preserved, independent of approximate material
correlation or sampling thresholds. To request projection explicitly, use
`ModeSpec(force_symmetry=True)` on the source or monitor (or the port used to
create it).

Projection acts on the solver's collocated cell-center fields **before** Yee
interpolation, aperture windowing, guarded Maxwell refinement, and discrete
unit-power normalization. It uses a common vector mirror parity, with electric
fields treated as polar vectors and magnetic fields as axial vectors. It never
reverses a compact staggered field array or modifies an accepted refinement.
The previous approximate correlation detector and source-only post-refinement
projection have been removed.

Each transverse axis is checked independently. The physical mirror center is
the requested plane center; direct internal `solve_modes` calls use the aperture
midpoint unless `mirror_center` is provided in metres and profile storage order.
All grid edges, all component coordinates, and the complex diagonal epsilon/mu
profiles must be mirror symmetric about that center, to numerical tolerance.
Discrete solves also check the constitutive arrays on their individual Yee
supports, using edge or center coordinates according to each component's
stagger. A symmetric graded mesh is supported; an asymmetric graded mesh,
clipped/off-center crop, asymmetric tensor, or unavailable/non-closed component
support is skipped. There is no approximate mirror interpolation. Full-tensor
ports use the separate coupled path described below and reject forced symmetry.

`ModeSource.solve_modes()` previews the solver representation, with optional
validated symmetry, before Yee interpolation, launch aperture processing,
Maxwell refinement, and launch power scaling. It retains the eigensolver's own
normalization. `ModeData.diagnostics` contains one immutable mapping per
frequency; `DiscreteMode.diagnostics` records the discrete processing result.
Both report `force_symmetry_requested`, `force_symmetry_applied`,
`force_symmetry_axes`, `force_symmetry_skipped`, and `symmetry_stage`. Refinement
and final power normalization remain independent of this option; an accepted
refinement need not retain exact seed parity. Raw `solve_grid` results remain
unprojected. One-dimensional line profiles used for 2D launches also remain
unprojected, with an explicit skip diagnostic when symmetry is requested.

## Full-tensor dielectric ports

Confined lossless, nonmagnetic 3D tensor ports on uniform or rectilinear grids use a coupled
discrete eigenproblem shared with the propagation constitutive operator. Use
`ModeSpec(polarization=None, force_symmetry=False)` explicitly on the port.
Source injection and monitor projection retain cross terms, normal Yee
averaging, time staggering, and reciprocal complex eigenfields. Tensor material
elsewhere in the device is allowed when the port's required support is locally
diagonal. Tensor ports require locally uniform normal spacing and longitudinally
invariant material; transverse cells may vary. Unsupported combinations fail explicitly.

See [tensor modal workflows](../../../docs/tensor-modal-workflows.md) for the
support matrix, mode ordering, power convention, result archives, and measured
validation against the actual time step and independent Meep/MPB modes.
