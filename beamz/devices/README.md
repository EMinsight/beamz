# Devices

Simulation devices define how fields enter, leave, and are sampled from a
simulation.

- `modes/`: native finite-difference mode solving, minimal labeled results,
  BeamZ-shaped discrete modes, and guarded Yee-grid refinement.
- `modes/plane.py`: shared finite-plane extraction and solving for source launch
  and modal analysis.
- `modes/specs.py`: immutable mode-selection and mode-data values shared by
  sources, monitors, and ports.
- `sources/specs.py`: immutable public source values.
- `sources/time.py`: sampled and analytic temporal waveforms.
- `sources/solve.py`: source-plane extraction and the 2D native-solver adapter.
- `sources/mode_launch.py`: 2D/3D launch planning, profile conversion, and power scaling.
- `sources/planar_tfsf.py`: discrete 3D total-field/scattered-field residuals.
- `sources/compiler.py`: the single lowering boundary into executable source plans.
- `monitors/monitors.py`: immutable public monitor specifications.
- `monitors/compiler.py`: grid placement and packed acquisition plans. Runtime
  accumulation belongs to `simulation.observe`, not to device specifications.
- `ports.py`: named modal port metadata used by S-parameter analysis.
- `boundaries.py`: immutable PEC, sponge `Absorber`, and PML specifications.
- `_placement.py`: shared grid-snapping rules for sources and monitors.
- `_boundary_compile.py`: grid-aware PEC/PML/absorber lowering kept separate from
  the public boundary values for the same reason as source and monitor compilation.
- `_immutable.py`: array freezing and canonicalization shared by every device spec.
- `visualization.py`: data-only visual descriptions consumed by analysis plotting.

These files separate public values, numerical planning, and runtime execution. Avoid
adding per-device facade modules; a new file should own a distinct numerical stage or
be folded into the nearest existing owner.

For CPML, the automatic complex-frequency shift is
`alpha_max = 0.1 * EPS_0 * LIGHT_SPEED / thickness` in S/m. Its decay rate is
one tenth of the inverse vacuum transit time through the layer; changing `dt`
does not change it. This is a nonzero heuristic, not a universal optimum.
Explicit `alpha_max` values (including zero) are used unchanged.

Use a fixed physical `thickness` for mesh convergence. Omitting thickness still
selects 12 cells, so refinement changes that layer's physical width and automatic
parameters. A thickness sweep changes both automatic sigma and alpha; specify
`alpha_max` explicitly when alpha must remain fixed. Inspect
`sim.pml_data["resolved_parameters"]` for the resolved parameters of each boundary;
`alpha_x`, `sigma_x`, `kappa_x`, etc. contain the sampled profiles. The specification
itself remains immutable. `PML()` continues to select the sponge formulation.

The [CPML validation report](../../docs/reviews/cpml-physical-alpha-2026-10-07.md)
records the convergence controls and their limits.
