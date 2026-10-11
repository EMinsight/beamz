# BeamZ optimization

Use `beamz.optimization` for topology inverse design with BeamZ simulations,
metre units, and JAX objectives. The same workflow supports full autodiff and
spectral adjoint gradients.

## Recommended workflow

Given an ordinary `simulation` with fixed mode sources and monitors:

```python
import jax.numpy as jnp
import beamz.optimization as opt

region = opt.TopologyDesignRegion(
    center=(2.5e-6, 2.5e-6, 1.1e-6),
    size=(3e-6, 3e-6, 200e-9),
    pixel_size=simulation.resolution,
    eps_bounds=(1, 4),
    transformations=[opt.FilterProject(radius=150e-9, beta=5)],
)
design = opt.InverseDesign(
    simulation=simulation,
    design_region=region,
    output_monitor_names=["input", "output"],
    gradient_backend="adjoint",  # or "autodiff"
)


def transmission(data):
    incoming = opt.utils.get_amps(data, "input", direction="+", mode_index=0)
    outgoing = opt.utils.get_amps(data, "output", direction="+", mode_index=0)
    return jnp.mean(jnp.abs(outgoing / incoming) ** 2)


optimizer = opt.AdamOptimizer(
    design=design,
    learning_rate=0.1,
)
result = optimizer.run(transmission, steps=10, checkpoint="results/inverse-design.npz")
result.plot_optimization()
final_data = result.simulation_data()
```

Parameters use `(y, x)` order. In 3D, the planar pattern is extruded through the
region's finite z thickness. The design must align to an unsharded uniform grid,
use scalar lossless materials, and stay outside the absorber and fixed mode
planes. The spectral adjoint additionally requires decayed fields and full-run
rectangular DFT monitors and nonperiodic boundaries. Use autodiff for periodic
boundaries; nonzero Bloch wavevectors do not support mode sources or monitors.

`FilterProject` and `ErosionDilationPenalty` provide differentiable smoothing,
projection, and soft geometric penalties. They do not guarantee minimum feature
sizes, connectivity, or a binary final design. Verify the exported geometry with
an independent simulation and resolution checks.

Use `OptimizationSchedule` with `LinearSchedule` or `StepSchedule` for projection,
penalty, and objective-weight continuation. A schedule has its own fixed horizon;
`steps` always means the target total number of updates:

```python
result = optimizer.run(transmission, steps=20, resume="results/inverse-design.npz")
result.final_params
result.history[-1].post_process_val
simulation = result.to_simulation()
```

By default, retain scalar history and the latest parameters, gradient, and optimizer
state. Set `store_full_results=True` to retain parameter/gradient snapshots.
`checkpoint_every=5` saves every five updates and at the final update; callbacks
receive `callback(result)`. Both workflows share the same result and checkpoint
format. The two checkpoint formats previously introduced on this branch are
rejected explicitly; rerun the examples to generate current checkpoints.
Numerical fingerprint 5 also rejects earlier histories, whose reference powers
could be paired with the wrong frequencies when monitor frequency orders differ.

See the [mode-converter notebook](../../examples/notebooks/topology_mode_converter.ipynb)
and [scheduled WDM notebook](../../examples/notebooks/topology_wdm_4channel.ipynb)
for complete inverse-design examples.

## Lower-level control

`TopologyProblem` binds a `TopologySpec` mask to a simulation and a composable
modal objective. Its existing API remains available:

```python
problem = opt.TopologyProblem(
    simulation,
    topology,  # TopologySpec containing the design mask and optimizer settings
    opt.ModePower("output", reference_monitor="input"),
    gradient_backend="autodiff",
)
result = problem.run(steps=20, checkpoint="results/topology.npz")
geometry = problem.export_design(result)
```

`ModePower` and `SoftMinModePower` combine using arithmetic, for example
`transmission - 0.2 * reflection`. `autodiff.py`, `projections.py`, and
`polygonize.py` contain the density transforms, SSP projection, and geometry export
helpers for custom loops.

## Export topology to GDS

Both workflows and their results accept the same geometry controls:

```python
from beamz import Material
from beamz.design import export_gds

pixels = result.export_design(mode="pixel_exact")
contours = result.export_design(mode="contour")  # default
refined = result.export_design(mode="contour", sample_spacing=10e-9)

export_gds(
    refined,
    "device.gds",
    precision=1e-9,  # metres; 1 nm database grid
    layer_map={Material(4.0): (1, 0)},  # select your core material/layer
)
```

- `pixel_exact` unions the full footprints of density cells strictly above the
  threshold. It preserves the thresholded mask, including its staircase edges.
- `contour` reconstructs boundaries by linear interpolation. Edge samples extend
  to the exact region boundary, so contour closure cannot inset connecting ports.
- `sample_spacing` sets the maximum spacing of optional bilinear sampling before
  contour extraction, in metres. It is only valid for contour mode and refines
  the representation without recovering missing physical information. It does
  not change the FDTD mesh or smooth the density with an additional filter.

All density arrays use `(y, x)` order. For `density_to_polygons` and
`density_to_shapely_geometry`, `x0, y0` are **cell-edge origins**, while density
values are at cell centers. `dx, dy` are the physical cell sizes. Cropped arrays
need their true edge origin; retain surrounding samples when external material
context matters. `region_mask` clips a full density field to an exact union of
cells, with the exterior values supplying interpolation context.

To hold a waveguide connection during optimization, pass disjoint boolean
`fixed_solid` and `fixed_void` arrays to `TopologyDesignRegion` or `TopologySpec`.
Region arrays have `params_shape`; spec arrays have `region_mask.shape` and must
lie inside that region. Fixed cells are applied before filtering and after
projection, have zero gradients with respect to their parameters, and retain
exact footprints during geometry export. They do not guarantee connectivity
elsewhere in the design. Masks are copied and participate in checkpoint identity.

`export_gds` preserves holes and uses a private database at the requested
precision, without changing gdsfactory's global grid. `layer_map` selects only
specified materials; omitting it retains automatic material-to-layer assignment.
Explicitly select the core to exclude region-clearing and background polygons.

Run the [export showcase](../../examples/optimization/topology_gds_export.py) to
optimize a small 2D mode converter, compare the previous export with all three
current variants, and reimport and simulate the actual GDS files on two meshes:

```bash
JAX_PLATFORMS=cpu uv run --extra gds python examples/optimization/topology_gds_export.py
```

The script writes GDS masks, boundary zooms, an interpolation comparison, and
measured transmission to the ignored `results/gds-export/` directory. Revalidate
geometry changes and GDS quantization separately from mesh convergence; smoother
contours alone do not establish better performance.
