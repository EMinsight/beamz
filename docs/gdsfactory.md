# GDSFactory components

BeamZ can prepare a GDSFactory component directly for local CPU or GPU FDTD.
The workflow lives in `beamz.design.gdsfactory`; BeamZ does not import or depend
on `gds_fdtd`. Install the released package with its optional layout dependency:

```bash
pip install "beamz[gdsfactory]"
```

`prepare()` turns the PDK component geometry into ordinary BeamZ `Design` and
`Port` values. It does not run or compile a simulation, so the domain, PML,
mode source, monitors, grid, and runtime remain visible and editable.

```python
from beamz.design.gdsfactory import Settings, prepare

settings = Settings(
    wavelengths=(1.50e-6, 1.60e-6),
    wavelength_points=21,
    xy_padding=3.0e-6,
    pml_thickness=1.0e-6,
)
setup = prepare(
    "straight",
    settings=settings,
    component_settings={"length": 10.0},
)

assert {port.name for port in setup.ports} == {"o1", "o2"}
simulation = setup.simulation_for("o1")
assert simulation.sources[0].signed_direction == setup.port("o1").signed_direction
```

Use the pre-run inspection tools before choosing execution resources:

```pycon
>>> setup.preview()                 # geometry, PML, source and monitor placement
>>> setup.estimate_resources()["grid_cells"]
123456
```

Run a named S-matrix locally when the mesh and source placement are satisfactory:

```pycon
>>> result = setup.run_sparameters(termination=None)
>>> result.sparameters.s_matrix[("o2", "o1")]
array([...])
>>> result.check_reciprocity()["max_abs_error"]
...
>>> result.check_passivity()["o1"]
array([...])
```

`S[out, in]` is the outgoing modal amplitude at the output port datum divided
by the incident modal amplitude at the input datum. BeamZ stores source/monitor
settings, layer, PML, frequency grid, and port order in `result.provenance`.
The first release supports axis-aligned optical ports, fundamental TE-like
mode 1, and local 3D FDTD. Use `setup.simulation_for(...)` plus
`Simulation.updated_copy(...)` to make non-default numerical choices explicit.
