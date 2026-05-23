import numpy as np
import xarray as xr

from beamz import (
    Design,
    GaussianSource,
    Material,
    Monitor,
    Simulation,
    field_intensity,
    poynting_vector,
    um,
)
from beamz.devices.sources.mode import ModeSource
from beamz.simulation.core import MonitorResults, SimulationResults


def _sim():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))
    return Simulation(
        design=design,
        sources=[],
        monitors=[],
        time=np.array([0.0, 1e-15]),
        resolution=0.25 * um,
    )


def test_simulation_results_fields_are_xarray_by_default():
    sim = _sim()
    fields = {"Ez": np.zeros((2, 4, 5))}
    results = SimulationResults(
        simulation=sim,
        fields=fields,
        field_times=np.array([1e-15, 2e-15]),
        field_steps=np.array([10, 20]),
    )

    ds = results.fields

    assert isinstance(ds, xr.Dataset)
    assert ds["Ez"].dims == ("t", "y", "x")
    assert ds["Ez"].coords["t"].attrs["units"] == "s"
    assert ds["Ez"].coords["x"].attrs["units"] == "m"
    assert ds["Ez"].attrs["component"] == "Ez"
    np.testing.assert_allclose(ds["Ez"].coords["t"], [1e-15, 2e-15])
    np.testing.assert_array_equal(ds["step"], [10, 20])
    assert results.to_xarray() is results.fields


def test_simulation_results_use_yee_component_coordinates():
    sim = _sim()
    ez = SimulationResults(simulation=sim, fields={"Ez": np.zeros((1, 5, 9))}).fields
    hx = SimulationResults(simulation=sim, fields={"Hx": np.zeros((1, 4, 9))}).fields
    hy = SimulationResults(simulation=sim, fields={"Hy": np.zeros((1, 5, 8))}).fields

    dx = float(ez["Ez"].coords["x"][1] - ez["Ez"].coords["x"][0])
    dy = float(ez["Ez"].coords["y"][1] - ez["Ez"].coords["y"][0])
    np.testing.assert_allclose(ez["Ez"].coords["x"][:2], [0.0, dx])
    np.testing.assert_allclose(hx["Hx"].coords["y"][:2], [0.5 * dy, 1.5 * dy])
    np.testing.assert_allclose(hy["Hy"].coords["x"][:2], [0.5 * dx, 1.5 * dx])


def test_monitor_to_xarray_labels_fields_and_power():
    monitor = Monitor(start=(0.0, 0.0), end=(1.0, 0.0), name="m")
    monitor.fields["t"].extend([0.0, 1e-15])
    monitor.fields["Ez"].extend([np.array([0.0, 1.0]), np.array([0.5, 0.0])])
    monitor.power_history.extend([1.0, 2.0])
    monitor.power_timestamps.extend([0.0, 1e-15])

    ds = monitor.to_xarray()

    assert ds["Ez"].dims == ("t", "s")
    assert ds["power"].dims == ("t",)
    assert ds.attrs["monitor_name"] == "m"
    np.testing.assert_allclose(ds["power"].coords["t"], [0.0, 1e-15])


def test_monitor_results_to_xarray_uses_snapshot_data():
    monitor = Monitor(start=(0.0, 0.0), end=(1.0, 0.0), name="m")
    monitor.fields["t"].append(0.0)
    monitor.fields["Ez"].append(np.array([0.0, 1.0]))
    monitor.power_history.append(1.0)
    monitor.power_timestamps.append(0.0)
    result = MonitorResults.from_monitor(monitor)

    ds = result.data

    assert ds["Ez"].dims == ("t", "s")
    assert ds["power"].dims == ("t",)
    assert result.to_xarray() is result.data


def test_monitor_to_xarray_exposes_dft_components():
    monitor = Monitor(
        start=(0.0, 0.0),
        end=(1.0 * um, 0.0),
        name="m",
        dft_enabled=True,
        dft_frequencies=[2.0e14, 2.1e14],
        dft_components=("Ez",),
    )
    monitor._dft_accum["Ez"] = np.ones((2, 3), dtype=np.complex128)
    monitor._dft_weight_sum = np.ones((2,), dtype=float)

    ds = monitor.to_xarray()

    assert ds["dft_Ez"].dims == ("f", "s")
    np.testing.assert_allclose(ds["dft_Ez"].coords["f"], [2.0e14, 2.1e14])
    np.testing.assert_allclose(ds["dft_Ez"].coords["s"], [0.0, 0.5 * um, 1.0 * um])


def test_source_to_xarray_labels_signal_time():
    t = np.array([0.0, 1e-15, 2e-15])
    source = GaussianSource(
        position=(0.0, 0.0),
        width=0.1 * um,
        signal=np.array([0.0, 1.0, 0.0]),
    )

    ds = source.to_xarray(t=t)

    assert ds["signal"].dims == ("t",)
    assert ds["signal"].coords["t"].attrs["units"] == "s"
    np.testing.assert_allclose(ds["signal"], [0.0, 1.0, 0.0])


def test_mode_source_to_xarray_contains_profile_and_signal():
    source = ModeSource.__new__(ModeSource)
    source._Ez_profile = np.array([0.0, 1.0, 0.0])
    source._jz_profile = None
    source.grid = None
    source.direction = "+x"
    source._neff = 2.4
    source.signal = np.array([0.0, 1.0, 0.0])

    ds = source.to_xarray()

    assert ds["profile"].dims == ("s",)
    assert ds["amplitude"].dims == ("s",)
    assert ds["signal"].dims == ("sample",)
    assert ds["profile"].attrs["neff"] == 2.4


def test_field_intensity_and_poynting_vector_use_xarray_fields():
    sim = _sim()
    shape = (1, 4, 8)
    fields = {
        "Ex": np.ones(shape),
        "Ey": np.zeros(shape),
        "Ez": np.zeros(shape),
        "Hx": np.zeros(shape),
        "Hy": np.ones(shape),
        "Hz": np.zeros(shape),
    }
    results = SimulationResults(simulation=sim, fields=fields)

    intensity = field_intensity(results)
    poynting = poynting_vector(results)

    assert intensity.name == "intensity"
    assert set(poynting.data_vars) == {"Sx", "Sy", "Sz"}
    np.testing.assert_allclose(poynting["Sz"], 1.0)
