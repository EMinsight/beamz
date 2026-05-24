import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from beamz import (
    BoundarySpec,
    Box,
    Design,
    FieldMonitor,
    GaussianSource,
    GridSpec,
    Material,
    Medium,
    Monitor,
    Rectangle,
    Simulation,
    Structure,
    mode_field_component_pairs,
    plot_mode_fields,
    plot_signal,
    plot_tidy3d_cross_sections,
    plot_tidy3d_field_frame,
    um,
)
from beamz.devices.sources.mode import ModeSource
from beamz.simulation.core import MonitorResults, SimulationResults


def _close(fig_ax):
    fig, ax = fig_ax
    plt.close(fig)
    return fig, ax


def test_design_show_returns_matplotlib_handles():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))
    design += Rectangle(
        position=(1 * um, 0.5 * um),
        width=0.5 * um,
        height=0.2 * um,
        material=Material(12.0),
    )

    fig, ax = _close(design.show(show=False))

    assert fig is ax.figure
    assert ax.get_title() == "Design Layout"


def test_design_plot_is_standard_non_showing_api():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))

    fig, ax = _close(design.plot())

    assert fig is ax.figure
    assert ax.get_title() == "Design Layout"


def test_grid_show_returns_matplotlib_handles():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))
    grid = design.rasterize(resolution=0.25 * um)

    fig, ax = _close(grid.show(show=False))

    assert fig is ax.figure
    assert ax.get_title() == "Rasterized Design Grid"


def test_grid_plot_accepts_standard_slice_kwargs():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))
    grid = design.rasterize(resolution=0.25 * um)

    fig, ax = _close(grid.plot(field="permittivity"))

    assert fig is ax.figure
    assert ax.get_title() == "Rasterized Design Grid"


def test_mode_source_show_uses_profile_data():
    source = ModeSource.__new__(ModeSource)
    source._Ez_profile = np.array([0.0, 1.0, 0.0])
    source._jz_profile = None
    source.grid = None
    source.direction = "+x"
    source._neff = 2.4

    fig, ax = _close(source.show(show=False))

    assert fig is ax.figure
    assert "Mode Source 1D Profile" in ax.get_title()


def test_mode_source_plot_eps_and_source_spectrum():
    source = ModeSource.__new__(ModeSource)
    source._eps_profile_2d = np.ones((3, 4)) * 12.0
    source.signal = np.sin(np.linspace(0.0, 2.0 * np.pi, 16))

    eps_fig, eps_ax = _close(source.plot_eps())
    spectrum_fig, spectrum_ax = _close(source.plot_spectrum(dt=1e-15))

    assert eps_fig is eps_ax.figure
    assert eps_ax.get_title() == "Mode Source Permittivity"
    assert spectrum_fig is spectrum_ax.figure
    assert spectrum_ax.get_title() == "ModeSource Spectrum"


def test_gaussian_source_plots_signal_and_spectrum():
    signal = np.sin(np.linspace(0.0, 2.0 * np.pi, 16))
    source = GaussianSource(position=(0.0, 0.0), width=0.1 * um, signal=signal)

    signal_fig, signal_ax = _close(source.plot_signal(t=np.arange(signal.size) * 1e-15))
    spectrum_fig, spectrum_ax = _close(source.plot_spectrum(dt=1e-15))

    assert signal_fig is signal_ax.figure
    assert signal_ax.get_title() == "GaussianSource Signal"
    assert spectrum_fig is spectrum_ax.figure
    assert spectrum_ax.get_title() == "GaussianSource Spectrum"


def test_monitor_show_and_power_show_return_matplotlib_handles():
    monitor = Monitor(start=(0.0, 0.0), end=(1.0, 0.0))
    monitor.fields["t"].append(0.0)
    monitor.fields["Ez"].append(np.array([0.0, 1.0, 0.0]))
    monitor.power_history.extend([1.0, 2.0, 1.0])

    field_fig, field_ax = _close(monitor.show(show=False))
    power_fig, power_ax = _close(monitor.show_power(show=False))

    assert field_fig is field_ax.figure
    assert "Ez at t" in field_ax.get_title()
    assert power_fig is power_ax.figure
    assert power_ax.get_title() == "Power vs Time"


def test_monitor_plot_fields_and_power_are_standard_non_showing_api():
    monitor = Monitor(start=(0.0, 0.0), end=(1.0, 0.0))
    monitor.fields["t"].append(0.0)
    monitor.fields["Ez"].append(np.array([0.0, 1.0, 0.0]))
    monitor.power_history.extend([1.0, 2.0, 1.0])

    field_fig, field_ax = _close(monitor.plot_fields())
    power_fig, power_ax = _close(monitor.plot_power())

    assert field_fig is field_ax.figure
    assert "Ez at t" in field_ax.get_title()
    assert power_fig is power_ax.figure
    assert power_ax.get_title() == "Power vs Time"


def test_monitor_results_are_plottable():
    monitor = Monitor(start=(0.0, 0.0), end=(1.0, 0.0))
    monitor.fields["t"].append(0.0)
    monitor.fields["Ez"].append(np.array([0.0, 1.0, 0.0]))
    monitor.power_history.extend([1.0, 2.0, 1.0])
    result = MonitorResults.from_monitor(monitor)

    field_fig, field_ax = _close(result.plot(field="Ez"))
    power_fig, power_ax = _close(result.plot_power())

    assert field_fig is field_ax.figure
    assert "Ez at t" in field_ax.get_title()
    assert power_fig is power_ax.figure
    assert power_ax.get_title() == "Power vs Time"


def test_simulation_show_is_matplotlib_and_show3d_remains_available():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))
    sim = Simulation(
        design=design,
        sources=[],
        monitors=[],
        time=np.array([0.0, 1e-15]),
        resolution=0.25 * um,
    )

    fig, ax = _close(sim.show(show=False))

    assert fig is ax.figure
    assert ax.get_title() == "Simulation Layout"
    assert hasattr(sim, "show3d")
    assert hasattr(sim, "view3d")


def test_simulation_plot_is_standard_non_showing_api():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))
    sim = Simulation(
        design=design,
        sources=[],
        monitors=[],
        time=np.array([0.0, 1e-15]),
        resolution=0.25 * um,
    )

    fig, ax = _close(sim.plot())

    assert fig is ax.figure
    assert ax.get_title() == "Simulation Layout"


def test_simulation_plot_eps_overlays_layout():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))
    design += Rectangle(
        position=(1 * um, 0.5 * um),
        width=0.5 * um,
        height=0.2 * um,
        material=Material(12.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        monitors=[],
        time=np.array([0.0, 1e-15]),
        resolution=0.25 * um,
    )

    fig, ax = _close(sim.plot_eps())

    assert fig is ax.figure
    assert ax.get_title() == "Permittivity"
    assert len(ax.patches) >= 1


def test_plot_signal_returns_matplotlib_handles():
    fig, ax = _close(
        plot_signal(
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1e-12, 2e-12]),
            show=False,
        )
    )

    assert fig is ax.figure
    assert ax.get_title() == "Signal"


def test_mode_field_component_pairs_are_physical_labels():
    assert mode_field_component_pairs(("Ey", "Ez"), direction="-x") == [
        ("Ey", "Ey"),
        ("Ez", "Ez"),
    ]
    assert mode_field_component_pairs(("Ey", "Ez"), direction="+y") == [
        ("Ey", "Ey"),
        ("Ez", "Ez"),
    ]
    explicit = [("E major", "Ez")]
    assert mode_field_component_pairs(display_components=explicit) == explicit


def test_mode_field_plot_defaults_to_tidy3d_percentile_abs_scale(monkeypatch):
    class Grid:
        permittivity = np.ones((2, 2, 1))
        resolution = 1.0

    e_fields = np.zeros((1, 3, 2, 2), dtype=np.complex128)
    e_fields[0, 1] = np.array([[0.0, 2.0], [3.0, 4.0]])

    def fake_solve_modes(**kwargs):
        del kwargs
        return np.array([2.0]), e_fields, e_fields, 0

    monkeypatch.setattr("beamz.devices.sources.solve.solve_modes", fake_solve_modes)

    fig, _axes, _neffs = plot_mode_fields(
        Grid(),
        plane_x=0.0,
        wavelength=1.55,
        num_modes=1,
        components=("Ey",),
        show=False,
    )

    expected_vmax = float(np.nanpercentile(np.abs(e_fields[0, 1]), 99.5))
    assert fig.axes[0].images[0].get_clim() == (0.0, expected_vmax)
    plt.close(fig)


def test_simulation_results_show_uses_stored_snapshots():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))
    sim = Simulation(
        design=design,
        sources=[],
        monitors=[],
        time=np.array([0.0, 1e-15]),
        resolution=0.25 * um,
    )
    snapshot = {
        "kind": "simulation_snapshot",
        "field": np.zeros((2, 2)),
        "field_name": "Ez",
        "time": 0.0,
        "step": 1,
        "num_steps": 1,
        "extent": (0.0, 2 * um, 0.0, 1 * um),
        "units": "V/um",
        "plane_2d": "xy",
        "layout": sim.to_plot_data(),
    }
    results = SimulationResults(simulation=sim, snapshots=(snapshot,))

    fig, ax = _close(results.show(clean_visualization=False, show=False))

    assert fig is ax.figure
    assert "Ez at t" in ax.get_title()


def test_simulation_results_show_snapshots_accepts_field_and_frame_aliases():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))
    sim = Simulation(
        design=design,
        sources=[],
        monitors=[],
        time=np.array([0.0, 1e-15]),
        resolution=0.25 * um,
    )
    snapshots = (
        {
            "kind": "simulation_snapshot",
            "field": np.zeros((2, 2)),
            "field_name": "Ez",
            "time": 0.0,
            "step": 1,
            "num_steps": 2,
            "extent": (0.0, 2 * um, 0.0, 1 * um),
            "units": "V/um",
            "plane_2d": "xy",
            "layout": sim.to_plot_data(),
        },
        {
            "kind": "simulation_snapshot",
            "field": np.ones((2, 2)),
            "field_name": "Ez",
            "time": 1e-15,
            "step": 2,
            "num_steps": 2,
            "extent": (0.0, 2 * um, 0.0, 1 * um),
            "units": "V/um",
            "plane_2d": "xy",
            "layout": sim.to_plot_data(),
        },
    )
    results = SimulationResults(simulation=sim, snapshots=snapshots)

    fig, ax = _close(
        results.show(field="Ez", frame=1, clean_visualization=False, show=False)
    )

    assert fig is ax.figure
    assert "step 2/2" in ax.get_title()

    with pytest.raises(ValueError, match="Snapshot field 'Hz' is not available"):
        results.show(field="Hz", show=False)


def test_simulation_results_plot_field_uses_stored_fields():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))
    sim = Simulation(
        design=design,
        sources=[],
        monitors=[],
        time=np.array([0.0, 1e-15]),
        resolution=0.25 * um,
    )
    design += Rectangle(
        position=(1 * um, 0.5 * um),
        width=0.5 * um,
        height=0.2 * um,
        material=Material(12.0),
    )
    fields = {"Ez": np.zeros((2, 4, 5))}
    results = SimulationResults(simulation=sim, fields=fields)

    fig, ax = _close(results.plot_field(field="Ez", time_index=-1))

    assert fig is ax.figure
    assert ax.get_title() == "Ez frame -1"
    assert len(ax.patches) >= 1


def test_simulation_results_plot_field_selects_physical_coordinate():
    design = Design(width=2 * um, height=1 * um, depth=1 * um, material=Material(1.0))
    sim = Simulation(
        design=design,
        sources=[],
        monitors=[],
        time=np.array([0.0, 1e-15]),
        resolution=0.25 * um,
    )
    fields = {"Ez": np.ones((2, 4, 4, 8))}
    results = SimulationResults(simulation=sim, fields=fields)

    fig, ax = _close(
        results.plot_field(field="Ez", plane="z", index=0.5 * um, show=False)
    )

    assert fig is ax.figure
    assert "z=" in ax.get_title()


def test_tidy3d_cross_sections_plot_grid_slices():
    design = Design(
        width=2 * um,
        height=1 * um,
        depth=1 * um,
        material=Material(1.0),
    )
    design += Rectangle(
        position=(0.0, 0.0, 0.0),
        width=2 * um,
        height=1 * um,
        depth=0.5 * um,
        material=Material(2.25),
    )
    design += Rectangle(
        position=(0.0, 0.4 * um, 0.5 * um),
        width=2 * um,
        height=0.2 * um,
        depth=0.2 * um,
        material=Material(12.0),
    )
    grid = design.rasterize(resolution=0.25 * um)

    fig, axes = plot_tidy3d_cross_sections(
        grid,
        z=0.5 * um,
        y=0.5 * um,
        origin=(1 * um, 0.5 * um, 0.5 * um),
        substrate_z=0.5 * um,
        pml_thickness=0.25 * um,
        xy_markers=({"x": 0.5, "span": (-0.4, 0.4), "color": "orange"},),
        show=False,
    )
    plt.close(fig)

    assert len(axes) == 2
    assert axes[0].get_title() == "cross section at z=0.00 (um)"
    assert axes[1].get_title() == "cross section at y=0.00 (um)"
    xy = np.asarray(axes[0].images[0].get_array())
    assert np.count_nonzero(xy == 1) > np.count_nonzero(xy == 0)


def test_simulation_plot_uses_tidy3d_cross_sections_for_3d_slices():
    si = Medium(permittivity=12.0)
    sio2 = Medium(permittivity=2.25)
    sim = Simulation(
        size=(3 * um, 2 * um, 1.5 * um),
        grid_spec=GridSpec.uniform(0.25 * um),
        structures=[
            Structure(
                geometry=Box(center=(0.0, 0.0, -0.75 * um), size=(3 * um, 2 * um, 1.5 * um)),
                medium=sio2,
            ),
            Structure(
                geometry=Box(center=(0.0, 0.0, 0.1 * um), size=(3 * um, 0.3 * um, 0.2 * um)),
                medium=si,
            ),
        ],
        monitors=[
            FieldMonitor(
                center=(0.5 * um, 0.0, 0.0),
                size=(0.0, 1.0 * um, 0.8 * um),
                freqs=[1.0],
                name="m",
            )
        ],
        sources=[],
        boundary_spec=BoundarySpec.all_sides(),
        run_time=1e-15,
    )

    fig, axes = sim.plot(z=0.0, y=0.0, show=False)
    plt.close(fig)

    assert len(axes) == 2
    assert axes[0].get_title() == "cross section at z=0.00 (um)"
    assert axes[1].get_title() == "cross section at y=0.00 (um)"
    assert axes[0].lines


def test_tidy3d_field_frame_uses_xarray_results():
    design = Design(
        width=2 * um,
        height=1 * um,
        depth=1 * um,
        material=Material(1.0),
    )
    sim = Simulation(
        design=design,
        sources=[],
        monitors=[],
        time=np.array([0.0, 1e-15]),
        resolution=0.25 * um,
    )
    fields = {"Ez": np.ones((2, 4, 4, 8))}
    results = SimulationResults(simulation=sim, fields=fields)

    fig, ax = _close(
        plot_tidy3d_field_frame(
            results,
            field="Ez",
            plane="z",
            index=0.5 * um,
            origin=(1 * um, 0.5 * um, 0.5 * um),
            show=False,
        )
    )

    assert fig is ax.figure
    assert ax.get_title() == "cross section at z=0.00 (um)"
