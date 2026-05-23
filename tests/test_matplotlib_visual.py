import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from beamz import Design, Material, Monitor, Rectangle, Simulation, plot_signal, um
from beamz.devices.sources.mode import ModeSource
from beamz.simulation.core import SimulationResults


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


def test_grid_show_returns_matplotlib_handles():
    design = Design(width=2 * um, height=1 * um, material=Material(1.0))
    grid = design.rasterize(resolution=0.25 * um)

    fig, ax = _close(grid.show(show=False))

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
