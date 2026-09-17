"""Public contracts for the transparent GDSFactory design workflow."""

from __future__ import annotations

import importlib.util

import pytest

from beamz import Simulation
from beamz.analysis import SParameterResult
from beamz.design.gdsfactory import Settings, prepare

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("gdsfactory") is None,
        reason="gdsfactory not installed",
    ),
    pytest.mark.filterwarnings(
        "ignore:Support for class-based `config` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Implicitly cleaning up <TemporaryDirectory.*:ResourceWarning"
    ),
    pytest.mark.filterwarnings("ignore:unclosed file .*gdsfactory.*:ResourceWarning"),
]


def _vertical_straight():
    import gdsfactory as gf

    container = gf.Component()
    reference = container << gf.components.straight(length=2.0)
    reference.drotate(90)
    container.add_ports(reference.ports)
    return container


def test_settings_exposes_ascending_frequency_grid_and_automatic_mesh():
    settings = Settings(wavelength_points=3)

    assert settings.frequencies.shape == (3,)
    assert settings.frequencies[0] < settings.frequencies[-1]
    assert settings.resolved_grid_spec().is_automatic


def test_prepare_returns_editable_native_beamz_objects_for_vertical_ports():
    settings = Settings(wavelength_points=2, run_time=2e-14)
    prepared = prepare(_vertical_straight(), settings=settings)

    assert {port.axis for port in prepared.ports} == {"y"}
    assert {port.signed_direction for port in prepared.ports} == {"+y", "-y"}
    assert {metadata.inward_direction for metadata in prepared.port_metadata} == {
        "+y",
        "-y",
    }
    assert {metadata.outward_direction for metadata in prepared.port_metadata} == {
        "+y",
        "-y",
    }

    simulation = prepared.simulation_for("o1")

    assert isinstance(simulation, Simulation)
    assert (
        simulation.sources[0].signed_direction == prepared.port("o1").signed_direction
    )
    assert {monitor.name for monitor in simulation.monitors} == {"o1", "o2"}
    assert simulation.updated_copy(run_time=3e-14).run_time == pytest.approx(3e-14)


def test_prepare_exposes_pre_run_resource_estimate_without_execution():
    prepared = prepare(
        "straight",
        settings=Settings(wavelength_points=2, run_time=2e-14),
        component_settings={"length": 2.0},
    )

    estimate = prepared.estimate_resources("o1")

    assert estimate["grid_cells"] > 0
    assert estimate["estimated_memory_bytes"] > 0
    assert estimate["n_simulations"] == 2


def test_run_sparameters_assembles_named_columns_and_physics_checks(monkeypatch):
    prepared = prepare(
        "straight",
        settings=Settings(wavelength_points=2, run_time=2e-14),
        component_settings={"length": 2.0},
    )

    class FakeSimulation:
        def __init__(self, source):
            self.source = source

        def run(self, **kwargs):
            assert kwargs["progress"] is False
            return {"source": self.source}

    def fake_simulation_for(self, source):
        return FakeSimulation(source)

    def fake_sparameters(result, *, source_port, ports, frequencies, **kwargs):
        del result, kwargs
        return SParameterResult(
            frequencies=frequencies,
            s_matrix={
                (port.name, source_port): (
                    [0.0j, 0.0j]
                    if port.name == source_port
                    else [0.9 + 0.0j, 0.9 + 0.0j]
                )
                for port in ports
            },
            diagnostics={"source": source_port},
        )

    monkeypatch.setattr(type(prepared), "simulation_for", fake_simulation_for)
    monkeypatch.setattr("beamz.design.gdsfactory.s_parameters", fake_sparameters)

    result = prepared.run_sparameters()

    assert set(result.native_results) == {"o1", "o2"}
    assert set(result.sparameters.s_matrix) == {
        ("o1", "o1"),
        ("o2", "o1"),
        ("o1", "o2"),
        ("o2", "o2"),
    }
    assert result.check_reciprocity()["max_abs_error"] == pytest.approx(0.0)
    assert result.check_passivity()["o1"].tolist() == pytest.approx([0.81, 0.81])
    assert result.provenance["component_name"] == prepared.component_name
