"""End-to-end numerical evidence for the explicitly supported tensor ports."""

import json
from pathlib import Path

import numpy as np
import pytest

from beamz.analysis.sparameters import _extract_port_waves_dft
from beamz.simulation.results import SimulationResults
from benchmarks.reports.issue361.validate_ports import run


def test_absolute_modal_watts_and_reference_api_for_nonunit_source():
    report, result, _, _ = run(source_power=2.5)
    near, far = result.mode("near"), result.mode("far")
    assert result.launched_power() == pytest.approx(2.5, rel=1e-10)
    assert report["ports"]["near"]["forward_power"] == pytest.approx(2.5, rel=1e-3)
    transmission = far.normalized_to(near)
    np.testing.assert_allclose(
        transmission.power.sel(direction="+", mode_index=0), 1, rtol=1e-3
    )
    assert abs(near.flux.item()) == pytest.approx(2.5, rel=1e-3)
    # Reference normalization scales amplitudes and measured flux together.
    np.testing.assert_allclose(transmission.modal_flux, transmission.flux, rtol=5e-5)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("direction", ["+", "-"])
def test_tensor_guide_launch_projection_cpml_and_archive(axis, direction, tmp_path):
    report, result, ports, waves = run(axis, direction)
    values = report["ports"]
    near = values["near"]
    assert abs(near["flux"]) == pytest.approx(result.launched_power(), rel=1e-3)
    assert near["forward_power"] > 0
    assert near["forward_power"] == pytest.approx(result.launched_power(), rel=1e-3)
    for name in ("near", "far", "shifted"):
        row = values[name]
        assert row["backward_power"] / row["forward_power"] < 1e-6
        # Off-grid sampling can change discrete flux; modal watts must track
        # the independently recorded flux at that same plane.
        assert row["forward_power"] == pytest.approx(abs(row["flux"]), rel=5e-5)
        if name != "shifted":
            assert row["forward_power"] == pytest.approx(
                near["forward_power"], rel=1e-3
            )
        assert row["sampled_modal_flux"] == pytest.approx(row["flux"], rel=5e-5)
        assert row["reconstruction_residual"] < 2e-3
    assert abs(values["back"]["flux"]) / abs(near["flux"]) < 1e-6
    assert near["other_mode_power"] / near["forward_power"] < 1e-6
    assert values["far"]["other_mode_power"] / near["forward_power"] < 1e-6
    positive, negative = ("a_plus", "a_minus") if axis == "z" else ("a_minus", "a_plus")
    forward = positive if direction == "+" else negative
    public = result.mode("near")
    assert public.monitor.mode_spec.polarization is None
    for index, name in enumerate(("near", "near_mode1")):
        np.testing.assert_allclose(
            public.amps.sel(direction=direction, mode_index=index),
            waves[name][forward],
            rtol=1e-10,
            atol=1e-10,
        )
    np.testing.assert_allclose(public.modal_flux, result["near"].flux, rtol=5e-5)
    np.testing.assert_allclose(
        public.power.sel(direction=direction, mode_index=0),
        near["forward_power"],
        rtol=1e-10,
    )
    assert np.max(abs(public.unresolved_flux)) < 5e-5
    transmission = waves["far"][forward][0] / waves["near"][forward][0]
    expected = np.exp(1j * 2 * np.pi / 1.55e-6 * near["neff"] * 0.8e-6)
    np.testing.assert_allclose(transmission, expected, rtol=0, atol=5e-5)
    archive = tmp_path / "results.npz"
    result.save(archive)
    loaded = SimulationResults.load(archive)
    np.testing.assert_allclose(
        loaded.mode("near").amps, public.amps, rtol=1e-12, atol=1e-12
    )
    restored = _extract_port_waves_dft(
        loaded, ports=ports, frequencies=[299792458 / 1.55e-6], return_power=True
    )
    # Repeat with the same modal grouping used above: archive must not affect any basis.
    original = _extract_port_waves_dft(
        result, ports=ports, frequencies=[299792458 / 1.55e-6], return_power=True
    )
    for port in ports:
        for key in ("a_plus", "a_minus", "mode_neff", "projected_signed_power"):
            np.testing.assert_allclose(
                restored[port.name][key],
                original[port.name][key],
                rtol=1e-12,
                atol=1e-12,
            )
        assert (
            not loaded[port.name]
            .material_region.yee_tensors["eps_node"]
            .flags.writeable
        )
    path = (
        Path("benchmarks/reports/issue361")
        / f"ports-{axis}-{'plus' if direction == '+' else 'minus'}.json"
    )
    # Tests do not modify repository artifacts. The benchmark CLI writes reports.
    (tmp_path / path.name).write_text(json.dumps(report, indent=2))


def test_zero_cross_terms_recover_diagonal_workflow():
    _, full, ports, full_waves = run(cross=False)
    _, diagonal, _, diagonal_waves = run(cross=False, smoothing="farjadpour_diagonal")
    for port in ports:
        for key in ("a_plus", "a_minus", "projected_signed_power"):
            np.testing.assert_allclose(
                full_waves[port.name][key],
                diagonal_waves[port.name][key],
                rtol=1e-7,
                atol=1e-9,
            )
        np.testing.assert_allclose(
            full[port.name].flux, diagonal[port.name].flux, rtol=1e-7, atol=1e-9
        )
        for component in full[port.name].dft_fields:
            np.testing.assert_allclose(
                full[port.name].dft_fields[component],
                diagonal[port.name].dft_fields[component],
                rtol=1e-7,
                atol=1e-9,
            )


def test_reciprocal_tensor_guide_complex_transmission():
    transmissions = []
    for direction in ("+", "-"):
        _, _, _, waves = run(direction=direction)
        forward = "a_minus" if direction == "+" else "a_plus"
        transmissions.append(waves["far"][forward][0] / waves["near"][forward][0])
    np.testing.assert_allclose(*transmissions, rtol=0, atol=5e-5)


def test_oblique_isotropic_interface_uses_coupled_port_operator():
    report, result, _, _ = run(axis="z", smoothed=True)
    assert np.any(result["near"].material_region.yee_tensors["eps_node"][3:] != 0)
    for name in ("near", "far", "shifted"):
        row = report["ports"][name]
        assert row["backward_power"] / row["forward_power"] < 1e-6
        assert row["reconstruction_residual"] < 2e-3
        assert row["sampled_modal_flux"] == pytest.approx(row["flux"], rel=5e-5)
    assert (
        report["ports"]["near"]["other_mode_power"]
        / report["ports"]["near"]["forward_power"]
        < 1e-6
    )
