"""Public tensor launch/monitor workflows with unequal physical cell sizes."""

import numpy as np
import pytest

from beamz.simulation.results import SimulationResults
from tests.validation.regression.test_tensor_port_workflow import run


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("direction", ["+", "-"])
def test_rectilinear_tensor_launch_monitors_and_archive(axis, direction, tmp_path):
    report, result, ports, waves = run(axis, direction, rectilinear=True)
    near = report["ports"]["near"]
    assert abs(near["independent_flux"]) == pytest.approx(
        result.launched_power(), rel=1e-3
    )
    assert near["forward_power"] == pytest.approx(result.launched_power(), rel=1e-3)
    for name in ("near", "far", "shifted"):
        row = report["ports"][name]
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
    for name in ("near", "far"):
        row = report["ports"][name]
        assert row["other_mode_power"] / row["forward_power"] < 1e-6
        assert row["independent_flux"] == pytest.approx(
            row["sampled_modal_flux"], rel=5e-5
        )
    assert abs(report["ports"]["back"]["flux"]) / abs(near["flux"]) < 1e-6
    assert report["preview_neffs"][0] == pytest.approx(near["neff"], rel=1e-8)
    near_a = complex(*near["forward_amplitude"])
    far_a = complex(*report["ports"]["far"]["forward_amplitude"])
    np.testing.assert_allclose(
        far_a / near_a,
        np.exp(1j * 2 * np.pi / 1.55e-6 * near["neff"] * 1.4e-6),
        atol=5e-5,
        rtol=0,
    )
    public = result.mode("near")
    np.testing.assert_allclose(
        public.amps.sel(direction=direction, mode_index=0),
        [near_a],
        atol=1e-10,
        rtol=1e-10,
    )
    assert result.metadata.grid.metric_kind == "rectilinear"
    archive = tmp_path / "rectilinear.npz"
    result.save(archive)
    restored = SimulationResults.load(archive)
    np.testing.assert_allclose(
        restored.mode("near").amps, public.amps, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_array_equal(
        restored.metadata.grid.x_edges, result.metadata.grid.x_edges
    )


def test_rectilinear_tensor_source_reciprocity():
    transmissions = []
    for direction in ("+", "-"):
        report, *_ = run("x", direction, rectilinear=True)
        ports = report["ports"]
        transmissions.append(
            complex(*ports["far"]["forward_amplitude"])
            / complex(*ports["near"]["forward_amplitude"])
        )
    np.testing.assert_allclose(*transmissions, atol=5e-5, rtol=0)


def test_tensor_propagates_through_longitudinal_grid_grading():
    report, *_ = run(rectilinear=True, graded_normal=True)
    near, far = (report["ports"][n] for n in ("near", "far"))
    # Longitudinal grid grading produces finite-resolution reflection.
    assert near["backward_power"] / near["forward_power"] < 1e-3
    np.testing.assert_allclose(
        far["independent_flux"], near["independent_flux"], rtol=1e-5
    )
    for row in (near, far):
        assert row["sampled_modal_flux"] == pytest.approx(
            row["independent_flux"], rel=1e-4
        )


def test_rectilinear_oblique_smoothed_isotropic_core():
    report, result, *_ = run(axis="z", rectilinear=True, smoothed=True)
    assert np.any(result["near"].material_region.yee_tensors["eps_node"][3:] != 0)
    for name in ("near", "far"):
        row = report["ports"][name]
        assert row["backward_power"] / row["forward_power"] < 1e-6
        assert row["sampled_modal_flux"] == pytest.approx(
            row["independent_flux"], rel=1e-4
        )


def test_zero_cross_terms_recover_rectilinear_diagonal_workflow():
    _, full, ports, _ = run(rectilinear=True, cross=False)
    _, diagonal, _, _ = run(
        rectilinear=True, cross=False, smoothing="farjadpour_diagonal"
    )
    for port in ports:
        np.testing.assert_allclose(
            full[port.name].flux, diagonal[port.name].flux, rtol=1e-7, atol=1e-9
        )
    np.testing.assert_allclose(
        full.mode("near").amps, diagonal.mode("near").amps, rtol=1e-7, atol=1e-9
    )


def test_partial_cell_aperture_uses_the_flux_monitor_area_weights():
    report, result, *_ = run(rectilinear=True, aperture_um=2.17)
    near = report["ports"]["near"]
    assert abs(near["independent_flux"]) == pytest.approx(
        result.launched_power(), rel=1e-3
    )
    for name in ("near", "far"):
        row = report["ports"][name]
        assert row["forward_power"] == pytest.approx(
            abs(row["independent_flux"]), rel=5e-5
        )
        assert row["sampled_modal_flux"] == pytest.approx(
            row["independent_flux"], rel=5e-5
        )
