"""Regression contracts for complex interpolation and projection diagnostics."""

import numpy as np
import pytest

from beamz.analysis.modal_projection.colocation import _interpolate_plane_matrix_2d
from beamz.analysis.modal_projection.diagnostics import (
    _modal_projection_reconstruction_diagnostics_from_matrix,
    _modal_projection_reconstruction_residual,
    _relative_residual,
)
from beamz.analysis.sparameters import _resample_complex_matrix


@pytest.mark.parametrize("src0", [[0.0, 0.3, 2.0], [0.3]])
def test_plane_interpolation_preserves_complex_bilinear_field_and_clamps(src0):
    src0 = np.asarray(src0)
    src1 = np.array([-1.0, 0.2, 3.0])
    dst0 = np.array([-2.0, 0.1, 0.7, 4.0])
    dst1 = np.array([-3.0, -0.4, 1.3, 5.0])

    def field(x, y):
        return (1 + 2j) + (3 - 4j) * x + (-2 + 5j) * y + (2 + 1j) * x * y

    values = field(src0[:, None], src1[None, :])
    actual = _interpolate_plane_matrix_2d(values, src0, src1, dst0, dst1)
    expected = field(
        np.clip(dst0, src0[0], src0[-1])[:, None],
        np.clip(dst1, src1[0], src1[-1])[None, :],
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert actual.dtype == np.complex128


@pytest.mark.parametrize(
    "destinations",
    [[0.0, 1.0, 1.7, 4.0, 5.0, 6.0], [1.7, 4.0], [], [2.5]],
)
def test_frequency_resampling_flattens_space_and_zero_fills_outside_band(destinations):
    frequencies = np.array([1.0, 2.5, 5.0])
    destinations = np.asarray(destinations)
    slopes = np.array([[1 + 2j, -3 + 4j], [2 - 1j, -1 - 3j]])
    intercepts = np.array([[2 - 3j, 5 + 1j], [-4 + 2j, 3 - 2j]])
    values = frequencies[:, None, None] * slopes + intercepts
    expected = (destinations[:, None, None] * slopes + intercepts).reshape(
        len(destinations), 4
    )
    expected[(destinations < frequencies[0]) | (destinations > frequencies[-1])] = 0
    actual = _resample_complex_matrix(frequencies, values, destinations)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert actual.dtype == np.complex128


def test_component_diagnostics_recover_independent_complex_scales():
    matrix = np.array([[1, 0], [2, 0], [0, 1j], [0, 2j]])
    coeff = np.ones(2)
    e_scale, h_scale = 2 + 1j, -1 + 0.5j
    field = np.array([e_scale, 2 * e_scale, 1j * h_scale, 2j * h_scale])
    result = _modal_projection_reconstruction_diagnostics_from_matrix(
        field, matrix, coeff, [("Ex", 0, 2), ("Hy", 2, 4)]
    )
    expected_residual = np.sqrt(
        (abs(e_scale - 1) ** 2 + abs(h_scale - 1) ** 2)
        / (abs(e_scale) ** 2 + abs(h_scale) ** 2)
    )
    assert result["residual"] == pytest.approx(expected_residual)
    assert result["residual_e"] == pytest.approx(abs(e_scale - 1) / abs(e_scale))
    assert result["residual_h"] == pytest.approx(abs(h_scale - 1) / abs(h_scale))
    assert result["e_scale"] == pytest.approx(e_scale)
    assert result["h_scale"] == pytest.approx(h_scale)
    for key in ("residual_balanced", "residual_e_scaled", "residual_h_scaled"):
        assert result[key] == pytest.approx(0, abs=1e-14)
    legacy = _modal_projection_reconstruction_residual(
        field, {"mode_matrix": matrix}, coeff
    )
    assert legacy == pytest.approx(expected_residual)


@pytest.mark.parametrize("target", [[], [0.0], [1e-31], [np.nan], [np.inf]])
def test_relative_residual_is_undefined_for_invalid_target_norm(target):
    target = np.asarray(target, dtype=np.complex128)
    assert np.isnan(_relative_residual(target, np.ones_like(target)))


@pytest.mark.parametrize("magnitude", [0.0, 1e-31])
def test_projection_diagnostics_handle_tiny_targets_and_missing_components(magnitude):
    field = np.full(2, magnitude, dtype=np.complex128)
    matrix = np.eye(2, dtype=np.complex128)
    coeff = np.ones(2)
    assert np.isnan(
        _modal_projection_reconstruction_residual(field, {"mode_matrix": matrix}, coeff)
    )
    result = _modal_projection_reconstruction_diagnostics_from_matrix(
        field, matrix, coeff
    )
    for key, value in result.items():
        assert np.isnan(value), key
