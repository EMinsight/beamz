"""Complex transmission and power compared with independent Fresnel optics."""

from dataclasses import replace

import numpy as np
import pytest

import beamz as bz
from tests.validation.analytical.bloch_slab_reference import (
    calibrated_transmission,
    oblique_slab_amplitudes,
    run_oblique_slab,
)


@pytest.mark.parametrize("polarization", ["s", "p"])
@pytest.mark.parametrize("dispersive", [False, True])
def test_oblique_slab_complex_transmission_and_power(polarization, dispersive):
    medium = (
        bz.PoleResidue.lorentz(
            1.0, strength=1, resonance=6e15, damping=3e14, frequency_range=(4e14, 8e14)
        )
        if dispersive
        else bz.Material(2.25)
    )
    frequency = 5e14
    epsilon = medium.eps_model(frequency) if dispersive else medium.permittivity
    angle = np.deg2rad(25)
    expected_r, expected_t = oblique_slab_amplitudes(
        epsilon, frequency, 160e-9, angle, polarization
    )
    reference = run_oblique_slab(None, 20e-9, polarization)
    device = run_oblique_slab(medium, 20e-9, polarization)
    t = calibrated_transmission(device, reference, polarization)
    reflected = replace(
        device["input"],
        dft_fields={
            c: device["input"].dft_fields[c] - reference["input"].dft_fields[c]
            for c in device["input"].dft_fields
        },
    )
    incident = reference["input"].flux[0]
    R = -reflected.flux[0] / incident
    T = device["output"].flux[0] / reference["output"].flux[0]
    np.testing.assert_allclose(t, expected_t, atol=0.025, rtol=0.025)
    np.testing.assert_allclose(
        [R, T], [abs(expected_r) ** 2, abs(expected_t) ** 2], atol=0.02, rtol=0.03
    )
    if not dispersive:
        np.testing.assert_allclose(R + T, 1, atol=0.015)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("graded", [False, True])
def test_empty_oblique_cell_directionality_and_phase(sign, graded):
    reference = run_oblique_slab(None, 20e-9, sign=sign, graded=graded)
    assert np.max(abs(reference["back"].flux / reference["input"].flux)) < 2e-4
    np.testing.assert_allclose(
        reference["output"].flux / reference["input"].flux, 1, atol=0.003
    )
    # In vacuum the projected plane wave accumulates k_z over a 1 um path.
    a = reference["input"].dft_fields["Ey"].reshape(-1)
    b = reference["output"].dft_fields["Ey"].reshape(-1)
    ratio = np.vdot(a, b) / np.vdot(a, a)
    expected = np.exp(
        1j * 2 * np.pi * 5e14 / bz.LIGHT_SPEED * np.cos(np.deg2rad(25)) * 1e-6
    )
    np.testing.assert_allclose(ratio, expected, atol=0.03)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_oblique_injection_supports_every_signed_normal(axis, sign):
    reference = run_oblique_slab(None, 40e-9, axis=axis, sign=sign)
    assert sign * reference["input"].flux[0] > 0
    assert np.max(abs(reference["back"].flux / reference["input"].flux)) < 2e-4
    np.testing.assert_allclose(
        reference["output"].flux / reference["input"].flux, 1, atol=0.003
    )
