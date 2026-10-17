"""Reciprocal transmission and direction-correct 2D modal extraction (#348)."""

from __future__ import annotations

import numpy as np
import pytest

from tests.modal_sparam_physical_case import (
    StraightWaveguideSParamConfig,
    run_straight_waveguide_sparam_case,
)


@pytest.fixture(
    scope="module", params=[("sponge", 10, 1.1), ("cpml", 10, 1.1), ("cpml", 14, 1.4)]
)
def reciprocal_straight_waveguide_results(request):
    """Run the smallest two source excitations once for this module."""
    absorber, ppw, monitor_offset = request.param
    config = StraightWaveguideSParamConfig(
        pml_formulation=absorber, input_monitor_offset_um=monitor_offset
    )
    forward = run_straight_waveguide_sparam_case(
        resolution_ppw=ppw,
        cfg=config,
        launch_direction="+",
    )
    reverse = run_straight_waveguide_sparam_case(
        resolution_ppw=ppw,
        cfg=config,
        launch_direction="-",
    )
    center = int(np.argmin(np.abs(forward.wavelengths_um - config.wavelength_um)))
    return config, forward, reverse, center


@pytest.mark.simulation
@pytest.mark.compiled
def test_straight_waveguide_complex_transmission_is_reciprocal(
    reciprocal_straight_waveguide_results,
    validation_metrics,
):
    """Opposite source excitations produce the same de-embedded transmission."""
    config, forward, reverse, center = reciprocal_straight_waveguide_results
    s21 = complex(forward.s21_by_monitor["far"][center])
    s12 = complex(reverse.s21_by_monitor["far"][center])
    reciprocity_residual = abs(s21 - s12)

    validation_metrics.check(
        "complex transmission reciprocity |S21-S12|",
        measured=reciprocity_residual,
        reference=0.0,
        tolerance="sparameter_reciprocity",
        resolution=f"{forward.resolution_ppw} ppw",
        metadata={
            "device": "symmetric straight slab waveguide",
            "wavelength_um": config.wavelength_um,
            "reference_plane_separation_um": abs(
                forward.monitor_x_um["far"] - forward.monitor_x_um["o1"]
            ),
            "forward_transmission": {
                "real": float(np.real(s21)),
                "imag": float(np.imag(s21)),
            },
            "reverse_transmission": {
                "real": float(np.real(s12)),
                "imag": float(np.imag(s12)),
            },
        },
    )


@pytest.mark.simulation
@pytest.mark.compiled
def test_reverse_mode_launch_has_low_spurious_counterpropagation(
    reciprocal_straight_waveguide_results,
):
    """Both directions must retain transmission and have low extracted reflection."""
    _config, forward, reverse, center = reciprocal_straight_waveguide_results
    for result in (forward, reverse):
        reflection_db = 20.0 * np.log10(max(float(abs(result.s11[center])), 1e-12))
        assert reflection_db < -35.0
        for transmission in result.s21_by_monitor.values():
            assert 0.95 < abs(transmission[center]) ** 2 < 1.05
