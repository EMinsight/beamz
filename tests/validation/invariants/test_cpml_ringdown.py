"""A shifted absorber must preserve the isolated annulus pole (issue #316)."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.benchmark_cpml_convergence import beamz_run, disk_pole, fit_single_mode


@pytest.mark.slow
@pytest.mark.parametrize("angular_phase", [0.0, np.pi / 2])
def test_default_cpml_preserves_annulus_decay_against_unshifted_control(angular_phase):
    """Compare source-free poles with only alpha changed; this is not a continuum oracle."""
    args = SimpleNamespace(
        resolution=160,
        courant=0.5,
        pml=0.4,
        padding=0.0,
        angular_phase=angular_phase,
        alpha_max=None,
        until=75.0,
        backend="jax",
        vertices=1024,
        order=6,
        inner_radius=0.25,
        shift=0.0,
        corrugation=0.0,
        smoothing="farjadpour_full",
        quality="balanced",
    )
    pole = disk_pole(order=args.order, inner_radius=args.inner_radius)
    fits = []
    for alpha in (None, 0.0):
        args.alpha_max = alpha
        values, _ = beamz_run(args, pole)
        windows = [
            fit_single_mode(
                values,
                2 * args.courant / args.resolution,
                window,
                frequency_band=(pole["frequency"] * 0.9, pole["frequency"] * 1.1),
            )
            for window in ((20, 60), (30, 70))
        ]
        for fit in windows:
            assert fit["relative_residual"] < 1e-3
        assert windows[1]["quality_factor"] == pytest.approx(
            windows[0]["quality_factor"], rel=2e-4
        )
        fits.append(windows[0])
    assert fits[0]["frequency"] == pytest.approx(fits[1]["frequency"], rel=1e-5)
    assert fits[0]["quality_factor"] == pytest.approx(
        fits[1]["quality_factor"], rel=2e-4
    )
