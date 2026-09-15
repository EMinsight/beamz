"""Equivalent material distributions must produce the same solver coefficients."""

import numpy as np
import pytest

from beamz.design import raster


MATERIALS = (raster.Material(1.44402**2), raster.Material(3.47644**2))


def compare_results(left, right):
    for collection in ("tensors", "yee_tensors"):
        for name, expected in getattr(left, collection).items():
            np.testing.assert_allclose(
                getattr(right, collection)[name],
                expected,
                rtol=2e-6,
                atol=2e-6,
                err_msg=f"{collection}.{name}",
            )


@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
@pytest.mark.parametrize("scale", [1.0, 1e-6])
def test_duplicate_diagonal_interface_preserves_cell_and_yee_tensors(smoothing, scale):
    geometry = raster.ExtrudedPolygon(
        raster.Polygon(
            tuple((x * scale, y * scale) for x, y in ((-2, -2), (2, -2), (-2, 2)))
        ),
        -2 * scale,
        2 * scale,
    )
    grid = raster.Grid.uniform((-0.5 * scale,) * 3, (0.5 * scale,) * 3, (1, 1, 1))
    options = raster.RasterOptions(smoothing=smoothing)
    results = [
        raster.rasterize(
            raster.Scene(
                MATERIALS, tuple(raster.Object(geometry, 1, id=i) for i in range(n))
            ),
            grid,
            options=options,
        )
        for n in (1, 2, 3)
    ]
    arithmetic = sum(m.epsilon_r[0] for m in MATERIALS) / 2
    harmonic = 2 / sum(1 / m.epsilon_r[0] for m in MATERIALS)
    expected = {
        "volume": [arithmetic],
        "farjadpour_diagonal": [(arithmetic + harmonic) / 2] * 2 + [arithmetic],
        "farjadpour_full": [(arithmetic + harmonic) / 2] * 2
        + [arithmetic, (harmonic - arithmetic) / 2, 0, 0],
    }[smoothing]
    for result in results:
        np.testing.assert_allclose(
            result.tensors["epsilon"].ravel(), expected, rtol=2e-6
        )
        compare_results(results[0], result)
        assert result.diagnostics["adaptive_samples"] == 0
        assert result.diagnostics["fallback_multiple_objects"] == 0


@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
def test_stepped_core_slab_union_matches_disjoint_decomposition(smoothing):
    slab = raster.Box((-2, -2, -0.3), (2, 2, 0.15))
    core = raster.Box((-0.2, -2, -0.3), (0.2, 2, 0.22))
    upper_core = raster.Box((-0.2, -2, 0.15), (0.2, 2, 0.22))
    # Include unequal widths and supports crossing the hidden slab/core seam.
    grid = raster.Grid([-0.5, -0.1, 0.3, 0.5], [-0.5, 0.1, 0.5], [-0.1, 0.1, 0.19, 0.3])
    representations = (
        (slab, core),
        (slab, upper_core),
        (core, slab),
        (slab, core, core),
    )
    results = [
        raster.rasterize(
            raster.Scene(
                MATERIALS, tuple(raster.Object(g, 1, id=i) for i, g in enumerate(gs))
            ),
            grid,
            options=raster.RasterOptions(smoothing=smoothing),
        )
        for gs in representations
    ]
    for result in results:
        compare_results(results[0], result)
        assert result.diagnostics["adaptive_samples"] == 0
        assert result.diagnostics["fallback_multiple_objects"] == 0
    if smoothing != "volume":
        assert results[0].diagnostics["fallback_multiple_orientations"] > 0
