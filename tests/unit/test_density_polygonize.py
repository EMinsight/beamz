import numpy as np
import pytest
from shapely.geometry import Point

from beamz import Design, Material
from beamz.optimization.polygonize import (
    density_to_polygons,
    density_to_shapely_geometry,
)

EPS_CORE = 3.48**2
EPS_CLAD = 1.0


def _iter_polygons(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _iter_polygons(part)


def test_density_to_shapely_geometry_preserves_holes_and_islands():
    density = np.zeros((60, 60), dtype=float)
    density[10:50, 10:50] = 1.0
    density[20:40, 20:40] = 0.0
    density[27:33, 27:33] = 1.0

    geometry = density_to_shapely_geometry(
        density,
        level=0.5,
        dx=1.0,
        min_area=1.0,
    )

    polygons = list(_iter_polygons(geometry))
    assert len(polygons) == 2
    assert sum(len(poly.interiors) for poly in polygons) == 1
    assert geometry.covers(Point(12.0, 12.0))
    assert not geometry.covers(Point(24.0, 24.0))
    assert geometry.covers(Point(30.0, 30.0))
    assert geometry.area == pytest.approx(40 * 40 - 20 * 20 + 6 * 6, abs=1.0)


def test_density_to_shapely_geometry_handles_boundary_touching_regions():
    density = np.zeros((40, 40), dtype=float)
    density[:, :20] = 1.0
    density[12:28, 6:14] = 0.0

    geometry = density_to_shapely_geometry(
        density,
        level=0.5,
        dx=1.0,
        min_area=1.0,
    )

    polygons = list(_iter_polygons(geometry))
    assert len(polygons) == 1
    assert len(polygons[0].interiors) == 1
    assert geometry.covers(Point(3.0, 3.0))
    assert not geometry.covers(Point(10.0, 20.0))
    assert not geometry.covers(Point(30.0, 20.0))


def test_density_to_shapely_geometry_respects_physical_offsets():
    density = np.zeros((12, 12), dtype=float)
    density[3:9, 2:7] = 1.0

    geometry = density_to_shapely_geometry(
        density,
        level=0.5,
        x0=10.0,
        y0=20.0,
        dx=0.5,
        min_area=0.25,
    )

    assert geometry.covers(Point(11.25, 22.25))
    assert not geometry.covers(Point(10.75, 21.25))
    assert not geometry.covers(Point(13.75, 22.25))


def test_density_to_polygons_rerasterizes_with_preserved_voids():
    density = np.zeros((60, 60), dtype=float)
    density[8:52, 8:52] = 1.0
    density[18:42, 18:42] = 0.0
    density[26:34, 26:34] = 1.0

    design = Design(width=60.0, height=60.0, material=Material(permittivity=EPS_CLAD))
    for polygon in density_to_polygons(
        density,
        material=Material(permittivity=EPS_CORE),
        level=0.5,
        dx=1.0,
        min_area=1.0,
    ):
        design += polygon
    design = design.unified_polygons()

    grid = design.rasterize(1.0, quality="reference", force_recompute=True)

    assert grid.permittivity[12, 12] == pytest.approx(EPS_CORE)
    assert grid.permittivity[24, 24] == pytest.approx(EPS_CLAD)
    assert grid.permittivity[30, 30] == pytest.approx(EPS_CORE)


def test_cropped_density_requires_true_cell_edge_origin():
    dx = 0.5
    density = np.zeros((16, 20), dtype=float)
    density[5:11, 6:15] = 1.0

    full_geometry = density_to_shapely_geometry(
        density,
        level=0.5,
        x0=0.0,
        y0=0.0,
        dx=dx,
        min_area=0.25,
    )

    i0, i1 = 4, 12
    j0, j1 = 5, 16
    cropped = density[i0:i1, j0:j1]
    cropped_geometry = density_to_shapely_geometry(
        cropped,
        level=0.5,
        x0=j0 * dx,
        y0=i0 * dx,
        dx=dx,
        min_area=0.25,
    )

    symmetric_diff = full_geometry.symmetric_difference(cropped_geometry)
    assert symmetric_diff.area == pytest.approx(0.0, abs=1e-9)


def test_subnanometre_contour_survives_native_rasterization():
    # Interpolating a value just above threshold can produce a small valid island.
    density = np.full((3, 3), 0.49)
    density[1, 1] = 0.5001
    polygons = density_to_polygons(
        density, material=Material(4.0), dx=50e-9, x0=1.5e-6, y0=1.25e-6
    )
    assert len(polygons) == 1
    design = Design(width=5e-6, height=5e-6, material=Material(1.0))
    design += polygons[0]
    raster = design.rasterize(25e-9).permittivity
    assert np.isfinite(raster).all()
    assert raster.max() > 1.0


@pytest.mark.parametrize("mode", ["contour", "pixel_exact"])
@pytest.mark.parametrize("level,value", [(0.5, 1), (0.5, 0.51), (0.8, 0.9)])
def test_solid_density_reaches_exact_cell_edges(mode, level, value):
    from shapely.geometry import box

    # Regression for #331: old padding inset a solid edge by dx/6 at level=.5.
    geometry = density_to_shapely_geometry(
        np.full((8, 12), value),
        level=level,
        mode=mode,
        x0=-300e-9,
        y0=200e-9,
        dx=60e-9,
        dy=80e-9,
    )
    expected = box(-300e-9, 200e-9, 420e-9, 840e-9)
    assert geometry.symmetric_difference(expected).area < 1e-26


def test_pixel_exact_preserves_diagonal_pixels_holes_and_nested_islands():
    from shapely.geometry import box
    from shapely.ops import unary_union

    density = np.zeros((12, 15))
    density[1:10, 2:13] = 1
    density[3:8, 4:11] = 0
    density[4:6, 5:7] = 1
    density[10, 13] = 1  # Corner-only contact must not become a finite-width bridge.
    geometry = density_to_shapely_geometry(density, mode="pixel_exact")
    expected = unary_union([box(x, y, x + 1, y + 1) for y, x in np.argwhere(density)])
    assert geometry.equals(expected)
    assert geometry.area == density.sum()
    assert len(list(_iter_polygons(geometry))) == 3


def test_contour_sampling_converges_to_bilinear_interface():
    from shapely.geometry import box

    yy, xx = np.mgrid[:8, :10] + 0.5
    density = xx * yy / 100
    domain = box(0.5, 0.5, 9.5, 7.5)
    # Analytic area above x*y=25, restricted to the sample-center domain.
    exact_area = 7.5 * (9.5 - 25 / 7.5) - 25 * np.log(9.5 / (25 / 7.5))
    errors = []
    for spacing in [1, 0.25, 0.0625]:
        geometry = density_to_shapely_geometry(
            density,
            level=0.25,
            sample_spacing=spacing,
        ).intersection(domain)
        errors.append(abs(geometry.area - exact_area))
    assert errors[1] < errors[0] / 8
    assert errors[2] < errors[1] / 8


@pytest.mark.parametrize("mode", ["contour", "pixel_exact"])
def test_fixed_cells_and_nonrectangular_region_are_exact(mode):
    from shapely.geometry import box

    density = np.full((10, 12), 0.51)
    solid = np.zeros_like(density, dtype=bool)
    void = solid.copy()
    solid[4:6, :3] = True
    void[:4, :3] = True
    void[6:, :3] = True
    region = np.ones_like(solid)
    region[8:, 8:] = False
    geometry = density_to_shapely_geometry(
        density,
        fixed_solid=solid,
        fixed_void=void,
        region_mask=region,
        mode=mode,
    )
    assert geometry.covers(box(0, 4, 3, 6))
    assert geometry.intersection(box(0, 0, 3, 4)).area == 0
    assert geometry.intersection(box(8, 8, 12, 10)).area == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dx": 0},
        {"dy": -1},
        {"dx": np.inf},
        {"x0": np.nan},
        {"level": np.nan},
        {"min_area": -1},
        {"mode": "smooth"},
        {"sample_spacing": 0},
        {"sample_spacing": np.inf},
        {"mode": "pixel_exact", "sample_spacing": 0.1},
        {"fixed_solid": np.ones((2, 2))},
    ],
)
def test_invalid_geometry_options_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        density_to_shapely_geometry(np.ones((2, 2)), **kwargs)


@pytest.mark.parametrize(
    "density", [np.empty((0, 2)), np.ones((2, 2, 2)), np.full((2, 2), np.nan)]
)
def test_invalid_density_fails_explicitly(density):
    with pytest.raises(ValueError):
        density_to_shapely_geometry(density)
