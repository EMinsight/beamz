"""Check opt-in region results against the normal full-grid raster."""

import numpy as np
import pytest

from beamz.design.discretization import MaterialGrid
from beamz.design.raster import (
    Box,
    Grid,
    Material,
    Object,
    RasterOptions,
    RasterResult,
    Scene,
)
from beamz.design.raster import rasterize as full_rasterize
from beamz.design.raster._regions import rasterize_regions


def _assert_material_fields_match(full, reduced):
    for group_name in ("tensors", "yee_tensors"):
        for name, expected in getattr(full, group_name).items():
            np.testing.assert_array_equal(expected, getattr(reduced, group_name)[name])


def test_tiled_x_matches_full_grid():
    scene = Scene(
        (Material(), Material(permittivity=4.0)),
        (Object(Box((0.2, 0.2, 0.2), (0.8, 0.8, 0.8)), material_id=1),),
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (12, 12, 12))
    options = RasterOptions()
    full = full_rasterize(scene, grid, options=options)
    reduced = rasterize_regions(scene, grid, options=options, tile_x_cells=3)
    assert isinstance(reduced, RasterResult)
    assert not reduced.cache_hit
    assert reduced.scene_hash == full.scene_hash
    _assert_material_fields_match(full, reduced)
    regular_grid = MaterialGrid.from_raster_result(full, dimensions=3)
    region_grid = MaterialGrid.from_raster_result(reduced, dimensions=3)
    np.testing.assert_array_equal(regular_grid.permittivity, region_grid.permittivity)


def test_split_substrate_matches_full_grid():
    substrate = Object(Box((0, 0, 0), (1, 1, 0.3)), material_id=2, priority=10)
    scene = Scene(
        (Material(), Material(permittivity=4.0), Material(permittivity=8.0)),
        (
            substrate,
            Object(Box((0.2, 0.2, 0.7), (0.8, 0.8, 0.8)), material_id=1, priority=20),
        ),
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (12, 12, 20))
    options = RasterOptions()
    full = full_rasterize(scene, grid, options=options)
    reduced = rasterize_regions(
        scene, grid, options=options, tile_x_cells=3, substrate_object=substrate
    )
    _assert_material_fields_match(full, reduced)


def test_region_rejects_periodic_and_anisotropic_background():
    obj = Object(Box((0.2, 0.2, 0.2), (0.8, 0.8, 0.8)), material_id=1)
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (8, 8, 8))
    scene = Scene((Material(), Material(permittivity=4.0)), (obj,))
    with pytest.raises(ValueError, match="periodic"):
        rasterize_regions(scene, grid, options=RasterOptions(periodic_axes=("x",)))
    anisotropic = Scene(
        (Material(permittivity=(1.0, 1.1, 1.2)), Material(permittivity=4.0)),
        (obj,),
    )
    with pytest.raises(ValueError, match="isotropic"):
        rasterize_regions(anisotropic, grid)


def test_nonuniform_grid_and_nonunit_permeability_match_full_raster():
    scene = Scene(
        (
            Material(permeability=2.0),
            Material(permittivity=4.0, permeability=3.0),
        ),
        (Object(Box((0.3, 0.2, 0.1), (0.7, 0.8, 0.9)), material_id=1),),
    )
    grid = Grid(
        np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]),
        np.array([0.0, 0.2, 0.6, 0.8, 1.0]),
        np.array([0.0, 0.1, 0.4, 0.6, 0.9, 1.0]),
    )
    full = full_rasterize(scene, grid)
    reduced = rasterize_regions(scene, grid, tile_x_cells=2)
    _assert_material_fields_match(full, reduced)
