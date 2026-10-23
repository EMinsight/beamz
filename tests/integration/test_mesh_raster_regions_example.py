"""Check that the standalone region example preserves full-grid fields."""

import numpy as np

from beamz.design.raster import Box, Grid, Material, Object, RasterOptions, Scene
from beamz.design.raster import rasterize as full_rasterize
from examples.mesh_raster_regions import rasterize_split_substrate, rasterize_tiled_x


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
    reduced, _ = rasterize_tiled_x(scene, grid, options, tile_cells=3)
    _assert_material_fields_match(full, reduced)


def test_split_substrate_matches_full_grid():
    scene = Scene(
        (Material(), Material(permittivity=4.0), Material(permittivity=8.0)),
        (
            Object(Box((0, 0, 0), (1, 1, 0.3)), material_id=2, priority=10),
            Object(Box((0.2, 0.2, 0.7), (0.8, 0.8, 0.8)), material_id=1, priority=20),
        ),
    )
    grid = Grid.uniform((0, 0, 0), (1, 1, 1), (12, 12, 20))
    options = RasterOptions()
    full = full_rasterize(scene, grid, options=options)
    reduced, _ = rasterize_split_substrate(scene, grid, options, core_tile_cells=3)
    _assert_material_fields_match(full, reduced)
