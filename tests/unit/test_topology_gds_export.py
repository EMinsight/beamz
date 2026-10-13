"""Read the serialized masks independently and verify topology and precision."""

import numpy as np
import pytest
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from beamz import Design, Material, Rectangle
from beamz.design import export_gds
from beamz.optimization import density_to_polygons


def read_geometry(path, layer=(7, 2)):
    kdb = pytest.importorskip("klayout.db")
    layout = kdb.Layout()
    layout.read(str(path))
    region = kdb.Region(layout.top_cell().begin_shapes_rec(layout.layer(*layer)))
    region.merge()
    scale = layout.dbu * 1e-6
    shapes = []
    for p in region.each():
        shapes.append(
            Polygon(
                [(v.x * scale, v.y * scale) for v in p.each_point_hull()],
                [
                    [(v.x * scale, v.y * scale) for v in p.each_point_hole(i)]
                    for i in range(p.holes())
                ],
            )
        )
    return unary_union(shapes), layout


@pytest.mark.parametrize(
    "mode,spacing", [("contour", None), ("contour", 10e-9), ("pixel_exact", None)]
)
def test_gds_roundtrip_preserves_port_connection_and_hole(tmp_path, mode, spacing):
    pytest.importorskip("gdsfactory")
    core, clad = Material(4), Material(1)
    density = np.ones((12, 16))
    density[4:8, 6:10] = 0
    design = Design(width=1.6e-6, height=0.72e-6, material=clad)
    design += Rectangle(
        position=(0, 0.24e-6), width=0.3e-6, height=0.24e-6, material=core
    )
    # The clearing polygon must not accidentally become part of the core mask.
    design += Rectangle(
        position=(0.3e-6, 0), width=0.96e-6, height=0.72e-6, material=clad
    )
    for p in density_to_polygons(
        density, material=core, x0=0.3e-6, dx=60e-9, mode=mode, sample_spacing=spacing
    ):
        design += p
    path = export_gds(design, tmp_path / f"{mode}.gds", layer_map={core: (7, 2)})
    geometry, layout = read_geometry(path)
    assert geometry.geom_type == "Polygon"  # Connected after write/read/union.
    assert len(geometry.interiors) == 1
    assert geometry.covers(box(0.299e-6, 0.3e-6, 0.301e-6, 0.36e-6))
    assert not geometry.contains(box(0.72e-6, 0.3e-6, 0.78e-6, 0.36e-6))
    assert [(info.layer, info.datatype) for info in layout.layer_infos()] == [(7, 2)]


def test_precision_is_local_and_not_limited_by_global_gdsfactory_grid(tmp_path):
    gf = pytest.importorskip("gdsfactory")
    original_dbu = gf.kcl.dbu
    core = Material(4)
    design = Design(width=1e-6, height=1e-6)
    design += Rectangle(
        position=(0.10025e-6, 0.20025e-6),
        width=0.5005e-6,
        height=0.4005e-6,
        material=core,
    )
    path = export_gds(
        design, tmp_path / "fine.gds", precision=0.25e-9, layer_map={core: (7, 2)}
    )
    geometry, layout = read_geometry(path)
    np.testing.assert_allclose(
        geometry.bounds, [0.10025e-6, 0.20025e-6, 0.60075e-6, 0.60075e-6], atol=1e-20
    )
    assert layout.dbu == pytest.approx(0.00025)
    assert gf.kcl.dbu == original_dbu
    with pytest.raises(ValueError, match="too fine"):
        export_gds(design, tmp_path / "overflow.gds", precision=1e-17)
    assert not (tmp_path / "overflow.gds").exists()


@pytest.mark.parametrize("precision", [0, -1e-9, np.nan, np.inf])
def test_rejects_invalid_precision(tmp_path, precision):
    with pytest.raises(ValueError, match="precision"):
        export_gds(Design(width=1, height=1), tmp_path / "bad.gds", precision=precision)
