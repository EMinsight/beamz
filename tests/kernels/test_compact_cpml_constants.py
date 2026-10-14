"""Uniform loss and permeability stay scalar under 3D CPML without changing physics."""

from __future__ import annotations

import jax
import numpy as np
import pytest

import beamz as bz
from beamz import PEC, PML
from beamz.const import LIGHT_SPEED
from beamz.design import MaterialGrid, raster
from beamz.devices import _boundary_compile
from beamz.lattice import component_shapes

SPACING = 0.1e-6
CELLS = 14


def _simulation(
    boundaries, *, smoothing="farjadpour_diagonal", conductivity=0.0, background=0.0
):
    size = (CELLS * SPACING,) * 3
    dt = 0.5 * SPACING / (LIGHT_SPEED * np.sqrt(3.0))
    sphere = raster.Sphere(tuple(0.5 * value for value in size), 0.35e-6)
    return bz.Simulation(
        scene=raster.Scene(
            (
                raster.Material(conductivity=background),
                raster.Material(4.0, conductivity=conductivity or background),
            ),
            (raster.Object(sphere, 1),),
        ),
        raster_grid=raster.Grid.uniform((0.0, 0.0, 0.0), size, (CELLS,) * 3),
        time=np.arange(8) * dt,
        boundaries=boundaries,
        sources=(),
        normalize_source=None,
        raster_options=raster.RasterOptions(smoothing=smoothing),
    )


def _lower(simulation):
    materials = simulation._material_grid()
    return materials, _boundary_compile.lower_boundaries(
        materials,
        component_shapes(materials.shape),
        simulation.boundaries,
        materials.grid.extent,
        simulation.dt,
    )


def _lower_both(simulation, monkeypatch):
    materials, compact = _lower(simulation)
    with monkeypatch.context() as patch:
        patch.setattr(
            _boundary_compile, "_scalar_constants_allowed", lambda *_args: False
        )
        _, dense = _lower(simulation)
    return materials, compact, dense


def _material_arrays(data):
    arrays = {
        "permittivity": data.permittivity,
        "conductivity": data.conductivity,
        "permeability": data.permeability,
    }
    return {**arrays, **data.yee_materials}


def _assert_same_values(compact, dense):
    compact_arrays, dense_arrays = _material_arrays(compact), _material_arrays(dense)
    assert compact_arrays.keys() == dense_arrays.keys()
    for name, expected in dense_arrays.items():
        expected = np.asarray(expected)
        actual = np.asarray(compact_arrays[name])
        assert actual.dtype == expected.dtype, name
        np.testing.assert_array_equal(
            np.broadcast_to(actual, expected.shape), expected, err_msg=name
        )
    assert compact.profiles.keys() == dense.profiles.keys()
    for name, expected in dense.profiles.items():
        if isinstance(expected, str):
            assert compact.profiles[name] == expected
        else:
            np.testing.assert_array_equal(compact.profiles[name], expected)
    assert compact.masks.keys() == dense.masks.keys()
    for name, expected in dense.masks.items():
        np.testing.assert_array_equal(compact.masks[name], expected)
    assert compact.metallic_edges == dense.metallic_edges
    assert compact.periodic_axes == dense.periodic_axes


@pytest.mark.parametrize(
    "smoothing", ["volume", "farjadpour_diagonal", "farjadpour_full"]
)
def test_uniform_cpml_constants_are_exact_scalars(smoothing, monkeypatch):
    materials, compact, dense = _lower_both(
        _simulation([PML(formulation="cpml")], smoothing=smoothing), monkeypatch
    )
    assert np.asarray(compact.conductivity).ndim == 0
    assert np.asarray(compact.permeability).ndim == 0
    assert np.asarray(compact.permittivity).ndim == 3
    assert np.asarray(dense.conductivity).ndim == 3
    if smoothing == "farjadpour_diagonal":
        assert materials.uses_direct_yee_materials
    constants = [
        name for name in compact.yee_materials if name.startswith(("sig_", "mu_"))
    ]
    assert len(constants) == (6 if materials.yee_materials else 0)
    for name, value in compact.yee_materials.items():
        expected = 0 if name in constants else 3
        assert np.asarray(value).ndim == expected, name
    _assert_same_values(compact, dense)


@pytest.mark.parametrize(
    "boundaries",
    [[PML(formulation="sponge")], [PEC(), PML(edges=("left",), formulation="cpml")]],
    ids=["sponge", "mixed-pec"],
)
def test_other_boundaries_keep_full_constants(boundaries, monkeypatch):
    _, compact, dense = _lower_both(_simulation(boundaries), monkeypatch)
    for name, value in _material_arrays(compact).items():
        assert np.asarray(value).ndim == 3, name
    _assert_same_values(compact, dense)


def test_nonuniform_conductivity_keeps_its_array(monkeypatch):
    _, compact, dense = _lower_both(
        _simulation([PML(formulation="cpml")], conductivity=50.0), monkeypatch
    )
    assert np.asarray(compact.conductivity).ndim == 3
    assert np.asarray(compact.permeability).ndim == 0
    for axis in "xyz":
        assert np.asarray(compact.yee_materials["sig_" + axis]).ndim == 3
        assert np.asarray(compact.yee_materials["mu_h" + axis]).ndim == 0
    _assert_same_values(compact, dense)


def test_gate_rejects_2d():
    materials = MaterialGrid(
        np.full((8, 10), 2.0), np.zeros((8, 10)), np.ones((8, 10)), SPACING, (8, 10)
    )
    assert not _boundary_compile._scalar_constants_allowed(
        materials, (PML(formulation="cpml"),)
    )


# A nonzero uniform background sends a nonzero scalar sig_* through the kernels.
@pytest.mark.parametrize("background", [0.0, 50.0])
def test_scalar_constants_preserve_diagonal_fields(background, monkeypatch):
    def advance():
        simulation = _simulation([PML(formulation="cpml")], background=background)
        simulation.clear_compiled_cache()
        state = simulation.initial_state()
        random = np.random.default_rng(41)
        state = state._replace(
            **{
                name: random.normal(size=getattr(state, name).shape).astype(np.float32)
                for name in ("ex", "ey", "ez", "hx", "hy", "hz")
            }
        )
        final = simulation.advance(state=state, backend="jax").state
        grid = simulation.compile(backend="jax").grid
        stored = (np.ndim(grid.permeability), np.ndim(grid.sig_x))
        simulation.clear_compiled_cache()
        return final, stored

    compact, compact_ndim = advance()
    with monkeypatch.context() as patch:
        patch.setattr(
            _boundary_compile, "_scalar_constants_allowed", lambda *_args: False
        )
        dense, dense_ndim = advance()
    assert compact_ndim == (0, 0)
    assert dense_ndim == (3, 3)
    for actual, expected in zip(
        jax.tree.leaves(compact), jax.tree.leaves(dense), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
