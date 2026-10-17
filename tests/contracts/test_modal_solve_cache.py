"""Candidate reuse must preserve the current polarization-aware modal basis."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from beamz.design.grid import RectilinearGrid
from beamz.devices.modes import discrete
from beamz.devices.modes._cache import _SolveCache
from beamz.devices.modes.discrete import ModePlaneSpec


def _spec(axis="x", polarization="te"):
    shape = (7, 8, 9)
    transverse = {"x": ("z", "y"), "y": ("z", "x"), "z": ("y", "x")}[axis]
    sizes = dict(zip("zyx", shape, strict=True))
    eps = np.full(tuple(sizes[a] for a in transverse), 1.44**2)
    eps[2:5, 2:6] = 2.2**2
    dx = 60e-9
    return ModePlaneSpec(
        scalar_permittivity=eps,
        frequency=299_792_458 / 1.55e-6,
        resolution=dx,
        dt=0.35 * dx / 299_792_458,
        axis=axis,
        direction="+" + axis,
        transverse_axes=transverse,
        grid_shape=shape,
        center=tuple(0.5 * n * dx for n in shape[::-1]),
        width=0.48e-6,
        height=0.48e-6,
        plane_index=3,
        offset_index=2,
        polarization=polarization,
        target_neff=2.0,
        num_modes=3,
    )


@pytest.fixture(autouse=True)
def _empty_caches():
    discrete._MODE_CACHE.clear()
    discrete._CANDIDATE_CACHE.clear()
    yield
    discrete._MODE_CACHE.clear()
    discrete._CANDIDATE_CACHE.clear()


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("polarization", ["te", "tm", None])
def test_selected_modes_share_exact_candidate_pool(monkeypatch, axis, polarization):
    spec = _spec(axis, polarization)
    specs = [replace(spec, mode_index=i) for i in range(3)]
    expected = [discrete.solve_beamz_mode(s) for s in specs]
    original = discrete.solve_grid
    counts = []

    def counted(**kwargs):
        counts.append(kwargs["num_modes"])
        return original(**kwargs)

    monkeypatch.setattr(discrete, "solve_grid", counted)
    for request, reference in zip(specs, expected, strict=True):
        actual = discrete._solve_cached_beamz_mode(request)
        np.testing.assert_equal(asdict(actual), asdict(reference))
        actual.profiles["Ey"][:] = 0
        actual.diagnostics["solver_info"].clear()
    assert counts == [3 if polarization is None else 11]
    assert discrete._CANDIDATE_CACHE.hits == 2
    # Fresh equivalent requests share complete bases, including normalization.
    for request, reference in zip(specs, expected, strict=True):
        actual = discrete._solve_cached_beamz_mode(replace(request))
        np.testing.assert_equal(asdict(actual), asdict(reference))
    assert discrete._MODE_CACHE.hits == 3
    assert len(counts) == 1


_CHANGES = {
    "scalar_permittivity": lambda s: s.scalar_permittivity + 0.01,
    "frequency": lambda s: s.frequency * 1.01,
    "resolution": lambda s: s.resolution * 1.01,
    "dt": lambda s: s.dt * 1.01,
    "direction": lambda s: "-x",
    "transverse_axes": lambda s: ("y", "z"),
    "grid_shape": lambda s: (7, 8, 10),
    "center": lambda s: (s.center[0] + 1e-9, *s.center[1:]),
    "width": lambda s: s.width * 0.9,
    "height": lambda s: s.height * 0.9,
    "plane_index": lambda s: 4,
    "offset_index": lambda s: 1,
    "mode_index": lambda s: 1,
    "polarization": lambda s: "tm",
    "target_neff": lambda s: 2.1,
    "num_modes": lambda s: 4,
    "solver_direction": lambda s: "-x",
    "aperture_pad_cells": lambda s: 3,
    "aperture_window_alpha": lambda s: 0.3,
    "component_permittivity": lambda s: {"Ey": np.ones((3, 3))},
    "component_permeability": lambda s: {"Hz": np.ones((3, 3))},
    "diagonal_permittivity": lambda s: {"yy": s.scalar_permittivity * 1.01},
    "diagonal_permeability": lambda s: {"zz": np.ones_like(s.scalar_permittivity)},
    "yee_refinement": lambda s: False,
    "grid": lambda s: RectilinearGrid.uniform(
        (0, 0, 0),
        tuple(n * s.resolution for n in s.grid_shape[::-1]),
        s.grid_shape[::-1],
    ),
}


@pytest.mark.parametrize("name", _CHANGES)
def test_completed_basis_invalidates_for_changed_spec(monkeypatch, name):
    spec = _spec()
    calls = []

    def fake_solve(request, **kwargs):
        calls.append(request)
        return {"call": len(calls)}

    monkeypatch.setattr(discrete, "_solve_beamz_mode", fake_solve)
    first = discrete._solve_cached_beamz_mode(spec)
    assert discrete._solve_cached_beamz_mode(replace(spec)) == first
    changed = replace(spec, **{name: _CHANGES[name](spec)})
    assert discrete._solve_cached_beamz_mode(changed) != first
    assert len(calls) == 2


def test_candidate_cache_keys_actual_solver_inputs(monkeypatch):
    calls = []

    def fake_solve(**kwargs):
        calls.append(kwargs)
        return len(calls)

    monkeypatch.setattr(discrete, "solve_grid", fake_solve)
    cache = discrete._CANDIDATE_CACHE
    kwargs = dict(
        eps_xx=np.ones((3, 4)),
        eps_yy=None,
        eps_zz=None,
        mu_xx=None,
        mu_yy=None,
        mu_zz=None,
        x_edges=(0, 1, 2, 3),
        y_edges=(0, 1, 2, 3, 4),
        freqs=[2e14],
        num_modes=11,
        target_neff=2.0,
        direction="+",
        normal_axis=0,
    )
    assert discrete._solve_candidate_pool(cache, **kwargs) == 1
    assert discrete._solve_candidate_pool(cache, **kwargs) == 1
    for name in kwargs:
        change = dict(kwargs)
        value = kwargs[name]
        if name.startswith(("eps", "mu")):
            change[name] = np.full((3, 4), 1.1)
        elif name.endswith("edges") or name == "freqs":
            change[name] = np.asarray(value) * 1.01
        elif name == "direction":
            change[name] = "-"
        else:
            change[name] = value + 1
        before = len(calls)
        discrete._solve_candidate_pool(cache, **change)
        assert len(calls) == before + 1, name
    # In-place material mutations also invalidate; keys never retain the array.
    kwargs["eps_xx"][0, 0] = 5
    before = len(calls)
    discrete._solve_candidate_pool(cache, **kwargs)
    assert len(calls) == before + 1


def test_cache_evicts_lru_and_bounds_payload_bytes():
    cache = _SolveCache(max_entries=2, max_bytes=1024)

    def compute():
        return np.ones(16)

    cache.get_or_compute("a", compute)
    cache.get_or_compute("b", compute)
    cache.get_or_compute("a", compute)
    cache.get_or_compute("c", compute)
    assert list(cache._entries) == ["a", "c"]
    cache.get_or_compute("large", lambda: np.ones(1000))
    assert list(cache._entries) == ["a", "c"]
    cache.max_bytes = cache.bytes
    cache.max_entries = 10
    cache.get_or_compute("d", compute)
    assert list(cache._entries) == ["c", "d"]
    assert cache.bytes <= cache.max_bytes
    cache.clear()
    assert not cache._entries and cache.bytes == cache.hits == cache.misses == 0


def test_failed_solves_are_not_cached():
    cache = _SolveCache(max_entries=2, max_bytes=1024)

    def fail():
        raise ValueError("failed solve")

    with pytest.raises(ValueError, match="failed solve"):
        cache.get_or_compute("key", fail)
    assert cache.get_or_compute("key", lambda: 42) == 42
    assert cache.misses == 2
