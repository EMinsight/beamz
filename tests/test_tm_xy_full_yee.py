from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from beamz.const import EPS_0, LIGHT_SPEED, MU_0
from beamz.simulation import ops
from beamz.simulation.boundaries import (
    full_pec_curl_e_to_h_2d_xy,
    full_pec_curl_h_to_e_2d_xy,
    initialize_tm_2d_xy_state,
    normalize_boundaries,
    tm_xy_curl_e_to_h_2d,
    tm_xy_curl_h_to_e_2d,
    xy_te_curl_e_to_h_2d,
    xy_te_curl_h_to_e_2d,
)
from beamz.simulation.fields import Fields


def _tm_xy_lossless_step(ez, hx, hy, *, dx, dt, metallic_edges=frozenset()):
    ez = jnp.asarray(ez)
    hx = jnp.asarray(hx)
    hy = jnp.asarray(hy)
    curl_hx, curl_hy = tm_xy_curl_e_to_h_2d(
        ez,
        dx,
        hx.shape,
        hy.shape,
        metallic_edges,
    )
    hx = ops.advance_h_field(hx, curl_hx, np.zeros_like(hx), dt)
    hy = ops.advance_h_field(hy, curl_hy, np.zeros_like(hy), dt)
    curl_ez = tm_xy_curl_h_to_e_2d(hx, hy, dx, ez.shape, metallic_edges)
    ez = ops.advance_e_field(
        ez,
        curl_ez,
        np.zeros_like(ez),
        np.ones_like(ez),
        dt,
        (slice(None), slice(None)),
    )
    return ez, hx, hy


def _tm_xy_full_pec_step(ez, hx, hy, *, dx, dt, ez_mask, hx_mask, hy_mask):
    ez = jnp.asarray(ez)
    hx = jnp.asarray(hx)
    hy = jnp.asarray(hy)
    curl_hx, curl_hy = full_pec_curl_e_to_h_2d_xy(ez, dx, hx.shape, hy.shape)
    hx = ops.advance_h_field(hx, curl_hx, np.zeros_like(hx), dt)
    hy = ops.advance_h_field(hy, curl_hy, np.zeros_like(hy), dt)
    hx = jnp.where(jnp.asarray(hx_mask), 0.0, hx)
    hy = jnp.where(jnp.asarray(hy_mask), 0.0, hy)
    curl_ez = full_pec_curl_h_to_e_2d_xy(hx, hy, dx, ez.shape)
    ez = ops.advance_e_field(
        ez,
        curl_ez,
        np.zeros_like(ez),
        np.ones_like(ez),
        dt,
        (slice(None), slice(None)),
    )
    ez = jnp.where(jnp.asarray(ez_mask), 0.0, ez)
    return ez, hx, hy


def _tm_xy_energy(ez, hx, hy, *, dx):
    electric = 0.5 * EPS_0 * np.sum(np.asarray(ez) ** 2)
    magnetic = 0.5 * MU_0 * (
        np.sum(np.asarray(hx) ** 2) + np.sum(np.asarray(hy) ** 2)
    )
    return float((electric + magnetic) * dx * dx)


def _te_xy_lossless_step(ex, ey, hz, *, dx, dt, metallic_edges=frozenset()):
    ex = jnp.asarray(ex)
    ey = jnp.asarray(ey)
    hz = jnp.asarray(hz)
    curl_hz = xy_te_curl_e_to_h_2d(ex, ey, dx, hz.shape)
    hz = ops.advance_h_field(hz, curl_hz, np.zeros_like(hz), dt)
    curl_ex, curl_ey = xy_te_curl_h_to_e_2d(hz, dx, ex.shape, ey.shape, metallic_edges)
    ex = ops.advance_e_field(
        ex,
        curl_ex,
        np.zeros_like(ex),
        np.ones_like(ex),
        dt,
        (slice(None), slice(None)),
    )
    ey = ops.advance_e_field(
        ey,
        curl_ey,
        np.zeros_like(ey),
        np.ones_like(ey),
        dt,
        (slice(None), slice(None)),
    )
    return ex, ey, hz


def _te_xy_pec_step(ex, ey, hz, *, dx, dt):
    ex = jnp.asarray(ex)
    ey = jnp.asarray(ey)
    hz = jnp.asarray(hz)
    curl_hz = xy_te_curl_e_to_h_2d(ex, ey, dx, hz.shape)
    hz = ops.advance_h_field(hz, curl_hz, np.zeros_like(hz), dt)
    curl_ex, curl_ey = xy_te_curl_h_to_e_2d(
        hz,
        dx,
        ex.shape,
        ey.shape,
        frozenset({"left", "right", "top", "bottom"}),
    )
    ex = ops.advance_e_field(
        ex,
        curl_ex,
        np.zeros_like(ex),
        np.ones_like(ex),
        dt,
        (slice(None), slice(None)),
    )
    ey = ops.advance_e_field(
        ey,
        curl_ey,
        np.zeros_like(ey),
        np.ones_like(ey),
        dt,
        (slice(None), slice(None)),
    )
    ex = ex.at[0, :].set(0.0)
    ex = ex.at[-1, :].set(0.0)
    ey = ey.at[:, 0].set(0.0)
    ey = ey.at[:, -1].set(0.0)
    return ex, ey, hz


def _te_xy_energy(ex, ey, hz, *, dx):
    electric = 0.5 * EPS_0 * (
        np.sum(np.asarray(ex) ** 2) + np.sum(np.asarray(ey) ** 2)
    )
    magnetic = 0.5 * MU_0 * np.sum(np.asarray(hz) ** 2)
    return float((electric + magnetic) * dx * dx)


def _x_centroid(field):
    weights = np.sum(np.asarray(field) ** 2, axis=0)
    x = np.arange(weights.shape[0], dtype=np.float64)
    return float(np.sum(weights * x) / np.sum(weights))


def _build_x_propagating_packet(direction: str, *, ny=12, nx=192, dx=1e-6):
    eta_0 = np.sqrt(MU_0 / EPS_0)
    x0 = 0.35 * nx * dx
    sigma = 12.0 * dx
    wavelength = 24.0 * dx
    k = 2.0 * np.pi / wavelength
    sign = -1.0 if direction == "+x" else 1.0

    x_ez = np.arange(nx + 1, dtype=np.float32) * dx
    x_hy = (np.arange(nx, dtype=np.float32) + 0.5) * dx

    ez_profile = np.exp(-((x_ez - x0) / sigma) ** 2) * np.cos(k * (x_ez - x0))
    hy_profile = (
        sign
        * np.exp(-((x_hy - x0) / sigma) ** 2)
        * np.cos(k * (x_hy - x0))
        / eta_0
    )

    ez = np.tile(ez_profile[None, :], (ny + 1, 1))
    hx = np.zeros((ny, nx + 1), dtype=np.float64)
    hy = np.tile(hy_profile[None, :], (ny + 1, 1))
    sample_x = int(np.argmin(np.abs(x_hy - x0)))
    return ez, hx, hy, sample_x


def _build_x_propagating_te_packet(direction: str, *, ny=13, nx=192, dx=1e-6):
    eta_0 = np.sqrt(MU_0 / EPS_0)
    x0 = 0.35 * nx * dx
    sigma = 12.0 * dx
    wavelength = 24.0 * dx
    k = 2.0 * np.pi / wavelength
    sign = 1.0 if direction == "+x" else -1.0

    x_ey = (np.arange(nx, dtype=np.float32) + 0.5) * dx
    x_hz = np.arange(nx - 1, dtype=np.float32) * dx

    ey_profile = np.exp(-((x_ey - x0) / sigma) ** 2) * np.cos(k * (x_ey - x0))
    hz_profile = (
        sign
        * np.exp(-((x_hz - x0) / sigma) ** 2)
        * np.cos(k * (x_hz - x0))
        / eta_0
    )

    ex = np.zeros((ny, nx - 1), dtype=np.float32)
    ey = np.tile(ey_profile[None, :], (ny - 1, 1))
    hz = np.tile(hz_profile[None, :], (ny - 1, 1))
    sample_x = int(np.argmin(np.abs(x_hz - x0)))
    return ex, ey, hz, sample_x


def test_ops_reject_generic_xy_tm_curls():
    ez = np.zeros((4, 4), dtype=np.float32)
    ex = np.zeros((4, 3), dtype=np.float32)
    ey = np.zeros((3, 4), dtype=np.float32)
    hx = np.zeros((4, 3), dtype=np.float32)
    hy = np.zeros((3, 4), dtype=np.float32)
    hz = np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="does not handle plane='xy'"):
        ops.curl_e_to_h_2d((ex, ey, ez), 1.0, plane="xy")

    with pytest.raises(ValueError, match="does not handle plane='xy'"):
        ops.curl_h_to_e_2d((hx, hy, hz), 1.0, (ex.shape, ey.shape, ez.shape), plane="xy")


def test_tm_xy_full_yee_curls_match_tmyz_identities():
    ny, nx = 5, 7
    y = np.arange(ny + 1, dtype=np.float32)[:, None]
    x = np.arange(nx + 1, dtype=np.float32)[None, :]
    ez = y + 2.0 * x

    curl_hx, curl_hy = tm_xy_curl_e_to_h_2d(
        ez,
        1.0,
        (ny, nx + 1),
        (ny + 1, nx),
        frozenset(),
    )
    np.testing.assert_allclose(curl_hx, np.ones((ny, nx + 1)))
    np.testing.assert_allclose(curl_hy, -2.0 * np.ones((ny + 1, nx)))

    hx = 3.0 * (np.arange(ny, dtype=np.float32)[:, None] + 0.5) * np.ones(
        (1, nx + 1), dtype=np.float32
    )
    hy = 5.0 * np.ones((ny + 1, 1), dtype=np.float32) * (
        np.arange(nx, dtype=np.float32)[None, :] + 0.5
    )
    curl_ez = tm_xy_curl_h_to_e_2d(hx, hy, 1.0, (ny + 1, nx + 1), frozenset())
    np.testing.assert_allclose(curl_ez[1:-1, 1:-1], 2.0)


def test_tm_xy_constant_fields_have_zero_curl():
    ny, nx = 6, 8
    ez = np.ones((ny + 1, nx + 1), dtype=np.float32)
    hx = np.ones((ny, nx + 1), dtype=np.float32)
    hy = np.ones((ny + 1, nx), dtype=np.float32)

    curl_hx, curl_hy = tm_xy_curl_e_to_h_2d(
        ez,
        1.0,
        hx.shape,
        hy.shape,
        frozenset(),
    )
    curl_ez = tm_xy_curl_h_to_e_2d(hx, hy, 1.0, ez.shape, frozenset())

    np.testing.assert_allclose(curl_hx, 0.0)
    np.testing.assert_allclose(curl_hy, 0.0)
    np.testing.assert_allclose(curl_ez, 0.0)


def test_tm_xy_plane_wave_packet_propagates_in_the_expected_x_direction():
    dx = 1e-6
    dt = 0.25 * dx / LIGHT_SPEED
    ez, hx, hy, _sample_x = _build_x_propagating_packet("+x", dx=dx)
    initial_centroid = _x_centroid(ez)

    for _ in range(32):
        ez, hx, hy = _tm_xy_lossless_step(ez, hx, hy, dx=dx, dt=dt)

    assert _x_centroid(ez) > initial_centroid + 2.0
    np.testing.assert_allclose(hx, 0.0, atol=1e-12)


def test_tm_xy_reversing_x_propagation_flips_hy_sign_and_motion():
    dx = 1e-6
    dt = 0.25 * dx / LIGHT_SPEED
    ez_plus, hx_plus, hy_plus, sample_x = _build_x_propagating_packet("+x", dx=dx)
    ez_minus, hx_minus, hy_minus, _ = _build_x_propagating_packet("-x", dx=dx)

    sample_y = hy_plus.shape[0] // 2
    assert hy_plus[sample_y, sample_x] < 0.0
    assert hy_minus[sample_y, sample_x] > 0.0

    plus_initial = _x_centroid(ez_plus)
    minus_initial = _x_centroid(ez_minus)

    for _ in range(32):
        ez_plus, hx_plus, hy_plus = _tm_xy_lossless_step(
            ez_plus, hx_plus, hy_plus, dx=dx, dt=dt
        )
        ez_minus, hx_minus, hy_minus = _tm_xy_lossless_step(
            ez_minus, hx_minus, hy_minus, dx=dx, dt=dt
        )

    assert _x_centroid(ez_plus) > plus_initial + 2.0
    assert _x_centroid(ez_minus) < minus_initial - 2.0


def test_tm_xy_closed_pec_domain_conserves_discrete_energy():
    ny, nx = 24, 30
    dx = 1e-6
    dt = 0.15 * dx / (LIGHT_SPEED * np.sqrt(2.0))

    fields = Fields(
        permittivity=np.ones((ny, nx), dtype=np.float64),
        conductivity=np.zeros((ny, nx), dtype=np.float64),
        permeability=np.ones((ny, nx), dtype=np.float64),
        resolution=dx,
        plane_2d="xy",
    )
    fields.boundaries = normalize_boundaries([], is_3d=False)
    state = initialize_tm_2d_xy_state(fields)

    y = np.arange(ny + 1, dtype=np.float32)[:, None]
    x = np.arange(nx + 1, dtype=np.float32)[None, :]
    state.Ez = jnp.asarray(
        np.sin(np.pi * y / ny) * np.sin(2.0 * np.pi * x / nx), dtype=jnp.float32
    )
    state.Hx = jnp.zeros_like(state.Hx)
    state.Hy = jnp.zeros_like(state.Hy)

    energies = [_tm_xy_energy(state.Ez, state.Hx, state.Hy, dx=dx)]
    for _ in range(120):
        state.Ez, state.Hx, state.Hy = _tm_xy_full_pec_step(
            state.Ez,
            state.Hx,
            state.Hy,
            dx=dx,
            dt=dt,
            ez_mask=state.ez_mask,
            hx_mask=state.hx_mask,
            hy_mask=state.hy_mask,
        )
        energies.append(_tm_xy_energy(state.Ez, state.Hx, state.Hy, dx=dx))

    energies = np.asarray(energies)
    relative_excursion = (np.max(energies) - np.min(energies)) / np.mean(energies)
    assert relative_excursion < 0.03


def test_xy_te_curls_match_tez_identities():
    ny, nx = 6, 8
    y_hz = np.arange(ny - 1, dtype=np.float32)[:, None]
    x_hz = np.arange(nx - 1, dtype=np.float32)[None, :]
    hz = y_hz + 2.0 * x_hz

    curl_ex, curl_ey = xy_te_curl_h_to_e_2d(
        hz,
        1.0,
        (ny, nx - 1),
        (ny - 1, nx),
        frozenset(),
    )
    np.testing.assert_allclose(curl_ex[1:-1, :], np.ones((ny - 2, nx - 1)))
    np.testing.assert_allclose(curl_ey[:, 1:-1], -2.0 * np.ones((ny - 1, nx - 2)))

    y_ex = np.arange(ny, dtype=np.float32)[:, None]
    x_ey = np.arange(nx, dtype=np.float32)[None, :]
    ex = 3.0 * y_ex * np.ones((1, nx - 1), dtype=np.float32)
    ey = 5.0 * np.ones((ny - 1, 1), dtype=np.float32) * x_ey
    curl_hz = xy_te_curl_e_to_h_2d(ex, ey, 1.0, (ny - 1, nx - 1))
    np.testing.assert_allclose(curl_hz, 2.0)


def test_xy_te_constant_fields_have_zero_curl():
    ny, nx = 7, 9
    ex = np.ones((ny, nx - 1), dtype=np.float32)
    ey = np.ones((ny - 1, nx), dtype=np.float32)
    hz = np.ones((ny - 1, nx - 1), dtype=np.float32)

    curl_hz = xy_te_curl_e_to_h_2d(ex, ey, 1.0, hz.shape)
    curl_ex, curl_ey = xy_te_curl_h_to_e_2d(hz, 1.0, ex.shape, ey.shape, frozenset())

    np.testing.assert_allclose(curl_hz, 0.0)
    np.testing.assert_allclose(curl_ex, 0.0)
    np.testing.assert_allclose(curl_ey, 0.0)


def test_xy_te_plane_wave_packet_propagates_in_the_expected_x_direction():
    dx = 1e-6
    dt = 0.25 * dx / LIGHT_SPEED
    ex, ey, hz, _sample_x = _build_x_propagating_te_packet("+x", dx=dx)
    initial_centroid = _x_centroid(ey)

    for _ in range(32):
        ex, ey, hz = _te_xy_lossless_step(ex, ey, hz, dx=dx, dt=dt)

    assert _x_centroid(ey) > initial_centroid + 2.0
    np.testing.assert_allclose(ex, 0.0, atol=1e-12)


def test_xy_te_reversing_x_propagation_flips_hz_sign_and_motion():
    dx = 1e-6
    dt = 0.25 * dx / LIGHT_SPEED
    ex_plus, ey_plus, hz_plus, sample_x = _build_x_propagating_te_packet("+x", dx=dx)
    ex_minus, ey_minus, hz_minus, _ = _build_x_propagating_te_packet("-x", dx=dx)

    sample_y = hz_plus.shape[0] // 2
    assert hz_plus[sample_y, sample_x] > 0.0
    assert hz_minus[sample_y, sample_x] < 0.0

    plus_initial = _x_centroid(ey_plus)
    minus_initial = _x_centroid(ey_minus)

    for _ in range(32):
        ex_plus, ey_plus, hz_plus = _te_xy_lossless_step(
            ex_plus, ey_plus, hz_plus, dx=dx, dt=dt
        )
        ex_minus, ey_minus, hz_minus = _te_xy_lossless_step(
            ex_minus, ey_minus, hz_minus, dx=dx, dt=dt
        )

    assert _x_centroid(ey_plus) > plus_initial + 2.0
    assert _x_centroid(ey_minus) < minus_initial - 2.0


def test_xy_te_closed_pec_domain_conserves_discrete_energy():
    ny, nx = 24, 30
    dx = 1e-6
    dt = 0.15 * dx / (LIGHT_SPEED * np.sqrt(2.0))

    y_hz = np.arange(ny - 1, dtype=np.float32)[:, None]
    x_hz = np.arange(nx - 1, dtype=np.float32)[None, :]
    hz = jnp.asarray(
        np.sin(np.pi * (y_hz + 1.0) / ny) * np.sin(2.0 * np.pi * (x_hz + 1.0) / nx),
        dtype=jnp.float32,
    )
    ex = jnp.zeros((ny, nx - 1), dtype=jnp.float32)
    ey = jnp.zeros((ny - 1, nx), dtype=jnp.float32)

    energies = [_te_xy_energy(ex, ey, hz, dx=dx)]
    for _ in range(120):
        ex, ey, hz = _te_xy_pec_step(ex, ey, hz, dx=dx, dt=dt)
        energies.append(_te_xy_energy(ex, ey, hz, dx=dx))

    energies = np.asarray(energies)
    relative_excursion = (np.max(energies) - np.min(energies)) / np.mean(energies)
    assert relative_excursion < 0.03
