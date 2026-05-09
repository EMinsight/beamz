import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation.boundaries import (
    _cpml_ab_from_profiles,
    build_h_boundary_views_for_e_3d,
    cpml_curl_e_to_h_3d,
    cpml_curl_h_to_e_3d,
)
from beamz.simulation.ops import curl_e_to_h_3d, curl_h_to_e_3d

pytestmark = pytest.mark.unit


def test_curl_e_to_h_3d_linear_field_has_constant_z_component():
    nz = ny = nx = 6
    ex = jnp.broadcast_to(
        jnp.arange(ny, dtype=jnp.float32)[None, :, None],
        (nz, ny, nx - 1),
    )
    ey = -jnp.broadcast_to(
        jnp.arange(nx, dtype=jnp.float32)[None, None, :],
        (nz, ny - 1, nx),
    )
    ez = jnp.zeros((nz - 1, ny, nx), dtype=jnp.float32)

    curl_ex, curl_ey, curl_ez = curl_e_to_h_3d(ex, ey, ez, resolution=1.0)

    np.testing.assert_allclose(np.asarray(curl_ex), 0.0)
    np.testing.assert_allclose(np.asarray(curl_ey), 0.0)
    np.testing.assert_allclose(np.asarray(curl_ez), -2.0, atol=1e-6)


def test_curl_h_to_e_3d_linear_field_has_constant_y_component():
    nz = ny = nx = 6
    hx = jnp.broadcast_to(
        jnp.arange(nz - 1, dtype=jnp.float32)[:, None, None],
        (nz - 1, ny - 1, nx),
    )
    hy = jnp.zeros((nz - 1, ny, nx - 1), dtype=jnp.float32)
    hz = -jnp.broadcast_to(
        jnp.arange(nx - 1, dtype=jnp.float32)[None, None, :],
        (nz, ny - 1, nx - 1),
    )

    boundary_views = build_h_boundary_views_for_e_3d(hx, hy, hz, boundaries=[])
    curl_hx, curl_hy, curl_hz = curl_h_to_e_3d(
        hx,
        hy,
        hz,
        resolution=1.0,
        ex_shape=(nz, ny, nx - 1),
        ey_shape=(nz, ny - 1, nx),
        ez_shape=(nz - 1, ny, nx),
        boundary_views=boundary_views,
    )

    np.testing.assert_allclose(np.asarray(curl_hx)[1:-1, 1:-1, :], 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(curl_hy)[1:-1, :, 1:-1], 2.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(curl_hz)[:, 1:-1, 1:-1], 0.0, atol=1e-6)


def _cpml_coefficients(shapes, *, sigma_value=0.5, kappa_value=1.5, alpha_value=0.1, dt=0.05):
    a_terms = []
    b_terms = []
    inv_kappa_terms = []
    for shape in shapes:
        sigma = jnp.full(shape, sigma_value, dtype=jnp.float32)
        kappa = jnp.full(shape, kappa_value, dtype=jnp.float32)
        alpha = jnp.full(shape, alpha_value, dtype=jnp.float32)
        a_term, b_term = _cpml_ab_from_profiles(sigma, kappa, alpha, dt)
        a_terms.append(a_term)
        b_terms.append(b_term)
        inv_kappa_terms.append(1.0 / kappa)
    return tuple(a_terms), tuple(b_terms), tuple(inv_kappa_terms)


def test_cpml_curl_e_to_h_3d_updates_psi_terms():
    ex = jnp.arange(3 * 4 * 4, dtype=jnp.float32).reshape(3, 4, 4)
    ey = jnp.arange(3 * 3 * 5, dtype=jnp.float32).reshape(3, 3, 5)
    ez = jnp.arange(2 * 4 * 5, dtype=jnp.float32).reshape(2, 4, 5)

    term_shapes = (
        (2, 3, 5),
        (2, 3, 5),
        (2, 4, 4),
        (2, 4, 4),
        (3, 3, 4),
        (3, 3, 4),
    )
    a_terms, b_terms, inv_kappa_terms = _cpml_coefficients(term_shapes)
    psi_init = tuple(jnp.zeros(shape, dtype=jnp.float32) for shape in term_shapes)

    curl_hx, curl_hy, curl_hz, psi_updated = cpml_curl_e_to_h_3d(
        ex,
        ey,
        ez,
        resolution=0.2,
        a_h_terms=a_terms,
        b_h_terms=b_terms,
        inv_kappa_h_terms=inv_kappa_terms,
        psi_h_terms=psi_init,
    )

    assert curl_hx.shape == term_shapes[0]
    assert curl_hy.shape == term_shapes[2]
    assert curl_hz.shape == term_shapes[4]
    assert any(not jnp.allclose(term, 0.0) for term in psi_updated)


def test_cpml_curl_h_to_e_3d_updates_psi_terms():
    hx = jnp.arange(2 * 3 * 5, dtype=jnp.float32).reshape(2, 3, 5)
    hy = jnp.arange(2 * 4 * 4, dtype=jnp.float32).reshape(2, 4, 4)
    hz = jnp.arange(3 * 3 * 4, dtype=jnp.float32).reshape(3, 3, 4)

    term_shapes = (
        (3, 4, 4),
        (3, 4, 4),
        (3, 3, 5),
        (3, 3, 5),
        (2, 4, 5),
        (2, 4, 5),
    )
    a_terms, b_terms, inv_kappa_terms = _cpml_coefficients(term_shapes)
    psi_init = tuple(jnp.zeros(shape, dtype=jnp.float32) for shape in term_shapes)

    curl_ex, curl_ey, curl_ez, psi_updated = cpml_curl_h_to_e_3d(
        hx,
        hy,
        hz,
        resolution=0.2,
        a_e_terms=a_terms,
        b_e_terms=b_terms,
        inv_kappa_e_terms=inv_kappa_terms,
        psi_e_terms=psi_init,
    )

    assert curl_ex.shape == term_shapes[0]
    assert curl_ey.shape == term_shapes[2]
    assert curl_ez.shape == term_shapes[4]
    assert any(not jnp.allclose(term, 0.0) for term in psi_updated)
