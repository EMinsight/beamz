"""Shared solver helper functions used by step and compiled engines."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from beamz.const import EPS_0, LIGHT_SPEED


@dataclass(frozen=True)
class CpmlTm2DxyTerms:
    sigma_h_terms: jnp.ndarray
    kappa_h_aux_terms: jnp.ndarray
    alpha_h_terms: jnp.ndarray
    kappa_h_direct_terms: jnp.ndarray
    sigma_e_terms: jnp.ndarray
    kappa_e_terms: jnp.ndarray
    alpha_e_terms: jnp.ndarray


@dataclass(frozen=True)
class Cpml3DTerms:
    a_h_terms: tuple[jnp.ndarray, ...]
    b_h_terms: tuple[jnp.ndarray, ...]
    inv_kappa_h_terms: tuple[jnp.ndarray, ...]
    a_e_terms: tuple[jnp.ndarray, ...]
    b_e_terms: tuple[jnp.ndarray, ...]
    inv_kappa_e_terms: tuple[jnp.ndarray, ...]


def full_tm_xy_component_to_centered_grid(component: str, values):
    """Project full-lattice TM samples onto centered monitor/sample points."""
    field = values
    if component == "Ez":
        if field.ndim != 2 or field.shape[0] < 2 or field.shape[1] < 2:
            raise ValueError(f"Ez full-TM field must be at least 2x2, got {field.shape}")
        return 0.25 * (
            field[:-1, :-1] + field[:-1, 1:] + field[1:, :-1] + field[1:, 1:]
        )
    if component == "Hx":
        if field.ndim != 2 or field.shape[1] < 2:
            raise ValueError(f"Hx full-TM field must have width >= 2, got {field.shape}")
        return 0.5 * (field[:, :-1] + field[:, 1:])
    if component == "Hy":
        if field.ndim != 2 or field.shape[0] < 2:
            raise ValueError(
                f"Hy full-TM field must have height >= 2, got {field.shape}"
            )
        return 0.5 * (field[:-1, :] + field[1:, :])
    raise ValueError(f"Unsupported full-TM centered-grid component {component!r}")


def is_full_tm_xy_lattice(ez, hx, hy) -> bool:
    """Return True when Ez/Hx/Hy follow BeamZ's physical xy-TM staggering."""
    return (
        getattr(ez, "ndim", None) == 2
        and getattr(hx, "ndim", None) == 2
        and getattr(hy, "ndim", None) == 2
        and hx.shape[0] == ez.shape[0] - 1
        and hx.shape[1] == ez.shape[1]
        and hy.shape[0] == ez.shape[0]
        and hy.shape[1] == ez.shape[1] - 1
    )


def embed_tm_xy_h_terms(
    term0: jnp.ndarray, term1: jnp.ndarray, ez_shape: tuple[int, int]
) -> jnp.ndarray:
    out = jnp.zeros((2, *ez_shape), dtype=term0.dtype)
    out = out.at[0, :-1, :].set(term0)
    out = out.at[1, :, :-1].set(term1)
    return out


def cpml_precompute_native_terms(
    sigma_terms: tuple[jnp.ndarray, ...],
    kappa_terms: tuple[jnp.ndarray, ...],
    alpha_terms: tuple[jnp.ndarray, ...],
    dt: float,
) -> tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]]:
    a_terms = []
    b_terms = []
    inv_kappa_terms = []
    dt_arr = jnp.asarray(dt, dtype=jnp.float32)
    eps0 = jnp.asarray(EPS_0, dtype=jnp.float32)
    for sigma, kappa, alpha in zip(sigma_terms, kappa_terms, alpha_terms, strict=True):
        sigma = jnp.asarray(sigma, dtype=jnp.float32)
        kappa = jnp.maximum(jnp.asarray(kappa, dtype=jnp.float32), 1.0)
        alpha = jnp.asarray(alpha, dtype=jnp.float32)
        decay = (sigma / kappa + alpha) * (dt_arr / eps0)
        b = jnp.expm1(-decay) + 1.0
        denom = sigma + kappa * alpha
        a = jnp.nan_to_num(
            ((b - 1.0) * sigma) / jnp.maximum(denom * kappa, 1e-30),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        a_terms.append(a)
        b_terms.append(b)
        inv_kappa_terms.append(1.0 / kappa)
    return tuple(a_terms), tuple(b_terms), tuple(inv_kappa_terms)


def build_tm_xy_cpml_terms(
    tm_xy: dict[str, jnp.ndarray] | None,
    *,
    ez_shape: tuple[int, int],
) -> CpmlTm2DxyTerms | None:
    if tm_xy is None:
        return None
    sigma_hx = jnp.asarray(tm_xy["Hx_y_sigma"], dtype=jnp.float32)
    kappa_hx = jnp.asarray(tm_xy["Hx_y_kappa"], dtype=jnp.float32)
    alpha_hx = jnp.asarray(tm_xy["Hx_y_alpha"], dtype=jnp.float32)
    sigma_hy = jnp.asarray(tm_xy["Hy_x_sigma"], dtype=jnp.float32)
    kappa_hy = jnp.asarray(tm_xy["Hy_x_kappa"], dtype=jnp.float32)
    alpha_hy = jnp.asarray(tm_xy["Hy_x_alpha"], dtype=jnp.float32)
    sigma_ez_x = jnp.asarray(tm_xy["Ez_x_sigma"], dtype=jnp.float32)
    kappa_ez_x = jnp.asarray(tm_xy["Ez_x_kappa"], dtype=jnp.float32)
    alpha_ez_x = jnp.asarray(tm_xy["Ez_x_alpha"], dtype=jnp.float32)
    sigma_ez_y = jnp.asarray(tm_xy["Ez_y_sigma"], dtype=jnp.float32)
    kappa_ez_y = jnp.asarray(tm_xy["Ez_y_kappa"], dtype=jnp.float32)
    alpha_ez_y = jnp.asarray(tm_xy["Ez_y_alpha"], dtype=jnp.float32)
    return CpmlTm2DxyTerms(
        sigma_h_terms=embed_tm_xy_h_terms(sigma_hx, sigma_hy, ez_shape),
        kappa_h_aux_terms=embed_tm_xy_h_terms(kappa_hx, kappa_hy, ez_shape),
        alpha_h_terms=embed_tm_xy_h_terms(alpha_hx, alpha_hy, ez_shape),
        kappa_h_direct_terms=embed_tm_xy_h_terms(
            kappa_ez_y[:-1, :], kappa_ez_x[:, :-1], ez_shape
        ),
        sigma_e_terms=jnp.stack((sigma_ez_x, sigma_ez_y), axis=0),
        kappa_e_terms=jnp.stack((kappa_ez_x, kappa_ez_y), axis=0),
        alpha_e_terms=jnp.stack((alpha_ez_x, alpha_ez_y), axis=0),
    )


def build_cpml_3d_terms(
    pml_data: dict[str, jnp.ndarray] | None,
    *,
    dt: float,
) -> Cpml3DTerms | None:
    if pml_data is None:
        return None
    sigma_h_terms = (
        jnp.asarray(pml_data["cpml3d_Hxy_sigma"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hxz_sigma"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hyz_sigma"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hyx_sigma"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hzx_sigma"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hzy_sigma"], dtype=jnp.float32),
    )
    kappa_h_terms = (
        jnp.asarray(pml_data["cpml3d_Hxy_kappa"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hxz_kappa"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hyz_kappa"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hyx_kappa"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hzx_kappa"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hzy_kappa"], dtype=jnp.float32),
    )
    alpha_h_terms = (
        jnp.asarray(pml_data["cpml3d_Hxy_alpha"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hxz_alpha"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hyz_alpha"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hyx_alpha"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hzx_alpha"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Hzy_alpha"], dtype=jnp.float32),
    )
    sigma_e_terms = (
        jnp.asarray(pml_data["cpml3d_Exy_sigma"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Exz_sigma"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Eyz_sigma"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Eyx_sigma"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Ezx_sigma"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Ezy_sigma"], dtype=jnp.float32),
    )
    kappa_e_terms = (
        jnp.asarray(pml_data["cpml3d_Exy_kappa"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Exz_kappa"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Eyz_kappa"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Eyx_kappa"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Ezx_kappa"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Ezy_kappa"], dtype=jnp.float32),
    )
    alpha_e_terms = (
        jnp.asarray(pml_data["cpml3d_Exy_alpha"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Exz_alpha"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Eyz_alpha"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Eyx_alpha"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Ezx_alpha"], dtype=jnp.float32),
        jnp.asarray(pml_data["cpml3d_Ezy_alpha"], dtype=jnp.float32),
    )
    a_h_terms, b_h_terms, inv_kappa_h_terms = cpml_precompute_native_terms(
        sigma_h_terms, kappa_h_terms, alpha_h_terms, dt
    )
    a_e_terms, b_e_terms, inv_kappa_e_terms = cpml_precompute_native_terms(
        sigma_e_terms, kappa_e_terms, alpha_e_terms, dt
    )
    return Cpml3DTerms(
        a_h_terms=a_h_terms,
        b_h_terms=b_h_terms,
        inv_kappa_h_terms=inv_kappa_h_terms,
        a_e_terms=a_e_terms,
        b_e_terms=b_e_terms,
        inv_kappa_e_terms=inv_kappa_e_terms,
    )


def poynting_magnitude_2d(ez, hx, hy):
    """Return |E x H| for 2D TM monitor samples."""
    sx = -ez * hy
    sy = ez * hx
    return (sx * sx + sy * sy) ** 0.5


def poynting_magnitude_3d(ex, ey, ez, hx, hy, hz):
    """Return |E x H| for 3D monitor samples."""
    sx = ey * hz - ez * hy
    sy = ez * hx - ex * hz
    sz = ex * hy - ey * hx
    return (sx * sx + sy * sy + sz * sz) ** 0.5


def meep_dft_sample_scale(weight, base_dt, record_interval, length_unit):
    """Return Meep-style DFT normalization scale for one sample weight."""
    return weight * (
        base_dt
        * record_interval
        * LIGHT_SPEED
        / length_unit
        / np.sqrt(2.0 * np.pi)
    )
