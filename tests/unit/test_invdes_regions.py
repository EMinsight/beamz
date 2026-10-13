"""Region transforms/penalties are differentiable, SI-based, and reproducible."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.optimization import (
    CustomInitializationSpec,
    ErosionDilationPenalty,
    FilterProject,
    RandomInitializationSpec,
    TopologyDesignRegion,
)


def test_filter_and_penalty_gradient_and_constant_limits():
    values = np.random.default_rng(5).uniform(0.2, 0.8, (11, 12)).astype(np.float32)
    transform = FilterProject(230e-9, beta=4)
    penalty = ErosionDilationPenalty(230e-9, beta=8, weight=0.8)

    def objective(x):
        density = transform(x, 100e-9)
        return jnp.mean(density**2) - penalty(density, 100e-9)

    gradient = jax.grad(objective)(values)
    direction = gradient / jnp.linalg.norm(gradient)
    h = 0.003
    fd = (objective(values + h * direction) - objective(values - h * direction)) / (
        2 * h
    )
    np.testing.assert_allclose(fd, jnp.sum(gradient * direction), rtol=0.002)
    for value in (0, 1):
        constant = jnp.full((11, 12), value, dtype=jnp.float32)
        np.testing.assert_allclose(transform(constant, 100e-9), value, atol=3e-7)
        assert np.all(np.asarray(transform(constant, 100e-9)) >= 0)
        assert np.all(np.asarray(transform(constant, 100e-9)) <= 1)
        assert abs(float(penalty(constant, 100e-9))) < 1e-6
        assert np.isfinite(jax.grad(lambda x: penalty(x, 100e-9))(constant)).all()


def test_region_initialization_and_transform_order():
    r = TopologyDesignRegion(
        size=(1.2e-6, 1e-6, np.inf),
        center=(0, 0, 0),
        eps_bounds=(1, 4),
        pixel_size=100e-9,
        transformations=(FilterProject(150e-9, beta=2), FilterProject(100e-9, beta=3)),
        initialization_spec=RandomInitializationSpec(seed=3),
    )
    x = r.initial_parameters
    assert x.shape == (10, 12)
    np.testing.assert_array_equal(x, r.initial_parameters)
    np.testing.assert_allclose(
        r.material_density(x),
        r.transformations[1](r.transformations[0](x, 100e-9), 100e-9),
    )
    copy = CustomInitializationSpec(x)
    x[:] = 0
    assert np.any(copy.params != 0)
    with pytest.raises(ValueError):
        copy.create_parameters((3, 4))


def test_unseeded_random_initialization_is_stable_within_spec():
    spec = RandomInitializationSpec(min_value=0.45, max_value=0.55)
    assert spec.seed is not None
    np.testing.assert_array_equal(
        spec.create_parameters((4, 5)), spec.create_parameters((4, 5))
    )


def test_fixed_port_masks_survive_filtering_and_have_zero_gradients():
    solid = np.zeros((10, 12), dtype=bool)
    solid[4:6, :2] = True
    void = np.zeros_like(solid)
    void[:4, :2] = True
    region = TopologyDesignRegion(
        size=(1.2e-6, 1e-6, np.inf),
        center=(0, 0, 0),
        eps_bounds=(1, 4),
        pixel_size=100e-9,
        transformations=(FilterProject(250e-9, beta=4),),
        fixed_solid=solid,
        fixed_void=void,
    )
    params = jnp.full((10, 12), 0.5)
    density = np.asarray(region.material_density(params))
    assert np.all(density[solid] == 1)
    assert np.all(density[void] == 0)
    gradient = np.asarray(
        jax.grad(lambda x: jnp.sum(region.material_density(x)))(params)
    )
    assert np.all(gradient[solid | void] == 0)
    assert np.any(gradient[~(solid | void)] > 0)
    solid[:] = False
    assert region.fixed_solid.any()  # Input mutation cannot change the specification.
    assert not region.fixed_solid.flags.writeable
    with pytest.raises(ValueError, match="overlap"):
        region.updated_copy(fixed_void=region.fixed_solid)
    with pytest.raises(ValueError, match="shape"):
        region.updated_copy(fixed_void=np.zeros((1, 2), dtype=bool))


def test_topology_transform_fixed_mask_vjp_matches_finite_difference():
    from beamz.optimization.autodiff import (
        compute_parameter_gradient_vjp,
        transform_density,
    )

    solid = np.zeros((10, 12), dtype=bool)
    solid[4:6, :2] = True
    void = np.zeros_like(solid)
    void[:4, :2] = True
    options = dict(
        mask=jnp.ones((10, 12), dtype=bool),
        beta=4,
        eta=0.5,
        radius=2,
        fixed_solid=solid,
        fixed_void=void,
    )
    params = jnp.full((10, 12), 0.5)
    physical = np.asarray(transform_density(params, **options))
    assert np.all(physical[solid] == 1)
    assert np.all(physical[void] == 0)
    grad = compute_parameter_gradient_vjp(params, jnp.ones_like(params), **options)
    assert np.all(np.asarray(grad)[solid | void] == 0)
    direction = grad / jnp.linalg.norm(grad)
    h = 0.01
    fd = (
        transform_density(params + h * direction, **options).sum()
        - transform_density(params - h * direction, **options).sum()
    ) / (2 * h)
    np.testing.assert_allclose(fd, jnp.sum(grad * direction), rtol=0.002)


def test_volumetric_filter_is_isotropic_and_has_correct_vjp():
    values = np.random.default_rng(6).uniform(0.2, 0.8, (7, 7, 7)).astype(np.float32)
    transform = FilterProject(230e-9, beta=3)

    def fun(x):
        return jnp.mean(transform(x, 100e-9) ** 2)

    gradient = jax.grad(fun)(values)
    direction = gradient / jnp.linalg.norm(gradient)
    fd = (fun(values + 0.003 * direction) - fun(values - 0.003 * direction)) / 0.006
    np.testing.assert_allclose(fd, jnp.sum(gradient * direction), rtol=0.003)
    filtered = np.asarray(transform(values, 100e-9))
    np.testing.assert_allclose(
        transform(values.transpose(2, 0, 1), 100e-9),
        filtered.transpose(2, 0, 1),
        atol=1e-6,
    )
    for value in (0, 1):
        np.testing.assert_allclose(
            transform(jnp.full(values.shape, value, dtype=jnp.float32), 100e-9),
            value,
            atol=5e-7,
        )
    region = TopologyDesignRegion(
        size=(1e-6, 0.8e-6, 0.6e-6),
        center=(0, 0, 0),
        eps_bounds=(1, 4),
        pixel_size=100e-9,
        parameterization="volumetric",
    )
    assert region.params_shape == (6, 8, 10)
    with pytest.raises(ValueError, match="positive z"):
        region.updated_copy(size=(1e-6, 0.8e-6, 0))
