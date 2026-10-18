"""GPU propagation and derivatives with compact JAX material coefficients."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.simulation.differentiation import build_differentiable_scan
from beamz.simulation.execute import build_program_scan
from beamz.simulation.material_blocks import MaterialBlocks, material_values
from beamz.simulation.sharding import place_tree
from benchmarks.cuda_material_blocks import simulation
from benchmarks.jax_material_blocks import dense_coefficients, executable_memory
from tests.jax_compat import enable_x64
from tests.performance.test_cuda_tensor import compare_states, seeded_state


@pytest.fixture(autouse=True)
def require_gpu():
    if jax.devices()[0].platform != "gpu":
        pytest.skip("GPU compiler/allocator validation requires a GPU")
    with enable_x64(False):
        yield


@pytest.mark.parametrize("full,bulk", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("pec", [False, True])
def test_compressed_scan_preserves_dense_operator(full, bulk, graded, pec):
    p = simulation(66, 8, full=full, bulk=bulk, graded=graded, pec=pec).compile(
        num_steps=8, backend="jax"
    )
    state = seeded_state(p)
    dense = place_tree(p, dense_coefficients(p.coefficients, state))
    packed = place_tree(p, p.coefficients)
    source = packed.e_inverse_diagonal_x if full else packed.e_permittivity_x
    assert source.ndim == 0 if bulk else isinstance(source, MaterialBlocks)
    run = build_program_scan(p)
    a, b = jax.block_until_ready((run(state, dense), run(state, packed)))
    compare_states(a, b)


@pytest.mark.parametrize("full", [False, True])
def test_dynamic_pool_values_and_gradients(full):
    p = simulation(48, 3, full=full, graded=True).compile(num_steps=3, backend="jax")
    state = seeded_state(p)
    coeffs = place_tree(p, p.coefficients)
    prefix = "e_inverse_diagonal_" if full else "e_permittivity_"
    assert isinstance(getattr(coeffs, prefix + "x"), MaterialBlocks)
    run = build_differentiable_scan(p, checkpoint_interval=1)

    def loss(scale, dense):
        changes = {}
        for c in "xyz":
            value = getattr(coeffs, prefix + c)
            value = replace(value, values=value.values * scale)
            changes[prefix + c] = material_values(value) if dense else value
        result = run(state, coeffs._replace(**changes))
        return jnp.mean(result.ex**2)

    functions = [
        jax.jit(jax.value_and_grad(lambda scale, dense=dense: loss(scale, dense)))
        for dense in (False, True)
    ]
    for scale in (1.0, 0.8, 1.2):
        a, b = jax.block_until_ready(tuple(f(jnp.float32(scale)) for f in functions))
        np.testing.assert_allclose(a, b, rtol=2e-4, atol=1e-12)
    primal = jax.jit(lambda scale: loss(scale, False))
    high, low = jax.block_until_ready((primal(1.005), primal(0.995)))
    difference = (float(high) - float(low)) / 0.01
    derivative = float(functions[0](jnp.float32(1))[1])
    np.testing.assert_allclose(derivative, difference, rtol=1e-3, atol=1e-12)


@pytest.mark.parametrize("full", [False, True])
def test_compiled_memory_really_decreases(full):
    p = simulation(96, 5, full=full).compile(num_steps=5, backend="jax")
    state = seeded_state(p)
    memories = []
    for coeffs in (p.coefficients, dense_coefficients(p.coefficients, state)):
        executable = build_program_scan(p).lower(state, place_tree(p, coeffs)).compile()
        memories.append(executable_memory(executable)["total"])
    assert memories[0] < 0.95 * memories[1], memories


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("spatial_conductivity", [False, True])
def test_literal_blocks_and_conductivity(graded, spatial_conductivity):
    from beamz.simulation.material_blocks import compact_material

    p = simulation(40, 5, graded=graded).compile(num_steps=5, backend="jax")
    state = seeded_state(p)
    dense = dense_coefficients(p.coefficients, state)
    changes, compressed = {}, {}
    for c in "xyz":
        key = "e_permittivity_" + c
        epsilon = getattr(dense, key).copy()
        epsilon.ravel()[100:500] = np.linspace(2, 3, 400, dtype=np.float32)
        changes[key] = epsilon
        compressed[key] = compact_material(epsilon)
        assert isinstance(compressed[key], MaterialBlocks)
        assert compressed[key].headers.dtype == np.int32
        sigma = (
            np.random.default_rng(42).uniform(0, 1e-3, epsilon.shape).astype(np.float32)
            if spatial_conductivity
            else np.float32(1e-3)
        )
        changes["e_conductivity_" + c] = sigma
    dense = dense._replace(**changes)
    packed = dense._replace(**compressed)
    run = build_program_scan(p)
    a, b = jax.block_until_ready(
        (run(state, place_tree(p, dense)), run(state, place_tree(p, packed)))
    )
    compare_states(a, b)
