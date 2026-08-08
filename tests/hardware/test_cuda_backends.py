"""Hardware parity gates for the optional CUDA FFI wheel."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from beamz.simulation.backend import cuda_backend_status
from beamz.simulation.execute import initial_program_state
from tests.performance.h100_workloads import H100Workload

STATUS = cuda_backend_status()
pytestmark = pytest.mark.skipif(
    not STATUS.available,
    reason=STATUS.reason or "CUDA backend unavailable",
)


def _simulation_and_seed(
    *,
    cpml: bool,
    source: bool = True,
    monitor: bool = True,
    heterogeneous: bool = True,
):
    workload = H100Workload(
        name="cuda_hardware_parity",
        shape_zyx=(18, 20, 35),
        # Cross the Gaussian source peak at step 24 and accumulate multiple DFT
        # phases rather than validating only the near-zero leading envelope.
        timesteps=32,
        resolution=80e-9,
        pml_cells=3,
        heterogeneous=heterogeneous,
        cpml=cpml,
        source=source,
        monitor=monitor,
    )
    simulation = workload.build()
    program = simulation.compile(backend="jax")
    state = initial_program_state(
        program,
        t=float(simulation.time[0]),
        current_step=0,
        monitor_steps=workload.timesteps,
    )
    rng = np.random.default_rng(20260803)
    fields = {
        name: rng.normal(size=np.asarray(getattr(state, name)).shape).astype(np.float32)
        * 1e-5
        for name in ("ex", "ey", "ez", "hx", "hy", "hz")
    }
    return simulation, state._replace(**fields)


def _copy_state(state):
    return jax.tree_util.tree_map(lambda value: np.array(value, copy=True), state)


def _assert_state_close(reference, actual):
    reference_leaves = jax.tree_util.tree_leaves(reference)
    actual_leaves = jax.tree_util.tree_leaves(actual)
    assert len(reference_leaves) == len(actual_leaves)
    for expected, observed in zip(reference_leaves, actual_leaves, strict=True):
        expected = np.asarray(expected)
        observed = np.asarray(observed)
        if np.issubdtype(expected.dtype, np.inexact):
            # CPML memories span roughly five orders of magnitude. Near-zero
            # elements can differ by a few float32 ULPs even when the complete
            # recurrence agrees to sub-ppm scale, so keep the strict relative
            # check and derive an absolute floor from the leaf's dynamic range.
            scale = float(np.max(np.abs(expected), initial=0.0))
            np.testing.assert_allclose(
                observed,
                expected,
                rtol=3e-5,
                atol=max(3e-6, 1e-6 * scale),
            )
        else:
            np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("cpml", [False, True], ids=["pec", "cpml"])
def test_streamed_cuda_matches_jax_complete_state(cpml):
    simulation, state = _simulation_and_seed(cpml=cpml)
    reference = simulation.advance(
        state=_copy_state(state), num_steps=simulation.num_steps, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        num_steps=simulation.num_steps,
        backend="cuda_streamed",
    ).state

    _assert_state_close(reference, actual)


@pytest.mark.parametrize("cpml", [False, True], ids=["pec_graph", "cpml_graph"])
def test_streamed_cuda_owns_source_free_constraints(cpml):
    simulation, state = _simulation_and_seed(
        cpml=cpml,
        source=False,
        monitor=False,
        heterogeneous=False,
    )
    reference = simulation.advance(
        state=_copy_state(state), num_steps=simulation.num_steps, backend="jax"
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        num_steps=simulation.num_steps,
        backend="cuda_streamed",
    ).state

    _assert_state_close(reference, actual)


@pytest.mark.skipif(
    not STATUS.compute_capabilities
    or any(capability < 90 for capability in STATUS.compute_capabilities),
    reason="Hopper tiled target requires SM90+",
)
@pytest.mark.parametrize("cpml", [False, True], ids=["pec", "cpml"])
def test_hopper_cuda_matches_streamed_complete_state(cpml):
    simulation, state = _simulation_and_seed(cpml=cpml)
    reference = simulation.advance(
        state=_copy_state(state),
        num_steps=simulation.num_steps,
        backend="cuda_streamed",
    ).state
    actual = simulation.advance(
        state=_copy_state(state),
        num_steps=simulation.num_steps,
        backend="cuda_hopper",
    ).state

    _assert_state_close(reference, actual)
