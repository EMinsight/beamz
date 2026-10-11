"""Composition semantics and immutable objective specifications."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamz.optimization import ModePower, SoftMinModePower, WeightedObjective
from beamz.optimization.data import ModeAmplitudes, ModeData


@pytest.mark.parametrize("objective_type", [ModePower, SoftMinModePower])
@pytest.mark.parametrize("frequencies", [None, (3e14, 2e14)])
@pytest.mark.parametrize("extra_frequency", [False, True])
def test_reference_power_aligns_frequency_coordinates(
    objective_type, frequencies, extra_frequency
):
    def monitor(values, coords):
        return ModeData(
            ModeAmplitudes(
                values[None, :, None],
                ("direction", "f", "mode_index"),
                {"f": coords, "direction": ("+",), "mode_index": (0,)},
            )
        )

    objective = objective_type(
        "output", reference_monitor="input", frequencies=frequencies
    )

    def evaluate(incoming, coords):
        data = {
            "output": monitor(jnp.array([2.0, 3.0]), (2e14, 3e14)),
            "input": monitor(incoming, coords),
        }
        return objective(data)

    expected, gradient = jax.value_and_grad(evaluate)(
        jnp.array([4.0, 6.0]), (2e14, 3e14)
    )
    actual, reordered_gradient = jax.value_and_grad(evaluate)(
        jnp.array([6.0, 4.0, 10.0] if extra_frequency else [6.0, 4.0]),
        (3e14, 2e14, 4e14) if extra_frequency else (3e14, 2e14),
    )
    np.testing.assert_allclose(actual, expected)
    expected_gradient = [gradient[1], gradient[0]] + ([0] if extra_frequency else [])
    np.testing.assert_allclose(reordered_gradient, expected_gradient)

    with pytest.raises(KeyError, match="not recorded"):
        evaluate(jnp.array([4.0]), (2e14,))


def test_objective_arithmetic_keeps_signs_and_spectral_identity():
    transmission = ModePower("output", frequencies=[2e14, 2.1e14])
    reflection = ModePower("input", direction="-")
    objective = 0.5 * (transmission + transmission) - 0.1 * reflection
    assert objective.terms == (transmission, transmission, reflection)
    assert objective.weights == (0.5, 0.5, -0.1)
    assert sum([transmission, reflection]).weights == (1.0, 1.0)
    assert hash(transmission)
    assert transmission.frequencies == (2e14, 2.1e14)


@pytest.mark.parametrize(
    "frequencies", [[], [0], [np.nan], [np.inf], [1e14, 1e14], [[1e14]], 1e14]
)
def test_rejects_invalid_frequency_selection(frequencies):
    with pytest.raises(ValueError, match="frequencies"):
        ModePower("output", frequencies=frequencies)


def test_rejects_invalid_combinations():
    term = ModePower("output")
    with pytest.raises(ValueError, match="finite"):
        term * np.nan
    with pytest.raises(ValueError, match="match"):
        WeightedObjective([term], [])
    with pytest.raises(ValueError, match="nonempty"):
        WeightedObjective([], [])
    with pytest.raises(TypeError):
        term + 1
    with pytest.raises(ValueError, match="reference_mode_index"):
        ModePower("output", reference_mode_index=-1)
