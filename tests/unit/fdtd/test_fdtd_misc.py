import numpy as np
import pytest

from beamz.simulation.boundaries import (
    initialize_full_pec_3d_state,
    sync_compact_fields_from_full_pec_3d,
    sync_full_pec_3d_from_compact,
)
from beamz.simulation.fields import Fields

pytestmark = pytest.mark.unit


def _uniform_3d_fields(n=4):
    permittivity = np.ones((n, n, n), dtype=np.float32)
    conductivity = np.zeros((n, n, n), dtype=np.float32)
    permeability = np.ones((n, n, n), dtype=np.float32)
    return Fields(
        permittivity=permittivity,
        conductivity=conductivity,
        permeability=permeability,
        resolution=1.0,
    )


def test_sync_full_pec_3d_from_compact_copies_owned_interior_with_masks():
    fields = _uniform_3d_fields()
    state = initialize_full_pec_3d_state(fields)

    for index, component in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), start=1):
        values = np.arange(np.prod(getattr(fields, component).shape), dtype=np.float32)
        values = values.reshape(getattr(fields, component).shape) + index
        setattr(fields, component, getattr(fields, component).at[:].set(values))

    sync_full_pec_3d_from_compact(fields, state)

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        compact = np.asarray(getattr(fields, component))
        full = np.asarray(getattr(state, component))
        mask = np.asarray(state.masks[component])

        expected = np.where(mask[:-1, :-1, :-1], 0.0, compact)
        np.testing.assert_allclose(full[:-1, :-1, :-1], expected)


def test_sync_full_then_compact_round_trip_restores_masked_compact_views():
    fields = _uniform_3d_fields()
    state = initialize_full_pec_3d_state(fields)

    originals = {}
    for index, component in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), start=1):
        values = np.arange(np.prod(getattr(fields, component).shape), dtype=np.float32)
        values = values.reshape(getattr(fields, component).shape) - index
        setattr(fields, component, getattr(fields, component).at[:].set(values))
        originals[component] = values

    sync_full_pec_3d_from_compact(fields, state)

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        setattr(fields, component, getattr(fields, component).at[:].set(99.0))

    sync_compact_fields_from_full_pec_3d(fields, state)

    for component in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        mask = np.asarray(state.masks[component][:-1, :-1, :-1])
        expected = np.where(mask, 0.0, originals[component])
        np.testing.assert_allclose(np.asarray(getattr(fields, component)), expected)
